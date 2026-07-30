"""MLP learning curves on EFP features built for a multi-GPU cluster, with an optional muP flag.

This code computes a learning curve of test MSE vs training-set
size T for a width-`W` ReLU MLP, with Navg independent fits averaged at small T —
and adds:

1. **Task-parallel multi-GPU.**  The experiment is a large set of *independent*
   fits indexed by (T, repeat).  Rather than data-parallel (DDP / Keras
   MirroredStrategy), which would split one small model's batch across GPUs and
   leave them under-utilized with sync overhead, we treat each (T, repeat) as a
   job and farm the jobs out to the GPUs through a shared queue — one model per
   GPU, no inter-GPU communication.  Every GPU stays busy on a different fit, so
   throughput scales with the number of GPUs across the loops over training
   sizes and averages.  (For the few largest-T fits the model is still small
   enough to train on a single GPU; data-parallelism would not help.)
   Multi-node clusters: split the job list with --num-shards / --shard-id, one
   shard per SLURM array task (each task then task-parallelizes over its node's
   GPUs).

2. **muP (Maximal Update Parametrization) flag.**  --mup switches the readout
   multiplier and per-layer learning rates from standard parametrization (SP) to
   muP, so a *wide* MLP trains in the rich feature-learning limit instead of the
   lazy / NTK / kernel limit reached by SP.  Run the same wide --width with and
   without --mup to compare the two limits.  Implemented self-contained (no `mup`
   dependency) for --optimizer sgd and the normalized-update family
   adam/adamw/adagrad; the per-layer LR rule differs between the two classes
   (see build_param_groups for the exact table and derivation).  AdamW's
   decoupled weight decay is rescaled per group so the relative shrinkage
   lr_group*wd is width-uniform.

The data, model (width 256, 3 inner layers, He init, ReLU, linear output), MSE
loss, 200-epoch / patience-20 early stopping, and the recorded result dict all
mirror the original.  Default optimizer is now AdamW(1e-3, wd=0.01) + cosine,
chosen by the scans in dnn_prototyping/ (see Cfg comments for the numbers).  Best weights (min val loss) are kept in
memory and restored — no per-improvement .h5 checkpoint I/O.  The one optimizer
change from the original is a default **cosine LR decay** to ~0 over --epochs
(--lr-schedule constant restores the fixed-lr behavior): a fixed Adam lr orbits
sharp minima at step size ~lr and can climb back out late in training, so
annealing lr->0 is what lets the training loss descend monotonically and settle.

The features and labels are read from a SINGLE file each (--x-file X.npy,
--y-file Y.npy); a fixed split holds out --n-test test rows and --n-val
validation rows, and the rest is the training pool that the T-sweep subsamples.
To make the MLP comparable to the ridge-regression notebook on the SAME log-EFP
features and thrust labels:
  --standardize        standardize X and center y by training-pool stats
                       (replicates ridge_learning_curve(scale=True))
  --no-shuffle-split   test = the last --n-test rows (matches the ridge test set)
A learning curve (loss vs epoch, with the early-stop epoch and saved test loss
marked) is dumped per T to <out>_curves.png / <out>_curves.pkl (disable with
--no-curves) so the reported loss can be eyeballed for reasonableness.
One representative trained network per training-set size T -- the rep-0 fit, at
its best-val (restored) weights -- is saved to <out>_model_T<T>.pt (disable with
--no-save-models).  Each .pt bundles the state_dict with the architecture kwargs
and the train-pool standardization stats (x mean/std, y mean), so the model can
be rebuilt with MLP(**payload['arch']) and applied to raw feature files.

Usage (single node, all visible GPUs):
    python EFP_DNN_GPU_batch_torch.py --x-file X.npy --y-file Y.npy \
        --n-val 10000 --n-test 30000 [--mup --width 1024]
SLURM array (one task per node):
    python EFP_DNN_GPU_batch_torch.py --x-file X.npy --y-file Y.npy ... \
        --num-shards $SLURM_ARRAY_TASK_COUNT --shard-id $SLURM_ARRAY_TASK_ID
Local smoke test (synthetic data, no files needed):
    python EFP_DNN_GPU_batch_torch.py --smoke-test
"""

from __future__ import annotations

import argparse
import copy
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Model: width-W MLP, SP or muP                                                 #
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """ReLU MLP: Linear(in,W) -> [Linear(W,W)] x n_hidden -> Linear(W,out).

    Matches the original ``myMLP`` (He fan-in init std = sqrt(2/fan_in), zero
    bias, linear output).  ``mup=True`` changes (i) the readout output is
    multiplied by base_width/width in forward(), (ii) the readout INIT std
    becomes width-independent (sqrt(2/base_width), coinciding with SP at
    width=base_width), and (iii) the per-layer learning rates (handled in
    build_param_groups, optimizer-dependent).

    Why (ii) matters -- the historical muP bug: with the SP He readout init
    (entries ~1/sqrt(width)) *and* the 1/m multiplier, the backprop signal
    mult*w reaching the hidden layers is Theta(width^-3/2) instead of muP's
    Theta(1/width).  Adam-family optimizers self-heal in one step (normalized
    updates regrow the readout scale), but under SGD the hidden layers then
    train only at second order in lr and feature learning is suppressed as
    width grows (coordinate check: activation updates ~ width^-1 instead of
    width^0).  A width-independent readout init (entries Theta(1)) restores
    the Theta(1/width) signal; the multiplier still guarantees the init output
    vanishes as sqrt(base_width/width), so the learned part dominates.
    """

    def __init__(self, in_dim: int, width: int = 256, n_hidden: int = 3,
                 out_dim: int = 1, act: str = "relu",
                 mup: bool = False, base_width: int = 64):
        super().__init__()
        self.in_dim, self.width, self.n_hidden = in_dim, width, n_hidden
        self.mup, self.base_width = mup, base_width
        self.readout_mult = (base_width / width) if mup else 1.0

        self.input = nn.Linear(in_dim, width)
        self.hidden = nn.ModuleList(
            [nn.Linear(width, width) for _ in range(n_hidden)])
        self.output = nn.Linear(width, out_dim)
        self.act_name = act
        self.act = {"relu": nn.functional.relu,
                    "gelu": nn.functional.gelu,
                    "tanh": torch.tanh}[act]
        self.reset_parameters()

    def reset_parameters(self):
        def he(layer, fan_in):
            if self.act_name=="relu":
                nn.init.normal_(layer.weight, 0.0, np.sqrt(2.0 / fan_in))
            else:
                nn.init.normal_(layer.weight, 0.0, np.sqrt(1.0 / fan_in))
            nn.init.zeros_(layer.bias)
        he(self.input, self.in_dim)
        for lin in self.hidden:
            he(lin, self.width)
        # muP readout: width-INDEPENDENT init std (= SP at width=base_width) so
        # the backprop signal mult*w is Theta(1/width); see class docstring.
        he(self.output, self.base_width if self.mup else self.width)

    def forward(self, x):
        h = self.act(self.input(x))
        for lin in self.hidden:
            h = self.act(lin(h))
        return self.output(h) * self.readout_mult


# Optimizers whose per-entry update magnitude is ~lr regardless of the gradient
# scale (gradient-normalized updates).  They share one muP LR table; SGD (update
# proportional to the gradient) has its own.
NORMALIZED_OPTS = ("adam", "adamw", "adagrad")


def build_param_groups(model: MLP, base_lr: float, optimizer: str = "adam",
                       weight_decay: float = 0.0):
    """Per-parameter-group learning rates (and AdamW weight decay) for SP or muP.

    SP: every parameter uses base_lr (matches the original single-LR optimizer).

    muP (width multiplier m = width/base_width): fan-in init for all layers and
    the 1/m readout output multiplier in forward() are optimizer-independent and
    set elsewhere; only the per-layer LR depends on the optimizer.  The LR keeps
    each layer's contribution to the activations / output Theta(1) per step (the
    feature-learning condition).  The rule differs by optimizer class because the
    SGD update is proportional to the gradient while Adam/AdamW/AdaGrad updates
    are gradient-normalized (per-entry magnitude ~lr independent of grad scale):

        layer                 Adam/AdamW/AdaGrad    SGD
        input  weights        base_lr               base_lr * m
        hidden weights        base_lr / m           base_lr
        output weights        base_lr               base_lr * m
        input+hidden biases   base_lr               base_lr * m
        output bias           base_lr               base_lr

    Derivation sketch (width n, m = n/n0). Backprop signal df/dz at hidden
    preactivations is Theta(1/n) (readout multiplier 1/m x readout weights), so
    SGD gradients of input weights and inner biases are Theta(1/n): moving their
    preactivations Theta(1) per step needs lr ~ m.  Hidden weight updates are
    outer-product correlated with h, so (dW h)_i picks up a factor n: SGD grad
    Theta(1/n) x lr Theta(1) x n = Theta(1).  The readout's 1/m suppression lives
    in the forward multiplier, so SGD scales its LR UP by m to compensate; its
    bias sees a Theta(1) gradient and stays at base_lr.  For the normalized
    family the per-entry update is ~lr, so only the hidden layers (whose update
    correlates over n entries of h) need the 1/m; everything else stays at
    base_lr.  NOTE: earlier versions kept ALL biases at base_lr under SGD; that
    freezes input/hidden biases relative to the weights as width grows (their
    gradients are Theta(1/n)) and is fixed here.

    weight_decay (AdamW only; 0 for the others): torch applies decoupled decay
    as W -= lr_group * wd * W, so with per-layer LRs the relative shrinkage
    would be width-dependent.  We rescale per group (wd_g = wd * base_lr/lr_g)
    to keep lr_g * wd_g = base_lr * wd uniform across groups and widths."""
    wd = weight_decay if optimizer == "adamw" else 0.0
    if not model.mup:
        return [{"params": list(model.parameters()), "lr": base_lr,
                 "weight_decay": wd}]
    m = model.width / model.base_width
    hidden_w = [lin.weight for lin in model.hidden]
    in_out_w = [model.input.weight, model.output.weight]
    inner_b = [model.input.bias] + [lin.bias for lin in model.hidden]
    out_b = [model.output.bias]
    if optimizer in NORMALIZED_OPTS:
        return [{"params": hidden_w, "lr": base_lr / m, "weight_decay": wd * m},
                {"params": in_out_w + inner_b + out_b, "lr": base_lr,
                 "weight_decay": wd}]
    if optimizer == "sgd":
        return [{"params": in_out_w + inner_b, "lr": base_lr * m,
                 "weight_decay": 0.0},
                {"params": hidden_w + out_b, "lr": base_lr, "weight_decay": 0.0}]
    raise ValueError(f"unknown optimizer {optimizer!r}")


# --------------------------------------------------------------------------- #
# Data                                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class Data:
    Xtr: torch.Tensor
    Ytr: torch.Tensor
    Xval: torch.Tensor
    Yval: torch.Tensor
    Xte: torch.Tensor
    Yte: torch.Tensor
    # train-pool preprocessing stats (None unless --standardize); saved with each
    # model so raw features can be mapped to network inputs at load time.
    x_mean: Optional[np.ndarray] = None
    x_std: Optional[np.ndarray] = None
    y_mean: Optional[float] = None

    @property
    def in_dim(self):
        return self.Xtr.shape[1]


def load_data(cfg, device) -> Data:
    """Load the full X / Y from one file each (or synthetic data) and carve out a
    held-out test set (n_test) and validation set (n_val); the remainder is the
    training pool that run_job subsamples.

    Split convention (test is always the LAST block):
      train = first (N - n_test - n_val) rows, val = next n_val, test = last n_test.
    With --no-shuffle-split the row order is untouched, so the test set is the
    literal last n_test rows -- matching the ridge notebook's
    ``X[-n_test:]`` convention (so both score on the same points).  Otherwise a
    single permutation seeded by ``cfg.split_seed`` is applied first; that split
    is IDENTICAL across all workers and runs and is independent of the per-fit
    ``--seed`` used for subsampling/init.

    With --standardize the features are standardized by the TRAINING-POOL mean/std
    and the labels are centered by the training-pool mean (std not rescaled),
    applied to all splits -- replicating ``ridge_learning_curve(scale=True)`` in
    thrust_scaling_utils.py, so the MLP and ridge see the same preprocessing."""
    if cfg.smoke_test:
        rng = np.random.default_rng(0)
        d = cfg.smoke_dim
        N = cfg.smoke_train + cfg.n_val + cfg.n_test
        X = rng.standard_normal((N, d)).astype(np.float32)
        w = rng.standard_normal(d).astype(np.float32)
        # a fixed nonlinear target so MLPs of different T/width differ
        Y = ((np.sin(X @ w) + 0.3 * X[:, 0] ** 2) > 0).astype(np.float32)[:, None]
    else:
        X = np.asarray(np.load(cfg.x_file), dtype=np.float32)
        Y = np.asarray(np.load(cfg.y_file), dtype=np.float32)
        if Y.ndim == 1:
            Y = Y[:, None]
        if len(X) != len(Y):
            raise SystemExit(f"X and Y have different lengths: {len(X)} vs {len(Y)}")

    N = X.shape[0]
    if cfg.n_test + cfg.n_val >= N:
        raise SystemExit(f"n_test ({cfg.n_test}) + n_val ({cfg.n_val}) "
                         f">= dataset size ({N}); leave rows for training.")
    perm = (np.arange(N) if cfg.no_shuffle_split
            else np.random.default_rng(cfg.split_seed).permutation(N))
    n_tr = N - cfg.n_test - cfg.n_val
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + cfg.n_val], perm[n_tr + cfg.n_val:]

    Xtr, Ytr = X[tr], Y[tr]
    Xva, Yva = X[va], Y[va]
    Xte, Yte = X[te], Y[te]
    x_mean = x_std = y_mean = None
    if cfg.standardize:
        mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True)
        sd[sd == 0] = 1.0
        Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
        ymu = Ytr.mean()                     # center y (std not rescaled), as in ridge
        Ytr, Yva, Yte = Ytr - ymu, Yva - ymu, Yte - ymu
        x_mean, x_std, y_mean = mu[0].copy(), sd[0].copy(), float(ymu)

    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    return Data(Xtr=t(Xtr), Ytr=t(Ytr), Xval=t(Xva), Yval=t(Yva),
                Xte=t(Xte), Yte=t(Yte),
                x_mean=x_mean, x_std=x_std, y_mean=y_mean)


# --------------------------------------------------------------------------- #
# Train one model (one fit on one device)                                       #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_mse(model, X, Y, chunk=65536):
    model.eval()
    tot, n = 0.0, X.shape[0]
    for s in range(0, n, chunk):
        pred = model(X[s:s + chunk])
        tot += nn.functional.mse_loss(pred, Y[s:s + chunk], reduction="sum").item()
    return tot / n


def train_one(data: Data, idx: torch.Tensor, cfg, batch_size: int,
              seed: int, device):
    """Train a fresh MLP on data[idx]; early-stop on val MSE; return
    (test_mse, train_mse, n_params, history, state) using the best (min-val)
    weights, where ``state`` is the restored state_dict on CPU (for
    --save-models).

    ``history`` records, per epoch, the validation MSE and the epoch-mean
    training (mini-batch) MSE, plus the best (restored) epoch and the epoch at
    which training stopped -- enough to plot the learning curve and check that
    the saved test loss matches the floor of the val curve at the best epoch."""
    torch.manual_seed(seed)
    Xtr, Ytr = data.Xtr[idx], data.Ytr[idx]
    model = MLP(data.in_dim, cfg.width, cfg.hidden_layers, out_dim=Ytr.shape[1],
                act=cfg.act, mup=cfg.mup, base_width=cfg.mup_base_width).to(device)
    groups = build_param_groups(model, cfg.lr, cfg.optimizer, cfg.weight_decay)
    if cfg.optimizer == "adam":
        opt = torch.optim.Adam(groups)
    elif cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(groups)   # per-group lr/wd set in build_param_groups
    elif cfg.optimizer == "adagrad":
        opt = torch.optim.Adagrad(groups)
    elif cfg.optimizer == "sgd":
        opt = torch.optim.SGD(groups, momentum=cfg.momentum)
    else:
        raise ValueError(f"unknown optimizer {cfg.optimizer!r}")
    # Cosine LR decay (default): each param group anneals from its OWN initial lr
    # (so the muP per-layer ratios are preserved) to ~0 over --epochs.  Driving
    # lr->0 lets the optimizer settle into the minimum instead of orbiting it at
    # step size ~lr, so the training loss descends monotonically.
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs,
                                                        eta_min=0.0)
             if cfg.lr_schedule == "cosine" else None)
    loss_fn = nn.MSELoss()

    n = Xtr.shape[0]
    best_val, best_state, best_epoch, wait = float("inf"), None, 0, 0
    val_hist, train_hist, lr_hist = [], [], []
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, generator=gen, device=device)
        ep_loss, nb = 0.0, 0
        for s in range(0, n, batch_size):
            b = perm[s:s + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(Xtr[b]), Ytr[b])
            loss.backward()
            opt.step()
            ep_loss += loss.item(); nb += 1
        val = eval_mse(model, data.Xval, data.Yval)
        val_hist.append(val)
        train_hist.append(ep_loss / max(nb, 1))
        lr_hist.append(max(g["lr"] for g in opt.param_groups))  # nominal lr this epoch
        if val < best_val - 1e-7:
            best_val, best_epoch, wait = val, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= cfg.patience:
                break
        if sched is not None:
            sched.step()
    if best_state is not None:
        model.load_state_dict(best_state)
    test_mse = eval_mse(model, data.Xte, data.Yte)
    train_mse = eval_mse(model, Xtr, Ytr)
    n_params = sum(p.numel() for p in model.parameters())
    history = {"val": val_hist, "train_batch": train_hist, "lr": lr_hist,
               "best_epoch": best_epoch, "stopped_epoch": len(val_hist) - 1,
               "best_val": best_val, "test": test_mse, "train": train_mse}
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return test_mse, train_mse, n_params, history, state


def save_model(state, job: "Job", cfg, data: Data, history=None):
    """Save the representative trained network for one training-set size to
    <out-stem>_model_T<T>.pt: the best-val state_dict plus everything needed to
    rebuild it and reproduce its predictions from a raw feature file --
    architecture kwargs (payload['arch'] unpacks straight into MLP(...)), the
    train-pool standardization stats, and the training provenance.  Load with:

        payload = torch.load(f, map_location='cpu')
        model = MLP(**payload['arch']); model.load_state_dict(payload['state_dict'])
        X = (X_raw - payload['preproc']['x_mean']) / payload['preproc']['x_std']
        pred = model(torch.as_tensor(X)) + payload['preproc']['y_mean']
    """
    stem = cfg.out[:-4] if cfg.out.endswith(".pkl") else cfg.out
    out_dim = state["output.weight"].shape[0]
    payload = {
        "state_dict": state,
        "T": job.T, "rep": job.rep,
        "arch": {"in_dim": data.in_dim, "width": cfg.width,
                 "n_hidden": cfg.hidden_layers, "out_dim": out_dim,
                 "act": cfg.act, "mup": cfg.mup,
                 "base_width": cfg.mup_base_width},
        "preproc": {"standardize": cfg.standardize, "x_mean": data.x_mean,
                    "x_std": data.x_std, "y_mean": data.y_mean},
        "train": {"optimizer": cfg.optimizer, "lr": cfg.lr,
                  "weight_decay": cfg.weight_decay,
                  "lr_schedule": cfg.lr_schedule, "batch_size": job.batch_size,
                  "epochs": cfg.epochs, "patience": cfg.patience,
                  "seed": cfg.seed, "x_file": cfg.x_file, "y_file": cfg.y_file,
                  "best_epoch": history.get("best_epoch") if history else None,
                  "test_mse": history.get("test") if history else None},
    }
    torch.save(payload, f"{stem}_model_T{job.T}.pt")


# --------------------------------------------------------------------------- #
# Jobs + task-parallel workers                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    T: int
    rep: int
    batch_size: int


def make_jobs(cfg) -> List[Job]:
    jobs = []
    for T in cfg.T_small:
        for rep in range(cfg.navg):
            jobs.append(Job(T, rep, cfg.batch_small))
    for T in cfg.T_large:
        for rep in range(cfg.navg_large):
            jobs.append(Job(T, rep, cfg.batch_large))
    return jobs


def pick_device(gpu_id):
    """gpu_id: int -> cuda:gpu_id (cluster); 'cpu' -> CPU (safe multi-worker on a
    box without CUDA); None -> MPS if available else CPU (local single worker)."""
    if isinstance(gpu_id, int) and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        return torch.device(f"cuda:{gpu_id}")
    if gpu_id == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_job(data: Data, job: Job, cfg, device):
    n_avail = data.Xtr.shape[0]
    T = min(job.T, n_avail)
    # subsample reproducibly per (T, rep); the full-data point uses all rows
    if T >= n_avail:
        idx = torch.arange(n_avail, device=device)
    else:
        g = torch.Generator().manual_seed(cfg.seed + 100003 * job.T + job.rep)
        idx = torch.from_numpy(
            np.random.default_rng(cfg.seed + 100003 * job.T + job.rep)
            .choice(n_avail, size=T, replace=False)).to(device)
    seed = cfg.seed + 7919 * job.T + job.rep
    return train_one(data, idx, cfg, job.batch_size, seed, device)


def worker(gpu_id, job_q, res_q, cfg):
    device = pick_device(gpu_id)
    data = load_data(cfg, device)
    while True:
        item = job_q.get()
        if item is None:
            break
        ji, job = item
        t0 = time.time()
        try:
            test_mse, train_mse, n_params, history, state = run_job(data, job, cfg, device)
            # keep the learning curve(s) for the dump: every repeat with
            # --all-curves, otherwise just the first repeat per T.
            keep = (not cfg.no_curves) and (cfg.all_curves or job.rep == 0)
            hist = history if keep else None
            # save the representative (rep-0) model here in the worker, which has
            # the data stats; distinct T -> distinct file, so no write races.
            if cfg.save_models and job.rep == 0:
                save_model(state, job, cfg, data, history)
            res_q.put((ji, job.T, job.rep, test_mse, train_mse, n_params,
                       job.batch_size, time.time() - t0, str(device), None, hist))
        except Exception as e:  # keep the pool alive; report the failure
            res_q.put((ji, job.T, job.rep, None, None, None,
                       job.batch_size, time.time() - t0, str(device), repr(e), None))


# --------------------------------------------------------------------------- #
# Orchestration                                                                 #
# --------------------------------------------------------------------------- #
def aggregate(records, cfg) -> List[dict]:
    by_T = {}
    for r in records:
        by_T.setdefault(r["T"], []).append(r)
    out = []
    for T in sorted(by_T):
        rs = [r for r in by_T[T] if r["loss"] is not None]
        if not rs:
            continue
        losses = np.array([r["loss"] for r in rs])
        trains = np.array([r["train_loss"] for r in rs])
        out.append({
            "T": T,
            "N": rs[0]["N"],
            "num_EFPs": rs[0]["num_EFPs"],
            "loss": float(losses.mean()),
            "loss_std": float(losses.std()),
            "train_loss": float(trains.mean()),
            "train_loss_std": float(trains.std()),
            "n_avg": len(rs),
            "batch_size": rs[0]["batch_size"],
            "hidden layers": cfg.hidden_layers,
            "width": cfg.width,
            "mup": cfg.mup,
            "mup_base_width": cfg.mup_base_width if cfg.mup else None,
        })
    return out


def run(cfg):
    jobs = make_jobs(cfg)
    # multi-node sharding: keep only this shard's jobs
    jobs = [j for i, j in enumerate(jobs) if i % cfg.num_shards == cfg.shard_id]
    print(f"[main] {len(jobs)} jobs this shard "
          f"(shard {cfg.shard_id}/{cfg.num_shards}); "
          f"mup={cfg.mup} width={cfg.width}", flush=True)

    n_gpu = torch.cuda.device_count()
    n_workers = cfg.num_workers or (n_gpu if n_gpu > 0 else 1)
    records = []
    # learning-curve store. Default: {T: history} (one representative run per T).
    # With --all-curves: {T: {rep: history}} (every run).
    curves = {}

    def record_curve(T, rep, history):
        if cfg.no_curves or history is None:
            return
        if cfg.all_curves:
            curves.setdefault(T, {})[rep] = history
        else:
            curves[T] = history
        dump_curves(curves, cfg, announce=False)   # flush as we go

    if n_workers <= 1:
        # single device (also the local MPS/CPU path)
        device = pick_device(0 if n_gpu > 0 else None)
        data = load_data(cfg, device)
        for ji, job in enumerate(jobs):
            t0 = time.time()
            test_mse, train_mse, n_params, history, state = run_job(data, job, cfg, device)
            if cfg.save_models and job.rep == 0:
                save_model(state, job, cfg, data, history)
            if cfg.all_curves or job.rep == 0:
                record_curve(job.T, job.rep, history)
            records.append(dict(T=job.T, rep=job.rep, loss=test_mse,
                                train_loss=train_mse, N=n_params,
                                num_EFPs=data.in_dim, batch_size=job.batch_size))
            print(f"[{device}] T={job.T} rep={job.rep} "
                  f"test={test_mse:.4e} train={train_mse:.4e} "
                  f"({time.time()-t0:.1f}s) [{ji+1}/{len(jobs)}]", flush=True)
            _save(aggregate(records, cfg), cfg)
    else:
        ctx = mp.get_context("spawn")
        job_q, res_q = ctx.Queue(), ctx.Queue()
        for ji, job in enumerate(jobs):
            job_q.put((ji, job))
        for _ in range(n_workers):
            job_q.put(None)  # one sentinel per worker
        procs = [ctx.Process(
                    target=worker,
                    args=((g % n_gpu) if n_gpu > 0 else "cpu", job_q, res_q, cfg))
                 for g in range(n_workers)]
        for p in procs:
            p.start()
        for done in range(len(jobs)):
            (ji, T, rep, test_mse, train_mse, n_params, bs, dt, dev, err, hist) = res_q.get()
            if err is not None:
                print(f"[{dev}] T={T} rep={rep} FAILED: {err}", flush=True)
            else:
                record_curve(T, rep, hist)
                records.append(dict(T=T, rep=rep, loss=test_mse,
                                    train_loss=train_mse, N=n_params,
                                    num_EFPs=cfg._in_dim, batch_size=bs))
                print(f"[{dev}] T={T} rep={rep} test={test_mse:.4e} "
                      f"train={train_mse:.4e} ({dt:.1f}s) "
                      f"[{done+1}/{len(jobs)}]", flush=True)
            _save(aggregate(records, cfg), cfg)
        for p in procs:
            p.join()

    results = aggregate(records, cfg)
    _save(results, cfg)
    if not cfg.no_curves and curves:
        dump_curves(curves, cfg)
    print(f"[main] done -> {cfg.out}", flush=True)
    for r in results:
        print(f"  T={r['T']:>8d}  test={r['loss']:.4e}+-{r['loss_std']:.1e}  "
              f"train={r['train_loss']:.4e}  (n_avg={r['n_avg']})")
    return results


def _save(results, cfg):
    with open(cfg.out, "wb") as f:
        pickle.dump(results, f)


def _runs_for_T(entry):
    """Normalize a curves[T] entry to a list of (rep, history).  The entry is
    either a single history dict ({T: history}, default mode) or a {rep: history}
    dict (--all-curves mode)."""
    if isinstance(entry, dict) and "val" in entry:      # single history
        return [(0, entry)]
    return sorted(entry.items())                        # {rep: history}


def dump_curves(curves, cfg, announce=True):
    """Save the learning curve(s) per T: a <out>_curves.pkl with the raw histories
    and a <out>_curves.png grid (one panel per T).  Each panel plots the
    validation and epoch-mean training MSE vs epoch with a marker at each run's
    best (restored) epoch, so the reported loss can be sanity-checked against the
    floor of the val curve.  Default: one representative run per T
    (curves = {T: history}).  With --all-curves: every (T, repeat) run is overlaid
    in its panel (curves = {T: {rep: history}}).

    Called incrementally as curves arrive (announce=False, quiet) and once at the
    end (announce=True), so a partial dump survives an early kill."""
    stem = cfg.out[:-4] if cfg.out.endswith(".pkl") else cfg.out
    with open(stem + "_curves.pkl", "wb") as f:
        pickle.dump(curves, f)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as e:
        print(f"[main] curves saved to {stem}_curves.pkl; skipping plot ({e})")
        return

    Ts = sorted(curves)
    ncol = min(4, len(Ts))
    nrow = (len(Ts) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow),
                             squeeze=False)
    tag = (f"muP w={cfg.width}/{cfg.mup_base_width}" if cfg.mup
           else f"SP w={cfg.width}")
    for ax in axes.flat:
        ax.set_visible(False)
    for i, T in enumerate(Ts):
        runs = _runs_for_T(curves[T])
        single = len(runs) == 1
        a = 1.0 if single else max(0.2, len(runs) ** -0.5)
        ax = axes[i // ncol][i % ncol]
        ax.set_visible(True)
        for _, h in runs:
            ep = range(len(h["val"]))
            ax.plot(ep, h["val"], "-", color="C0", lw=1, alpha=a,
                    marker="o" if single else None, ms=3)
            ax.plot(ep, h["train_batch"], "-", color="C1", lw=1, alpha=0.8 * a)
            ax.plot(h["best_epoch"], h["best_val"], "v", color="k", ms=5,
                    alpha=min(1.0, 1.5 * a), zorder=5)
        ax.set_yscale("log")
        ax.set_xlabel("epoch"); ax.set_ylabel("MSE")
        ax.grid(alpha=0.3, which="both")
        handles = [Line2D([0], [0], color="C0", marker="o", ms=3, label="val"),
                   Line2D([0], [0], color="C1", label="train (batch mean)"),
                   Line2D([0], [0], color="k", marker="v", ls="", label="best epoch")]
        tests = [h["test"] for _, h in runs]
        if single:
            h = runs[0][1]
            ax.axhline(h["test"], ls=":", color="r", lw=1)
            handles.append(Line2D([0], [0], color="r", ls=":",
                                  label=f"saved test={h['test']:.2e}"))
            ax.set_title(f"T={T}  (best val={h['best_val']:.2e})", fontsize=9)
        else:
            ax.set_title(f"T={T}  ({len(runs)} runs, "
                         f"test={np.mean(tests):.2e}±{np.std(tests):.0e})", fontsize=9)
        ax.legend(handles=handles, fontsize=6)
        h0 = runs[0][1]
        if h0.get("lr"):                          # show the LR schedule on a twin axis
            ax2 = ax.twinx()
            ax2.plot(range(len(h0["lr"])), h0["lr"], color="green", lw=1, alpha=0.4)
            ax2.set_ylabel("lr", color="green", fontsize=7)
            ax2.tick_params(axis="y", labelcolor="green", labelsize=6)
    fig.suptitle(f"Learning curves ({tag}, lr_schedule={cfg.lr_schedule}); "
                 f"reported loss = test MSE at the early-stop epoch", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(stem + "_curves.png", dpi=110)
    plt.close(fig)
    if announce:
        print(f"[main] learning curves -> {stem}_curves.png (+ {stem}_curves.pkl)")


# --------------------------------------------------------------------------- #
# Config / CLI                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class Cfg:
    x_file: str = "X.npy"          # single file: all features  (N, d)
    y_file: str = "Y.npy"          # single file: all labels    (N,) or (N, 1)
    n_val: int = 10000             # held-out validation rows (early stopping)
    n_test: int = 30000            # held-out test rows (reported loss)
    split_seed: int = 0            # seed for the (fixed) train/val/test split
    no_shuffle_split: bool = False # test = last n_test rows (matches ridge)
    standardize: bool = False      # standardize X / center y by train-pool stats
    no_curves: bool = False        # disable the per-T learning-curve dump
    all_curves: bool = False       # dump every (T, rep) run, not just one per T
    save_models: bool = True       # save the rep-0 trained model per T (.pt)
    out: str = "resultsEFP_DNN_torch.pkl"
    width: int = 256
    hidden_layers: int = 3
    act: str = "relu"
    # Defaults from the 2026-07 local scan on log_EFPs_special_d20_E700_qq_N15
    # (see dnn_prototyping/): AdamW+cosine beat Adam at T=5000 (2.3e-5 vs
    # 4.1e-5, Adam early-stops unstably at lr 3e-3) and AdaGrad by ~5x; lr 1e-3
    # transfers across muP widths 256->1024 (3e-3 is slightly better for SP
    # width 256 but sits at the wide-muP stability edge).  SGD: max stable
    # CONSTANT lr is 3e-3 with batch >= 32 (1e-2 diverges at every batch size;
    # batch 16 diverges even at 3e-3 -- heavy-tailed features); overfitting at
    # small T is visible at lr 3e-3 (val_end/val_best ~ 1.1) but not below.
    lr: float = 1e-3
    optimizer: str = "adamw"       # adam | adamw | adagrad | sgd (muP LR rules differ)
    weight_decay: float = 0.01     # decoupled wd, adamw only (see build_param_groups)
    momentum: float = 0.9          # SGD momentum (ignored otherwise)
    lr_schedule: str = "cosine"    # "cosine" (anneal lr->0 over epochs) or "constant"
    epochs: int = 200
    patience: int = 20
    navg: int = 20
    navg_large: int = 1
    batch_small: int = 32
    batch_large: int = 128
    T_small: List[int] = field(default_factory=lambda: [100, 150, 200, 300, 500, 700, 1000])
    T_large: List[int] = field(default_factory=lambda: [3000, 5000, 10000,
                                                        50000, 100000, 200000, 300000,
                                                        500000, 1000000])
    mup: bool = False
    mup_base_width: int = 64
    num_workers: int = 0          # 0 -> auto (= #CUDA GPUs, else 1)
    num_shards: int = 1
    shard_id: int = 0
    seed: int = 0
    smoke_test: bool = False
    smoke_dim: int = 20
    smoke_train: int = 20000
    _in_dim: int = 0


def parse_args() -> Cfg:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--x-file", default=None,
                   help="single .npy with all features, shape (N, d)")
    p.add_argument("--y-file", default=None,
                   help="single .npy with all labels, shape (N,) or (N, 1)")
    p.add_argument("--n-val", type=int, default=10000,
                   help="held-out validation rows (used for early stopping)")
    p.add_argument("--n-test", type=int, default=30000,
                   help="held-out test rows (reported loss)")
    p.add_argument("--split-seed", type=int, default=0,
                   help="seed for the fixed train/val/test split (shared by all "
                        "workers; independent of --seed)")
    p.add_argument("--no-shuffle-split", action="store_true",
                   help="don't shuffle: test = last n_test rows, val = the n_val "
                        "before it (matches the ridge notebook's test set)")
    p.add_argument("--standardize", action="store_true",
                   help="standardize X and center y by training-pool stats, "
                        "replicating ridge_learning_curve(scale=True)")
    p.add_argument("--no-curves", action="store_true",
                   help="skip the per-T learning-curve (loss vs epoch) dump")
    p.add_argument("--all-curves", action="store_true",
                   help="dump the loss-vs-epoch curve for EVERY (T, repeat) run "
                        "(overlaid per-T panel; pkl = {T: {rep: history}}), not "
                        "just the first repeat per T")
    p.add_argument("--no-save-models", action="store_true",
                   help="don't save the representative (rep-0) trained network "
                        "per T to <out>_model_T<T>.pt")
    p.add_argument("--out", default="resultsEFP_DNN_torch.pkl")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--hidden-layers", type=int, default=3)
    p.add_argument("--act", default="relu", choices=["relu", "gelu", "tanh"])
    p.add_argument("--lr", type=float, default=1e-3,
                   help="base learning rate (at the muP base width).  Scan-chosen "
                        "defaults on the log-EFP/thrust data: adamw/adam 1e-3 "
                        "(3e-3 ok for SP width 256, unstable for wide muP); "
                        "sgd 3e-3 max stable CONSTANT lr (needs batch >= 32); "
                        "adagrad 1e-2.")
    p.add_argument("--optimizer", default="adamw",
                   choices=["adam", "adamw", "adagrad", "sgd"],
                   help="adam/adamw/adagrad share the normalized-update muP LR "
                        "table; sgd has its own (see build_param_groups).")
    p.add_argument("--weight-decay", type=float, default=0.01,
                   help="decoupled weight decay for --optimizer adamw (ignored "
                        "otherwise); per-group rescaled under muP so lr*wd is "
                        "width-uniform.")
    p.add_argument("--momentum", type=float, default=0.9,
                   help="SGD momentum (ignored for the adam family).")
    p.add_argument("--lr-schedule", default="cosine", choices=["cosine", "constant"],
                   help="cosine: anneal each param group's lr to ~0 over --epochs "
                        "(default; lets Adam settle into the minimum). "
                        "constant: fixed lr (the original behavior).")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--navg", type=int, default=5, help="repeats averaged at small T")
    p.add_argument("--navg-large", type=int, default=1, help="repeats at large T")
    p.add_argument("--batch-small", type=int, default=16,
                   help="batch size for T_small fits.  WARNING: with "
                        "--optimizer sgd use >= 32 -- batch 16 diverges on the "
                        "heavy-tailed log-EFP features even at lr 3e-3.")
    p.add_argument("--batch-large", type=int, default=128)
    p.add_argument("--T-small", type=int, nargs="+", default=None)
    p.add_argument("--T-large", type=int, nargs="+", default=None)
    p.add_argument("--mup", action="store_true",
                   help="use muP (feature-learning limit) instead of SP (lazy limit)")
    p.add_argument("--mup-base-width", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0,
                   help="0 = auto (one per CUDA GPU, else single device)")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke-test", action="store_true",
                   help="run on small synthetic data, no .npy files needed")
    a = p.parse_args()
    if not a.smoke_test and (a.x_file is None or a.y_file is None):
        p.error("--x-file and --y-file are required (unless --smoke-test)")
    cfg = Cfg(x_file=a.x_file, y_file=a.y_file, n_val=a.n_val, n_test=a.n_test,
              split_seed=a.split_seed, no_shuffle_split=a.no_shuffle_split,
              standardize=a.standardize, no_curves=a.no_curves,
              all_curves=a.all_curves, save_models=not a.no_save_models,
              out=a.out, width=a.width,
              hidden_layers=a.hidden_layers, act=a.act, lr=a.lr,
              optimizer=a.optimizer, weight_decay=a.weight_decay,
              momentum=a.momentum,
              lr_schedule=a.lr_schedule, epochs=a.epochs,
              patience=a.patience, navg=a.navg, navg_large=a.navg_large,
              batch_small=a.batch_small, batch_large=a.batch_large,
              mup=a.mup, mup_base_width=a.mup_base_width,
              num_workers=a.num_workers, num_shards=a.num_shards,
              shard_id=a.shard_id, seed=a.seed, smoke_test=a.smoke_test)
    if a.T_small is not None:
        cfg.T_small = a.T_small
    if a.T_large is not None:
        cfg.T_large = a.T_large
    if a.smoke_test:
        cfg.T_small = a.T_small or [50, 100, 300]
        cfg.T_large = a.T_large or [1000, 5000]
        cfg.epochs = min(cfg.epochs, 60)
        cfg.n_val = min(cfg.n_val, 2000)
        cfg.n_test = min(cfg.n_test, 2000)
    return cfg


def probe_in_dim(cfg) -> int:
    """Input dimension without loading the full array (mmap header read)."""
    if cfg.smoke_test:
        return cfg.smoke_dim
    return int(np.load(cfg.x_file, mmap_mode="r").shape[1])


def main():
    cfg = parse_args()
    cfg._in_dim = probe_in_dim(cfg)   # for the result dicts; workers load data themselves
    run(cfg)


if __name__ == "__main__":
    main()
