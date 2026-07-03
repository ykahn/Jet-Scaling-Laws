import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import pickle

print("starting")

def _compute_z_theta_t(events):
    E  = events[..., 0]
    p3 = events[..., 1:4]
    z = 2.0 * E / E.sum(dim=-1, keepdim=True)
    pmag = torch.linalg.norm(p3, dim=-1)
    pdot = torch.einsum('...ia,...ja->...ij', p3, p3)
    denom = pmag.unsqueeze(-1) * pmag.unsqueeze(-2)
    cos_th = torch.where(denom > 0, pdot / denom.clamp_min(1e-30),
                         torch.zeros_like(pdot))
    N = events.shape[-2]
    eye = torch.eye(N, dtype=cos_th.dtype, device=cos_th.device).expand_as(cos_th)
    cos_th = torch.where(eye.bool(), torch.ones_like(cos_th), cos_th)
    return z, 0.5 * (1.0 - cos_th)


def compute_special_torch(events_np, dmax, device='cuda', dtype=torch.float64):
    """Note: ordered by degree rather than type, this is different from previous versions"""
    events = torch.as_tensor(events_np, dtype=dtype, device=device)
    z, th = _compute_z_theta_t(events)
    zz = z.unsqueeze(-1) * z.unsqueeze(-2)

    # powers of theta: Ap[k] = theta^(k+2)
    Ap = [th * th]
    for _ in range(dmax - 2):
        Ap.append(Ap[-1] * th)

    # type 1: sum_ij z_i z_j theta_ij^m
    type1 = torch.stack([(zz * a).sum(dim=(-2, -1)) for a in Ap], dim=-1)

    # s_cache[k] = sum_j z_j theta_ij^(k+2)
    s_cache = [(a * z.unsqueeze(-2)).sum(dim=-1) for a in Ap]
    t2_keys = [(p, q) for p in range(2, dmax + 1)
                      for q in range(p, dmax + 1) if p + q <= dmax]
    type2 = torch.stack([(z * s_cache[p - 2] * s_cache[q - 2]).sum(dim=-1)
                         for p, q in t2_keys], dim=-1)

    # type 3 triangle: for each (s, t) precompute M = A^s @ (diag(z) A^t)
    A = [th]
    for _ in range(dmax - 1):
        A.append(A[-1] * th)
    t3_keys = [(r, s, t) for r in range(1, dmax + 1)
                         for s in range(r, dmax + 1)
                         for t in range(s, dmax + 1) if r + s + t <= dmax]
    M_cache, t3_cols = {}, []
    for r, s, t in t3_keys:
        if (s, t) not in M_cache:
            M_cache[(s, t)] = A[s - 1] @ (z.unsqueeze(-1) * A[t - 1])
        t3_cols.append((zz * A[r - 1] * M_cache[(s, t)]).sum(dim=(-2, -1)))
    type3 = torch.stack(t3_cols, dim=-1)
    feats = torch.cat([type1, type2, type3], dim=-1)

    # --- reorder columns by degree (then type, then original key order) ---
    # degree = total power of theta; type is the tiebreak so each degree block
    # is ordered type1 -> type2 -> type3, matching the original within-type order.
    labels  = [(k + 2,     1) for k in range(len(Ap))]          # type1: deg k+2
    labels += [(p + q,     2) for (p, q) in t2_keys]            # type2: deg p+q
    labels += [(r + s + t, 3) for (r, s, t) in t3_keys]         # type3: deg r+s+t
    order = sorted(range(len(labels)), key=lambda i: labels[i])  # stable -> keeps key order
    order_t = torch.as_tensor(order, device=feats.device)
    return feats.index_select(-1, order_t)

@torch.no_grad()
def efps_chunked(events, dmax, device='cuda', chunk=20000):
    """compute_special_torch in chunks over events, offloading each chunk to CPU,
    so the ~130 (B,M,M) intermediates stay small."""
    events = torch.as_tensor(events, dtype=torch.float32).cpu()  # keep full tensor on host
    parts = []
    for s in range(0, events.shape[0], chunk):
      efp = compute_special_torch(events[s:s+chunk], dmax, device=device)
      parts.append(efp.cpu())
      del efp
      torch.cuda.empty_cache()
    return torch.cat(parts, 0)

def compute_thrust_torch(events, device='cuda', dtype=torch.float64, chunk=20000):
      """Thrust for a batch of e+e- events, GPU-vectorized.
      Matches ThrustScalingLaws.compute_thrust exactly (candidate axes = the
      particle directions). events: (B, M, 4) = (E, px, py, pz). Returns (B,) numpy."""
      ev = torch.as_tensor(events, dtype=dtype)
      out = torch.empty(ev.shape[0], dtype=dtype)
      for s in range(0, ev.shape[0], chunk):
          p = ev[s:s+chunk, :, 1:4].to(device)                 # (b, M, 3)
          norms = p.norm(dim=-1)                               # (b, M)
          denom = norms.sum(dim=-1)                            # (b,)
          # candidate axes = unit particle directions; zero-momentum -> zero axis
          axes = torch.where(norms.unsqueeze(-1) > 1e-10,
                             p / norms.unsqueeze(-1).clamp_min(1e-30),
                             torch.zeros_like(p))              # (b, M, 3)
          proj = torch.einsum('bjd,bid->bji', p, axes).abs()   # (b, M_j, M_i)=|p_j·n_i|
          T = proj.sum(dim=1) / denom.unsqueeze(-1)            # (b, M_i): thrust per axis
          out[s:s+chunk] = T.max(dim=1).values.cpu()
      return out.numpy()

@torch.no_grad()
def ridge_learning_curve_torch(X, y, train_sizes, *, n_test=5000, alpha=1e-12,
                                 n_repeats=200, scale=True, seed_base=0,
                                 device="cuda", dtype=torch.float64, verbose=True):
    X = torch.as_tensor(X, dtype=dtype, device=device)            # no-op if already on GPU
    y = torch.as_tensor(y, dtype=dtype, device=device).reshape(-1)
    Xp, yp, Xt, yt = X[:-n_test], y[:-n_test], X[-n_test:], y[-n_test:]
    if scale:
      mu, sd = Xp.mean(0), Xp.std(0); sd = torch.where(sd == 0, torch.ones_like(sd), sd)
      Xp, Xt = (Xp - mu) / sd, (Xt - mu) / sd
      ymu = yp.mean(); yp, yt = yp - ymu, yt - ymu
    pool, p = Xp.shape
    eye = torch.eye(p, device=device, dtype=dtype)
    sizes = list(train_sizes); means, stds = [], []
    for n in sizes:
        t0 = time.perf_counter(); losses = torch.empty(n_repeats, device=device, dtype=dtype)
        for s in range(n_repeats):
            g = torch.Generator(device=device).manual_seed(seed_base + s)
            idx = torch.randint(pool, (n,), generator=g, device=device)   # n << pool
            Xs, ys = Xp[idx], yp[idx]

            lam = alpha * n
            if n < p:                                   # dual: n×n system (n << p here)
                K = Xs @ Xs.T                           # (n, n)   cost ~ n^2 * p
                K.diagonal().add_(lam)
                a = torch.linalg.solve(K,ys)
                beta = Xs.T @ a                         # (p,)
            else:                                       # primal: p×p (only when n >= p)
                G = Xs.T @ Xs
                G.diagonal().add_(lam)
                beta = torch.linalg.solve(G,Xs.T @ ys)
            losses[s] = ((Xt @ beta - yt) ** 2).mean()
        means.append(losses.mean().item()); stds.append(losses.std().item())
        if verbose: print(f"{n:>8d}{means[-1]:>15.6e}{stds[-1]:>15.6e}{time.perf_counter()-t0:>12.2f}")
    return np.asarray(means), np.asarray(stds)

Ns = [5,10,15] # [3,5,10,15]
print(Ns)

maxdat = 100000

EFPs={}
thrust={}
ndat={}
print("Loading data and constructing EFPs...")

for N in Ns:
    print(N)
#    zqq = torch.load(f'/home/yfkahn/projects/aip-yfkahn/shared/intermediate_by_nbranch/ee2qqbar_S=2000GeV_a=0p1365_NBranch={N-2}_Nparticles={N}.pt',map_location='cpu')
    zqq = torch.as_tensor(np.load(f'/home/yfkahn/projects/aip-yfkahn/yfkahn/JetScalingLaws/zqq_dat/splittings_{N}.npy'),device='cpu')
     #compute EFPs up to max degree d = 13

    EFPs['d13', f'N{N}'] = efps_chunked(zqq, 13, chunk=5000,device='cuda').numpy() #109 such EFPs
    numEFPs = EFPs['d13', f'N{N}'].shape[1]
     #construct composite EFPs
    ndat = min(maxdat,EFPs['d13', f'N{N}'].shape[0])
    print(ndat)
    EFPs['composite',f'N{N}'] = np.einsum('na,nb->nab',EFPs['d13',f'N{N}'][:ndat],EFPs['d13',f'N{N}'][:ndat]).reshape((ndat,numEFPs**2))
    print(EFPs['composite',f'N{N}'].shape)
    thrust[f'N{N}']          = (compute_thrust_torch(zqq,chunk=5000))[:ndat] 
    del zqq
    torch.cuda.empty_cache()

ntest = 5000
train_sizes_t = np.logspace(2,np.log10(ndat-5000),30).astype(int)
mean_mse_thrust={}
std_mse_thrust={}

alphaval = 1e-10

print('starting regression')
for N in Ns:
    print(N)   
    mean_mse_thrust['composite',f'N{N}'], std_mse_thrust['composite',f'N{N}'] = ridge_learning_curve_torch(
        EFPs['composite',f'N{N}'], thrust[f'N{N}'],
        train_sizes=train_sizes_t,
        n_test=ntest,
        alpha=alphaval,
        n_repeats=100,
        scale=True
        )

    with open(f'thrust_regression_nolog_overparam_alpha{alphaval:g}_mean.pkl', "wb") as f:
        pickle.dump(mean_mse_thrust, f)

    with open(f'thrust_regression_nolog_overparam_alpha{alphaval:g}_std.pkl', "wb") as f:
        pickle.dump(std_mse_thrust, f)