#!/usr/bin/env python3
"""Analyze EFP covariance spectra from Yoni's shower-depth snapshots.

Input files are produced by ../generate_pythia_zqq_splittings.py:
    pythia_zqq_splittings/splittings_k.npy

For each requested shower step k, this script computes the same style of EFPs
used in EFP_covariance_spectrum.ipynb:
    EFPSet(('d<=', 6), measure='eeefm', beta=2, coords='epxpypz')

It can also whiten/orthonormalize the EFP basis on RAMBO phase-space events of
the same multiplicity and apply that fixed transform to the PYTHIA snapshots.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from energyflow import EFPSet
from energyflow.utils.event_utils import gen_massless_phase_space


def compute_efps(events: np.ndarray, dmax: int, beta: float, n_jobs: int) -> np.ndarray:
    efpset = EFPSet(("d<=", dmax), measure="eeefm", beta=beta, coords="epxpypz")
    return efpset.batch_compute(events, n_jobs=n_jobs)


def centered_covariance(features: np.ndarray) -> np.ndarray:
    x = features - features.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / len(x)
    return 0.5 * (cov + cov.T)


def spectrum(features: np.ndarray) -> np.ndarray:
    vals = np.linalg.eigvalsh(centered_covariance(features))
    return np.clip(vals[::-1], 0.0, None)


def fit_whitener(features: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = features.mean(axis=0)
    cov = centered_covariance(features)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    keep = vals > eps * vals[0]
    return mu, vecs[:, keep], vals[keep]


def apply_whitener(features: np.ndarray, mu: np.ndarray, vecs: np.ndarray, vals: np.ndarray) -> np.ndarray:
    return (features - mu) @ vecs / np.sqrt(vals)


def load_step(data_dir: Path, step: int, max_events: int | None) -> np.ndarray:
    path = data_dir / f"splittings_{step}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run generate_pythia_zqq_splittings.py first.")
    arr = np.load(path)
    if max_events is not None:
        arr = arr[:max_events]
    return arr.astype(np.float64, copy=False)


def plot_step(
    raw: np.ndarray,
    whitened: np.ndarray | None,
    ref_white: np.ndarray | None,
    outpath: Path,
    title: str,
    max_i: int | None,
) -> None:
    fig, ax = plt.subplots(figsize=(5.3, 4.0), constrained_layout=True)
    curves = [(raw, "PYTHIA EFP covariance", "C0", "-")]
    if whitened is not None:
        curves.append((whitened, "PYTHIA after RAMBO whitening", "C3", "-"))
    if ref_white is not None:
        curves.append((ref_white, "RAMBO after whitening", "0.35", "--"))

    n = min(len(y) for y, *_ in curves)
    if max_i is not None:
        n = min(n, max_i)
    xs = np.arange(1, n + 1)
    for y, label, color, linestyle in curves:
        yy = y[:n]
        pos = yy > 0
        ax.loglog(xs[pos], yy[pos], label=label, color=color, linestyle=linestyle, lw=2)

    ax.set_xlim(1, n)
    ax.set_xlabel(r"$i$")
    ax.set_ylabel(r"$\lambda_i$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def parse_steps(text: str) -> list[int]:
    out: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = (int(x) for x in chunk.split("-", 1))
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(chunk))
    return sorted(set(out))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "pythia_zqq_splittings")
    parser.add_argument("--steps", default="1,2,3,4", help="Comma/range list, e.g. 1,2,5-8.")
    parser.add_argument("--max-events", type=int, default=5000)
    parser.add_argument("--dmax", type=int, default=6)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--whiten", action="store_true")
    parser.add_argument("--n-reference", type=int, default=20000)
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-i", type=int, default=200)
    parser.add_argument("--outdir", type=Path, default=ROOT / "ItayCode" / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    steps = parse_steps(args.steps)

    for step in steps:
        events = load_step(args.data_dir, step=step, max_events=args.max_events)
        n_events, multiplicity, _ = events.shape
        print(f"step {step}: events shape {events.shape}")
        features = compute_efps(events, dmax=args.dmax, beta=args.beta, n_jobs=args.n_jobs)
        raw_spec = spectrum(features)

        white_spec = None
        ref_white_spec = None
        whitening_rank = 0
        if args.whiten:
            print(f"  generating RAMBO reference: N={args.n_reference}, M={multiplicity}")
            reference = gen_massless_phase_space(
                args.n_reference,
                multiplicity,
                energy=1.0,
                seed=args.seed + step,
            )
            ref_features = compute_efps(reference, dmax=args.dmax, beta=args.beta, n_jobs=args.n_jobs)
            mu, vecs, vals = fit_whitener(ref_features, eps=args.eps)
            white_features = apply_whitener(features, mu, vecs, vals)
            ref_white_features = apply_whitener(ref_features, mu, vecs, vals)
            white_spec = spectrum(white_features)
            ref_white_spec = spectrum(ref_white_features)
            whitening_rank = len(vals)

        stem = f"step{step}_d{args.dmax}_beta{args.beta:g}_n{n_events}"
        if args.whiten:
            stem += "_rambo_whitened"
        npz_out = args.outdir / f"{stem}.npz"
        png_out = args.outdir / f"{stem}_xmax{args.max_i}.png"
        np.savez(
            npz_out,
            raw=raw_spec,
            whitened=white_spec,
            ref_white=ref_white_spec,
            step=step,
            multiplicity=multiplicity,
            n_events=n_events,
            dmax=args.dmax,
            beta=args.beta,
            whitening_rank=whitening_rank,
            eps=args.eps,
        )
        plot_step(
            raw_spec,
            white_spec,
            ref_white_spec,
            png_out,
            title=f"EFP covariance at shower step {step} (M={multiplicity})",
            max_i=args.max_i,
        )
        print(f"  saved {npz_out}")
        print(f"  saved {png_out}")


if __name__ == "__main__":
    main()
