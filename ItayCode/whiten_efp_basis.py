#!/usr/bin/env python3
"""Whiten EFPs on a reference distribution and apply to top-tagging jets."""

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

from efp_spectrum import load_random_events_both


def compute_efps(events: np.ndarray | list[np.ndarray], beta: float, n_jobs: int) -> np.ndarray:
    efpset = EFPSet("d<=6", measure="hadr", beta=beta, normed=True, coords="epxpypz")
    features = efpset.batch_compute(events, n_jobs=n_jobs)
    if features.shape[1] != 314:
        raise RuntimeError(f"Expected 314 EFPs including trivial one, got {features.shape[1]}.")
    return features[:, 1:]


def centered_spectrum(features: np.ndarray) -> np.ndarray:
    x = features - np.mean(features, axis=0, keepdims=True)
    cov = (x.T @ x) / len(x)
    vals = np.linalg.eigvalsh(cov)
    return np.clip(vals[::-1], 0.0, None)


def fit_whitener(features: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.mean(features, axis=0)
    x = features - mu
    cov = (x.T @ x) / len(x)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    keep = vals > eps * vals[0]
    return mu, vecs[:, keep], vals[keep]


def apply_whitener(features: np.ndarray, mu: np.ndarray, vecs: np.ndarray, vals: np.ndarray) -> np.ndarray:
    return (features - mu) @ vecs / np.sqrt(vals)


def plot_compare(
    raw_qcd: np.ndarray,
    raw_top: np.ndarray,
    white_qcd: np.ndarray,
    white_top: np.ndarray,
    outpath: Path,
    max_i: int,
) -> None:
    fig, ax = plt.subplots(figsize=(5.3, 4.0), constrained_layout=True)
    curves = [
        (raw_qcd[:max_i], "raw EFP QCD", "C0", "-"),
        (raw_top[:max_i], "raw EFP top", "C0", "--"),
        (white_qcd[:max_i], "whitened EFP QCD", "C3", "-"),
        (white_top[:max_i], "whitened EFP top", "C3", "--"),
    ]
    n = min(len(y) for y, *_ in curves)
    xs = np.arange(1, n + 1)
    for y, label, color, ls in curves:
        y = y[:n]
        positive = y > 0
        ax.loglog(xs[positive], y[positive], label=label, color=color, linestyle=ls, lw=2)

    ax.set_xlim(1, n)
    ax.set_xlabel(r"$i$")
    ax.set_ylabel(r"$\lambda_i$")
    ax.set_title("EFP covariance before/after reference whitening")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=ROOT / "data" / "train.parquet")
    parser.add_argument("--n-real", type=int, default=100, help="Top/QCD events per class.")
    parser.add_argument("--n-reference", type=int, default=2000, help="Reference/noise events.")
    parser.add_argument("--n-reference-particles", type=int, default=50)
    parser.add_argument("--reference-energy", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-10, help="Eigenvalue cutoff s/smax.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "whitening")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Sampling real jets...")
    qcd_events, top_events, _, _ = load_random_events_both(
        args.parquet,
        n_events=args.n_real,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    print("Generating RAMBO reference events...")
    ref_events = gen_massless_phase_space(
        args.n_reference,
        args.n_reference_particles,
        energy=args.reference_energy,
        seed=args.seed + 1,
    )

    print("Computing EFPs...")
    ref_features = compute_efps(ref_events, beta=args.beta, n_jobs=args.n_jobs)
    qcd_features = compute_efps(qcd_events, beta=args.beta, n_jobs=args.n_jobs)
    top_features = compute_efps(top_events, beta=args.beta, n_jobs=args.n_jobs)

    print("Fitting whitener...")
    mu, vecs, vals = fit_whitener(ref_features, eps=args.eps)
    qcd_white = apply_whitener(qcd_features, mu, vecs, vals)
    top_white = apply_whitener(top_features, mu, vecs, vals)

    raw_qcd = centered_spectrum(qcd_features)
    raw_top = centered_spectrum(top_features)
    white_qcd = centered_spectrum(qcd_white)
    white_top = centered_spectrum(top_white)
    ref_white = centered_spectrum(apply_whitener(ref_features, mu, vecs, vals))

    npz_out = args.outdir / "efp_reference_whitening.npz"
    png_out = args.outdir / f"efp_reference_whitening_t{args.n_real}_xmax{args.n_real}.png"
    np.savez(
        npz_out,
        raw_qcd=raw_qcd,
        raw_top=raw_top,
        white_qcd=white_qcd,
        white_top=white_top,
        ref_white=ref_white,
        whitening_eigenvalues=vals,
        whitening_rank=len(vals),
        beta=args.beta,
        eps=args.eps,
        n_reference=args.n_reference,
        n_reference_particles=args.n_reference_particles,
        n_real=args.n_real,
        seed=args.seed,
    )
    plot_compare(raw_qcd, raw_top, white_qcd, white_top, png_out, max_i=args.n_real)

    print(f"Reference features: {ref_features.shape}")
    print(f"QCD/top features: {qcd_features.shape} / {top_features.shape}")
    print(f"Whitening rank: {len(vals)} / {ref_features.shape[1]}")
    print(f"Reference whitened eigenvalue range: {ref_white[0]:.3g} .. {ref_white[len(vals)-1]:.3g}")
    print(f"Saved arrays to {npz_out}")
    print(f"Saved plot to {png_out}")


if __name__ == "__main__":
    main()
