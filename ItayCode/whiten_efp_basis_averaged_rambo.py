#!/usr/bin/env python3
"""Whiten EFPs using an averaged RAMBO covariance estimate.

Averaging rotation matrices from separate covariance diagonalizations is not
well-defined because eigenvector signs, ordering, and near-degenerate subspaces
can fluctuate. This script instead averages the reference distribution itself:
it accumulates first and second EFP moments over many independent RAMBO batches,
then diagonalizes the resulting covariance once.
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
from energyflow.utils.event_utils import gen_massless_phase_space

from efp_spectrum import load_random_events_both
from whiten_efp_basis import apply_whitener, centered_spectrum, compute_efps, plot_compare


def accumulate_rambo_moments(
    n_batches: int,
    batch_size: int,
    n_particles: int,
    beta: float,
    seed: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    sum_x = None
    sum_xx = None
    total = 0

    for batch_idx in range(n_batches):
        print(f"RAMBO batch {batch_idx + 1}/{n_batches}")
        events = gen_massless_phase_space(
            batch_size,
            n_particles,
            energy=1.0,
            seed=seed + batch_idx,
        )
        features = compute_efps(events, beta=beta, n_jobs=n_jobs)
        if sum_x is None:
            n_features = features.shape[1]
            sum_x = np.zeros(n_features, dtype=np.float64)
            sum_xx = np.zeros((n_features, n_features), dtype=np.float64)
        sum_x += features.sum(axis=0)
        sum_xx += features.T @ features
        total += len(features)

    assert sum_x is not None and sum_xx is not None
    return sum_x, sum_xx, total


def fit_whitener_from_moments(
    sum_x: np.ndarray, sum_xx: np.ndarray, total: int, eps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = sum_x / total
    cov = sum_xx / total - np.outer(mu, mu)
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    keep = vals > eps * vals[0]
    return mu, vecs[:, keep], vals[keep]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=ROOT / "data" / "train.parquet")
    parser.add_argument("--n-real", type=int, default=100)
    parser.add_argument("--n-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--n-reference-particles", type=int, default=50)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--parquet-batch-size", type=int, default=8192)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "whitening_rambo_avg100")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Sampling real jets...")
    qcd_events, top_events, _, _ = load_random_events_both(
        args.parquet,
        n_events=args.n_real,
        seed=args.seed,
        batch_size=args.parquet_batch_size,
    )

    print("Accumulating RAMBO moments...")
    sum_x, sum_xx, total = accumulate_rambo_moments(
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        n_particles=args.n_reference_particles,
        beta=args.beta,
        seed=args.seed + 10_000,
        n_jobs=args.n_jobs,
    )

    print("Fitting whitener from averaged covariance...")
    mu, vecs, vals = fit_whitener_from_moments(sum_x, sum_xx, total, eps=args.eps)

    print("Computing real EFPs...")
    qcd_features = compute_efps(qcd_events, beta=args.beta, n_jobs=args.n_jobs)
    top_features = compute_efps(top_events, beta=args.beta, n_jobs=args.n_jobs)
    qcd_white = apply_whitener(qcd_features, mu, vecs, vals)
    top_white = apply_whitener(top_features, mu, vecs, vals)

    raw_qcd = centered_spectrum(qcd_features)
    raw_top = centered_spectrum(top_features)
    white_qcd = centered_spectrum(qcd_white)
    white_top = centered_spectrum(top_white)

    # Independent heldout RAMBO check with the same total number of events as one pass.
    print("Computing heldout RAMBO cross-check...")
    heldout = gen_massless_phase_space(
        total,
        args.n_reference_particles,
        energy=1.0,
        seed=args.seed + 20_000,
    )
    heldout_features = compute_efps(heldout, beta=args.beta, n_jobs=args.n_jobs)
    heldout_white = centered_spectrum(apply_whitener(heldout_features, mu, vecs, vals))

    npz_out = args.outdir / "efp_avg_rambo_whitening.npz"
    png_out = args.outdir / f"efp_avg_rambo_whitening_t{args.n_real}_xmax{len(vals)}.png"
    np.savez(
        npz_out,
        raw_qcd=raw_qcd,
        raw_top=raw_top,
        white_qcd=white_qcd,
        white_top=white_top,
        heldout_rambo_white=heldout_white,
        whitening_eigenvalues=vals,
        whitening_rank=len(vals),
        total_reference=total,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        n_reference_particles=args.n_reference_particles,
        beta=args.beta,
        eps=args.eps,
        seed=args.seed,
    )
    plot_compare(raw_qcd, raw_top, white_qcd, white_top, png_out, max_i=args.n_real)

    # Separate heldout reference diagnostic.
    xs = np.arange(1, len(heldout_white) + 1)
    fig, ax = plt.subplots(figsize=(5.3, 4.0), constrained_layout=True)
    ax.semilogx(xs, heldout_white, lw=2, label="heldout RAMBO after averaged whitening")
    ax.axhline(1.0, color="0.3", linestyle=":", label="ideal")
    ax.set_xlim(1, len(heldout_white))
    ax.set_xlabel(r"$i$")
    ax.set_ylabel(r"$\lambda_i$")
    ax.set_title("Heldout RAMBO check for averaged EFP whitening")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    heldout_png = args.outdir / f"heldout_rambo_check_xmax{len(vals)}.png"
    fig.savefig(heldout_png, dpi=200)
    plt.close(fig)

    print(f"Total reference events: {total}")
    print(f"Whitening rank: {len(vals)} / {len(mu)}")
    print(f"Heldout RAMBO eig median: {np.median(heldout_white):.3g}")
    print(f"Heldout RAMBO eig central 90%: {np.quantile(heldout_white, [0.05, 0.95])}")
    print(f"Saved arrays to {npz_out}")
    print(f"Saved plot to {png_out}")
    print(f"Saved heldout check to {heldout_png}")


if __name__ == "__main__":
    main()
