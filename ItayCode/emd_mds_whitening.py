#!/usr/bin/env python3
"""Reference-whiten an EMD-induced embedding with classical MDS.

EMD is a metric, not an explicit finite feature basis like EFPs. This script
constructs an analogous basis by:

1. Computing EMD distances among reference/noise events.
2. Turning the reference distance matrix into a classical-MDS embedding.
3. Whitening the positive-eigenvalue MDS coordinates on the reference events.
4. Projecting real jets into that same embedding using EMD distances to the
   reference set.
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
import wasserstein
from energyflow.emd import emd
from energyflow.utils.event_utils import gen_massless_phase_space
from energyflow.utils.particle_utils import center_ptyphims, ptyphims_from_p4s

from efp_spectrum import load_random_events_both

wasserstein.without_openmp()


def to_centered_ptyphi(events: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for event in events:
        keep = event[:, 0] > 0.0
        ptyphim = ptyphims_from_p4s(event[keep], phi_ref="hardest", mass=True)
        centered = center_ptyphims(ptyphim, center="ptscheme")
        arr = centered[:, :3]
        arr[:, 0] /= arr[:, 0].sum()
        out.append(arr)
    return out


def emd_distance(a: np.ndarray, b: np.ndarray, radius: float, beta: float) -> float:
    return float(
        emd(
            a,
            b,
            R=radius,
            beta=beta,
            norm=False,
            gdim=2,
            periodic_phi=False,
        )
    )


def pairwise_emd(events: list[np.ndarray], radius: float, beta: float) -> np.ndarray:
    n = len(events)
    distances = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        if i % max(1, n // 10) == 0:
            print(f"  reference pairwise row {i + 1}/{n}")
        for j in range(i + 1, n):
            d = emd_distance(events[i], events[j], radius=radius, beta=beta)
            distances[i, j] = distances[j, i] = d
    return distances


def cross_emd(
    events: list[np.ndarray], reference: list[np.ndarray], radius: float, beta: float
) -> np.ndarray:
    out = np.empty((len(events), len(reference)), dtype=np.float64)
    for i, event in enumerate(events):
        if i % max(1, len(events) // 10) == 0:
            print(f"  cross distances row {i + 1}/{len(events)}")
        for j, ref in enumerate(reference):
            out[i, j] = emd_distance(event, ref, radius=radius, beta=beta)
    return out


def fit_reference_mds(distances: np.ndarray, eps: float) -> dict[str, np.ndarray]:
    d2 = distances**2
    row_mean = d2.mean(axis=1)
    grand_mean = d2.mean()
    centered = -0.5 * (d2 - row_mean[:, None] - row_mean[None, :] + grand_mean)
    vals, vecs = np.linalg.eigh(0.5 * (centered + centered.T))
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    keep = vals > eps * vals[0]
    vals = vals[keep]
    vecs = vecs[:, keep]
    n_ref = len(distances)
    ref_raw = vecs * np.sqrt(vals)
    ref_white = vecs * np.sqrt(n_ref)
    return {
        "eigenvalues": vals,
        "eigenvectors": vecs,
        "row_mean_d2": row_mean,
        "grand_mean_d2": np.array(grand_mean),
        "ref_raw": ref_raw,
        "ref_white": ref_white,
    }


def project_mds(cross_distances: np.ndarray, model: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    d2 = cross_distances**2
    mean_new = d2.mean(axis=1)
    k = -0.5 * (
        d2
        - model["row_mean_d2"][None, :]
        - mean_new[:, None]
        + float(model["grand_mean_d2"])
    )
    vecs = model["eigenvectors"]
    vals = model["eigenvalues"]
    raw = (k @ vecs) / np.sqrt(vals)
    white = (k @ vecs) * np.sqrt(len(model["row_mean_d2"])) / vals
    return raw, white


def centered_spectrum(features: np.ndarray) -> np.ndarray:
    x = features - features.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / len(x)
    vals = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    return np.clip(vals[::-1], 0.0, None)


def plot_spectra(curves: list[tuple[np.ndarray, str, str, str]], outpath: Path) -> None:
    n = min(len(y) for y, *_ in curves)
    xs = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(5.3, 4.0), constrained_layout=True)
    for y, label, color, ls in curves:
        y = y[:n]
        pos = y > 0
        ax.loglog(xs[pos], y[pos], color=color, linestyle=ls, lw=2, label=label)
    ax.set_xlim(1, n)
    ax.set_xlabel(r"$i$")
    ax.set_ylabel(r"$\lambda_i$")
    ax.set_title("EMD-MDS covariance before/after reference whitening")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=ROOT / "data" / "train.parquet")
    parser.add_argument("--n-real", type=int, default=30, help="QCD/top events per class.")
    parser.add_argument("--n-reference", type=int, default=40)
    parser.add_argument("--n-reference-particles", type=int, default=50)
    parser.add_argument("--radius", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "emd_mds_whitening")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Sampling real jets...")
    _, _, qcd, top = load_random_events_both(
        args.parquet,
        n_events=args.n_real,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    for events in (qcd, top):
        for event in events:
            event[:, 0] /= event[:, 0].sum()

    print("Generating RAMBO reference events...")
    ref_p4 = gen_massless_phase_space(
        args.n_reference,
        args.n_reference_particles,
        energy=1.0,
        seed=args.seed + 1,
    )
    reference = to_centered_ptyphi(ref_p4)

    print("Computing reference EMD matrix...")
    d_ref = pairwise_emd(reference, radius=args.radius, beta=args.beta)
    model = fit_reference_mds(d_ref, eps=args.eps)
    print(f"Retained MDS rank: {len(model['eigenvalues'])}/{args.n_reference}")

    print("Computing QCD-reference EMDs...")
    d_qcd = cross_emd(qcd, reference, radius=args.radius, beta=args.beta)
    print("Computing top-reference EMDs...")
    d_top = cross_emd(top, reference, radius=args.radius, beta=args.beta)

    qcd_raw, qcd_white = project_mds(d_qcd, model)
    top_raw, top_white = project_mds(d_top, model)
    ref_raw = model["ref_raw"]
    ref_white = model["ref_white"]

    specs = {
        "ref_raw": centered_spectrum(ref_raw),
        "ref_white": centered_spectrum(ref_white),
        "qcd_raw": centered_spectrum(qcd_raw),
        "top_raw": centered_spectrum(top_raw),
        "qcd_white": centered_spectrum(qcd_white),
        "top_white": centered_spectrum(top_white),
    }

    stem = f"emd_mds_ref{args.n_reference}_real{args.n_real}_beta{args.beta:g}"
    npz_out = args.outdir / f"{stem}.npz"
    png_out = args.outdir / f"{stem}_xmax{len(model['eigenvalues'])}.png"
    np.savez(
        npz_out,
        **specs,
        d_ref=d_ref,
        eigenvalues=model["eigenvalues"],
        n_real=args.n_real,
        n_reference=args.n_reference,
        n_reference_particles=args.n_reference_particles,
        beta=args.beta,
        radius=args.radius,
        seed=args.seed,
    )
    plot_spectra(
        [
            (specs["qcd_raw"], "raw EMD-MDS QCD", "C0", "-"),
            (specs["top_raw"], "raw EMD-MDS top", "C0", "--"),
            (specs["qcd_white"], "whitened EMD-MDS QCD", "C3", "-"),
            (specs["top_white"], "whitened EMD-MDS top", "C3", "--"),
            (specs["ref_white"], "whitened reference", "0.4", ":"),
        ],
        png_out,
    )
    print(f"Reference whitened eig range: {specs['ref_white'][0]:.3g} .. {specs['ref_white'][-1]:.3g}")
    print(f"Saved arrays to {npz_out}")
    print(f"Saved plot to {png_out}")


if __name__ == "__main__":
    main()
