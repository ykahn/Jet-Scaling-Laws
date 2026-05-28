#!/usr/bin/env python3
"""Reproduce the EFP part of Fig. 4 left in arXiv:2312.02264.

The script uses the top-tagging train parquet file, computes the first
313 nontrivial EFPs (degree <= 6), and plots the uncentered feature covariance
spectrum from Eq. (8) separately for QCD and top jets.
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
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import wasserstein
from energyflow import EFPSet
from energyflow.emd import emd
from energyflow.utils.particle_utils import center_ptyphims, ptyphims_from_p4s

wasserstein.without_openmp()


N_CONSTITUENTS = 200
FOUR_VECTOR_COLUMNS = [
    f"{component}_{idx}"
    for idx in range(N_CONSTITUENTS)
    for component in ("E", "PX", "PY", "PZ")
]
LABEL_COLUMN = "is_signal_new"


def _arrays_to_events(arr: np.ndarray) -> list[np.ndarray]:
    events: list[np.ndarray] = []
    for event in arr:
        # The dataset is zero-padded to 200 constituents. Remove padding before
        # passing events to EnergyFlow.
        keep = event[:, 0] > 0.0
        events.append(event[keep])
    return events


def _arrays_to_centered_ptyphi(arr: np.ndarray) -> list[np.ndarray]:
    """Convert Cartesian four-vectors to centered [pT, y, phi] jets."""
    events: list[np.ndarray] = []
    for event in arr:
        keep = event[:, 0] > 0.0
        ptyphim = ptyphims_from_p4s(event[keep], phi_ref="hardest", mass=True)
        centered = center_ptyphims(ptyphim, center="ptscheme")
        events.append(centered[:, :3])
    return events


def load_head_events(parquet_path: Path, label: int, n_events: int) -> list[np.ndarray]:
    """Load the first n_events jets with a given label."""
    dataset = ds.dataset(parquet_path, format="parquet")
    table = dataset.scanner(
        columns=FOUR_VECTOR_COLUMNS + [LABEL_COLUMN],
        filter=ds.field(LABEL_COLUMN) == label,
        batch_size=max(1024, n_events),
    ).head(n_events)

    if table.num_rows != n_events:
        raise RuntimeError(
            f"Requested {n_events} events with label {label}, got {table.num_rows}."
        )

    arr = table.select(FOUR_VECTOR_COLUMNS).to_pandas().to_numpy(dtype=np.float64)
    arr = arr.reshape(n_events, N_CONSTITUENTS, 4)
    return _arrays_to_events(arr)


def load_head_events_ptyphi(parquet_path: Path, label: int, n_events: int) -> list[np.ndarray]:
    """Load the first n_events jets with a given label as centered [pT, y, phi]."""
    dataset = ds.dataset(parquet_path, format="parquet")
    table = dataset.scanner(
        columns=FOUR_VECTOR_COLUMNS + [LABEL_COLUMN],
        filter=ds.field(LABEL_COLUMN) == label,
        batch_size=max(1024, n_events),
    ).head(n_events)

    if table.num_rows != n_events:
        raise RuntimeError(
            f"Requested {n_events} events with label {label}, got {table.num_rows}."
        )

    arr = table.select(FOUR_VECTOR_COLUMNS).to_pandas().to_numpy(dtype=np.float64)
    arr = arr.reshape(n_events, N_CONSTITUENTS, 4)
    return _arrays_to_centered_ptyphi(arr)


def load_random_events(
    parquet_path: Path, n_events: int, seed: int, batch_size: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Uniformly sample n_events QCD and top jets in one streaming parquet pass."""
    rng = np.random.default_rng(seed)
    samples = {0: [], 1: []}
    seen = {0: 0, 1: 0}

    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(
        batch_size=batch_size, columns=FOUR_VECTOR_COLUMNS + [LABEL_COLUMN]
    ):
        frame = batch.to_pandas()
        labels = frame[LABEL_COLUMN].to_numpy(dtype=np.int8)
        arr = frame[FOUR_VECTOR_COLUMNS].to_numpy(dtype=np.float64)
        arr = arr.reshape(len(frame), N_CONSTITUENTS, 4)

        for event, label_value in zip(arr, labels):
            label = int(label_value)
            seen[label] += 1

            if len(samples[label]) < n_events:
                samples[label].append(event)
                continue

            replacement = rng.integers(seen[label])
            if replacement < n_events:
                samples[label][replacement] = event

    for label in (0, 1):
        if len(samples[label]) != n_events:
            raise RuntimeError(
                f"Requested {n_events} events with label {label}, got {len(samples[label])}."
            )

    return _arrays_to_events(np.asarray(samples[0])), _arrays_to_events(np.asarray(samples[1]))


def load_random_events_both(
    parquet_path: Path, n_events: int, seed: int, batch_size: int
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Randomly sample events and return both Cartesian and centered pT-y-phi views."""
    rng = np.random.default_rng(seed)
    samples = {0: [], 1: []}
    seen = {0: 0, 1: 0}

    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(
        batch_size=batch_size, columns=FOUR_VECTOR_COLUMNS + [LABEL_COLUMN]
    ):
        frame = batch.to_pandas()
        labels = frame[LABEL_COLUMN].to_numpy(dtype=np.int8)
        arr = frame[FOUR_VECTOR_COLUMNS].to_numpy(dtype=np.float64)
        arr = arr.reshape(len(frame), N_CONSTITUENTS, 4)

        for event, label_value in zip(arr, labels):
            label = int(label_value)
            seen[label] += 1

            if len(samples[label]) < n_events:
                samples[label].append(event)
                continue

            replacement = rng.integers(seen[label])
            if replacement < n_events:
                samples[label][replacement] = event

    qcd_arr = np.asarray(samples[0])
    top_arr = np.asarray(samples[1])
    return (
        _arrays_to_events(qcd_arr),
        _arrays_to_events(top_arr),
        _arrays_to_centered_ptyphi(qcd_arr),
        _arrays_to_centered_ptyphi(top_arr),
    )


def compute_efps(events: list[np.ndarray], n_jobs: int | None) -> np.ndarray:
    """Compute the 313 nontrivial EFP features used in the paper."""
    efpset = EFPSet("d<=6", measure="hadr", beta=1, normed=True, coords="epxpypz")
    features = efpset.batch_compute(events, n_jobs=n_jobs)

    if features.shape[1] != 314:
        raise RuntimeError(f"Expected 314 EFPs including trivial one, got {features.shape[1]}.")

    # The first EFP is the trivial sum_i z_i = 1, excluded in the paper.
    return features[:, 1:]


def make_reference_jet(n_ref: int, radius: float) -> np.ndarray:
    """Create an equal-weight reference jet on a uniform y-phi square grid."""
    side = int(np.ceil(np.sqrt(n_ref)))
    coords_1d = np.linspace(-radius, radius, side)
    yy, pp = np.meshgrid(coords_1d, coords_1d, indexing="ij")
    coords = np.column_stack([yy.ravel(), pp.ravel()])

    # Keep the most central n_ref grid points when side**2 is larger than n_ref.
    order = np.argsort(np.sum(coords**2, axis=1))
    coords = coords[order[:n_ref]]
    weights = np.full((n_ref, 1), 1.0 / n_ref)
    return np.column_stack([weights, coords])


def compute_lot(
    events_ptyphi: list[np.ndarray],
    n_ref: int,
    radius: float,
    use_displacements: bool,
) -> np.ndarray:
    """Compute LOT coordinates from jets to a fixed reference jet."""
    reference = make_reference_jet(n_ref=n_ref, radius=radius)
    ref_weights = reference[:, 0]
    ref_coords = reference[:, 1:3]
    features = np.empty((len(events_ptyphi), n_ref * 2), dtype=np.float64)

    for idx, event in enumerate(events_ptyphi):
        event = event[event[:, 0] > 0.0].copy()
        event[:, 0] /= np.sum(event[:, 0])

        _, flow = emd(
            reference,
            event,
            R=2.0 * radius,
            beta=2.0,
            norm=False,
            gdim=2,
            periodic_phi=False,
            return_flow=True,
        )
        flow = flow[:n_ref, : len(event)]
        transported = (flow @ event[:, 1:3]) / ref_weights[:, None]
        if use_displacements:
            transported = transported - ref_coords
        features[idx] = transported.reshape(-1)

    return features


def covariance_spectrum(features: np.ndarray) -> np.ndarray:
    """Eigenvalues of (1/T) sum_alpha x_i,alpha x_j,alpha, sorted descending."""
    t = features.shape[0]
    covariance = (features.T @ features) / t
    eigvals = np.linalg.eigvalsh(covariance)
    eigvals = np.clip(eigvals, 0.0, None)
    return eigvals[::-1]


def plot_spectra(
    spectra: list[tuple[np.ndarray, str, str, str]],
    outpath: Path,
    n_events: int,
) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    max_rank = min([n_events] + [len(spectrum) for spectrum, _, _, _ in spectra])
    xs = np.arange(1, max_rank + 1)

    for spectrum, label, color, linestyle in spectra:
        spectrum = spectrum[:max_rank]
        positive = spectrum > 0
        ax.loglog(
            xs[positive],
            spectrum[positive],
            label=label,
            lw=2,
            color=color,
            linestyle=linestyle,
        )

    ax.set_xlabel(r"$i$")
    ax.set_ylabel(r"$\lambda_i$")
    ax.set_title(f"Data covariance spectrum (T = {n_events})")
    ax.set_xlim(1, max_rank)
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.25)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        type=Path,
        default=ROOT / "data" / "train.parquet",
        help="Path to top-tagging train parquet file.",
    )
    parser.add_argument("--n-events", type=int, default=100, help="Events per class.")
    parser.add_argument("--seed", type=int, default=12345, help="Random sampling seed.")
    parser.add_argument(
        "--sampling",
        choices=("random", "head"),
        default="random",
        help="Use random reservoir sampling over the full train file, or fast file-order head sampling.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Parquet batch size for random reservoir sampling.",
    )
    parser.add_argument("--include-lot", action="store_true", help="Also compute LOT spectra.")
    parser.add_argument("--lot-ref-size", type=int, default=200, help="LOT reference jet size.")
    parser.add_argument("--lot-radius", type=float, default=0.8, help="Reference grid half-width.")
    parser.add_argument(
        "--lot-displacements",
        action="store_true",
        help="Use z_i - X_i instead of z_i as the LOT coordinate.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="EnergyFlow worker processes. Use -1/0? No: pass a positive int.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "outputs",
        help="Directory for plot and spectrum arrays.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.sampling == "head":
        qcd_events = load_head_events(args.parquet, label=0, n_events=args.n_events)
        top_events = load_head_events(args.parquet, label=1, n_events=args.n_events)
        qcd_ptyphi = load_head_events_ptyphi(args.parquet, label=0, n_events=args.n_events)
        top_ptyphi = load_head_events_ptyphi(args.parquet, label=1, n_events=args.n_events)
    else:
        qcd_events, top_events, qcd_ptyphi, top_ptyphi = load_random_events_both(
            args.parquet,
            n_events=args.n_events,
            seed=args.seed,
            batch_size=args.batch_size,
        )

    qcd_features = compute_efps(qcd_events, n_jobs=args.n_jobs)
    top_features = compute_efps(top_events, n_jobs=args.n_jobs)

    qcd_spectrum = covariance_spectrum(qcd_features)
    top_spectrum = covariance_spectrum(top_features)

    spectra = [
        (qcd_spectrum, "EFP (QCD jets)", "C0", "-"),
        (top_spectrum, "EFP (top jets)", "C0", "--"),
    ]

    lot_qcd_spectrum = None
    lot_top_spectrum = None
    if args.include_lot:
        qcd_lot = compute_lot(
            qcd_ptyphi,
            n_ref=args.lot_ref_size,
            radius=args.lot_radius,
            use_displacements=args.lot_displacements,
        )
        top_lot = compute_lot(
            top_ptyphi,
            n_ref=args.lot_ref_size,
            radius=args.lot_radius,
            use_displacements=args.lot_displacements,
        )
        lot_qcd_spectrum = covariance_spectrum(qcd_lot)
        lot_top_spectrum = covariance_spectrum(top_lot)
        spectra.extend(
            [
                (lot_qcd_spectrum, "LOT (QCD jets)", "green", "-"),
                (lot_top_spectrum, "LOT (top jets)", "green", "--"),
            ]
        )

    npz_out = args.outdir / f"efp_spectrum_t{args.n_events}.npz"
    png_out = args.outdir / f"efp_spectrum_t{args.n_events}_xmax{args.n_events}.png"

    np.savez(
        npz_out,
        efp_qcd=qcd_spectrum,
        efp_top=top_spectrum,
        lot_qcd=lot_qcd_spectrum,
        lot_top=lot_top_spectrum,
        n_events=args.n_events,
        seed=args.seed,
        sampling=args.sampling,
        lot_ref_size=args.lot_ref_size,
        lot_radius=args.lot_radius,
        lot_displacements=args.lot_displacements,
    )
    plot_spectra(
        spectra,
        png_out,
        n_events=args.n_events,
    )

    print(f"QCD features: {qcd_features.shape}")
    print(f"Top features: {top_features.shape}")
    if args.include_lot:
        print(f"QCD LOT features: {qcd_lot.shape}")
        print(f"Top LOT features: {top_lot.shape}")
    print(f"Saved spectra to {npz_out}")
    print(f"Saved plot to {png_out}")


if __name__ == "__main__":
    main()
