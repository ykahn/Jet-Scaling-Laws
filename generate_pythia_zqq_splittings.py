"""Generate e+e- -> Z -> q qbar at the Z pole (ISR off, no hadronization) and
save parton-level 4-momentum snapshots after each FSR splitting.

For every event with at least K total final-state-radiation branchings, the
parton 4-momenta after the k-th branching are extracted for k = 1, ..., K.
Snapshots from different events that share the same shower step are stacked
into a single float32 array of shape (N_k, 2 + k, 4).

Outputs (written to --output-dir):

    splittings_k.npy  float32, shape (N_k, 2 + k, 4)
                      each row holds the (E, px, py, pz) of the 2 + k
                      active partons after the k-th branching.
    event_ids_k.npy   int64, shape (N_k,)
                      original PYTHIA event index for each row of
                      splittings_k.npy (so the same event can be tracked
                      across snapshot files).
    metadata.json     run configuration and per-step shape summary.

These are plain numpy files so the script has no torch dependency and can
run in the same env as pythia8. Load as torch tensors with:
    torch.from_numpy(np.load("splittings_k.npy"))

A per-snapshot energy-momentum conservation check is run before writing,
comparing the sum of parton 4-momenta in each row to the expected total
(E_cm, 0, 0, 0). The script exits with a nonzero status if any snapshot
falls outside the user-specified tolerance.

Usage:
    python generate_pythia_zqq_splittings.py [-n NEVENTS] [--ecm ECM]
                                             [--seed SEED]
                                             [--output-dir DIR]
                                             [--tolerance TOL]
"""

from __future__ import annotations

import argparse
import json
import secrets
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

# PYTHIA accepts seeds in [1, 900_000_000] when Random:setSeed = on.
PYTHIA_SEED_MAX = 900_000_000

try:
    import pythia8
except ImportError as exc:
    raise SystemExit(
        "pythia8 Python bindings are not importable. On this machine they "
        "live in the `madgraph` conda env; run with "
        "/opt/anaconda3/envs/madgraph/bin/python (or `conda activate madgraph` first)."
    ) from exc


def configure_pythia(seed: int, ecm: float) -> "pythia8.Pythia":
    """Configure a Pythia instance for e+e- -> gamma*/Z -> q qbar at sqrt(s) = ecm
    with ISR off, MPI off, FSR on, and hadronization off."""
    p = pythia8.Pythia()
    settings = [
        "Random:setSeed = on",
        f"Random:seed = {seed}",
        "Beams:idA = 11",
        "Beams:idB = -11",
        f"Beams:eCM = {ecm}",
        # e+ e- -> gamma*/Z; force Z (id 23) to decay only to quarks.
        "WeakSingleBoson:ffbar2gmZ = on",
        "23:onMode = off",
        "23:onIfAny = 1 2 3 4 5",
        # Make incoming leptons point-like with the full beam energy. Without
        # this the default lepton PDF puts a bremsstrahlung tail on the e+ /
        # e- energy and leaks momentum into status-63 beam photons even with
        # PartonLevel:ISR off, so the Z would not sit at rest in the CM frame.
        "PDF:lepton = off",
        # Turn off the QCD/QED ISR shower and MPI as well.
        "PartonLevel:ISR = off",
        "PartonLevel:MPI = off",
        # Keep final-state radiation on so we actually have splittings.
        "PartonLevel:FSR = on",
        # Stop at parton level: no hadronization, no hadron decays.
        "HadronLevel:all = off",
        # Quiet banner / per-event info.
        "Init:showProcesses = off",
        "Init:showMultipartonInteractions = off",
        "Init:showChangedSettings = off",
        "Init:showChangedParticleData = off",
        "Next:numberShowEvent = 0",
        "Next:numberShowInfo = 0",
        "Next:numberShowProcess = 0",
        "Next:numberCount = 0",
    ]
    for s in settings:
        p.readString(s)
    if not p.init():
        raise RuntimeError("Pythia.init() failed; check the configuration above.")
    return p


def find_initial_zqq(event) -> List[int]:
    """Return the event-record indices of the two outgoing partons emitted
    directly by a Z (id 23). Returns an empty list if no such pair exists."""
    out: List[int] = []
    for i in range(1, event.size()):
        p = event[i]
        if p.statusAbs() != 23:
            continue
        m1 = p.mother1()
        if m1 > 0 and event[m1].id() == 23:
            out.append(i)
    return out


def extract_snapshots(event) -> List[np.ndarray]:
    """Return parton-level 4-momentum snapshots after each FSR branching.

    snapshots[k - 1] has shape (2 + k, 4) and contains the (E, px, py, pz) of
    the active partons immediately after the k-th branching. Snapshots are
    ordered by event-record index within each row, which is also a valid
    proxy for branching time order under PYTHIA's pT-ordered shower.

    The algorithm walks the event record forward starting just after the
    initial Z -> q qbar pair. A "branching" is identified as a maximal run of
    consecutive entries whose mother1() is in the current active set; this
    captures the 2 status-51 daughters of the emitter and (when present) the
    single status-52 daughter of the recoiler in PYTHIA's default pT-ordered
    dipole shower, without relying on the exact status codes.
    """
    initial = find_initial_zqq(event)
    if len(initial) != 2:
        return []

    active: Set[int] = set(initial)
    snapshots: List[np.ndarray] = []
    i = max(initial) + 1
    size = event.size()
    while i < size:
        if event[i].mother1() not in active:
            i += 1
            continue
        # Collect a maximal run of consecutive new partons whose mother1()
        # is currently active. That run is one shower step.
        j = i
        new_idx: List[int] = []
        mothers_replaced: Set[int] = set()
        while j < size and event[j].mother1() in active:
            new_idx.append(j)
            mothers_replaced.add(event[j].mother1())
            j += 1
        active -= mothers_replaced
        active |= set(new_idx)
        # Each FSR branching should add exactly one parton overall.
        if len(active) != 2 + len(snapshots) + 1:
            return snapshots
        snap = np.array(
            [[event[k].e(), event[k].px(), event[k].py(), event[k].pz()]
             for k in sorted(active)],
            dtype=np.float64,
        )
        snapshots.append(snap)
        i = j
    return snapshots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--n-events", type=int, default=100_000,
                        help="Number of events to generate.")
    parser.add_argument("--ecm", type=float, default=91.188,
                        help="Center-of-mass energy in GeV (default = Z mass).")
    parser.add_argument("--seed", type=int, default=42,
                        help="PYTHIA random seed (1 to 900_000_000). "
                             "Pass -1 to draw a fresh seed; the value used is "
                             "recorded in metadata.json for reproducibility.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("pythia_zqq_splittings"))
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="Per-component conservation tolerance (GeV).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed == -1:
        args.seed = secrets.randbelow(PYTHIA_SEED_MAX) + 1
        print(f"Drew random seed: {args.seed}")
    elif not 1 <= args.seed <= PYTHIA_SEED_MAX:
        raise SystemExit(
            f"--seed must be -1 (random) or in [1, {PYTHIA_SEED_MAX}]; got {args.seed}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pythia = configure_pythia(args.seed, args.ecm)

    # buffers[k] holds (event_id, snapshot_array) pairs for shower step k.
    buffers: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    n_generated = 0
    n_skipped = 0
    report_every = max(1, args.n_events // 20)

    for iev in range(args.n_events):
        if not pythia.next():
            n_skipped += 1
            continue
        n_generated += 1
        for step, snap in enumerate(extract_snapshots(pythia.event), start=1):
            buffers[step].append((iev, snap))
        if (iev + 1) % report_every == 0:
            print(f"  ... event {iev + 1} / {args.n_events}")

    pythia.stat()
    max_step = max(buffers) if buffers else 0
    print(f"Generated {n_generated} events ({n_skipped} skipped); "
          f"up to {max_step} branchings observed.")

    # In the lab/CM frame the Z is at rest (ISR off, beams aligned), so the
    # total 4-momentum is (E_cm, 0, 0, 0). Each shower step conserves it.
    expected = np.array([args.ecm, 0.0, 0.0, 0.0], dtype=np.float64)

    metadata: Dict = {
        "n_events_requested": args.n_events,
        "n_events_generated": n_generated,
        "n_events_skipped": n_skipped,
        "ecm_GeV": args.ecm,
        "seed": args.seed,
        "tolerance_GeV": args.tolerance,
        "expected_total_4momentum": expected.tolist(),
        "snapshot_files": {},
    }

    bad_steps: List[Tuple[int, float, int]] = []

    for step in sorted(buffers):
        ids = np.fromiter((eid for eid, _ in buffers[step]),
                          dtype=np.int64, count=len(buffers[step]))
        arr = np.stack([snap for _, snap in buffers[step]], axis=0)
        assert arr.shape[1] == 2 + step, (
            f"internal error: expected {2 + step} partons at step {step}, "
            f"got {arr.shape[1]}"
        )

        total = arr.sum(axis=1)
        dev = np.abs(total - expected)
        max_dev = float(dev.max())
        n_bad = int(np.any(dev > args.tolerance, axis=1).sum())
        flag = "OK" if n_bad == 0 else "FAIL"
        print(f"  step {step:2d}: N = {arr.shape[0]:7d}  shape = {tuple(arr.shape)}  "
              f"max |sum p - p_Z| = {max_dev:.3e} GeV  "
              f"({n_bad} rows outside tol) [{flag}]")
        if n_bad:
            bad_steps.append((step, max_dev, n_bad))

        snap_file = args.output_dir / f"splittings_{step}.npy"
        ids_file = args.output_dir / f"event_ids_{step}.npy"
        np.save(snap_file, arr.astype(np.float32))
        np.save(ids_file, ids)
        metadata["snapshot_files"][step] = {
            "snapshot_file": snap_file.name,
            "event_ids_file": ids_file.name,
            "shape": list(arr.shape),
            "max_conservation_deviation_GeV": max_dev,
            "rows_outside_tolerance": n_bad,
        }

    with open(args.output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if bad_steps:
        msg = ", ".join(
            f"step {s} ({n} rows, max dev {d:.3e} GeV)" for s, d, n in bad_steps
        )
        raise SystemExit(f"Energy-momentum conservation check FAILED: {msg}")
    print("All snapshots conserve 4-momentum within tolerance.")


if __name__ == "__main__":
    main()
