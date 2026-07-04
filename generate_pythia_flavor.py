"""Item 4: hard-parton flavor scan at a common CM energy via a Higgs resonance.

Both samples are produced through e+e- -> H with the Higgs pole mass set equal to
the CM energy (``25:m0 = eCM``), so the colorless H sits on resonance and is
produced at rest -- the total 4-momentum is (eCM, 0, 0, 0) and the existing
conservation check applies unchanged.  The light-quark reference at the same
energy is the gamma*/Z -> qqbar sample from generate_pythia_ecm_scan.py (eCM=700).

    --flavor gluon : force H -> g g.  The two status-23 gluons shower via ordinary
                     FSR; snapshots use the standard FSR extractor (seeds = the
                     gluons, mother id 25).

    --flavor top   : force H -> t tbar, with W -> hadrons forced.  The tops radiate,
                     decay (t -> b W), the W's decay (W -> q qbar), and everything
                     showers.  Snapshots use the full decay+shower-tree extractor,
                     segmenting by total object multiplicity k.  Low-k snapshots are
                     the (pre-decay) t tbar + gluons system - kinematically like the
                     light-quark events - while high-k snapshots are the fully
                     decayed "fat" top final state (b + b + W-decay quarks + QCD
                     radiation); both regimes are needed in the scaling-law fits.

Usage:
    python generate_pythia_flavor.py --flavor top  [-n N] [--ecm 700] [--seed S]
    python generate_pythia_flavor.py --flavor gluon [-n N] [--ecm 700] [--seed S]
"""

from __future__ import annotations

import pythia8

from pythia_snapshot_common import (
    COMMON_EE_SETTINGS,
    build_base_parser,
    extract_fsr_snapshots,
    extract_tree_snapshots,
    find_hard_outgoing,
    resolve_seed,
    run_generation,
)


def find_higgs_children(event, child_abs_id: int):
    """Indices of partons with |id| == child_abs_id whose mother is the Higgs
    (id 25).  Used for the t/tbar seeds (status -22, so the status-23 finder
    does not apply)."""
    out = []
    for i in range(1, event.size()):
        if abs(event[i].id()) == child_abs_id:
            m = event[i].mother1()
            if m > 0 and event[m].id() == 25:
                out.append(i)
    return out


def configure_pythia(seed, ecm, flavor):
    p = pythia8.Pythia("", False)
    settings = [
        "Random:setSeed = on",
        f"Random:seed = {seed}",
        "Beams:idA = 11",
        "Beams:idB = -11",
        f"Beams:eCM = {ecm}",
        # s-channel e+e- -> H (electron Yukawa); put H on resonance at eCM.
        "HiggsSM:ffbar2H = on",
        f"25:m0 = {ecm}",
        "25:onMode = off",
        "HadronLevel:all = off",
    ]
    if flavor == "gluon":
        settings.append("25:onIfMatch = 21 21")        # H -> g g
    elif flavor == "top":
        settings.append("25:onIfMatch = 6 -6")          # H -> t tbar
        settings += ["24:onMode = off", "24:onIfAny = 1 2 3 4"]  # W -> hadrons
    else:
        raise SystemExit(f"unknown flavor {flavor!r}")
    settings.extend(COMMON_EE_SETTINGS)
    for s in settings:
        p.readString(s)
    if not p.init():
        raise RuntimeError("Pythia.init() failed; check the configuration above.")
    return p


def main() -> None:
    parser = build_base_parser(
        description=__doc__,
        default_output_dir="pythia_flavor",
        default_ecm=700.0,
    )
    parser.add_argument("--flavor", choices=("gluon", "top"), required=True,
                        help="Hard-parton flavor: gluon (H->gg) or top (H->ttbar).")
    args = parser.parse_args()
    args.seed = resolve_seed(args.seed)

    pythia = configure_pythia(args.seed, args.ecm, args.flavor)

    if args.flavor == "gluon":
        # H -> g g: status-23 gluons, pure FSR.
        extract = lambda ev: extract_fsr_snapshots(ev, find_hard_outgoing(ev, 25))
        process = "e+e- -> H -> gg (H mass = eCM)"
    else:
        # H -> t tbar: full decay+shower tree from the two tops.
        extract = lambda ev: extract_tree_snapshots(ev, find_higgs_children(ev, 6))
        process = "e+e- -> H -> ttbar (H mass = eCM), W -> hadrons"

    extra_metadata = {
        "process": process,
        "item": "4 (hard-parton flavor)",
        "flavor": args.flavor,
        "higgs_m0_GeV": args.ecm,
    }

    run_generation(
        pythia, extract,
        n_events=args.n_events, ecm=args.ecm, seed=args.seed,
        tolerance=args.tolerance, output_dir=args.output_dir,
        save_event_record=args.save_event_record, extra_metadata=extra_metadata,
    )


if __name__ == "__main__":
    main()
