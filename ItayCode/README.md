# ItayCode

Small scripts for trying EFP covariance/whitening experiments on top of Yoni's
`Jet-Scaling-Laws` repo.

## Generate shower-depth data

The repo-level generator writes `pythia_zqq_splittings/splittings_k.npy`:

```bash
python generate_pythia_zqq_splittings.py -n 100000 --output-dir pythia_zqq_splittings
```

On this machine, the current Miniconda Python does not import `pythia8`; use an
environment with PYTHIA8 Python bindings.

## Analyze EFP spectra

Once the `.npy` files exist:

```bash
/Users/itaybloch/miniconda3/bin/python ItayCode/analyze_splitting_efps.py --steps 1-4 --whiten
```

This uses Yoni's notebook convention:

```python
EFPSet(("d<=", 6), measure="eeefm", beta=2, coords="epxpypz")
```

The whitening reference is RAMBO phase space with the same particle
multiplicity as the shower snapshot.

## Other Reproduction Scripts

These scripts were copied here from the earlier scratch workspace so this folder
is the central place for our code:

- `efp_spectrum.py`: reproduces Fig. 4-left style EFP/LOT spectra on the
  top-tagging parquet dataset.
- `whiten_efp_basis.py`: fits an EFP whitening transform on one RAMBO sample
  and applies it to top-tagging jets.
- `whiten_efp_basis_averaged_rambo.py`: estimates the RAMBO covariance by
  averaging many independent RAMBO batches, then whitens.
- `emd_mds_whitening.py`: EMD analogue using classical MDS on a RAMBO reference
  distance matrix.

The large top-tagging parquet data file is still in the scratch data directory:

`/Users/itaybloch/Documents/scaling_laws_repro/data/train.parquet`

Pass that path via `--parquet` if running the parquet-based scripts from this
repo.
