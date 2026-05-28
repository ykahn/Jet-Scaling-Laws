# Analytic EFP Covariance Normalization

This note records the clean analytic object we want if we want to replace the
Monte Carlo RAMBO whitening with an analytic normalization.

## Setup

Let the feature vector be the selected EFPs

```text
phi(x) = (EFP_1(x), ..., EFP_d(x)).
```

Pick a reference/noise distribution `P0`, for example RAMBO phase space or a
large-`N` thermal phase-space limit. Define

```text
mu_i = E_0[phi_i]
C_ij = E_0[(phi_i - mu_i)(phi_j - mu_j)].
```

If `C` is known, diagonalize

```text
C = U diag(lambda) U^T
```

and keep only numerically/analytically nonzero modes. The orthonormalized EFP
coordinates are

```text
psi_a(x) = sum_i U_{ia} (phi_i(x) - mu_i) / sqrt(lambda_a).
```

Equivalently, with `W = U diag(lambda^{-1/2})`,

```text
psi(x) = (phi(x) - mu)^T W,
E_0[psi_a psi_b] = delta_ab.
```

Applying this same `W` to a real distribution `P` gives

```text
C_real^(orth) = W^T Cov_P[phi] W.
```

If `C_real^(orth)` is still structured/power-law-like, the structure is not
just the non-orthogonality of the original EFP basis relative to `P0`.

## Why EFPs Help

The product of two EFPs is another EFP: graph multiplication is disjoint union.
Therefore

```text
E_0[EFP_G EFP_H] = E_0[EFP_{G union H}]
```

where `G union H` is the disconnected graph obtained by multiplying the two
observables. Thus

```text
C_GH = m_{G union H} - m_G m_H,
m_G = E_0[EFP_G].
```

So analytic whitening reduces to analytic EFP moments `m_G` under the reference
distribution. This is the sense in which the covariance normalization can be
done analytically if those moments are known.

## Caveats

- The covariance should be centered if the goal is an orthonormal basis.
- The paper's Fig. 4 uses an uncentered `E[x_i x_j]` spectrum; for basis
  orthonormalization, centered covariance is the more natural object.
- The EFP basis is overcomplete, so `C` is singular or ill-conditioned. The
  analytic version still needs a rank/null-space prescription.
- Averaging eigenvector rotation matrices is not well-defined; average
  moments/covariances first, then diagonalize once.

## Practical Next Step

For full `EFPSet(("d<=", 6), measure="eeefm", beta=2, coords="epxpypz")`, the
matrix entries can in principle be filled by computing moments of the product
graphs. The immediately useful target is:

```text
for all selected graphs G, H:
    C_GH = moment(disjoint_union(G, H)) - moment(G) moment(H)
```

where `moment(...)` is either:

1. An analytic formula from the phase-space calculation.
2. A high-statistics Monte Carlo estimate, which is what
   `whiten_efp_basis_averaged_rambo.py` currently approximates.

