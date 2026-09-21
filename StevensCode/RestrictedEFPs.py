import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import pickle

# From Yoni's ThrustScalingLaws_overparam.py


def compute_special_torch(events_np, dmax, device='cpu', dtype=torch.float64):
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
    
    # type 2: 
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


def efps_chunked(events, dmax, device='cpu', chunk=20000):
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