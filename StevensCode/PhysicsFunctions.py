import numpy as np



def compute_thrust(event):
    """Thrust for an e+e- event. event: (N,4) array (E,px,py,pz)."""
    momenta = event[:, 1:4]                      # \vec{p}
    norms = np.linalg.norm(momenta, axis=1)      # |\vec{p}|
    momenta = momenta[norms> 1e-10]             # get rid of 
    norms = norms[norms > 1e-10]                 # particles with 0 norm ... 
    denom = np.sum(norms)                        # sum_i |p_i| 
    best_T = 0.
    for i in range(len(momenta)):                  # only scanning over momenta in the event ...  
        n = momenta[i] / norms[i]                  # \hat{p}
        T = np.sum(np.abs(momenta @ n)) / denom    # sum_i |  |
        if T > best_T:
            best_T = T
    return best_T


def compute_thrust_iterative(event, max_iter=20):
    """Exact thrust via iterated hemisphere refinement. event: (N,4) array (E,px,py,pz)."""
    momenta = event[:, 1:4]                        # \vec{p}
    norms = np.linalg.norm(momenta, axis=1)        # |\vec{p}|
    momenta = momenta[norms > 1e-10]
    norms = norms[norms > 1e-10]
    denom = np.sum(norms)     
    best_T = 0.
    for i in range(len(momenta)):
        n = momenta[i] / norms[i]
        for _ in range(max_iter):
            signs = np.sign(momenta @ n)       # assigns each momenta to a hemisphere +/-
            signs[signs == 0] = 1.
            n_new = signs @ momenta            # optimal axis is sum of momenta on + hemisphere
            norm_new = np.linalg.norm(n_new) 
            if norm_new < 1e-10:
                break
            n_new /= norm_new
            if np.abs(1. - np.abs(n_new @ n)) < 1e-12:  # check convergence
                break
            n = n_new
        T = np.sum(np.abs(momenta @ n)) / denom
        if T > best_T:
            best_T = T
    return best_T

## Invariant Mass pairs given an event 

def pair_inv(ev):   # (n, N, 4) -> sorted s_ij, i<j
    E, p = ev[..., 0], ev[..., 1:]
    s = 2*(E[:, :, None]*E[:, None, :] - np.einsum('nia,nja->nij', p, p))
    iu = np.triu_indices(ev.shape[1], 1)
    return -np.sort(-s[:, iu[0], iu[1]], axis=1)



### Larkowski QCD School Notes Eq. 56
def dim_QCD_qq(Q, alpha_s): 
    CF = 4/3
    return -16*alpha_s*CF/np.pi*np.log(Q) - 6*alpha_s*CF/np.pi

