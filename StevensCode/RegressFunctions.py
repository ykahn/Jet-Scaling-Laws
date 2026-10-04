import numpy as np
import torch
from scipy.optimize import curve_fit, brentq
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import LinearOperator, eigsh






def ridge_fit(X_train, y_train, X_test, λ=1e-8, center=False, y_mean=None, kernel=None):
    X = torch.tensor(X_train, dtype=torch.float64)
    Xt = torch.tensor(X_test, dtype=torch.float64)
    y = torch.tensor(y_train, dtype=torch.float64)
    P, N  = X.shape[0], X.shape[1]
    if center and y_mean is None:
        y_mean = y.mean() 
    y_mean = y_mean if center else 0.0 
    y = y - y_mean
    ## Kernel Regression ## 
    if kernel=="NTK": 
        A = ntk(X, X) + λ * P * torch.eye(P, dtype=torch.float64)
        return (ntk(Xt, X) @ torch.linalg.solve(A, y) + y_mean).numpy()
    ## Linear Regression ## 
    if λ==0: 
        w = torch.linalg.lstsq(X, y).solution
    elif P < N:  # Overparam
        A = X @ X.T + λ * P * torch.eye(P, dtype=torch.float64)
        w = X.T @ torch.linalg.solve(A, y)
    elif P >= N: # Underparam
        XtX = X.T @ X + λ * P * torch.eye(N, dtype=torch.float64)
        w = torch.linalg.solve(XtX, X.T @ y)
    return (torch.tensor(X_test, dtype=torch.float64) @ w + y_mean).numpy()



def run_regression_sweep(X, y, train_sizes, λ=1e-8, n_repeats=250, test_size=0.2, w_true=None, 
                         filter_zero_var=False, center=False, scale=False, kern=None):
    X, y = np.array(X), np.array(y)
    idx = np.random.default_rng(42).permutation(len(X))
    X, y = X[idx], y[idx]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    y_global_mean = y_train.mean() if center else None
    if filter_zero_var:
        mask = np.std(X_train, axis=0) > 1e-10
        X_train, X_test = X_train[:, mask], X_test[:, mask]
        if w_true is not None:
            w_true = np.asarray(w_true)[mask]

    if center or scale:
        scaler = StandardScaler(with_mean=center, with_std=scale)
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)
    else:
        X_train_s, X_test_s = X_train, X_test

    if w_true is not None: 
        w_true = np.asarray(w_true)
        y_true_test = X_test @ w_true 
########## Define NTK kernel  ################
    if kern == "NTK":
        print(X_train_s.shape, X_test_s.shape, f"{8*len(X_train_s)**2/1e9:.1f} GB per train Gram")
        Xtr_t = torch.tensor(X_train_s, dtype=torch.float64)
        Xte_t = torch.tensor(X_test_s,  dtype=torch.float64)
        # G_tr = ntk(Xtr_t, Xtr_t)
        # G_te = ntk(Xte_t, Xtr_t)
##############################################
    losses, stds = [], []
    for n in train_sizes:
        # n_reps = min(n_repeats, max(50, 50_000 // n))  # fewer repeats at large n
        ls = []
        reps = n_repeats(n) if callable(n_repeats) else n_repeats
        for s in range(reps):
            sub_idx = np.random.default_rng(s).choice(len(X_train_s), n, replace=False)
            # y_pred = ridge_fit(X_train_s[sub_idx], y_train[sub_idx], X_test_s, λ, center=center, y_mean=y_global_mean, kernel=kern)
################################################################
            if kern == "NTK":
                s_ = torch.as_tensor(sub_idx)
                Ktr, Kte = ntk(Xtr_t[s_], Xtr_t[s_]), ntk(Xte_t, Xtr_t[s_])
                A  = Ktr + λ * n * torch.eye(n, dtype=torch.float64)
                ym = y_global_mean if center else 0.0
                yt = torch.tensor(y_train[sub_idx] - ym, dtype=torch.float64)
                # y_pred = (G_te[:, s_] @ torch.linalg.solve(A, yt)).numpy()
                y_pred = (Kte @ torch.linalg.solve(A, yt)).numpy() + ym
            else:
                y_pred = ridge_fit(X_train_s[sub_idx], y_train[sub_idx], X_test_s, λ, center=center, y_mean=y_global_mean, kernel=kern)
################################################################
            target = y_true_test if w_true is not None else y_test
            ls.append(mean_squared_error(target, y_pred))
        losses.append(np.mean(ls))
        stds.append(np.std(ls))
        print(f'  {n:6d}  |  {losses[-1]:.6f}')
    return np.array(losses), np.array(stds)




def fit_floor(P, losses, fit_min, fit_max, p0=None, fix_L_inf=None, err=None):
    """Fit losses(P) = A * P**-alpha + L_inf via nonlinear least squares, using only P in [fit_min, fit_max].
    If fix_L_inf is given, L_inf is held fixed at that value (only A, alpha are fit).
    If err is given (same length as P), points are weighted by inverse variance (absolute_sigma=True).
    Returns (A, alpha, L_inf)."""
    P = np.asarray(P, dtype=float)
    losses = np.asarray(losses, dtype=float)
    mask = (P >= fit_min) & (P <= fit_max)
    P_fit, losses_fit = P[mask], losses[mask]
    err_fit = np.asarray(err, dtype=float)[mask] if err is not None else None

    if fix_L_inf is not None:
        model = lambda P, A, alpha: A * P**(-alpha) + fix_L_inf
        if p0 is None:
            p0 = [max(losses_fit[0] - fix_L_inf, 1e-12), 1.0]
        (A, alpha), _ = curve_fit(model, P_fit, losses_fit, p0=p0, sigma=err_fit, absolute_sigma=err_fit is not None)
        return A, alpha, fix_L_inf

    def model(P, A, alpha, L_inf):
        return A * P**(-alpha) + L_inf

    if p0 is None:
        p0 = [max(losses_fit[0] - losses_fit[-1], 1e-12), 1.0, max(losses_fit[-1], 1e-12)]

    (A, alpha, L_inf), _ = curve_fit(model, P_fit, losses_fit, p0=p0, bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
                                      sigma=err_fit, absolute_sigma=err_fit is not None)
    return A, alpha, L_inf









# def run_sweep(X, y, w_true, r, Ns, n_repeats, push):
#     excess_loss = []
#     X = torch.as_tensor(X, dtype=torch.float64)
#     y = torch.as_tensor(y, dtype=torch.float64)
#     for i, N in enumerate(Ns):
#         print(f'on N {i} out of {len(Ns)}')
#         losses = []
#         w_true = torch.as_tensor(w_true, dtype=torch.float64)
#         for _ in range(n_repeats(N)):
#             X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=N)
#             w_hat = ridge_fit(X_train, y_train, r, push)
#             losses.append(torch.mean((X_test @ (w_hat - w_true)) ** 2).item())
#         excess_loss.append(np.mean(losses))
#     return np.array(excess_loss)



#########################
#### ===== NTK ===== #### 
#########################

## Plot NTK Spectrum ## 


def preprocess(Z):
    Z = Z[:, Z.std(0) > 0]                      # drop constant columns (d=0 EFP, β=2 constants, all-zero Q entries)
    Z = (Z - Z.mean(0)) / Z.std(0)              # standardize each column
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)   # unit-normalize each event

def ntk(x, xp):
    nx, nxp = x.norm(dim=1, keepdim=True), xp.norm(dim=1, keepdim=True)
    u = (x @ xp.T / (nx * nxp.T)).clamp(-1, 1)
    # u = x @ xp.T / (nx * nxp.T)
    th = torch.arccos(u)
    return (nx * nxp.T) * (torch.sin(th) + 2*(torch.pi - th)*u) / (2*torch.pi)


def kernel_spectrum(X, kernel=None, M=5000, seed=0):
    """Empirical Mercer spectrum under the distribution of rows of X."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(M, len(X)), replace=False)
    Xs = torch.as_tensor(np.asarray(X)[idx], dtype=torch.float64)
    K = ntk(Xs, Xs) if kernel == "NTK" else Xs @ Xs.T
    # return np.clip(np.linalg.eigvalsh((K / len(idx)).numpy())[::-1], 1e-300, None)
    return np.linalg.eigvalsh((K / len(idx)).numpy())[::-1]


def kernel_spectrum_lanczos(X, k=2000, chunk=5000, dev='cuda'):   # top-k NTK eigenvalues using all N rows, never forms NxN
    X = torch.as_tensor(X, dtype=torch.float64, device=dev)
    def Kv(v):
        v = torch.as_tensor(np.asarray(v, dtype=np.float64), device=dev).ravel(); 
        out = torch.empty(len(X), dtype=torch.float64, device=dev)
        for i in range(0, len(X), chunk): out[i:i+chunk] = ntk(X[i:i+chunk], X) @ v
        return (out / len(X)).cpu().numpy()
    return eigsh(LinearOperator((len(X), len(X)), matvec=Kv, dtype=np.float64), k=k, which='LA', return_eigenvectors=False)[::-1]


def fit_spectrum_exponent(ev, lo=10, hi=None):
    """Fit ev[i] ~ i**-b over the reliable window; b is your effective 'a'."""
    hi = hi or len(ev) // 4
    i = np.arange(lo, hi)
    return -np.polyfit(np.log(i + 1.), np.log(ev[lo:hi]), 1)[0]











########### Cengiz Eq 25, ****  written from Claude and not verified ***** #############
def solve_kappa(eigs, P, λ):
    """Renormalized ridge κ from Eq. 23 of arXiv:2405.00592:  κ(1 - (1/P) Σ η/(η+κ)) = λ."""
    eigs = np.asarray(eigs, dtype=float)
    f = lambda kap: kap * (1.0 - np.sum(eigs / (eigs + kap)) / P) - λ
    hi = max(10 * eigs.max(), 10 * λ, 1.0)
    while f(hi) < 0:
        hi *= 10.0
    return brentq(f, 1e-300, hi, rtol=1e-15, maxiter=500)


def eg_theory(eigs, target_power, P, λ, sigma2=0.0):
    """Generalization error, Eq. 25 of arXiv:2405.00592 (Atanasov, Zavatone-Veth, Pehlevan):

        E_g = κ²/(1-γ) Σ_k  v_k² / (κ + η_k)²  +  σ² γ/(1-γ)

    eigs         : η_k, covariance eigenvalues (linear) or Mercer eigenvalues (kernel).
    target_power : v_k² = η_k * w_k², the target's power in eigenmode k.
    λ            : ridge, in the same convention as ridge_fit (A = K + λ P I).

    Takes the full spectrum, so it needs no power-law assumption. Returns (E_g, κ, γ)."""
    eigs = np.asarray(eigs, dtype=float)
    v2 = np.asarray(target_power, dtype=float)
    kap = solve_kappa(eigs, P, λ)
    gam = np.sum(eigs**2 / (eigs + kap)**2) / P
    Eg = kap**2 / (1 - gam) * np.sum(v2 / (eigs + kap)**2) + sigma2 * gam / (1 - gam)
    return Eg, kap, gam


def spectrum_and_target(X, y, kernel=None):
    """Mercer eigenvalues and per-mode target power (η_k, v_k²) for use with eg_theory.
    For kernel != None the target must be projected onto the gram eigenvectors, so this
    needs eigenvectors and is therefore more expensive than eigenvalues alone."""
    Xs = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    K = ntk(Xs, Xs) if kernel == "NTK" else Xs @ Xs.T
    M = len(Xs)
    ev, U = np.linalg.eigh((K / M).numpy())
    ev, U = ev[::-1], U[:, ::-1]
    return np.clip(ev, 0.0, None), (U.T @ np.asarray(y, dtype=float))**2 / M

