import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split




def ridge_fit(X_train, y_train, X_test, λ=1e-8, center=False):
    X = torch.tensor(X_train, dtype=torch.float64)
    y = torch.tensor(y_train, dtype=torch.float64)
    P, N  = X.shape[0], X.shape[1]
    y_mean = y.mean() if center else 0.0
    y = y - y_mean
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
                         filter_zero_var=False, center=False, scale=False):
    X, y = np.array(X), np.array(y)
    idx = np.random.default_rng(42).permutation(len(X))
    X, y = X[idx], y[idx]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

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
    losses, stds = [], []
    for n in train_sizes:
        # n_reps = min(n_repeats, max(50, 50_000 // n))  # fewer repeats at large n
        ls = []
        reps = n_repeats(n) if callable(n_repeats) else n_repeats
        for s in range(reps):
            sub_idx = np.random.default_rng(s).choice(len(X_train_s), n, replace=False)
            y_pred = ridge_fit(X_train_s[sub_idx], y_train[sub_idx], X_test_s, λ, center=center)
            target = y_true_test if w_true is not None else y_test
            ls.append(mean_squared_error(target, y_pred))
        losses.append(np.mean(ls))
        stds.append(np.std(ls))
        print(f'  {n:6d}  |  {losses[-1]:.6f}')
    return np.array(losses), np.array(stds)


def run_sweep(X, y, w_true, r, Ns, n_repeats, push):
    excess_loss = []
    X = torch.as_tensor(X, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    for i, N in enumerate(Ns):
        print(f'on N {i} out of {len(Ns)}')
        losses = []
        w_true = torch.as_tensor(w_true, dtype=torch.float64)
        for _ in range(n_repeats(N)):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=N)
            w_hat = ridge_fit(X_train, y_train, r, push)
            losses.append(torch.mean((X_test @ (w_hat - w_true)) ** 2).item())
        excess_loss.append(np.mean(losses))
    return np.array(excess_loss)

# def run_sklearn_regression(X, y, train_sizes, r=1e-8, n_repeats=500, test_size=20000):
#     X, y = np.array(X), np.array(y)
#     idx = np.random.default_rng(42).permutation(len(X))
#     X, y = X[idx], y[idx]
#     X_test, y_test   = X[-test_size:], y[-test_size:]
#     X_train, y_train = X[:-test_size], y[:-test_size]
#     scaler = StandardScaler()
#     X_train_s = scaler.fit_transform(X_train)
#     X_test_s  = scaler.transform(X_test)
#     losses = []
#     for n in train_sizes:
#         ls = []
#         for s in range(n_repeats):
#             sub_idx = np.random.default_rng(s).choice(len(X_train_s), n, replace=False)
#             reg = Ridge(alpha=r).fit(X_train_s[sub_idx], y_train[sub_idx])
#             ls.append(mean_squared_error(y_test, reg.predict(X_test_s)))
#         losses.append(np.mean(ls))
#         print(f'  {n:6d}  |  {losses[-1]:.6f}')
#     return losses
