import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler



# def ridge_predict_torch(X_train, y_train, X_test, r=1e-8):
#     X = torch.tensor(X_train, dtype=torch.float64)
#     y = torch.tensor(y_train, dtype=torch.float64)
#     XtX = X.T @ X + r * torch.eye(X.shape[1], dtype=torch.float64)
#     Xty = X.T @ y
#     w = torch.linalg.solve(XtX, Xty)
#     return (torch.tensor(X_test, dtype=torch.float64) @ w).numpy()

def ridge_predict_torch(X_train, y_train, X_test, r=1e-8):
    X = torch.tensor(X_train, dtype=torch.float64)
    y = torch.tensor(y_train, dtype=torch.float64)
    y_mean = y.mean()
    y = y - y_mean
    XtX = X.T @ X + r * torch.eye(X.shape[1], dtype=torch.float64)
    w = torch.linalg.solve(XtX, X.T @ y)
    return (torch.tensor(X_test, dtype=torch.float64) @ w + y_mean).numpy()


# def ridge_predict_torch(X_train, y_train, X_test, r=1e-8):
#     X = torch.tensor(X_train, dtype=torch.float64)
#     y = torch.tensor(y_train, dtype=torch.float64)
#     y_mean = y.mean()
#     y = y - y_mean
#     n, p = X.shape
#     if n < p:
#         K = X @ X.T + r * torch.eye(n, dtype=torch.float64)
#         alpha = torch.linalg.solve(K, y)
#         w = X.T @ alpha
#     else:
#         w = torch.linalg.solve(X.T @ X + r * torch.eye(p, dtype=torch.float64), X.T @ y)
#     return (torch.tensor(X_test, dtype=torch.float64) @ w + y_mean).numpy()



def run_torch_regression(X, y, train_sizes, r=1e-8, n_repeats=500, test_size=20000):
    X, y = np.array(X), np.array(y)
    idx = np.random.default_rng(42).permutation(len(X))
    X, y = X[idx], y[idx]
    X_test, y_test   = X[-test_size:], y[-test_size:]
    X_train, y_train = X[:-test_size], y[:-test_size]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    losses = []
    for n in train_sizes:
        ls = []
        for s in range(n_repeats):
            sub_idx = np.random.default_rng(s).choice(len(X_train_s), n, replace=False)
            y_pred = ridge_predict_torch(X_train_s[sub_idx], y_train[sub_idx], X_test_s, r)
            ls.append(mean_squared_error(y_test, y_pred))
        losses.append(np.mean(ls))
        print(f'  {n:6d}  |  {losses[-1]:.6f}')
    return losses


def run_sklearn_regression(X, y, train_sizes, r=1e-8, n_repeats=500, test_size=20000):
    X, y = np.array(X), np.array(y)
    idx = np.random.default_rng(42).permutation(len(X))
    X, y = X[idx], y[idx]
    X_test, y_test   = X[-test_size:], y[-test_size:]
    X_train, y_train = X[:-test_size], y[:-test_size]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    losses = []
    for n in train_sizes:
        ls = []
        for s in range(n_repeats):
            sub_idx = np.random.default_rng(s).choice(len(X_train_s), n, replace=False)
            reg = Ridge(alpha=r).fit(X_train_s[sub_idx], y_train[sub_idx])
            ls.append(mean_squared_error(y_test, reg.predict(X_test_s)))
        losses.append(np.mean(ls))
        print(f'  {n:6d}  |  {losses[-1]:.6f}')
    return losses
