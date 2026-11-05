"""Problem 3: Random Kitchen Sinks (RKS) + ridge vs. a 1-hidden-layer MLP on 2-D toy data.

Run from this folder:  python p3_rks_mlp.py
Outputs: figures/*.png, rks_test_predictions.csv, mlp_test_predictions.csv
"""
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataloader import HW2Dataset
from plotting import plot_surface_2d

# ---------------- hyperparameters
SEED = 0
RKS_D = 1000                          # total random features (= 2h)
RKS_LAMBDA = 1e-3                     # ridge strength
RKS_SIGMAS = np.logspace(-2, 1, 16)   # sweep: 0.01 ... 10
# MLP: sweep the hidden width, keep the others fixed
MLP_LR = 1e-2
MLP_BATCH_SIZE = 64
MLP_EPOCHS = 600
MLP_WIDTHS = [2, 4, 8, 16, 32, 64, 128, 256, 512]   # sweep

os.makedirs("figures", exist_ok=True)
ds = HW2Dataset.from_csv("p3/train.csv", "p3/val.csv", "p3/test.csv")
X_train, y_train = ds.train.X, ds.train.y
X_val, y_val = ds.val.X, ds.val.y
X_test = ds.test.X


def mse(y, pred):
    return float(np.mean((y - pred) ** 2))


# ---------------- part (a) ----------------#
fig = plt.figure(figsize=(11, 4.5))
ax = fig.add_subplot(1, 2, 1)
points = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=18)
fig.colorbar(points, ax=ax, label="y")
ax.set(xlabel="x1", ylabel="x2", title="Training data (color = y)")
ax = fig.add_subplot(1, 2, 2, projection="3d")
ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c=y_train, s=8)
ax.set(xlabel="x1", ylabel="x2", zlabel="y", title="Training data (3-D)")
plt.tight_layout()
plt.savefig("figures/a_training_data.png", dpi=200)
plt.close()


# ---------------- part (b) ----------------#
class RandomKitchenSinks:
    def __init__(self, D, sigma, ridge_lambda, seed=SEED):
        self.h, self.sigma, self.lam, self.seed = D // 2, sigma, ridge_lambda, seed

    def _features(self, X):
        proj = X @ self.W + self.b
        return np.hstack([np.sin(proj), np.cos(proj)]) / np.sqrt(self.h)

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(0, 1 / self.sigma, size=(X.shape[1], self.h))   # w_j ~ N(0, I / sigma^2)
        self.b = rng.uniform(0, 2 * np.pi, size=self.h)                    # b_j ~ U[0, 2 pi]
        phi = self._features(X)
        # ridge with an unpenalized intercept: center, solve (P^T P + lambda I) beta = P^T y
        self.phi_mean, self.y_mean = phi.mean(0), y.mean()
        P = phi - self.phi_mean
        self.coef = np.linalg.solve(P.T @ P + self.lam * np.eye(P.shape[1]), P.T @ (y - self.y_mean))
        return self

    def predict(self, X):
        return (self._features(X) - self.phi_mean) @ self.coef + self.y_mean


def plot_sweep(values, train_losses, val_losses, xlabel, title, filename, best):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, losses, name in zip(axs, (train_losses, val_losses), ("Training", "Validation")):
        ax.plot(values, losses, "o-")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvline(best, ls="--", c="gray")
        ax.set(xlabel=xlabel, ylabel="MSE", title=f"{title}: {name} MSE")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


rks_train, rks_val = [], []
for sigma in RKS_SIGMAS:
    model = RandomKitchenSinks(RKS_D, sigma, RKS_LAMBDA).fit(X_train, y_train)
    rks_train.append(mse(y_train, model.predict(X_train)))
    rks_val.append(mse(y_val, model.predict(X_val)))
best_sigma = RKS_SIGMAS[np.argmin(rks_val)]

print("Part (b): RKS sweep over sigma")
for s, tr, va in zip(RKS_SIGMAS, rks_train, rks_val):
    print(f"  sigma {s:7.4f}   train {tr:.4f}   val {va:.4f}")
print(f"  best sigma = {best_sigma:.3g} (val MSE {min(rks_val):.4f})")
plot_sweep(RKS_SIGMAS, rks_train, rks_val, "kernel scale sigma", "RKS", "figures/b_rks_sweep.png", best_sigma)


# ---------------- part (c) ----------------#
class MLP(nn.Module):
    def __init__(self, hidden, seed=SEED):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, X):
        return self.net(X)

    def fit(self, X, y, epochs=MLP_EPOCHS, batch_size=MLP_BATCH_SIZE, lr=MLP_LR):
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # decay lr to 0
        self.train()
        for _ in range(epochs):
            order = torch.randperm(len(y))
            for i in range(0, len(y), batch_size):
                batch = order[i : i + batch_size]
                loss = nn.functional.mse_loss(self.net(X[batch]), y[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
        self.eval()
        return self

    def predict(self, X):
        with torch.no_grad():
            return self.net(torch.tensor(X, dtype=torch.float32)).numpy().ravel()


mlp_train, mlp_val = [], []
for width in MLP_WIDTHS:
    model = MLP(width).fit(X_train, y_train)
    mlp_train.append(mse(y_train, model.predict(X_train)))
    mlp_val.append(mse(y_val, model.predict(X_val)))
best_width = MLP_WIDTHS[int(np.argmin(mlp_val))]

print("\nPart (c): MLP sweep over hidden width")
for w, tr, va in zip(MLP_WIDTHS, mlp_train, mlp_val):
    print(f"  width {w:4d}   train {tr:.4f}   val {va:.4f}")
print(f"  best width = {best_width} (val MSE {min(mlp_val):.4f})")
plot_sweep(MLP_WIDTHS, mlp_train, mlp_val, "hidden width", "MLP", "figures/c_mlp_sweep.png", best_width)



# ---------------- part (d) ----------------#
rks = RandomKitchenSinks(RKS_D, best_sigma, RKS_LAMBDA).fit(X_train, y_train)
mlp = MLP(best_width).fit(X_train, y_train)

print("\nPart (d): validation MSE of the best models")
print(f"  constant baseline   {mse(y_val, np.full_like(y_val, y_train.mean())):.4f}")
print(f"  RKS (sigma {best_sigma:.3g})    {mse(y_val, rks.predict(X_val)):.4f}")
print(f"  MLP (width {best_width})      {mse(y_val, mlp.predict(X_val)):.4f}")

# check the gap isn't just luck: repeat with 5 different random seeds
rks_seeds = [mse(y_val, RandomKitchenSinks(RKS_D, best_sigma, RKS_LAMBDA, s).fit(X_train, y_train).predict(X_val))
             for s in range(5)]
mlp_seeds = [mse(y_val, MLP(best_width, s).fit(X_train, y_train).predict(X_val)) for s in range(5)]
print(f"  over 5 seeds: RKS {np.mean(rks_seeds):.4f} ± {np.std(rks_seeds):.4f}, "
      f"MLP {np.mean(mlp_seeds):.4f} ± {np.std(mlp_seeds):.4f}")

# test predictions, one per row in test.csv order
np.savetxt("rks_test_predictions.csv", rks.predict(X_test), delimiter=",")
np.savetxt("mlp_test_predictions.csv", mlp.predict(X_test), delimiter=",")
