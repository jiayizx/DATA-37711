import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ---------------- hyperparameters
SEED = 0             # fixes the data split, initial weights and batch shuffling
HIDDEN = 128         # width of each hidden layer
EPOCHS = 30          # passes over the training set
BATCH_SIZE = 2048    # examples per gradient step
LR = 1e-3            # Adam learning rate
THRESHOLD = 45       # minutes, for parts (d) and (e)

torch.manual_seed(SEED)

# ---------------- load and split the data (70% train, 15% val, 15% test)
df = pd.read_parquet("p1/taxi_trips.parquet")
y = df["trip_duration"].to_numpy(np.float32)
X = df.drop(columns="trip_duration").to_numpy(np.float32)

X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.3, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=SEED)


y_mean, y_std = y_train.mean(), y_train.std()


# ---------------- the model: g(x) in R^2, then the link functions from part (a)
class MeanStdMLP(nn.Module):
    def __init__(self, d_in, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        out = self.net(x)
        mu = out[:, 0]                          # identity link
        sigma = F.softplus(out[:, 1]) + 1e-6    # softplus link, keeps sigma > 0
        return mu, sigma


def nll_loss(y, mu, sigma):
    """Gaussian negative log-likelihood from part (b), constant dropped."""
    return (torch.log(sigma) + (y - mu) ** 2 / (2 * sigma ** 2)).mean()


def train(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    """Train with mini-batch Adam; return the model at the epoch with the lowest validation NLL."""
    Xtr, ytr = torch.tensor(X_train), torch.tensor((y_train - y_mean) / y_std)
    Xva, yva = torch.tensor(X_val), torch.tensor((y_val - y_mean) / y_std)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_weights = float("inf"), None

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(ytr))
        for i in range(0, len(order), batch_size):
            batch = order[i : i + batch_size]
            mu, sigma = model(Xtr[batch])
            loss = nll_loss(ytr[batch], mu, sigma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = nll_loss(yva, *model(Xva)).item()
        print(f"epoch {epoch:2d}  validation NLL {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)
    return model


def predict(model, X):
    """Return the predicted mean and std in minutes."""
    model.eval()
    with torch.no_grad():
        mu, sigma = model(torch.tensor(X))
    return mu.numpy() * y_std + y_mean, sigma.numpy() * y_std


model = MeanStdMLP(d_in=X.shape[1], hidden=HIDDEN)
model = train(model, X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)

mu_val, _ = predict(model, X_val)
mu_test, sigma_test = predict(model, X_test)


# ---------------- part (c): compare against baselines on the test set
def gaussian_nll(y, mu, sigma):
    return np.mean(np.log(sigma) + (y - mu) ** 2 / (2 * sigma ** 2) + 0.5 * np.log(2 * np.pi))


linreg = LinearRegression().fit(X_train, y_train)
ones = np.ones_like(y_test)

# each entry: (name, predicted mean, predicted std) on the test set
models = [
    ("Constant: train mean and std", y_train.mean() * ones, y_train.std() * ones),
    ("Linear mean + constant std", linreg.predict(X_test), np.std(y_train - linreg.predict(X_train)) * ones),
    ("MLP mean + MLP std (ours)", mu_test, sigma_test),
]

print("\nPart (c): test set")
print(f"{'model':32s} {'RMSE':>7s} {'NLL':>7s}")
for name, mu, sigma in models:
    rmse = np.sqrt(np.mean((y_test - mu) ** 2))
    print(f"{name:32s} {rmse:7.3f} {gaussian_nll(y_test, mu, sigma):7.3f}")
print(f"predicted std (minutes): min {sigma_test.min():.2f}, median {np.median(sigma_test):.2f}, max {sigma_test.max():.2f}")

# ---------------- parts (d) and (e): P(trip < 45 minutes) = Phi((45 - mu) / sigma)
is_short = (y_test < THRESHOLD).astype(float)
print(f"\nPart (e): {is_short.mean():.2%} of test trips are under {THRESHOLD} minutes")
print(f"{'model':32s} {'Brier':>8s} {'LogLoss':>8s} {'Accuracy':>9s}")

base_rate = (y_train < THRESHOLD).mean()
prob_models = [("Base rate (always the same p)", base_rate * ones)]
prob_models += [(name, norm.cdf((THRESHOLD - mu) / sigma)) for name, mu, sigma in models]

for name, p in prob_models:
    p = np.clip(p, 1e-12, 1 - 1e-12)                      # avoid log(0)
    brier = np.mean((p - is_short) ** 2)
    logloss = -np.mean(is_short * np.log(p) + (1 - is_short) * np.log(1 - p))
    accuracy = np.mean((p >= 0.5) == is_short)
    print(f"{name:32s} {brier:8.5f} {logloss:8.4f} {accuracy:9.4f}")
