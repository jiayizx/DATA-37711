import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import category_encoders as ce
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

# ---------------- hyperparameters
SEED = 0             # fixes the data split, initial weights and batch shuffling
EMB_DIM = 16         # embedding dimension for each feature
HIDDEN = 64          # width of each hidden layer
EPOCHS = 15          # passes over the training set
BATCH_SIZE = 1024    # examples per gradient step
LR = 2e-3            # Adam learning rate
SMOOTHING = 10       # target-encoding smoothing toward the global mean

torch.manual_seed(SEED)
CATS = ["StockCode", "CustomerID"]

# ---------------- part (a): load data, look at cardinality, split 70 / 15 / 15
df = pd.read_csv("../Homework_2/p2/retail.csv")
df["StockCode"] = df["StockCode"].astype(str)
df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
df["y"] = np.log1p(df["Quantity"])

print("Part (a)")
for c in CATS:
    counts = df[c].value_counts()
    print(f"  {c}: {df[c].nunique()} unique values, median {counts.median():.0f} rows per value, "
          f"{(counts < 10).mean():.0%} of values have < 10 rows")
print(f"  Quantity: median {df.Quantity.median():.0f}, 99th percentile {df.Quantity.quantile(0.99):.0f}, "
      f"max {df.Quantity.max()}")

train_df, rest_df = train_test_split(df, test_size=0.3, random_state=SEED)
val_df, test_df = train_test_split(rest_df, test_size=0.5, random_state=SEED)
y_train, y_val, y_test = train_df.y.values, val_df.y.values, test_df.y.values
print(f"  train {len(train_df)}, val {len(val_df)}, test {len(test_df)}")

# ---------------- part (b) ----------------#
def evaluate(name, pred):
    """MSE on the log scale, plus MAE in the original units (items)."""
    mse_log = np.mean((y_test - pred) ** 2)
    mae = np.mean(np.abs(np.expm1(y_test) - np.expm1(pred)))
    print(f"  {name:40s} MSE(log) {mse_log:.4f}   MAE {mae:6.2f}")

print("\nPart (b): Test results")
evaluate("Baseline: predict the train mean", np.full(len(y_test), y_train.mean()))

encoder = ce.TargetEncoder(cols=CATS, smoothing=SMOOTHING)
te_train = encoder.fit_transform(train_df[CATS], y_train).values
te_val = encoder.transform(val_df[CATS]).values
te_test = encoder.transform(test_df[CATS]).values

gbm = HistGradientBoostingRegressor(max_iter=300, random_state=SEED)
gbm.fit(te_train, y_train)
pred_te = gbm.predict(te_test)
evaluate("Target encoding + GBM", pred_te)

# ---------------- shared by (c) and (d)----------------#
index = {c: {v: i + 1 for i, v in enumerate(train_df[c].unique())} for c in CATS}

def to_indices(d):
    cols = [d[c].map(index[c]).fillna(0).astype(int).values for c in CATS]
    return torch.tensor(np.stack(cols, axis=1))

idx_train, idx_val, idx_test = to_indices(train_df), to_indices(val_df), to_indices(test_df)
yt_train = torch.tensor(y_train, dtype=torch.float32)
yt_val = torch.tensor(y_val, dtype=torch.float32)

class EmbeddingNet(nn.Module):
    def __init__(self, emb_dim, hidden, n_extra=0):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(len(index[c]) + 1, emb_dim) for c in CATS])
        self.net = nn.Sequential(
            nn.Linear(emb_dim * len(CATS) + n_extra, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, idx, extra=None):
        parts = [emb(idx[:, j]) for j, emb in enumerate(self.embs)]
        if extra is not None:
            parts.append(extra)
        return self.net(torch.cat(parts, dim=1)).squeeze(1)


def train(model, inputs_train, inputs_val, epochs, batch_size, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val, best_weights = float("inf"), None

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(yt_train))
        for i in range(0, len(order), batch_size):
            batch = order[i : i + batch_size]
            loss = loss_fn(model(*[x[batch] for x in inputs_train]), yt_train[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(*inputs_val), yt_val).item()
        print(f"    epoch {epoch:2d}  validation MSE {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)
    return model


def predict(model, inputs):
    model.eval()
    with torch.no_grad():
        return model(*inputs).numpy()

# ---------------- part (c) ----------------#
print("  training (c) embeddings only...")
emb_model = EmbeddingNet(EMB_DIM, HIDDEN)
emb_model = train(emb_model, (idx_train,), (idx_val,), EPOCHS, BATCH_SIZE, LR)
pred_emb = predict(emb_model, (idx_test,))
evaluate("(c) Embeddings + MLP", pred_emb)

# ---------------- part (d) ----------------#
te_mean, te_std = te_train.mean(0), te_train.std(0)
te_train_t, te_val_t, te_test_t = [torch.tensor((te - te_mean) / te_std, dtype=torch.float32)
                                   for te in (te_train, te_val, te_test)]

print("  training (d) embeddings + target encoding...")
both_model = EmbeddingNet(EMB_DIM, HIDDEN, n_extra=len(CATS))
both_model = train(both_model, (idx_train, te_train_t), (idx_val, te_val_t), EPOCHS, BATCH_SIZE, LR)
pred_both = predict(both_model, (idx_test, te_test_t))
evaluate("(d) Embeddings + target encoding + MLP", pred_both)

# ---------------- part (e) ----------------#
train_counts = test_df["StockCode"].map(train_df["StockCode"].value_counts()).fillna(0).values
print("\nPart (e): test MSE(log) by how often the StockCode appears in training")
print(f"  {'rows in train':>14s} {'n_test':>7s} {'TE+GBM':>7s} {'Emb':>7s} {'Emb+TE':>7s}")
for label, lo, hi in [("0 (unseen)", -1, 0), ("1-20", 0, 20), ("21-100", 20, 100), (">100", 100, np.inf)]:
    m = (train_counts > lo) & (train_counts <= hi)
    mses = [np.mean((y_test[m] - p[m]) ** 2) for p in (pred_te, pred_emb, pred_both)]
    print(f"  {label:>14s} {m.sum():7d} " + " ".join(f"{v:7.4f}" for v in mses))