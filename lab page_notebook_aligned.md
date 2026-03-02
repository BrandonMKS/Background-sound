# 🧠 Phase 1 — Neural Networks Implementation

---

<table>
<tr>
<td width="35%" valign="top">
<p></p><br />
<p></p><br />
<p></p><br />
<p></p><br />
<p></p><br />
<p></p><br />
<p></p><br />
<h3>Table of Contents</h3>
<ol>
<li>🧠 <a href="#-phase-1-overview">Phase 1 Overview</a></li>
<li>📊 <a href="#-dataset--splitting-strategy">Dataset &amp; Splitting Strategy</a></li>
<li>🛠️ <a href="#%EF%B8%8F-feature-engineering--preprocessing">Feature Engineering &amp; Preprocessing</a></li>
<li>🧱 <a href="#-classical-baseline">Classical Baseline</a></li>
<li>🧠 <a href="#-neural-network-architecture">Neural Network Architecture</a></li>
<li>🏋️ <a href="#%EF%B8%8F-neural-network-training-procedure">Neural Network Training Procedure</a></li>
<li>🔍 <a href="#-hyperparameter-tuning">Hyperparameter Tuning</a></li>
<li>📈 <a href="#-final-evaluation">Final Evaluation</a></li>
<li>🧭 <a href="#-feature-importance">Feature Importance</a></li>
<li>🧪 <a href="#-hypothesis-h1-test">Hypothesis H1 Test</a></li>
</ol>
</td>
<td width="65%" valign="top" align="center">

<h3>Phase 1 Flowchart</h3>

<img alt="Solution diagram" src="https://github.com/user-attachments/assets/9d123311-035f-4c5c-bca9-154a140940e6" style="width:100%; max-width:450px; height:auto; display:block; margin:0 auto;"/>

</td>
</tr>
</table>

---

## 🗺️ How This Flowchart Maps to Our Implementation

| Flowchart Block | Lab Section |
|----------------|-------------|
| Data Load | Dataset & Splitting Strategy |
| Baseline Model | Classical Baseline |
| NN Design | Neural Network Architecture |
| Tuning | Hyperparameter Tuning |
| Final Test | Final Evaluation |
| Interpretation | Feature Importance + H1 Test |

The flowchart visually represents the controlled workflow required for Phase 1.  
Each block corresponds exactly to one section below, ensuring our implementation follows a clean, reproducible, and logically ordered pipeline.

---

## 🧠 Phase 1 Overview

### 🎯 Objective

In Phase 1, we implement and evaluate a **tabular neural network regressor** for the NYC Taxi Trip Duration dataset using a **reproducible, leakage-safe pipeline**.

This phase answers:

- **RQ1:** How well can a neural network predict taxi trip duration from engineered tabular features?
- **RQ2:** Which engineered features contribute the most to predictive performance?
- **H1:** A properly tuned neural network will outperform a tuned classical baseline (Ridge Regression).

### 🧩 What This Lab Implements (Notebook-Aligned)

The lab page is intentionally written to mirror the notebook execution order while keeping the same 10 section headers from the Table of Contents.

Notebook-aligned pipeline:

1. Reproducible setup (constants + device)
2. Deterministic seeding across RNGs
3. Dataset download (if missing), **load first 1,000,000 rows**, deterministic shuffle
4. Strict three-way split: **70% train / 15% validation / 15% test (holdout)**
5. Feature engineering (temporal + spatial + categorical proxies)
6. Column alignment for one-hot features across splits
7. Standardization using **train-only** statistics + artifact persistence
8. Log-space safety utilities + shared evaluation metrics
9. Tuned Ridge baseline (validation-selected α)
10. MLP neural net training + validation-only hyperparameter tuning + single final holdout evaluation
11. Feature importance (permutation importance in log-space)
12. H1 test support via bootstrap confidence intervals

---

### 🧰 Code: Reproducible Setup (Notebook Cell 1)

```python
from pathlib import Path
import numpy as np
import pandas as pd

# Imports for this section (torch, torch.utils.data).
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Imports for this section (sklearn.model_selection, sklearn.linear_model, sklearn.metrics).
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Set global configuration/constants used throughout the notebook.
SEED = 42
NROWS = 1_000_000
TARGET = "trip_duration"

# Set global configuration/constants used throughout the notebook.
DATA_PATH = Path("../data/train.csv")

# Kaggle NYC Taxi Trip Duration schema (train.csv)
DATA_URL = "https://github.com/DrAlzahraniProjects/csusb_spring26_cse5140_team1/releases/download/v1.0/train.csv"

# Set global configuration/constants used throughout the notebook.
ARTIFACTS_DIR = Path("../artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Print key configuration so runs are easy to reproduce/debug.
print("Config:")
print("  DATA_PATH:", DATA_PATH)
print("  NROWS:", NROWS)
print("  SEED:", SEED)

# Select computation device (GPU if available, otherwise CPU).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
```

### 🧬 Code: Deterministic Seeding (Notebook Cell 2)

```python
def seed_everything(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Next section: compute the step below.
seed_everything(SEED)
```

---

# 📊 Dataset & Splitting Strategy

## Why Only 1 Million Rows?

We enforce the course fairness constraint by loading only the first **NROWS = 1,000,000** rows.

This prevents:
- compute imbalance between teams
- “winning” purely via more data/longer runs
- uncontrolled runtime differences

## Why Shuffle After Loading?

Even when loading the first 1M rows, the original file can still be ordered (e.g., by time).  
We shuffle with a **fixed seed** so the split is more representative and still reproducible.

## Why a 70/15/15 Split?

The notebook uses:

- **70% Train** — fit model parameters  
- **15% Validation** — select/tune hyperparameters (baseline + NN)  
- **15% Test (Holdout)** — used once at the end for the final comparison

This keeps model selection separate from final generalization testing.

---

## 🔢 Code: Load, Shuffle, and Slice (Notebook Cell 3)

```python
import urllib.request

# Make sure the directory exists
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

#If it does not exist then it will request to download from GitHub
if not DATA_PATH.exists():
    print("Downloading dataset...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print("Download complete.")
else:
    print("Dataset already exists. Skipping download.")

print("Loading dataset into memory...")
# Load 1,000,000 rows
df = pd.read_csv(DATA_PATH, nrows=NROWS)
print("Loaded df:", df.shape)
df.head()

# Next section: compute the step below.
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
print("Shuffled with seed:", SEED)

```

## 🔢 Code: Train/Validation/Test Partition (Notebook Cell 4)

```python
# Splits (overall):
# - 70% Train
# - 15% Validation
# - 15% Test (stored in `holdout_df` and only used in the final comparison)

# Split the dataset into development (train+val) and separate test set.
dev_df, holdout_df = train_test_split(df, test_size=0.15, random_state=SEED)

# Split dev into train/val so the overall proportions are 70/15 (i.e., val is 0.15/0.85 of dev).
train_df, val_df = train_test_split(dev_df, test_size=(0.15/0.85), random_state=SEED)

# Optional clarity alias (so later text can say "test" while keeping existing variables).
test_df = holdout_df

# Print key configuration so runs are easy to reproduce/debug.
print("train_df:", train_df.shape)
print("val_df:", val_df.shape)
print("holdout_df (test):", holdout_df.shape)

```

---

# 🛠️ Feature Engineering & Preprocessing

## Feature Families Implemented (Notebook-Aligned)

### 🕒 Temporal Features
From `pickup_datetime`:

- `pickup_hour`, `pickup_dow`, `pickup_month`
- cyclical encodings: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`

### 📍 Spatial Features
From pickup/dropoff lat/lon:

- `delta_lat`, `delta_lon`
- `haversine_km` (great-circle distance proxy)

### 👥 Context + Simple Flags

- `passenger_count`
- `store_and_fwd_Y`
- one-hot encoding for `vendor_id`

---

## 🔢 Code: Feature Builder (Notebook Cell 5)

```python
def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized Haversine distance in km."""
    R = 6371.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# Define helper function `build_features` used in later cells.
def build_features(dfin: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=dfin.index)

    # Temporal
    dt = pd.to_datetime(dfin["pickup_datetime"], errors="coerce")
    X["pickup_hour"]  = dt.dt.hour.fillna(0).astype(int)
    X["pickup_dow"]   = dt.dt.dayofweek.fillna(0).astype(int)
    X["pickup_month"] = dt.dt.month.fillna(0).astype(int)

    # Next section: compute the step below.
    X["hour_sin"] = np.sin(2*np.pi*X["pickup_hour"]/24)
    X["hour_cos"] = np.cos(2*np.pi*X["pickup_hour"]/24)
    X["dow_sin"]  = np.sin(2*np.pi*X["pickup_dow"]/7)
    X["dow_cos"]  = np.cos(2*np.pi*X["pickup_dow"]/7)

    # Spatial
    X["delta_lat"] = (dfin["dropoff_latitude"] - dfin["pickup_latitude"]).astype(float)
    X["delta_lon"] = (dfin["dropoff_longitude"] - dfin["pickup_longitude"]).astype(float)
    X["haversine_km"] = haversine_km(
        dfin["pickup_latitude"].astype(float),
        dfin["pickup_longitude"].astype(float),
        dfin["dropoff_latitude"].astype(float),
        dfin["dropoff_longitude"].astype(float),
    )

    # Proxies
    X["passenger_count"] = pd.to_numeric(dfin["passenger_count"], errors="coerce").fillna(0.0)
    X["store_and_fwd_Y"] = (dfin["store_and_fwd_flag"].astype(str).str.upper() == "Y").astype(int)

    # Next section: compute the step below.
    vendor_oh = pd.get_dummies(dfin["vendor_id"].astype(str), prefix="vendor", drop_first=False)
    X = pd.concat([X, vendor_oh], axis=1)

    # Set global configuration/constants used throughout the notebook.
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X
```

## 🧩 Column Alignment + Target Extraction (Notebook Cell 6)

The notebook locks the **training** feature columns, then reindexes validation/test to match.
This prevents one-hot mismatch when a category appears in one split but not another.

```python
# Build train/val features and ALIGN columns (critical for get_dummies)
X_train = build_features(train_df)
feature_cols = X_train.columns

# Build engineered features and ensure train/val/holdout columns align.
X_val = build_features(val_df).reindex(columns=feature_cols, fill_value=0.0)

# Next section: compute the step below.
y_train = train_df[TARGET].to_numpy().astype(np.float64)
y_val   = val_df[TARGET].to_numpy().astype(np.float64)

# Print key configuration so runs are easy to reproduce/debug.
print("X_train:", X_train.shape, "X_val:", X_val.shape)
print("y_train:", y_train.shape, "y_val:", y_val.shape)
```

## 📏 Standardization + Artifact Saving (Notebook Cell 7)

We standardize with **train-only** mean/std and persist artifacts for reproducibility.

```python
# Scale using TRAIN stats only
mu = X_train.mean()
sigma = X_train.std().replace(0, 1)

# Apply standardization using training mean/std so models train stably.
X_train_s = (X_train - mu) / sigma
X_val_s   = (X_val   - mu) / sigma

# Save artifacts (optional)
mu.to_csv(ARTIFACTS_DIR / "mu.csv")
sigma.to_csv(ARTIFACTS_DIR / "sigma.csv")
pd.Series(feature_cols, name="feature").to_csv(ARTIFACTS_DIR / "feature_cols.csv", index=False)

# Print key configuration so runs are easy to reproduce/debug.
print("Saved artifacts to:", ARTIFACTS_DIR.resolve())
```

## 🧾 Target Transform + Shared Metrics Utilities (Notebook Cell 8)

We train and evaluate primarily in **log-space** (`log1p`) to stabilize heavy-tailed durations, and use:

- `safe_expm1` for a numerically safe inverse transform (with clipping)
- `mape` for relative error
- `eval_regression` for consistent metric reporting (log + original scale)

```python
CLIP_MIN = -2.0
CLIP_MAX = 13.0  # tighter cap; prevents absurd durations

# Define helper function `safe_expm1` used in later cells.
def safe_expm1(yhat_log, clip_min=CLIP_MIN, clip_max=CLIP_MAX):
    yhat_log = np.asarray(yhat_log).reshape(-1)
    yhat_log = np.clip(yhat_log, clip_min, clip_max)
    return np.expm1(yhat_log)

# Define helper function `mape` used in later cells.
def mape(y_true, y_pred, eps=1.0):
    y_true = np.asarray(y_true).reshape(-1).astype(np.float64)
    y_pred = np.asarray(y_pred).reshape(-1).astype(np.float64)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

# Define helper function `eval_regression` used in later cells.
def eval_regression(y_true_log, y_pred_log, y_true_orig=None, label=""):
    """Shared evaluation: log-space + optional original-scale metrics. Prints and returns dict."""
    y_pred_log = np.asarray(y_pred_log).reshape(-1)
    y_true_log = np.asarray(y_true_log).reshape(-1)
    metrics = {
        "R2_log": r2_score(y_true_log, y_pred_log),
        "RMSE_log": mean_squared_error(y_true_log, y_pred_log, squared=False),
    }
    if y_true_orig is not None:
        y_pred_orig = safe_expm1(y_pred_log)
        y_true_orig = np.asarray(y_true_orig).reshape(-1)
        metrics["R2"] = r2_score(y_true_orig, y_pred_orig)
        metrics["RMSE"] = mean_squared_error(y_true_orig, y_pred_orig, squared=False)
        metrics["MAE"] = mean_absolute_error(y_true_orig, y_pred_orig)
        metrics["MAPE(%)"] = mape(y_true_orig, y_pred_orig)
    if label:
        print(label)
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    return metrics
```

---

# 🧱 Classical Baseline

## Baseline Definition (Notebook-Aligned)

The baseline is **Ridge Regression** trained to predict **log1p(trip_duration)** using standardized features.

We tune `alpha` via a small validation grid search, then keep the best α (chosen by validation R² on original scale).

---

## 🔢 Code: Tuned Ridge (Notebook Cell 9)

```python
y_train_log = np.log1p(y_train)
y_val_log   = np.log1p(y_val)

# --- Ridge alpha grid search ---
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
ridge_results = []

# Iterate through this section's loop to compute/update results.
for a in alphas:
    mdl = Ridge(alpha=a, random_state=SEED)
    mdl.fit(X_train_s, y_train_log)
    pred_log = np.clip(mdl.predict(X_val_s), CLIP_MIN, CLIP_MAX)
    pred_orig = safe_expm1(pred_log)

    # Next section: compute the step below.
    row = {
        "alpha": a,
        "R2": r2_score(y_val, pred_orig),
        "RMSE": mean_squared_error(y_val, pred_orig, squared=False),
        "MAE": mean_absolute_error(y_val, pred_orig),
        "R2_log": r2_score(y_val_log, pred_log),
        "RMSE_log": mean_squared_error(y_val_log, pred_log, squared=False),
    }
    ridge_results.append((mdl, row))

# Next section: compute the step below.
ridge_grid_df = pd.DataFrame([r for _, r in ridge_results])
print("Ridge alpha grid search (validation):")
display(ridge_grid_df)

# Select best alpha by validation R²
best_idx = ridge_grid_df["R2"].idxmax()
ridge, best_ridge_row = ridge_results[best_idx]
print(f"\nBest alpha = {best_ridge_row['alpha']}  (val R2 = {best_ridge_row['R2']:.6f})")

# Recompute full validation metrics for best Ridge
val_pred_log_ridge = np.clip(ridge.predict(X_val_s), CLIP_MIN, CLIP_MAX)
baseline_val_metrics = eval_regression(y_val_log, val_pred_log_ridge, y_true_orig=y_val, label="\nVALIDATION — Tuned Ridge:")
```

---

# 🧠 Neural Network Architecture

## Architecture Design (Notebook-Aligned)

The notebook uses a **fully-connected MLP** (ReLU + Dropout) for tabular regression, created by a helper function that accepts:

- `layers`: tuple of hidden widths (e.g., `(256, 128)`)
- `dropout`: dropout probability
- input dimension determined by engineered feature count

### 🔢 Code: MLP Builder (Excerpt from Notebook Cell 11)

```python
def build_mlp(in_dim: int, layers=(256, 128), dropout=0.10):
    net = []
    prev = in_dim
    for h in layers:
        net.append(nn.Linear(prev, h))
        net.append(nn.ReLU())
        net.append(nn.Dropout(dropout))
        prev = h
    net.append(nn.Linear(prev, 1))
    return nn.Sequential(*net).to(device)
```

(Full tuning + training logic that calls this builder is included verbatim in the **Hyperparameter Tuning** section.)

---

## Why This Architecture?

MLPs are a standard first-choice neural baseline for tabular regression because they can learn:

- nonlinear feature interactions
- threshold-like behaviors (via ReLU)
- robust generalization with dropout + weight decay + early stopping

---

## 🏋️ Neural Network Training Procedure

## Data Interface (Mini-batches)

The notebook wraps arrays into a `Dataset` and uses `DataLoader` for mini-batch SGD-style training.

### 🔢 Code: Dataset + Loaders (Notebook Cell 10)

```python
class TabularDataset(Dataset):
    def __init__(self, X, y_log):
        self.X = torch.tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(y_log), dtype=torch.float32).view(-1, 1)

    # Define helper function `__len__` used in later cells.
    def __len__(self):
        return self.X.shape[0]

    # Define helper function `__getitem__` used in later cells.
    def __getitem__(self, i):
        return self.X[i], self.y[i]

# Next section: compute the step below.
train_ds = TabularDataset(X_train_s.values, y_train_log)
val_ds   = TabularDataset(X_val_s.values,   y_val_log)

# Wrap arrays as PyTorch Datasets/DataLoaders for efficient mini-batch training.
train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=4096, shuffle=False, num_workers=0)
```

## Training Loop Mechanics (Notebook-Aligned)

Inside each hyperparameter trial, training uses:

- **SmoothL1Loss(beta=0.5)** (robust to outliers vs MSE)
- **Adam** optimizer
- **weight_decay** (L2-style regularization)
- **gradient clipping** (`max_norm=1.0`)
- **early stopping** based on validation RMSE in log-space (`val_rmse_log`)
- per-epoch logging to CSV so learning curves can be plotted

(These mechanics appear in full in Notebook Cell 11 below.)

---

# 🔍 Hyperparameter Tuning

## Tuning Protocol (Notebook-Aligned)

- Fixed trial budget: `TRIALS = 20`
- Discrete search space of safe configs
- Each trial trains a **fresh** model (no cross-trial contamination)
- Selection criterion: **highest validation R²** on original target scale
- Best model retained as `best_model`
- Results persisted to `phase2_validation_results.csv` (artifact name retained to match notebook)

---

## 🔢 Code: Full Tuning + Trial Training (Notebook Cell 11)

```python
import os
from pathlib import Path
import random

# -----------------------------
# 2.2 Hyperparameter Tuning Loop
# -----------------------------
TRIALS = 20  # set to 20–50

# Set global configuration/constants used throughout the notebook.
PLOTS_DIR = ARTIFACTS_DIR / "plots"
LOG_DIR   = ARTIFACTS_DIR / "trial_logs"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Define helper function `build_mlp` used in later cells.
def build_mlp(in_dim: int, layers=(256, 128), dropout=0.10):
    net = []
    prev = in_dim
    for h in layers:
        net.append(nn.Linear(prev, h))
        net.append(nn.ReLU())
        net.append(nn.Dropout(dropout))
        prev = h
    net.append(nn.Linear(prev, 1))
    return nn.Sequential(*net).to(device)

# Define helper function `evaluate_nn_log` used in later cells.
def evaluate_nn_log(model, Xs_np):
    model.eval()
    with torch.no_grad():
        pred_log = model(torch.tensor(Xs_np, dtype=torch.float32).to(device)).cpu().numpy().reshape(-1)
    pred_log = np.clip(pred_log, CLIP_MIN, CLIP_MAX)
    return pred_log

# Define helper function `train_one_trial` used in later cells.
def train_one_trial(trial_id: int, cfg: dict):
    # Reset seeds per trial (prevents cross-trial contamination + improves reproducibility)
    seed_everything(SEED + trial_id)

    # Fresh model + optimizer per trial
    model = build_mlp(in_dim=X_train_s.shape[1], layers=cfg["layers"], dropout=cfg["dropout"])
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # Per-trial loaders (batch size can be tuned too)
    train_ds = TabularDataset(X_train_s.values, y_train_log)
    val_ds   = TabularDataset(X_val_s.values,   y_val_log)

    # Wrap arrays as PyTorch Datasets/DataLoaders for efficient mini-batch training.
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=4096, shuffle=False, num_workers=0)

    # Early stopping on validation RMSE in log-space
    best_state = None
    best_val_rmse_log = float("inf")
    patience = cfg["patience"]
    pat = 0

    # Next section: compute the step below.
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total = 0.0
        n = 0

        # Iterate through this section's loop to compute/update results.
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).view(-1)

            # Reset gradients before the forward/backward pass for this batch.
            optimizer.zero_grad()
            pred_log = model(xb).view(-1)
            pred_log = torch.clamp(pred_log, CLIP_MIN, CLIP_MAX)

            # Compute the training loss for this batch.
            loss = loss_fn(pred_log, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Next section: compute the step below.
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        # Next section: compute the step below.
        train_loss = total / n

        # Validate
        val_pred_log = evaluate_nn_log(model, X_val_s.values.astype(np.float32))
        val_rmse_log = mean_squared_error(y_val_log, val_pred_log, squared=False)

        # Log per-epoch metrics so training curves can be plotted later.
        history.append({
            "trial_id": trial_id,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_rmse_log": val_rmse_log,
        })

        # Early stopping check
        if val_rmse_log < best_val_rmse_log - 1e-4:
            best_val_rmse_log = val_rmse_log
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience:
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final validation metrics (log + original scale)
    val_pred_log = evaluate_nn_log(model, X_val_s.values.astype(np.float32))
    val_pred     = safe_expm1(val_pred_log)

    # Next section: compute the step below.
    val_rmse = mean_squared_error(y_val, val_pred, squared=False)
    val_mae  = mean_absolute_error(y_val, val_pred)
    val_r2   = r2_score(y_val, val_pred)

    # Save per-epoch history for Brandon plots
    hist_path = LOG_DIR / f"trial_{trial_id}_history.csv"
    pd.DataFrame(history).to_csv(hist_path, index=False)

    # Next section: compute the step below.
    result = {
        "trial_id": trial_id,
        "layers": str(cfg["layers"]),
        "dropout": cfg["dropout"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "batch_size": cfg["batch_size"],
        "epochs_cap": cfg["epochs"],
        "patience": cfg["patience"],
        "best_val_rmse_log": best_val_rmse_log,
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "val_r2": val_r2,
        "history_csv": str(hist_path),
    }
    return model, result

# -----------------------------
# Search space (simple + safe)
# -----------------------------
search_space = [
    {"layers": (256,128), "dropout": 0.05, "lr": 1e-3, "weight_decay": 1e-6, "batch_size": 2048, "epochs": 20, "patience": 3},
    {"layers": (256,128), "dropout": 0.10, "lr": 5e-4, "weight_decay": 1e-5, "batch_size": 2048, "epochs": 20, "patience": 3},
    {"layers": (512,256), "dropout": 0.10, "lr": 5e-4, "weight_decay": 1e-5, "batch_size": 2048, "epochs": 25, "patience": 4},
    {"layers": (512,256), "dropout": 0.20, "lr": 3e-4, "weight_decay": 1e-4, "batch_size": 2048, "epochs": 25, "patience": 4},
    {"layers": (128,64),  "dropout": 0.05, "lr": 1e-3, "weight_decay": 1e-6, "batch_size": 1024, "epochs": 20, "patience": 3},
    {"layers": (128,64),  "dropout": 0.10, "lr": 5e-4, "weight_decay": 1e-5, "batch_size": 1024, "epochs": 20, "patience": 3},
]
## This is to print safe space necessary for budget transparency accocrding to chatGPT
print("Search space size:", len(search_space))
print("Example configs:", search_space[:3])

# If TRIALS > len(search_space), we'll sample with replacement
seed_everything(SEED)

# Next section: compute the step below.
all_results = []
best_model = None
best_row = None

# Print key configuration so runs are easy to reproduce/debug.
print(f"Running {TRIALS} hyperparameter trials (validation-only selection)...")

#Start TIMER
import time
tuning_start = time.time()

# Run multiple randomized trials and keep the best model by validation R².
for t in range(1, TRIALS + 1):
    cfg = random.choice(search_space)

    # Next section: compute the step below.
    model, row = train_one_trial(t, cfg)
    all_results.append(row)

    # Print key configuration so runs are easy to reproduce/debug.
    print(f"Trial {t:02d} | val_R2={row['val_r2']:.4f} | val_RMSE={row['val_rmse']:.2f} | best_val_RMSE_log={row['best_val_rmse_log']:.4f} | cfg={cfg}")

    # Selection criterion: validation R² (strictly validation-based)
    if best_row is None or row["val_r2"] > best_row["val_r2"]:
        best_row = row
        best_model = model

# Persist preprocessing artifacts to disk for reuse and reproducibility.
results_df = pd.DataFrame(all_results).sort_values("val_r2", ascending=False)
results_path = ARTIFACTS_DIR / "phase2_validation_results.csv"
results_df.to_csv(results_path, index=False)

# Print key configuration so runs are easy to reproduce/debug.
print("\nSaved tuning results to:", results_path.resolve())
print("\nBest trial selected by validation R²:")
display(results_df.head(5))
print("\nBEST TRIAL:", best_row)

# END TIMER
tuning_end = time.time()
total_time = tuning_end - tuning_start

print("\n=== COMPUTE SUMMARY ===")
print("Trials run:", TRIALS)
print("Search space size:", len(search_space))
print("Total tuning time (seconds):", round(total_time, 2))
print("Total tuning time (minutes):", round(total_time/60, 2))
print("Device used:", device)
```

---

## 📈 Diagnostics & Plots (Notebook Cells 12–14)

These cells read the saved tuning artifacts and visualize:

- validation R² over trials
- training vs validation curves for top trials
- a quick “generalization gap” table for top trials

### 🔢 Code: Validation R² Across Trials (Notebook Cell 12)

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV into a DataFrame for preprocessing and modeling.
df = pd.read_csv(ARTIFACTS_DIR / "phase2_validation_results.csv").sort_values("trial_id")

# Visualize results to compare trials and diagnose training dynamics.
plt.figure()
plt.plot(df["trial_id"], df["val_r2"], marker="o")
plt.xlabel("Trial ID")
plt.ylabel("Validation R² (original scale)")
plt.title("Validation R² Across Trials")
plt.grid(True)
plt.show()
```

### 🔢 Code: Training Curves for Top Trials (Notebook Cell 13)

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV into a DataFrame for preprocessing and modeling.
df = pd.read_csv(ARTIFACTS_DIR / "phase2_validation_results.csv")
top = df.sort_values("val_r2", ascending=False).head(3)

# Visualize results to compare trials and diagnose training dynamics.
plt.figure()
for _, row in top.iterrows():
    hist = pd.read_csv(row["history_csv"])
    plt.plot(hist["epoch"], hist["train_loss"], label=f"Trial {int(row['trial_id'])} train")
    plt.plot(hist["epoch"], hist["val_rmse_log"], linestyle="--", label=f"Trial {int(row['trial_id'])} val_rmse_log")

# Visualize results to compare trials and diagnose training dynamics.
plt.xlabel("Epoch")
plt.ylabel("Loss / RMSE_log")
plt.title("Training Loss vs Validation RMSE_log (Top Trials)")
plt.grid(True)
plt.legend()
plt.show()
```

### 🔢 Code: Generalization Gap Table (Notebook Cell 14)

```python
import pandas as pd

# Load data from CSV into a DataFrame for preprocessing and modeling.
df = pd.read_csv(ARTIFACTS_DIR / "phase2_validation_results.csv")
top = df.sort_values("val_r2", ascending=False).head(5)

# Display: sets a table with coparing the top trials.
rows = []
for _, r in top.iterrows():
    hist = pd.read_csv(r["history_csv"])
    final = hist.iloc[-1]
    rows.append({
        "trial_id": int(r["trial_id"]),
        "val_r2": r["val_r2"],
        "final_train_loss": final["train_loss"],
        "final_val_rmse_log": final["val_rmse_log"],
    })

# Output: displays the table.
gap_df = pd.DataFrame(rows).sort_values("val_r2", ascending=False)
gap_df
```

---

## ✅ Best NN Validation Metrics (Notebook Cell 15)

After selecting the best NN from tuning, we evaluate it on validation:

```python
# NN validation metrics (log space + original scale)
val_pred_log_nn = evaluate_nn_log(best_model, X_val_s.values.astype(np.float32))

# Print key configuration so runs are easy to reproduce/debug.
print("Max log prediction:", float(np.max(val_pred_log_nn)))
print("Min log prediction:", float(np.min(val_pred_log_nn)))

# Next section: compute the step below.
nn_val_metrics = eval_regression(y_val_log, val_pred_log_nn, y_true_orig=y_val, label="VALIDATION — Neural Net:")
```

---

# 📈 Final Evaluation

## One-Time Holdout Test (Notebook-Aligned)

The **test/holdout set (15%)** is only used at the end to compare:

- tuned Ridge baseline
- best NN (selected via validation-only tuning)

### 🔢 Code: Final Holdout Evaluation + Summary Table (Notebook Cell 17)

```python
# Build test features ONLY here
X_holdout = build_features(holdout_df).reindex(columns=feature_cols, fill_value=0.0)
y_holdout = holdout_df[TARGET].to_numpy().astype(np.float64)
y_holdout_log = np.log1p(y_holdout)

# Apply standardization using training mean/std so models train stably.
X_holdout_s = (X_holdout - mu) / sigma

# Baseline test
hold_pred_log_ridge = ridge.predict(X_holdout_s)
hold_pred_log_ridge = np.clip(hold_pred_log_ridge, CLIP_MIN, CLIP_MAX)
hold_ridge_metrics = eval_regression(y_holdout_log, hold_pred_log_ridge, y_true_orig=y_holdout, label="TEST — Tuned Ridge:")

# NN test
hold_pred_log_nn = evaluate_nn_log(best_model, X_holdout_s.to_numpy().astype(np.float32))
hold_nn_metrics = eval_regression(y_holdout_log, hold_pred_log_nn, y_true_orig=y_holdout, label="\nTEST — Neural Net:")

# --- Summary table ---
summary = pd.DataFrame([
    {"Model": "Tuned Ridge", **hold_ridge_metrics},
    {"Model": "Best NN", **hold_nn_metrics},
])

# Print key configuration so runs are easy to reproduce/debug.
print("\n=== Test Summary (Final Comparison) ===")
display(summary)
```

---

# 🧭 Feature Importance

## Why Log-Space Permutation Importance?

Trip duration is heavy-tailed. The notebook measures feature importance in **log-space**, and does so for:

- tuned Ridge (log prediction function)
- best NN (log prediction function)

Importance is the mean drop in **R²_log** when a feature column is permuted.

### 🔢 Code: Permutation Importance (Notebook Cell 16)

```python
import pandas as pd

# Define helper function `permutation_importance_r2log` used in later cells.
def permutation_importance_r2log(predict_log_fn, X_df, y_true_log, n_repeats=5, seed=SEED):
    rng = np.random.default_rng(seed)

    # Next section: compute the step below.
    base_pred_log = predict_log_fn(X_df)
    base_r2 = r2_score(y_true_log, base_pred_log)

    # Next section: compute the step below.
    importances = []
    X_work = X_df.copy()

    # Iterate through this section's loop to compute/update results.
    for col in X_df.columns:
        drops = []
        for _ in range(n_repeats):
            saved = X_work[col].to_numpy().copy()
            X_work[col] = rng.permutation(X_work[col].to_numpy())
            pred_log = predict_log_fn(X_work)
            drops.append(base_r2 - r2_score(y_true_log, pred_log))
            X_work[col] = saved
        importances.append((col, float(np.mean(drops)), float(np.std(drops))))

    # Next section: compute the step below.
    imp = (pd.DataFrame(importances, columns=["feature", "mean_r2log_drop", "std_r2log_drop"])
           .sort_values("mean_r2log_drop", ascending=False)
           .reset_index(drop=True))
    return imp, base_r2

# Ridge predict (log space) — keep DataFrame to preserve feature names
def ridge_predict_log_fn(Xdf_raw):
    Xs_df = (Xdf_raw - mu) / sigma
    pred_log = ridge.predict(Xs_df)
    return np.clip(pred_log, CLIP_MIN, CLIP_MAX)

# NN predict (log space)
def nn_predict_log_fn(Xdf_raw):
    Xs = ((Xdf_raw - mu) / sigma).to_numpy().astype(np.float32)
    return evaluate_nn_log(best_model, Xs)

# Next section: compute the step below.
imp_ridge, ridge_r2log = permutation_importance_r2log(ridge_predict_log_fn, X_val, y_val_log, n_repeats=5, seed=SEED)
print("Permutation importance — Tuned Ridge on VAL (log space)")
print("Baseline VAL R2_log:", ridge_r2log)
display(imp_ridge.head(15))

# Next section: compute the step below.
imp_nn, nn_r2log = permutation_importance_r2log(nn_predict_log_fn, X_val, y_val_log, n_repeats=5, seed=SEED)
print("\nPermutation importance — Neural Net on VAL (log space)")
print("NN VAL R2_log:", nn_r2log)
display(imp_nn.head(15))
```

---

# 🧪 Hypothesis H1 Test

## H1 Definition

**H1:** A properly tuned neural network will outperform a tuned classical baseline model.

## Statistical Support via Bootstrap (Notebook Cell 18)

Beyond a single-point metric comparison, the notebook uses bootstrap resampling (1,000 resamples) on the test set to estimate confidence intervals for:

- ΔR²_log (NN − Ridge)
- ΔMAPE (NN − Ridge)

```python


from sklearn.metrics import r2_score

N_BOOTSTRAP = 1_000
rng = np.random.RandomState(SEED)

n = len(y_holdout_log)

delta_r2_log_samples = []
delta_mape_samples = []

for i in range(N_BOOTSTRAP):
    idx = rng.randint(0, n, size=n)  # bootstrap resample indices

    # Resample true values
    y_true_log_b = y_holdout_log[idx]
    y_true_orig_b = y_holdout[idx]

    # Resample predictions
    nn_pred_log_b = hold_pred_log_nn[idx]
    ridge_pred_log_b = hold_pred_log_ridge[idx]

    # R²_log for both models
    r2_nn = r2_score(y_true_log_b, nn_pred_log_b)
    r2_ridge = r2_score(y_true_log_b, ridge_pred_log_b)
    delta_r2_log_samples.append(r2_nn - r2_ridge)

    # MAPE for both models (original scale)
    mape_nn = mape(y_true_orig_b, safe_expm1(nn_pred_log_b))
    mape_ridge = mape(y_true_orig_b, safe_expm1(ridge_pred_log_b))
    delta_mape_samples.append(mape_nn - mape_ridge)

delta_r2_log_samples = np.array(delta_r2_log_samples)
delta_mape_samples = np.array(delta_mape_samples)

# 95% Confidence Intervals
ci_r2 = np.percentile(delta_r2_log_samples, [2.5, 97.5])
ci_mape = np.percentile(delta_mape_samples, [2.5, 97.5])

# --- Print Results ---
print("=" * 60)
print("📊 Bootstrap Statistical Validation on TEST (1,000 resamples)")
print("=" * 60)

print(f"\nΔR²_log (NN − Ridge):")
print(f"  Mean:   {delta_r2_log_samples.mean():.6f}")
print(f"  95% CI: [{ci_r2[0]:.6f}, {ci_r2[1]:.6f}]")
if ci_r2[0] > 0:
    print("  ✅ 0 is NOT in the interval → NN improvement is statistically significant")
else:
    print("  ⚠️ 0 is inside the interval → NN improvement is NOT statistically significant")

print(f"\nΔMAPE (NN − Ridge):")
print(f"  Mean:   {delta_mape_samples.mean():.4f}%")
print(f"  95% CI: [{ci_mape[0]:.4f}%, {ci_mape[1]:.4f}%]")
if ci_mape[1] < 0:
    print("  ✅ 0 is NOT in the interval → NN has significantly lower MAPE (better)")
else:
    print("  ⚠️ 0 is inside the interval → MAPE difference is NOT statistically significant")

print("=" * 60)
```


