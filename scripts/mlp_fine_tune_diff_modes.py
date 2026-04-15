import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit, GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIG =================

#EMB_DIR      = "output/ko_embeddings_650m"
EMB_DIR      = "/hpc/group/youlab/jlei912/taylor_law/kegg/output/ko_embeddings_650m"
SUMMARY_FILE = "data/refined_taylor_KO_Habitat_Category_Summary_with_a_b_bko.csv"
OUT_DIR      = "results/mlp_finetune_v2"
OUT_ROOT = "results/mlp_finetune_v2"

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42
TEST_SIZE    = 0.2
VAL_SIZE     = 0.15

TARGET = "b_ko_hab"
ETYPE  = "mean"

FEATURE_MODE = "both"   # "both", "embedding_only", "habitat_only"
FEATURE_MODES = ["habitat_only", "embedding_only", "both"]

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")

# ================= FINETUNE SEARCH SPACE =================

FINETUNE_SEARCH_SPACE = {
    "hidden_sizes": [ #BOGO architectures being tested
        (128, 64, 32),
        (256, 128, 64, 32),
        (512, 128, 32),
        (64, 32, 16),
        (256, 32),
        (128, 32),
    ],
    "lr":           [1e-4, 5e-4, 1e-3],
    "dropout":      [0.3, 0.4, 0.5],
    "weight_decay": [5e-4, 1e-3, 5e-3, 1e-2],
    "batch_size":   [128, 256, 512],
    "optimizer":    ["adam", "adamw"],
    "use_scheduler":[True, False],
    "epochs":       [500],
    "patience":     [30],
}

N_ITER = 50

# ================= MLP MODEL =================

class TorchMLP(BaseEstimator, RegressorMixin):

    def __init__(self, hidden_sizes=(128, 64, 32), lr=5e-4,
                 dropout=0.4, epochs=500, patience=30,
                 weight_decay=1e-3, batch_size=256,
                 optimizer="adamw", use_scheduler=True):
        self.hidden_sizes  = hidden_sizes
        self.lr            = lr
        self.dropout       = dropout
        self.epochs        = epochs
        self.patience      = patience
        self.weight_decay  = weight_decay
        self.batch_size    = batch_size
        self.optimizer     = optimizer
        self.use_scheduler = use_scheduler
        self.model_        = None

    def _build_network(self, input_dim: int) -> nn.Sequential:
        layers = []
        d = input_dim
        for h in self.hidden_sizes:
            layers += [
                nn.Linear(d, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ]
            d = h
        layers.append(nn.Linear(d, 1))
        return nn.Sequential(*layers).to(DEVICE)

    def _build_optimizer(self, model):
        if self.optimizer == "adamw":
            return optim.AdamW(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            return optim.Adam(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

    def fit(self, X, y, X_val=None, y_val=None, **kwargs):
        if X_val is None or y_val is None:
            from sklearn.model_selection import train_test_split
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=0.1, random_state=RANDOM_STATE
            )
        else:
            X_tr, y_tr = X, y

        X_t = torch.tensor(X_tr,  dtype=torch.float32).to(DEVICE)
        y_t = torch.tensor(y_tr,  dtype=torch.float32).view(-1, 1).to(DEVICE)
        X_v = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
        y_v = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(DEVICE)

        self.model_ = self._build_network(X_t.shape[1])
        opt         = self._build_optimizer(self.model_)
        loss_fn     = nn.MSELoss()

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=10
        ) if self.use_scheduler else None

        best_val_loss = float("inf")
        best_state    = None
        patience_ctr  = 0

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,
        )

        for epoch in range(self.epochs):
            self.model_.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(self.model_(xb), yb).backward()
                opt.step()

            self.model_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model_(X_v), y_v).item()

            if scheduler is not None:
                scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = copy.deepcopy(self.model_.state_dict())
                patience_ctr  = 0
            else:
                patience_ctr += 1
            if patience_ctr >= self.patience:
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            return self.model_(X_t).cpu().numpy().flatten()


# ================= HELPERS =================

def load_embeddings(etype: str) -> dict:
    ko_emb = {}
    edir = os.path.join(EMB_DIR, etype)
    for fname in tqdm(os.listdir(edir), desc=f"Loading [{etype}]"):
        if fname.endswith(".pt"):
            ko = fname.replace(".pt", "")
            ko_emb[ko] = torch.load(
                os.path.join(edir, fname),
                map_location="cpu",
                weights_only=True,
            ).numpy().astype(np.float32)
    return ko_emb


def ko_split(ko_list):
    rng    = np.random.default_rng(RANDOM_STATE)
    unique = np.array(list(set(ko_list)))
    rng.shuffle(unique)
    n_test = max(1, int(len(unique) * TEST_SIZE))
    return set(unique[n_test:]), set(unique[:n_test])


#mode-aware
def collect_rows(df, ko_emb, target, habitat_enc=None, feature_mode="both"):
    rows = []
    for _, row in df.iterrows():
        ko = row["KO"]
        if ko not in ko_emb or pd.isna(row[target]):
            continue

        emb = ko_emb[ko].copy().astype(np.float32)

        hab_vec = None
        if habitat_enc is not None:
            hab_vec = habitat_enc.transform(
                pd.DataFrame([[row["habitat"]]], columns=["habitat"])
            )[0].astype(np.float32)

        if feature_mode == "embedding_only":
            feat = emb
        elif feature_mode == "habitat_only":
            if hab_vec is None:
                continue
            feat = hab_vec
        elif feature_mode == "both":
            if hab_vec is None:
                feat = emb
            else:
                feat = np.concatenate([emb, hab_vec])
        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")

        rows.append((ko, feat, float(row[target])))

    if not rows:
        return [], None, None

    return (
        [r[0] for r in rows],
        np.array([r[1] for r in rows], dtype=np.float32),
        np.array([r[2] for r in rows], dtype=np.float32),
    )


def evaluate(y_true, y_pred) -> dict:
    r,   p_r   = pearsonr(y_true,  y_pred)
    rho, p_rho = spearmanr(y_true, y_pred)
    return dict(
        r2         = r2_score(y_true, y_pred),
        rmse       = root_mean_squared_error(y_true, y_pred),
        mae        = mean_absolute_error(y_true, y_pred),
        pearson_r  = r,
        pearson_p  = p_r,
        spearman_r = rho,
        spearman_p = p_rho,
    )


def plot_regression(y_true, y_pred, metrics: dict, name: str, out_dir: str):
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, s=15)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", lw=1)
    plt.text(
        0.05, 0.95,
        f"$R^2$={metrics['r2']:.3f}\n"
        f"Pearson r={metrics['pearson_r']:.3f}\n"
        f"Spearman ρ={metrics['spearman_r']:.3f}",
        transform=plt.gca().transAxes, va="top", fontsize=9,
    )
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=150)
    plt.close()


# ================= BASELINE EVALUATION =================
#mode-aware version
def run_baselines(X_tr_s, X_te_s, y_tr, y_te, n_emb, feature_mode, out_dir):
    print(f"\n{'='*50}")
    print(f"BASELINES [{feature_mode}]")
    print(f"{'='*50}")
    print(f"{'Model':<25} {'R²':>7} {'Spearman ρ':>11}")
    print(f"{'-'*45}")

    results = []

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_tr_s, y_tr)
    m = evaluate(y_te, dummy.predict(X_te_s))
    print(f"{'Dummy (mean)':<25} {m['r2']:>7.3f} {m['spearman_r']:>11.3f}")
    results.append({"model": "dummy_mean", "feature_mode": feature_mode, **m})

    if feature_mode == "habitat_only":
        ridge = Ridge()
        ridge.fit(X_tr_s, y_tr)
        m = evaluate(y_te, ridge.predict(X_te_s))
        print(f"{'Ridge (habitat only)':<25} {m['r2']:>7.3f} {m['spearman_r']:>11.3f}")
        results.append({"model": "ridge_habitat_only", "feature_mode": feature_mode, **m})

    elif feature_mode == "embedding_only":
        ridge = Ridge()
        ridge.fit(X_tr_s, y_tr)
        m = evaluate(y_te, ridge.predict(X_te_s))
        print(f"{'Ridge (embedding only)':<25} {m['r2']:>7.3f} {m['spearman_r']:>11.3f}")
        results.append({"model": "ridge_embedding_only", "feature_mode": feature_mode, **m})

    elif feature_mode == "both":
        ridge_hab = Ridge()
        ridge_hab.fit(X_tr_s[:, n_emb:], y_tr)
        m = evaluate(y_te, ridge_hab.predict(X_te_s[:, n_emb:]))
        print(f"{'Ridge (habitat only)':<25} {m['r2']:>7.3f} {m['spearman_r']:>11.3f}")
        results.append({"model": "ridge_habitat_only", "feature_mode": feature_mode, **m})

        ridge_emb = Ridge()
        ridge_emb.fit(X_tr_s[:, :n_emb], y_tr)
        m = evaluate(y_te, ridge_emb.predict(X_te_s[:, :n_emb]))
        print(f"{'Ridge (embedding only)':<25} {m['r2']:>7.3f} {m['spearman_r']:>11.3f}")
        results.append({"model": "ridge_embedding_only", "feature_mode": feature_mode, **m})

        ridge_all = Ridge()
        ridge_all.fit(X_tr_s, y_tr)
        m = evaluate(y_te, ridge_all.predict(X_te_s))
        print(f"{'Ridge (emb + habitat)':<25} {m['r2']:>7.3f} {m['spearman_r']:>11.3f}")
        results.append({"model": "ridge_emb_and_habitat", "feature_mode": feature_mode, **m})

    pd.DataFrame(results).to_csv(
        os.path.join(out_dir, "baseline_results.csv"), index=False
    )
    print(f"{'-'*45}")
    print("(MLP results printed below)\n")

    return results


# ================= FINETUNE PIPELINE =================

def run_finetune(X_tr, X_te, y_tr, y_te, ko_tr, n_emb, feature_mode, out_root):
    out_dir = os.path.join(out_root, feature_mode)
    os.makedirs(out_dir, exist_ok=True)

    # --- Standardize ---
    x_scaler = StandardScaler()
    X_tr_s   = x_scaler.fit_transform(X_tr)
    X_te_s   = x_scaler.transform(X_te)

    y_scaler = StandardScaler()
    y_tr_s   = y_scaler.fit_transform(y_tr.reshape(-1, 1)).flatten()

    # --- Baselines ---
    run_baselines(X_tr_s, X_te_s, y_tr, y_te, n_emb, feature_mode, out_dir)

    # --- KO-aware val split ---
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE
    )
    train_idx, val_idx = next(gss.split(X_tr_s, y_tr_s, groups=ko_tr))

    test_fold = np.full(X_tr_s.shape[0], fill_value=-1)
    test_fold[val_idx] = 0
    ps = PredefinedSplit(test_fold)

    print(f"Split sizes — train: {len(train_idx)}  val: {len(val_idx)}  test: {len(y_te)}")

    # --- Phase 1: Hyperparameter search ---
    print(f"\nPhase 1 — hyperparam search ({N_ITER} iterations)...", end=" ", flush=True)

    search = RandomizedSearchCV(
        TorchMLP(),
        FINETUNE_SEARCH_SPACE,
        n_iter=N_ITER,
        cv=ps,
        scoring="r2",
        random_state=RANDOM_STATE,
        refit=False,
    )
    search.fit(X_tr_s, y_tr_s)

    best_params = search.best_params_
    cv_score    = search.best_score_
    print(f"best CV R²={cv_score:.3f}")
    print(f"Best params: {best_params}")

    # --- Phase 2: Final refit ---
    print(f"\nPhase 2 — final refit...", end=" ", flush=True)

    X_refit_tr  = X_tr_s[train_idx]
    y_refit_tr  = y_tr_s[train_idx]
    X_refit_val = X_tr_s[val_idx]
    y_refit_val = y_tr_s[val_idx]

    best_mlp = TorchMLP(**best_params)
    best_mlp.fit(
        X_refit_tr, y_refit_tr,
        X_val=X_refit_val,
        y_val=y_refit_val,
    )
    print("done.")

    # --- Predict and inverse-transform ---
    pred_tr = y_scaler.inverse_transform(
        best_mlp.predict(X_tr_s).reshape(-1, 1)
    ).flatten()
    pred_te = y_scaler.inverse_transform(
        best_mlp.predict(X_te_s).reshape(-1, 1)
    ).flatten()

    # --- Evaluate ---
    metrics_tr = evaluate(y_tr, pred_tr)
    metrics_te = evaluate(y_te, pred_te)

    print(f"\n{'='*60}")
    print(f"RESULTS [{feature_mode}]")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'Train':>8} {'Test':>8}")
    print(f"{'-'*35}")
    for key in ["r2", "rmse", "mae", "pearson_r", "spearman_r"]:
        print(f"{key:<15} {metrics_tr[key]:>8.3f} {metrics_te[key]:>8.3f}")
    print(f"{'gap (R²)':<15} {metrics_tr['r2'] - metrics_te['r2']:>8.3f}")

    plot_regression(
        y_tr, pred_tr, metrics_tr,
        f"finetune_v2_{feature_mode}_train",
        out_dir
    )
    plot_regression(
        y_te, pred_te, metrics_te,
        f"finetune_v2_{feature_mode}_test",
        out_dir
    )

    pd.DataFrame({
        "true": y_te,
        "pred": pred_te
    }).to_csv(os.path.join(out_dir, "mlp_preds.csv"), index=False)

    pd.DataFrame([dict(
        model="MLP_finetune_v2",
        feature_mode=feature_mode,
        etype=ETYPE,
        target=TARGET,
        hypothesis="H2_global",
        input_dim=X_tr.shape[1],
        r2_cv=cv_score,
        **{f"{k}_train": v for k, v in metrics_tr.items()},
        **{f"{k}_test": v for k, v in metrics_te.items()},
        **best_params,
    )]).to_csv(
        os.path.join(out_dir, "finetune_v2_results.csv"),
        index=False
    )

    print(f"\n✅ Results saved to {out_dir}")

    return {
        "feature_mode": feature_mode,
        "input_dim": X_tr.shape[1],
        "r2_cv": cv_score,
        **{f"{k}_train": v for k, v in metrics_tr.items()},
        **{f"{k}_test": v for k, v in metrics_te.items()},
        **best_params,
    }


# ================= MAIN =================
if __name__ == "__main__":
    FEATURE_MODES = ["habitat_only", "embedding_only", "both"]
    OUT_ROOT = "results/mlp_finetune_v2"

    os.makedirs(OUT_ROOT, exist_ok=True)

    df     = pd.read_csv(SUMMARY_FILE)
    ko_emb = load_embeddings(ETYPE)

    h_enc = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    ).fit(df[["habitat"]])

    emb_dim = next(iter(ko_emb.values())).shape[0]

    all_results = []

    for feature_mode in FEATURE_MODES:
        print(f"\n\n{'#'*70}")
        print(f"RUNNING FEATURE MODE: {feature_mode}")
        print(f"{'#'*70}")

        ko_l, X, y = collect_rows(
            df,
            ko_emb,
            TARGET,
            habitat_enc=h_enc,
            feature_mode=feature_mode
        )

        print(f"Total points: {len(ko_l)}")

        tr_kos, te_kos = ko_split(ko_l)
        tr_mask = np.array([k in tr_kos for k in ko_l])
        te_mask = np.array([k in te_kos for k in ko_l])

        X_tr, X_te = X[tr_mask], X[te_mask]
        y_tr, y_te = y[tr_mask], y[te_mask]
        ko_tr      = np.array(ko_l)[tr_mask]

        if feature_mode == "both":
            n_emb = emb_dim
        elif feature_mode == "embedding_only":
            n_emb = emb_dim
        else:  # habitat_only
            n_emb = 0

        print(f"Input dim: {X.shape[1]}")
        print(f"Train: {len(y_tr)}  Test: {len(y_te)}")

        result = run_finetune(
            X_tr, X_te, y_tr, y_te, ko_tr,
            n_emb=n_emb,
            feature_mode=feature_mode,
            out_root=OUT_ROOT
        )
        all_results.append(result)

    pd.DataFrame(all_results).to_csv(
        os.path.join(OUT_ROOT, "all_mlp_results.csv"),
        index=False
    )
    print(f"\n✅ Combined results saved to {OUT_ROOT}/all_mlp_results.csv")
