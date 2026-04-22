import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit, GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIG =================

# EMB_DIR      = "output/ko_embeddings_650m"
EMB_DIR      = "/hpc/group/youlab/jlei912/taylor_law/kegg/output/ko_embeddings_650m"
SUMMARY_FILE = "data/refined_taylor_KO_Habitat_Category_Summary_with_a_b_bko.csv"
OUT_ROOT     = "results/mlp_classifier_b_ranges"

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42
TEST_SIZE    = 0.2
VAL_SIZE     = 0.15

TARGET = "b_ko_hab"
ETYPE  = "mean"

FEATURE_MODES = ["habitat_only", "embedding_only", "both"]

# ---------- Classification target config ----------
CLASS_TARGET = "b_class"
BINNING_MODE = "quantile"   # "quantile" or "fixed"
N_BINS       = 3

# Used only if BINNING_MODE == "fixed"
# Example: [-inf, 0.9), [0.9, 1.1), [1.1, inf)
FIXED_BINS   = [-np.inf, 0.9, 1.1, np.inf]

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

os.makedirs(OUT_ROOT, exist_ok=True)
print(f"Using device: {DEVICE}")

# ================= FINETUNE SEARCH SPACE =================

"""FINETUNE_SEARCH_SPACE = {
    "hidden_sizes": [ #BOGO architectures being tested
        (128, 64, 32),
        (256, 128, 64, 32),
        (512, 128, 32),
        (64, 32, 16),
        #(256, 32),
        # (128, 32),
    ],
    "lr":           [1e-4, 5e-4, 1e-3], #0.001
    "dropout":      [0.3, 0.4, 0.5], #0.5
    "weight_decay": [5e-4, 1e-3, 5e-3, 1e-2],# 0.0005
    "batch_size":   [256] #[128, 256, 512], #256
    "optimizer":    ['adam'] #["adam", "adamw"], #adam
    "use_scheduler":[False] #[True, False], #False
    "epochs":       [500],
    "patience":     [30],
}

N_ITER = 30
"""
FINETUNE_SEARCH_SPACE = {
    "hidden_sizes": [
        (512, 128, 32),
    ],
    "lr":           [1e-3], #0.001
    "dropout":      [0.5], #0.5
    "weight_decay": [5e-4],# 0.0005
    "batch_size":   [256], #[128, 256, 512], #256
    "optimizer":    ['adam'],
    "use_scheduler":[False],
    "epochs":       [500],
    "patience":     [30],
}
N_ITER = 1
# ================= CLASSIFIER MODEL =================

class TorchMLPClassifier(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        hidden_sizes=(128, 64, 32),
        lr=5e-4,
        dropout=0.4,
        epochs=500,
        patience=30,
        weight_decay=1e-3,
        batch_size=256,
        optimizer="adamw",
        use_scheduler=True,
        n_classes=3,
    ):
        self.hidden_sizes  = hidden_sizes
        self.lr            = lr
        self.dropout       = dropout
        self.epochs        = epochs
        self.patience      = patience
        self.weight_decay  = weight_decay
        self.batch_size    = batch_size
        self.optimizer     = optimizer
        self.use_scheduler = use_scheduler
        self.n_classes     = n_classes
        self.model_        = None
        self.classes_      = np.arange(self.n_classes)

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
        layers.append(nn.Linear(d, self.n_classes))
        return nn.Sequential(*layers).to(DEVICE)

    def _build_optimizer(self, model):
        if self.optimizer == "adamw":
            return optim.AdamW(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        return optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def fit(self, X, y, X_val=None, y_val=None, **kwargs):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        if X_val is None or y_val is None:
            from sklearn.model_selection import train_test_split
            X_tr, X_val, y_tr, y_val = train_test_split(
                X,
                y,
                test_size=0.1,
                random_state=RANDOM_STATE,
                stratify=y if len(np.unique(y)) > 1 else None,
            )
        else:
            X_tr, y_tr = X, y
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.int64)

        X_t = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
        y_t = torch.tensor(y_tr, dtype=torch.long).to(DEVICE)
        X_v = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
        y_v = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

        self.classes_ = np.unique(y)

        self.model_ = self._build_network(X_t.shape[1])
        opt = self._build_optimizer(self.model_)

        class_counts = np.bincount(y_tr, minlength=self.n_classes)
        class_weights = len(y_tr) / (self.n_classes * np.maximum(class_counts, 1))
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

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
                logits = self.model_(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

            self.model_.eval()
            with torch.no_grad():
                val_logits = self.model_(X_v)
                val_loss = loss_fn(val_logits, y_v).item()

            if scheduler is not None:
                scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                patience_ctr = 0
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
            X_t = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32).to(DEVICE)
            logits = self.model_(X_t)
            return torch.argmax(logits, dim=1).cpu().numpy()

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32).to(DEVICE)
            logits = self.model_(X_t)
            probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy()

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


def make_class_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    df = df[~pd.isna(df[target_col])].copy()

    if BINNING_MODE == "quantile":
        labels = list(range(N_BINS))
        df[CLASS_TARGET] = pd.qcut(
            df[target_col],
            q=N_BINS,
            labels=labels,
            duplicates="drop",
        )
        df = df[~pd.isna(df[CLASS_TARGET])].copy()
        df[CLASS_TARGET] = df[CLASS_TARGET].astype(int)

    elif BINNING_MODE == "fixed":
        labels = list(range(len(FIXED_BINS) - 1))
        df[CLASS_TARGET] = pd.cut(
            df[target_col],
            bins=FIXED_BINS,
            labels=labels,
            include_lowest=True,
            right=False,
        )
        df = df[~pd.isna(df[CLASS_TARGET])].copy()
        df[CLASS_TARGET] = df[CLASS_TARGET].astype(int)

    else:
        raise ValueError(f"Unknown BINNING_MODE: {BINNING_MODE}")

    n_classes = int(df[CLASS_TARGET].nunique())
    return df, n_classes


def ko_split(ko_list):
    rng = np.random.default_rng(RANDOM_STATE)
    unique = np.array(list(set(ko_list)))
    rng.shuffle(unique)
    n_test = max(1, int(len(unique) * TEST_SIZE))
    return set(unique[n_test:]), set(unique[:n_test])


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

        rows.append((ko, feat, int(row[target])))

    if not rows:
        return [], None, None

    return (
        [r[0] for r in rows],
        np.array([r[1] for r in rows], dtype=np.float32),
        np.array([r[2] for r in rows], dtype=np.int64),
    )


def evaluate_classifier(y_true, y_pred) -> dict:
    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        f1_macro=f1_score(y_true, y_pred, average="macro"),
        f1_weighted=f1_score(y_true, y_pred, average="weighted"),
    )


def plot_confusion(y_true, y_pred, name: str, out_dir: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=150)
    plt.close()

# ================= BASELINE EVALUATION =================

def run_baselines(X_tr_s, X_te_s, y_tr, y_te, n_emb, feature_mode, out_dir):
    print(f"\n{'='*70}")
    print(f"BASELINES [{feature_mode}]")
    print(f"{'='*70}")
    print(f"{'Model':<30} {'Acc':>8} {'Bal Acc':>10} {'F1 macro':>10}")
    print(f"{'-'*70}")

    results = []

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_tr_s, y_tr)
    m = evaluate_classifier(y_te, dummy.predict(X_te_s))
    print(
        f"{'Dummy (most frequent)':<30} "
        f"{m['accuracy']:>8.3f} {m['balanced_accuracy']:>10.3f} {m['f1_macro']:>10.3f}"
    )
    results.append({"model": "dummy_most_frequent", "feature_mode": feature_mode, **m})

    if feature_mode == "habitat_only":
        clf = LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
        clf.fit(X_tr_s, y_tr)
        m = evaluate_classifier(y_te, clf.predict(X_te_s))
        print(
            f"{'LogReg (habitat only)':<30} "
            f"{m['accuracy']:>8.3f} {m['balanced_accuracy']:>10.3f} {m['f1_macro']:>10.3f}"
        )
        results.append({"model": "logreg_habitat_only", "feature_mode": feature_mode, **m})

    elif feature_mode == "embedding_only":
        clf = LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
        clf.fit(X_tr_s, y_tr)
        m = evaluate_classifier(y_te, clf.predict(X_te_s))
        print(
            f"{'LogReg (embedding only)':<30} "
            f"{m['accuracy']:>8.3f} {m['balanced_accuracy']:>10.3f} {m['f1_macro']:>10.3f}"
        )
        results.append({"model": "logreg_embedding_only", "feature_mode": feature_mode, **m})

    elif feature_mode == "both":
        clf_hab = LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
        clf_hab.fit(X_tr_s[:, n_emb:], y_tr)
        m = evaluate_classifier(y_te, clf_hab.predict(X_te_s[:, n_emb:]))
        print(
            f"{'LogReg (habitat only)':<30} "
            f"{m['accuracy']:>8.3f} {m['balanced_accuracy']:>10.3f} {m['f1_macro']:>10.3f}"
        )
        results.append({"model": "logreg_habitat_only", "feature_mode": feature_mode, **m})

        clf_emb = LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
        clf_emb.fit(X_tr_s[:, :n_emb], y_tr)
        m = evaluate_classifier(y_te, clf_emb.predict(X_te_s[:, :n_emb]))
        print(
            f"{'LogReg (embedding only)':<30} "
            f"{m['accuracy']:>8.3f} {m['balanced_accuracy']:>10.3f} {m['f1_macro']:>10.3f}"
        )
        results.append({"model": "logreg_embedding_only", "feature_mode": feature_mode, **m})

        clf_all = LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
        clf_all.fit(X_tr_s, y_tr)
        m = evaluate_classifier(y_te, clf_all.predict(X_te_s))
        print(
            f"{'LogReg (emb + habitat)':<30} "
            f"{m['accuracy']:>8.3f} {m['balanced_accuracy']:>10.3f} {m['f1_macro']:>10.3f}"
        )
        results.append({"model": "logreg_emb_and_habitat", "feature_mode": feature_mode, **m})

    pd.DataFrame(results).to_csv(
        os.path.join(out_dir, "baseline_results.csv"),
        index=False
    )

    print(f"{'-'*70}")
    print("(MLP classification results printed below)\n")

    return results

# ================= FINETUNE PIPELINE =================

def run_finetune(X_tr, X_te, y_tr, y_te, ko_tr, n_emb, feature_mode, out_root, n_classes):
    out_dir = os.path.join(out_root, feature_mode)
    os.makedirs(out_dir, exist_ok=True)

    # --- Standardize X only ---
    x_scaler = StandardScaler()
    X_tr_s = x_scaler.fit_transform(X_tr)
    X_te_s = x_scaler.transform(X_te)

    # --- Baselines ---
    run_baselines(X_tr_s, X_te_s, y_tr, y_te, n_emb, feature_mode, out_dir)

    # --- KO-aware val split ---
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
    )
    train_idx, val_idx = next(gss.split(X_tr_s, y_tr, groups=ko_tr))

    test_fold = np.full(X_tr_s.shape[0], fill_value=-1)
    test_fold[val_idx] = 0
    ps = PredefinedSplit(test_fold)

    print(f"Split sizes — train: {len(train_idx)}  val: {len(val_idx)}  test: {len(y_te)}")
    print("Train class counts:", pd.Series(y_tr).value_counts().sort_index().to_dict())
    print("Test  class counts:", pd.Series(y_te).value_counts().sort_index().to_dict())

    # --- Phase 1: Hyperparameter search ---
    print(f"\nPhase 1 — hyperparam search ({N_ITER} iterations)...", end=" ", flush=True)

    search = RandomizedSearchCV(
        TorchMLPClassifier(n_classes=n_classes),
        FINETUNE_SEARCH_SPACE,
        n_iter=N_ITER,
        cv=ps,
        scoring="balanced_accuracy",
        random_state=RANDOM_STATE,
        refit=False,
        n_jobs=1,
    )
    search.fit(X_tr_s, y_tr)

    best_params = search.best_params_
    cv_score = search.best_score_
    print(f"best CV balanced_accuracy={cv_score:.3f}")
    print(f"Best params: {best_params}")

    # --- Phase 2: Final refit ---
    print(f"\nPhase 2 — final refit...", end=" ", flush=True)

    X_refit_tr = X_tr_s[train_idx]
    y_refit_tr = y_tr[train_idx]
    X_refit_val = X_tr_s[val_idx]
    y_refit_val = y_tr[val_idx]

    best_mlp = TorchMLPClassifier(**best_params, n_classes=n_classes)
    best_mlp.fit(
        X_refit_tr,
        y_refit_tr,
        X_val=X_refit_val,
        y_val=y_refit_val,
    )
    print("done.")

    # --- Predict ---
    pred_tr = best_mlp.predict(X_tr_s)
    pred_te = best_mlp.predict(X_te_s)
    probs_te = best_mlp.predict_proba(X_te_s)

    # --- Evaluate ---
    metrics_tr = evaluate_classifier(y_tr, pred_tr)
    metrics_te = evaluate_classifier(y_te, pred_te)

    print(f"\n{'='*70}")
    print(f"RESULTS [{feature_mode}]")
    print(f"{'='*70}")
    print(f"{'Metric':<20} {'Train':>8} {'Test':>8}")
    print(f"{'-'*40}")
    for key in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]:
        print(f"{key:<20} {metrics_tr[key]:>8.3f} {metrics_te[key]:>8.3f}")

    print("\nTest classification report:")
    print(classification_report(y_te, pred_te, digits=3))

    plot_confusion(
        y_tr,
        pred_tr,
        f"mlp_classifier_{feature_mode}_train",
        out_dir,
    )
    plot_confusion(
        y_te,
        pred_te,
        f"mlp_classifier_{feature_mode}_test",
        out_dir,
    )

    pred_df = pd.DataFrame({
        "true": y_te,
        "pred": pred_te,
    })
    for c in range(n_classes):
        pred_df[f"prob_class_{c}"] = probs_te[:, c]

    pred_df.to_csv(os.path.join(out_dir, "mlp_preds.csv"), index=False)

    pd.DataFrame([dict(
        model="MLP_classifier",
        feature_mode=feature_mode,
        etype=ETYPE,
        target=CLASS_TARGET,
        raw_target=TARGET,
        binning_mode=BINNING_MODE,
        input_dim=X_tr.shape[1],
        cv_balanced_accuracy=cv_score,
        **{f"{k}_train": v for k, v in metrics_tr.items()},
        **{f"{k}_test": v for k, v in metrics_te.items()},
        **best_params,
    )]).to_csv(
        os.path.join(out_dir, "mlp_classifier_results.csv"),
        index=False
    )

    print(f"\n✅ Results saved to {out_dir}")

    return {
        "feature_mode": feature_mode,
        "input_dim": X_tr.shape[1],
        "cv_balanced_accuracy": cv_score,
        **{f"{k}_train": v for k, v in metrics_tr.items()},
        **{f"{k}_test": v for k, v in metrics_te.items()},
        **best_params,
    }

# ================= MAIN =================

if __name__ == "__main__":
    os.makedirs(OUT_ROOT, exist_ok=True)

    df = pd.read_csv(SUMMARY_FILE)
    df, n_classes = make_class_target(df, TARGET)

    print(f"\nBinning mode: {BINNING_MODE}")
    print(f"Raw target: {TARGET}")
    print(f"Class target: {CLASS_TARGET}")
    print(f"Number of classes: {n_classes}")
    print("Overall class counts:", df[CLASS_TARGET].value_counts().sort_index().to_dict())

    if BINNING_MODE == "fixed":
        print(f"Fixed bins: {FIXED_BINS}")

    ko_emb = load_embeddings(ETYPE)

    h_enc = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
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
            CLASS_TARGET,
            habitat_enc=h_enc,
            feature_mode=feature_mode,
        )

        if X is None or y is None or len(ko_l) == 0:
            print(f"No valid rows for feature_mode={feature_mode}; skipping.")
            continue

        print(f"Total points: {len(ko_l)}")

        tr_kos, te_kos = ko_split(ko_l)
        tr_mask = np.array([k in tr_kos for k in ko_l])
        te_mask = np.array([k in te_kos for k in ko_l])

        X_tr, X_te = X[tr_mask], X[te_mask]
        y_tr, y_te = y[tr_mask], y[te_mask]
        ko_tr = np.array(ko_l)[tr_mask]

        if feature_mode in ["both", "embedding_only"]:
            n_emb = emb_dim
        else:
            n_emb = 0

        print(f"Input dim: {X.shape[1]}")
        print(f"Train: {len(y_tr)}  Test: {len(y_te)}")
        print("Train class counts:", pd.Series(y_tr).value_counts().sort_index().to_dict())
        print("Test  class counts:", pd.Series(y_te).value_counts().sort_index().to_dict())

        # Skip if any split is missing a class
        train_classes = set(np.unique(y_tr))
        test_classes = set(np.unique(y_te))
        all_classes = set(range(n_classes))

        if train_classes != all_classes:
            print(f"Skipping {feature_mode}: training split missing classes {sorted(all_classes - train_classes)}")
            continue
        if len(test_classes) < 2:
            print(f"Skipping {feature_mode}: test split has fewer than 2 classes.")
            continue

        result = run_finetune(
            X_tr,
            X_te,
            y_tr,
            y_te,
            ko_tr,
            n_emb=n_emb,
            feature_mode=feature_mode,
            out_root=OUT_ROOT,
            n_classes=n_classes,
        )
        all_results.append(result)

    if all_results:
        pd.DataFrame(all_results).to_csv(
            os.path.join(OUT_ROOT, "all_mlp_classifier_results.csv"),
            index=False
        )
        print(f"\n✅ Combined results saved to {OUT_ROOT}/all_mlp_classifier_results.csv")
    else:
        print("\n⚠️ No successful runs completed.")