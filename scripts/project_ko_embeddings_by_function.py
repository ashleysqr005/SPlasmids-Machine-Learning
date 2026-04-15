#!/usr/bin/env python3
"""
Project KO embeddings with UMAP and t-SNE.

KEY DESIGN:
- UMAP/t-SNE computed on UNIQUE KOs only (one point per KO, ~12k points).
- Global plots: colored by Level3_clean (categorical) or mean b_ko_hab across habitats.
- Per-habitat plots: each KO colored by its b_ko_hab in that habitat (each
  habitat uses its own color-scale limits); KOs absent from a habitat in grey.

GPU-accelerated via cuML (RAPIDS). Falls back to CPU if unavailable.

Usage:
  python scripts/project_ko_embeddings_by_function.py \
    --embeddings-dir /hpc/group/youlab/jlei912/taylor_law/kegg/output/ko_embeddings_650m/mean \
    --metadata-csv data/refined_taylor_KO_Habitat_Category_Summary_with_a_b_bko.csv \
    --output-dir results/project_embeddings \
    --umap-n-neighbors 30 --umap-min-dist 0.3 --umap-metric cosine \
    --plot-per-category --tsne-perplexity 15

  # Add --load-coords to skip recomputing and only redo plots.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Optional GPU backends
# ---------------------------------------------------------------------------
try:
    from cuml.manifold import UMAP as cuUMAP
    from cuml.manifold import TSNE as cuTSNE
    CUML_AVAILABLE = True
    print("[GPU] cuML detected — UMAP and t-SNE will run on GPU.", flush=True)
except ImportError:
    CUML_AVAILABLE = False
    print("[CPU] cuML not found — falling back to umap-learn / sklearn.", flush=True)

if not CUML_AVAILABLE:
    try:
        import umap as umap_learn
    except ImportError as e:
        raise SystemExit(
            "Neither cuML nor umap-learn found.\n"
            "  GPU path : pip install cuml-cu12\n"
            "  CPU path : pip install umap-learn"
        ) from e


# ---------------------------------------------------------------------------
# GPU PCA
# ---------------------------------------------------------------------------
def pca_torch(x: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    device = torch.device("cuda")
    torch.manual_seed(seed)
    X = torch.from_numpy(x).to(device, dtype=torch.float32)
    X = X - X.mean(dim=0, keepdim=True)
    try:
        U, S, _ = torch.linalg.svd(X, full_matrices=False)
        coords = U[:, :n_components] * S[:n_components]
    except Exception:
        U, S, _ = torch.pca_lowrank(X, q=n_components, niter=4)
        coords = U * S
    return coords.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Data loading & preparation
# ---------------------------------------------------------------------------
def resolve_embeddings_dir(repo_root: Path) -> Path:
    for name in ("ko_embeddings", "ko_embeddings_150m", "ko_embeddings_650m"):
        p = repo_root / name
        if p.is_dir() and any(p.glob("*.pt")):
            return p
    raise FileNotFoundError(f"No *.pt directory found under {repo_root}")


def load_embeddings(embed_dir: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for path in sorted(embed_dir.glob("*.pt")):
        ko = path.stem
        try:
            t = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            t = torch.load(path, map_location="cpu")
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t)
        out[ko] = t.detach().float().cpu().numpy().ravel().astype(np.float32)
    return out


def prepare_data(
    raw_df: pd.DataFrame,
    emb: dict[str, np.ndarray],
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        x         : (n_ko, dim) embedding matrix, one row per unique KO
        ko_df     : one row per unique KO
        hab_pivot : rows=KO, cols=habitat, values=b_ko_hab
        long_df   : long table with one row per KO x habitat
                    columns: KO, Level3_clean, habitat, b_ko_hab
    """
    df = raw_df[raw_df["KO"].astype(str).isin(emb.keys())].copy()
    df["KO"] = df["KO"].astype(str)

    # Long-format deduplicated KO x habitat x category table
    long_df = (
        df[["KO", "Level3_clean", "habitat", "b_ko_hab"]]
        .drop_duplicates(subset=["KO", "habitat"])
        .copy()
    )

    # Optional thresholding
    min_b_ko_hab = 1.2
    long_df.loc[long_df["b_ko_hab"] < min_b_ko_hab, "b_ko_hab"] = np.nan
    long_df = long_df.dropna(subset=["b_ko_hab"])

    # pivot: rows=KO, cols=habitat, values=b_ko_hab
    hab_pivot = long_df.pivot(index="KO", columns="habitat", values="b_ko_hab")
    hab_pivot.columns.name = None

    # KO-level metadata
    ko_meta = (
        df[["KO", "Level3_clean"]]
        .drop_duplicates(subset="KO")
        .set_index("KO")
    )
    ko_meta["mean_b_ko_hab"] = hab_pivot.mean(axis=1)

    ko_order = sorted(hab_pivot.index)
    hab_pivot = hab_pivot.loc[ko_order]
    ko_meta = ko_meta.loc[ko_order]

    x = np.vstack([emb[ko] for ko in ko_order]).astype(np.float32)
    ko_df = ko_meta.reset_index()

    return x, ko_df, hab_pivot.reset_index(drop=True), long_df

def make_coord_df(z: np.ndarray, ko_df: pd.DataFrame) -> pd.DataFrame:
    out = ko_df[["KO", "Level3_clean"]].copy()
    out["dim1"] = z[:, 0]
    out["dim2"] = z[:, 1]
    return out

def make_level3_lut(ko_df: pd.DataFrame) -> dict[str, tuple]:
    labels = ko_df["Level3_clean"].astype(str).fillna("Unknown")
    uniq = pd.Index(labels.unique()).sort_values()
    palette = sns.color_palette("husl", n_colors=max(len(uniq), 1))
    return {lab: palette[i % len(palette)] for i, lab in enumerate(uniq)}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(x, l2_normalize, pca_dim, seed, use_gpu_pca):
    if l2_normalize:
        x = normalize(x, norm="l2", axis=1)
    if pca_dim and pca_dim > 0:
        if pca_dim > x.shape[1]:
            raise ValueError("pca_dim cannot exceed embedding dimension.")
        if use_gpu_pca and torch.cuda.is_available():
            print(f"  PCA on GPU ({x.shape} -> {pca_dim} dims) …", flush=True)
            x = pca_torch(x, pca_dim, seed)
        else:
            print(f"  PCA on CPU ({x.shape} -> {pca_dim} dims) …", flush=True)
            x = PCA(n_components=pca_dim, random_state=seed).fit_transform(x)
    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------
def run_umap(x, seed, n_neighbors, min_dist, metric):
    if CUML_AVAILABLE:
        print("  UMAP on GPU (cuML) …", flush=True)
        r      = cuUMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        metric=metric, random_state=seed, verbose=False)
        result = r.fit_transform(x.astype(np.float32))
        return result.get() if hasattr(result, "get") else np.asarray(result)
    else:
        print("  UMAP on CPU (umap-learn) …", flush=True)
        r = umap_learn.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                            metric=metric, random_state=seed,
                            low_memory=False, n_jobs=-1, verbose=False)
        return r.fit_transform(x)


def run_tsne(x, seed, perplexity):
    n    = x.shape[0]
    perp = float(min(perplexity, max(5, (n - 1) // 3)))
    if CUML_AVAILABLE:
        print(f"  t-SNE on GPU (cuML, perplexity={perp}) …", flush=True)
        t      = cuTSNE(n_components=2, perplexity=perp, learning_rate=200.0,
                        max_iter=2000, random_state=seed, verbose=False, method="fft")
        result = t.fit_transform(x.astype(np.float32))
        return result.get() if hasattr(result, "get") else np.asarray(result)
    else:
        print(f"  t-SNE on CPU (sklearn, perplexity={perp}) …", flush=True)
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, perplexity=perp, learning_rate="auto",
                    init="pca", random_state=seed, max_iter=1000).fit_transform(x)


# ---------------------------------------------------------------------------
# Plotting — global
# ---------------------------------------------------------------------------
def plot_categorical(z, labels, title, out_path):
    uniq    = pd.Index(labels.astype(str).unique()).sort_values()
    palette = sns.color_palette("husl", n_colors=max(len(uniq), 1))
    lut     = {lab: palette[i % len(palette)] for i, lab in enumerate(uniq)}
    colors  = labels.astype(str).map(lut)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(z[:, 0], z[:, 1], c=colors.tolist(), s=4, alpha=0.75, linewidths=0)
    ax.set_title(title); ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    handles = [plt.Line2D([0], [0], marker="o", color="w", label=str(lab),
               markerfacecolor=lut[lab], markersize=8) for lab in uniq]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
              borderaxespad=0, fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_continuous(z, values, title, out_path, colorbar_label):
    v = np.asarray(values, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(z[:, 0], z[:, 1], c=v, s=4, alpha=0.75, linewidths=0, cmap="plasma")
    ax.set_title(title); ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plotting — per functional category
# ---------------------------------------------------------------------------

def plot_per_functional_category_by_habitat(z, ko_df, long_df, method, out_dir):
    """
    One PNG per functional category (Level3_clean).
    Within each plot, points are colored by habitat.
    Each point is a KO-habitat observation located at that KO's embedding coords.
    """
    cat_dir = out_dir / f"{method}_per_functional_category"
    cat_dir.mkdir(parents=True, exist_ok=True)

    coord_df = make_coord_df(z, ko_df)

    # Join coords onto KO x habitat rows
    plot_df = long_df.merge(coord_df[["KO", "dim1", "dim2"]], on="KO", how="inner")

    habitats = pd.Index(plot_df["habitat"].astype(str).unique()).sort_values()
    palette = sns.color_palette("husl", n_colors=max(len(habitats), 1))
    hab_lut = {hab: palette[i % len(palette)] for i, hab in enumerate(habitats)}

    categories = pd.Index(plot_df["Level3_clean"].astype(str).unique()).sort_values()

    for cat in categories:
        sub = plot_df[plot_df["Level3_clean"].astype(str) == str(cat)].copy()
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot one layer per habitat
        for hab in habitats:
            hab_sub = sub[sub["habitat"].astype(str) == str(hab)]
            if hab_sub.empty:
                continue

            ax.scatter(
                hab_sub["dim1"].values,
                hab_sub["dim2"].values,
                c=[hab_lut[hab]],
                s=12,
                alpha=0.5,
                linewidths=0,
                label=str(hab)
            )

        ax.set_title(f"{method.upper()} — {cat} colored by habitat")
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")

        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
            fontsize=7,
            frameon=False
        )

        fig.tight_layout()
        safe = str(cat).replace(" ", "_").replace("/", "-")
        fig.savefig(cat_dir / f"{method}_habitat_by_function_{safe}.png",
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"  Per-functional-category plots -> {cat_dir}", flush=True)

def plot_functional_category_grid_by_habitat(z, ko_df, long_df, method, out_dir):
    """
    Grid of functional categories, colored by habitat.
    """
    coord_df = make_coord_df(z, ko_df)
    plot_df = long_df.merge(coord_df[["KO", "dim1", "dim2"]], on="KO", how="inner")

    habitats = pd.Index(plot_df["habitat"].astype(str).unique()).sort_values()
    palette = sns.color_palette("husl", n_colors=max(len(habitats), 1))
    hab_lut = {hab: palette[i % len(palette)] for i, hab in enumerate(habitats)}

    categories = pd.Index(plot_df["Level3_clean"].astype(str).unique()).sort_values()
    n = len(categories)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.8 * ncols, 4 * nrows),
        squeeze=False
    )
    fig.suptitle(f"{method.upper()} — functional categories colored by habitat", fontsize=14, y=1.01)

    for idx, cat in enumerate(categories):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        sub = plot_df[plot_df["Level3_clean"].astype(str) == str(cat)].copy()

        for hab in habitats:
            hab_sub = sub[sub["habitat"].astype(str) == str(hab)]
            if hab_sub.empty:
                continue

            ax.scatter(
                hab_sub["dim1"].values,
                hab_sub["dim2"].values,
                c=[hab_lut[hab]],
                s=6,
                alpha=0.5,
                linewidths=0
            )

        ax.set_title(f"{cat}\n(n={len(sub)})", fontsize=9)
        ax.set_xlabel("dim 1", fontsize=7)
        ax.set_ylabel("dim 2", fontsize=7)
        ax.tick_params(labelsize=6)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=str(hab),
                   markerfacecolor=hab_lut[hab], markersize=6)
        for hab in habitats
    ]
    fig.legend(handles=handles, bbox_to_anchor=(1.02, 0.5),
               loc="center left", fontsize=7, frameon=False)

    fig.tight_layout()
    grid_path = out_dir / f"{method}_functional_category_grid_colored_by_habitat.png"
    fig.savefig(grid_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Functional-category grid -> {grid_path}", flush=True)

# ---------------------------------------------------------------------------
# Plotting — per habitat
# ---------------------------------------------------------------------------
def _b_ko_hab_colormap_limits(vals: np.ndarray) -> tuple[float, float]:
    """Per-habitat vmin/vmax; expands degenerate range so the colormap is valid."""
    if vals.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return 0.0, 1.0
    if vmin == vmax:
        eps = max(abs(vmin) * 1e-6, 1e-9)
        vmin -= eps
        vmax += eps
    return vmin, vmax


def plot_per_habitat(z, ko_df, hab_pivot, method, out_dir):
    """
    One PNG per habitat.
    KOs present in habitat: colored by Level3_clean.
    KOs absent from habitat: light grey.
    """
    habitats = list(hab_pivot.columns)
    hab_dir = out_dir / f"{method}_per_habitat"
    hab_dir.mkdir(parents=True, exist_ok=True)

    labels = ko_df["Level3_clean"].astype(str).fillna("Unknown")
    lut = make_level3_lut(ko_df)

    for hab in habitats:
        b_series = hab_pivot[hab]
        present = b_series.notna().values
        absent = ~present

        fig, ax = plt.subplots(figsize=(10, 8))

        # Absent KOs in grey
        if absent.any():
            ax.scatter(
                z[absent, 0], z[absent, 1],
                c="lightgrey", s=3, alpha=0.25, linewidths=0, zorder=1
            )

        # Present KOs colored by functional group
        if present.any():
            present_labels = labels[present]
            present_colors = present_labels.map(lut)

            ax.scatter(
                z[present, 0], z[present, 1],
                c=present_colors.tolist(),
                s=10, alpha=0.85, linewidths=0, zorder=2
            )

        ax.set_title(f"{method.upper()} — {hab} (n={int(present.sum())}) | Level3_clean")
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")

        # Only show categories present in this habitat
        present_uniq = pd.Index(labels[present].unique()).sort_values()
        handles = [
            plt.Line2D(
                [0], [0], marker="o", color="w", label=str(lab),
                markerfacecolor=lut[lab], markersize=7
            )
            for lab in present_uniq
        ]
        if handles:
            ax.legend(
                handles=handles,
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                borderaxespad=0,
                fontsize=7,
                frameon=False
            )

        fig.tight_layout()
        safe = hab.replace(" ", "_").replace("/", "-")
        fig.savefig(hab_dir / f"{method}_level3_{safe}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"  Per-habitat plots colored by Level3_clean -> {hab_dir}", flush=True)


def plot_per_habitat_grid(z, ko_df, hab_pivot, method, out_dir):
    """
    Summary grid: one subplot per habitat; points colored by Level3_clean.
    """
    habitats = list(hab_pivot.columns)
    n = len(habitats)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    labels = ko_df["Level3_clean"].astype(str).fillna("Unknown")
    lut = make_level3_lut(ko_df)
    uniq_labels = pd.Index(labels.unique()).sort_values()

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.8 * ncols, 4 * nrows),
        squeeze=False
    )
    fig.suptitle(f"{method.upper()} — Level3_clean per habitat", fontsize=14, y=1.01)

    for idx, hab in enumerate(habitats):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        b_series = hab_pivot[hab]
        present = b_series.notna().values
        absent = ~present

        if absent.any():
            ax.scatter(
                z[absent, 0], z[absent, 1],
                c="lightgrey", s=2, alpha=0.2, linewidths=0, zorder=1
            )

        if present.any():
            present_labels = labels[present]
            present_colors = present_labels.map(lut)

            ax.scatter(
                z[present, 0], z[present, 1],
                c=present_colors.tolist(),
                s=6, alpha=0.8, linewidths=0, zorder=2
            )

        ax.set_title(f"{hab}\n(n={int(present.sum())})", fontsize=9)
        ax.set_xlabel("dim 1", fontsize=7)
        ax.set_ylabel("dim 2", fontsize=7)
        ax.tick_params(labelsize=6)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", label=str(lab),
            markerfacecolor=lut[lab], markersize=6
        )
        for lab in uniq_labels
    ]
    fig.legend(
        handles=handles,
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        fontsize=7,
        frameon=False
    )

    fig.tight_layout()
    grid_path = out_dir / f"{method}_level3_per_habitat_grid.png"
    fig.savefig(grid_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Per-habitat grid colored by Level3_clean -> {grid_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root",        type=Path,  default=None)
    parser.add_argument("--embeddings-dir",   type=Path,  default=None)
    parser.add_argument("--metadata-csv",     type=Path,  default=None)
    parser.add_argument("--output-dir",       type=Path,  default=None)
    parser.add_argument("--random-seed",      type=int,   default=42)
    parser.add_argument("--max-samples",      type=int,   default=None,
                        help="Subsample KOs for fast iteration.")
    parser.add_argument("--pca-dim",          type=int,   default=200)
    parser.add_argument("--no-l2-normalize",  action="store_true")
    parser.add_argument("--no-gpu-pca",       action="store_true")
    parser.add_argument("--tsne-perplexity",  type=float, default=30.0) #bogo
    parser.add_argument("--umap-n-neighbors", type=int,   default=15) #bogo
    parser.add_argument("--umap-min-dist",    type=float, default=0.1)
    parser.add_argument("--umap-metric",      type=str,   default="cosine",
                        choices=["euclidean", "cosine"])
    parser.add_argument("--skip-umap",        action="store_true")
    parser.add_argument("--skip-tsne",        action="store_true")
    parser.add_argument("--plot-per-category", action="store_true",
                        help="Produce per-habitat b_ko_hab plots.")
    parser.add_argument("--load-coords",      action="store_true",
                        help="Load saved coords instead of recomputing.")
    args = parser.parse_args()

    if args.skip_umap and args.skip_tsne:
        print("Specify at most one of --skip-umap / --skip-tsne.", file=sys.stderr)
        return 1

    if torch.cuda.is_available():
        print(f"[GPU] CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("[CPU] No CUDA device found.", flush=True)

    repo    = args.repo_root or Path(__file__).resolve().parents[1]
    emb_dir = args.embeddings_dir or resolve_embeddings_dir(repo)
    meta    = args.metadata_csv or (
        repo / "data/refined_taylor_KO_Habitat_Category_Summary_with_a_b_bko.csv"
    )
    out_dir = args.output_dir or (repo / "figures" / "ko_embedding_projections")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not meta.is_file():
        print(f"Metadata CSV not found: {meta}", file=sys.stderr)
        return 1

    print(f"Loading embeddings from {emb_dir} …", flush=True)
    emb = load_embeddings(emb_dir)
    print(f"  {len(emb)} KO tensors loaded.", flush=True)

    raw_df = pd.read_csv(meta)
    x, ko_df, hab_pivot, long_df = prepare_data(raw_df, emb)
    print(f"Unique KOs: {len(ko_df)}  |  Habitats: {list(hab_pivot.columns)}", flush=True)

    """# Flatten all b_ko_hab values and drop NaNs
    all_b = hab_pivot.values.ravel()
    all_b = all_b[~np.isnan(all_b)]

    plt.figure(figsize=(8, 5))
    plt.hist(all_b, bins=100)
    plt.xlabel("b_ko_hab")
    plt.ylabel("Count")
    plt.title("Distribution of b_ko_hab (all habitats)")
    plt.tight_layout()
    plt.savefig(out_dir / "b_ko_hab_distribution_hist.png", dpi=200)
    plt.close()"""

    if args.max_samples and args.max_samples < len(ko_df):
        rng       = np.random.default_rng(args.random_seed)
        idx       = rng.choice(len(ko_df), size=args.max_samples, replace=False)
        x         = x[idx]
        ko_df     = ko_df.iloc[idx].reset_index(drop=True)
        hab_pivot = hab_pivot.iloc[idx].reset_index(drop=True)
        print(f"Subsampled to {args.max_samples} KOs.", flush=True)

    x_proc = preprocess(x,
                        l2_normalize=not args.no_l2_normalize,
                        pca_dim=args.pca_dim,
                        seed=args.random_seed,
                        use_gpu_pca=not args.no_gpu_pca)

    z_umap: np.ndarray | None = None
    z_tsne: np.ndarray | None = None

    if args.load_coords:
        umap_path = out_dir / "coords_umap.npy"
        tsne_path = out_dir / "coords_tsne.npy"
        if umap_path.is_file() and not args.skip_umap:
            z_umap = np.load(umap_path)
            print(f"Loaded UMAP coords  shape={z_umap.shape}", flush=True)
        elif not args.skip_umap:
            print(f"Warning: {umap_path} not found.", file=sys.stderr)
        if tsne_path.is_file() and not args.skip_tsne:
            z_tsne = np.load(tsne_path)
            print(f"Loaded t-SNE coords  shape={z_tsne.shape}", flush=True)
        elif not args.skip_tsne:
            print(f"Warning: {tsne_path} not found.", file=sys.stderr)
    else:
        if not args.skip_umap:
            print("Running UMAP …", flush=True)
            z_umap = run_umap(x_proc, args.random_seed,
                              args.umap_n_neighbors, args.umap_min_dist, args.umap_metric)
            np.save(out_dir / "coords_umap.npy", z_umap)

        if not args.skip_tsne:
            print("Running t-SNE …", flush=True)
            z_tsne = run_tsne(x_proc, args.random_seed, args.tsne_perplexity)
            np.save(out_dir / "coords_tsne.npy", z_tsne)

    # global plots
    for method, z in [("umap", z_umap), ("tsne", z_tsne)]:
        if z is None:
            continue
        plot_categorical(z, ko_df["Level3_clean"],
                         f"{method.upper()} — colored by Level3_clean",
                         out_dir / f"{method}_level3_clean.png")
        plot_continuous(z, ko_df["mean_b_ko_hab"],
                        f"{method.upper()} — mean b_ko_hab across habitats",
                        out_dir / f"{method}_mean_b_ko_hab.png",
                        colorbar_label="mean b_ko_hab")

    # per-habitat plots
    if args.plot_per_category:
        print("Generating per-habitat plots colored by functional group …", flush=True)
        for method, z in [("umap", z_umap), ("tsne", z_tsne)]:
            if z is None:
                continue
            plot_per_habitat(z, ko_df, hab_pivot, method, out_dir)
            plot_per_habitat_grid(z, ko_df, hab_pivot, method, out_dir)

    print(f"\nDone. All outputs saved to: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())