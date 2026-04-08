#!/usr/bin/env python3
"""
Project KO embeddings with UMAP and t-SNE, with plots colored by habitat,
Level3_clean, and b_ko_hab.

Expects one PyTorch tensor per KO in the embeddings directory (e.g. K00001.pt).
Metadata is joined on column KO from the refined Taylor summary CSV.
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
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

try:
    import umap
except ImportError as e:
    raise SystemExit(
        "umap-learn is required. Install with: pip install umap-learn"
    ) from e


COLOR_SPECS = (
    ("habitat", "habitat", "categorical"),
    ("level3_clean", "Level3_clean", "categorical"),
    ("b_ko_hab", "b_ko_hab", "continuous"),
)


def resolve_embeddings_dir(repo_root: Path) -> Path:
    for name in ("ko_embeddings", "ko_embeddings_150m"):
        p = repo_root / name
        if p.is_dir() and any(p.glob("*.pt")):
            return p
    raise FileNotFoundError(
        f"No directory with *.pt files found under {repo_root} "
        "(tried ko_embeddings, ko_embeddings_150m)."
    )


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
        vec = t.detach().float().cpu().numpy().ravel()
        out[ko] = vec.astype(np.float32, copy=False)
    return out


def build_matrix(
    df: pd.DataFrame,
    emb: dict[str, np.ndarray],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Stack embedding rows aligned with df (one row per metadata row)."""
    kos = df["KO"].astype(str)
    if not kos.isin(emb.keys()).all():
        n_bad = int((~kos.isin(emb.keys())).sum())
        raise ValueError(f"{n_bad} rows reference KOs without embedding files.")
    x = np.vstack(kos.map(emb).to_numpy())
    return x.astype(np.float32, copy=False), df.reset_index(drop=True)


def maybe_subsample(
    x: np.ndarray,
    df: pd.DataFrame,
    max_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    if max_samples is None or len(df) <= max_samples:
        return x, df
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=max_samples, replace=False)
    return x[idx], df.iloc[idx].reset_index(drop=True)


def preprocess(
    x: np.ndarray,
    l2_normalize: bool,
    pca_dim: int | None,
    seed: int,
) -> np.ndarray:
    if l2_normalize:
        x = normalize(x, norm="l2", axis=1)
    if pca_dim is not None and pca_dim > 0:
        if pca_dim > x.shape[1]:
            raise ValueError("pca_dim cannot exceed embedding dimension.")
        x = PCA(n_components=pca_dim, random_state=seed).fit_transform(x)
    return x.astype(np.float32, copy=False)


def run_umap(x: np.ndarray, seed: int) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=50, # can tune
        min_dist=0.1,
        metric="euclidean",
        random_state=seed,
        verbose=False,
    )
    return reducer.fit_transform(x)


def run_tsne(x: np.ndarray, seed: int, perplexity: float) -> np.ndarray:
    n = x.shape[0]
    perp = min(perplexity, max(5, (n - 1) // 3))
    return TSNE(
        n_components=2,
        perplexity=perp,
        learning_rate="auto",
        init="pca",
        random_state=seed,
        max_iter=1000,
    ).fit_transform(x)


def plot_categorical(
    z: np.ndarray,
    labels: pd.Series,
    title: str,
    out_path: Path,
) -> None:
    uniq = pd.Index(labels.astype(str).unique()).sort_values()
    n = len(uniq)
    palette = sns.color_palette("husl", n_colors=max(n, 1))
    lut = {lab: palette[i % len(palette)] for i, lab in enumerate(uniq)}
    colors = labels.astype(str).map(lut)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        z[:, 0],
        z[:, 1],
        c=colors.tolist(),
        s=4,
        alpha=0.75,
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=str(lab),
                   markerfacecolor=lut[lab], markersize=8)
        for lab in uniq
    ]
    ax.legend(
        handles=handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_continuous(
    z: np.ndarray,
    values: pd.Series,
    title: str,
    out_path: Path,
    colorbar_label: str,
) -> None:
    v = values.astype(np.float64).values
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(z[:, 0], z[:, 1], c=v, s=4, alpha=0.75, linewidths=0, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (parent of embeddings dir). Default: parent of scripts/.",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=None,
        help="Directory of K*.pt tensors. Default: ko_embeddings or ko_embeddings_150m.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Path to refined_taylor_KO_Habitat_Category_Summary_with_a_b_bko.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write PNGs. Default: <repo>/figures/ko_embedding_projections",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on rows (random subsample after filtering to KOs with embeddings).",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=50,
        help="PCA components before UMAP/t-SNE (0 to disable). Default: 50.",
    )
    parser.add_argument(
        "--no-l2-normalize",
        action="store_true",
        help="Do not L2-normalize embedding rows before PCA (default: normalize).",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0, #can tune
        help="t-SNE perplexity (capped automatically vs sample size).",
    )
    parser.add_argument(
        "--skip-umap",
        action="store_true",
        help="Only run t-SNE (skip UMAP).",
    )
    parser.add_argument(
        "--skip-tsne",
        action="store_true",
        help="Only run UMAP (skip t-SNE; useful when full-data t-SNE is slow).",
    )
    args = parser.parse_args()

    if args.skip_umap and args.skip_tsne:
        print("Specify at most one of --skip-umap / --skip-tsne.", file=sys.stderr)
        return 1

    repo = args.repo_root
    if repo is None:
        repo = Path(__file__).resolve().parents[1]

    emb_dir = args.embeddings_dir or resolve_embeddings_dir(repo)
    meta = args.metadata_csv or (
        repo / "refined_taylor_KO_Habitat_Category_Summary_with_a_b_bko.csv"
    )
    out_dir = args.output_dir or (repo / "figures" / "ko_embedding_projections")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not meta.is_file():
        print(f"Metadata CSV not found: {meta}", file=sys.stderr)
        return 1

    print(f"Loading embeddings from {emb_dir} …", flush=True)
    emb = load_embeddings(emb_dir)
    print(f"  {len(emb)} KO tensors", flush=True)

    df = pd.read_csv(meta)
    df = df[df["KO"].astype(str).isin(emb.keys())].copy()
    print(f"Metadata rows with embeddings: {len(df)}", flush=True)

    x, df = build_matrix(df, emb)
    x, df = maybe_subsample(x, df, args.max_samples, args.random_seed)
    print(f"Samples used: {len(df)}", flush=True)

    pca_dim = args.pca_dim if args.pca_dim and args.pca_dim > 0 else None
    x_proc = preprocess(
        x,
        l2_normalize=not args.no_l2_normalize,
        pca_dim=pca_dim,
        seed=args.random_seed,
    )
    if pca_dim:
        print(f"PCA -> {pca_dim} dimensions", flush=True)
    else:
        print("No PCA (using full dimension)", flush=True)

    z_umap: np.ndarray | None = None
    z_tsne: np.ndarray | None = None

    if not args.skip_umap:
        print("Running UMAP …", flush=True)
        z_umap = run_umap(x_proc, args.random_seed)
        np.save(out_dir / "coords_umap.npy", z_umap)

    if not args.skip_tsne:
        print("Running t-SNE …", flush=True)
        z_tsne = run_tsne(x_proc, args.random_seed, args.tsne_perplexity)
        np.save(out_dir / "coords_tsne.npy", z_tsne)

    for stem, col, kind in COLOR_SPECS:
        if col not in df.columns:
            print(f"Missing column {col}, skipping.", file=sys.stderr)
            continue
        if kind == "categorical":
            if z_umap is not None:
                plot_categorical(
                    z_umap,
                    df[col],
                    f"UMAP — colored by {stem}",
                    out_dir / f"umap_{stem}.png",
                )
            if z_tsne is not None:
                plot_categorical(
                    z_tsne,
                    df[col],
                    f"t-SNE — colored by {stem}",
                    out_dir / f"tsne_{stem}.png",
                )
        else:
            if z_umap is not None:
                plot_continuous(
                    z_umap,
                    df[col],
                    f"UMAP — colored by {stem}",
                    out_dir / f"umap_{stem}.png",
                    colorbar_label=stem,
                )
            if z_tsne is not None:
                plot_continuous(
                    z_tsne,
                    df[col],
                    f"t-SNE — colored by {stem}",
                    out_dir / f"tsne_{stem}.png",
                    colorbar_label=stem,
                )

    print(f"Wrote figures and coords to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
