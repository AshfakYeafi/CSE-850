from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    DATASET_TAG,
    PLOT_DIR,
    RESULT_DIR,
)


def _read_csv(path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def _save_clean_vs_robust_scatter(eval_df, out_dir):
    if eval_df is None or eval_df.empty:
        return

    clean_df = eval_df[(eval_df["attack"] == "Clean") & (eval_df["epsilon"] == 0.0)]
    robust_df = eval_df[(eval_df["attack"] == "PGD") & (eval_df["epsilon"] == eval_df["epsilon"].max())]
    merged = clean_df[["model", "accuracy"]].rename(columns={"accuracy": "clean_acc"}).merge(
        robust_df[["model", "accuracy"]].rename(columns={"accuracy": "pgd_acc"}),
        on="model",
        how="inner",
    )
    if merged.empty:
        return

    plt.figure(figsize=(6, 5))
    plt.scatter(merged["clean_acc"], merged["pgd_acc"], s=90)
    for _, row in merged.iterrows():
        plt.annotate(row["model"], (row["clean_acc"], row["pgd_acc"]), xytext=(5, 5), textcoords="offset points")
    plt.xlabel("Clean Accuracy (%)")
    plt.ylabel("PGD Accuracy at strongest epsilon (%)")
    plt.title("Robustness-Generalization Tradeoff")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"clean_vs_robust_tradeoff_{DATASET_TAG}.png", dpi=200)
    plt.close()


def _save_transfer_heatmap(transfer_df, out_dir, attack_name):
    if transfer_df is None or transfer_df.empty:
        return

    subset = transfer_df[transfer_df["attack"] == attack_name].copy()
    if subset.empty:
        return
    eps = subset["epsilon"].max()
    subset = subset[subset["epsilon"] == eps]
    pivot = subset.pivot(index="source_model", columns="target_model", values="accuracy")
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"Transfer {attack_name} Accuracy Heatmap (eps={eps:.6f})")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.1f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, label="Accuracy (%)")
    plt.tight_layout()
    plt.savefig(out_dir / f"transfer_heatmap_{attack_name.lower()}_{DATASET_TAG}.png", dpi=200)
    plt.close()


def _save_per_class_grouped(per_class_df, out_dir):
    if per_class_df is None or per_class_df.empty:
        return

    subset = per_class_df[(per_class_df["attack"] == "PGD") & (per_class_df["epsilon"] == per_class_df["epsilon"].max())]
    if subset.empty:
        return
    pivot = subset.pivot(index="class_name", columns="model", values="accuracy")
    if pivot.empty:
        return

    pivot.plot(kind="bar", figsize=(12, 5))
    plt.title("Per-Class PGD Robustness at Strongest Epsilon")
    plt.xlabel("Class")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"per_class_grouped_pgd_{DATASET_TAG}.png", dpi=200)
    plt.close()


def _save_resolution_dual(resolution_df, out_dir):
    if resolution_df is None or resolution_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for model_name, subset in resolution_df.groupby("model"):
        subset = subset.sort_values("resolution")
        axes[0].plot(subset["resolution"], subset["clean_accuracy"], marker="o", label=model_name)
        axes[1].plot(subset["resolution"], subset["pgd_accuracy"], marker="o", label=model_name)

    axes[0].set_title("Clean Accuracy vs Input Resolution")
    axes[0].set_xlabel("Resolution (px)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("PGD Accuracy vs Input Resolution")
    axes[1].set_xlabel("Resolution (px)")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / f"resolution_dual_plot_{DATASET_TAG}.png", dpi=200)
    plt.close()


def generate_report_visualizations():
    out_dir = PLOT_DIR / "report"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    eval_df = _read_csv(RESULT_DIR / f"evaluation_results_{DATASET_TAG}.csv")
    transfer_df = _read_csv(RESULT_DIR / f"transfer_attack_results_{DATASET_TAG}.csv")
    per_class_df = _read_csv(RESULT_DIR / f"per_class_results_{DATASET_TAG}.csv")
    resolution_df = _read_csv(RESULT_DIR / f"resolution_robustness_{DATASET_TAG}.csv")

    _save_clean_vs_robust_scatter(eval_df, out_dir)
    _save_transfer_heatmap(transfer_df, out_dir, attack_name="FGSM")
    _save_transfer_heatmap(transfer_df, out_dir, attack_name="PGD")
    _save_per_class_grouped(per_class_df, out_dir)
    _save_resolution_dual(resolution_df, out_dir)

    print(f"Saved report visualizations to {out_dir}")
