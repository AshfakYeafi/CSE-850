import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import MODEL_DIR, PLOT_DIR, RESULT_DIR


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_directories():
    for path in (MODEL_DIR, PLOT_DIR, RESULT_DIR):
        Path(path).mkdir(parents=True, exist_ok=True)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def save_results_to_csv(dataframe, path):
    dataframe.to_csv(path, index=False)


def plot_results(results_df, plot_path):
    plt.figure(figsize=(9, 5))

    for (model_name, attack_name), subset in results_df.groupby(["model", "attack"]):
        if attack_name == "Clean":
            continue
        subset = subset.sort_values("epsilon")
        label = f"{model_name} - {attack_name}"
        plt.plot(subset["epsilon"], subset["accuracy"], marker="o", label=label)

    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy (%)")
    plt.title("CIFAR-10 Robustness Under FGSM and PGD")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def print_summary_table(results_df):
    summary = results_df.pivot_table(
        index=["attack", "epsilon"],
        columns="model",
        values="accuracy",
    )
    pd.set_option("display.max_rows", None)
    print(summary.round(2))


def plot_per_class_results(per_class_df, plot_path):
    if per_class_df.empty:
        return

    subset = per_class_df[
        (per_class_df["attack"] == "PGD")
        & (per_class_df["epsilon"] == per_class_df["epsilon"].max())
    ]

    pivot = subset.pivot_table(index="class_name", columns="model", values="accuracy")
    pivot.plot(kind="bar", figsize=(11, 5))
    plt.title("Per-Class Robustness at Strongest PGD Attack")
    plt.xlabel("Class")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def plot_resolution_results(resolution_df, plot_path):
    if resolution_df.empty:
        return

    plt.figure(figsize=(9, 5))
    for model_name, subset in resolution_df.groupby("model"):
        subset = subset.sort_values("resolution")
        plt.plot(
            subset["resolution"],
            subset["pgd_accuracy"],
            marker="o",
            label=f"{model_name} - PGD",
        )
        plt.plot(
            subset["resolution"],
            subset["clean_accuracy"],
            marker="x",
            linestyle="--",
            label=f"{model_name} - Clean",
        )

    plt.xlabel("Input resolution (px)")
    plt.ylabel("Accuracy (%)")
    plt.title("Robustness sensitivity to input resolution (PGD)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def plot_trades_training_curves(training_df, plot_path):
    if training_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(training_df["epoch"], training_df["total_loss"], marker="o", label="Total loss")
    axes[0].plot(
        training_df["epoch"],
        training_df["natural_loss"],
        marker="x",
        linestyle="--",
        label="Natural CE loss",
    )
    axes[0].plot(
        training_df["epoch"],
        training_df["robust_loss"],
        marker="s",
        linestyle=":",
        label="Robust KL loss",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("TRADES training losses")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        training_df["epoch"],
        training_df["clean_test_acc"],
        marker="o",
        color="tab:green",
        label="Clean test accuracy",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("TRADES clean accuracy over epochs")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
