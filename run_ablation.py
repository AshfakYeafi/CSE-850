import argparse
from pathlib import Path

import pandas as pd
import torch

from src.attacks import pgd_attack
from src.config import DEVICE, EPOCHS, LEARNING_RATE, RESULT_DIR, WEIGHT_DECAY
from src.data_loader import get_data_loaders
from src.evaluate import evaluate_under_attack
from src.model import build_model
from src.train import evaluate_clean
from src.utils import ensure_directories, set_seed


def train_one_setting(train_loader, test_loader, steps, epsilon=8 / 255, alpha=2 / 255):
    model = build_model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for _ in range(EPOCHS):
        model.train()
        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            adv_images = pgd_attack(model, images, labels, epsilon, alpha, steps, DEVICE)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(adv_images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

    clean_acc = evaluate_clean(model, test_loader)
    pgd_acc = evaluate_under_attack(model, test_loader, "PGD", epsilon)
    return clean_acc, pgd_acc


def parse_args():
    parser = argparse.ArgumentParser(description="Run PGD-step ablation.")
    parser.add_argument("--steps", nargs="+", type=int, default=[3, 7, 10])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_directories()
    train_loader, test_loader = get_data_loaders()

    rows = []
    for steps in args.steps:
        clean_acc, pgd_acc = train_one_setting(train_loader, test_loader, steps=steps)
        rows.append(
            {
                "pgd_train_steps": steps,
                "clean_accuracy": clean_acc,
                "pgd_accuracy_at_8_255": pgd_acc,
            }
        )
        print(f"steps={steps}: clean={clean_acc:.2f}% pgd@8/255={pgd_acc:.2f}%")

    df = pd.DataFrame(rows)
    out_path = Path(RESULT_DIR) / "ablation_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved ablation results to {out_path}")


if __name__ == "__main__":
    main()
