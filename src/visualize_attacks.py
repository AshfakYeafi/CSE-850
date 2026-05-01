from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.attacks import fgsm_attack, pgd_attack
from src.config import CLASS_NAMES, DEVICE, PGD_ALPHA, PGD_STEPS


def _generate_adv_images(model, images, labels, attack_name, epsilon):
    if attack_name == "FGSM":
        return fgsm_attack(model, images, labels, epsilon, DEVICE)
    if attack_name == "PGD":
        return pgd_attack(
            model,
            images,
            labels,
            epsilon=epsilon,
            alpha=PGD_ALPHA,
            steps=PGD_STEPS,
            device=DEVICE,
        )
    raise ValueError(f"Unsupported attack: {attack_name}")


def save_attack_visualization(
    model,
    data_loader,
    attack_name,
    epsilon,
    model_name,
    output_path,
    num_examples=8,
):
    model.eval()
    collected = []

    for images, labels in data_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        adv_images = _generate_adv_images(model, images, labels, attack_name, epsilon)

        with torch.no_grad():
            clean_preds = model(images).argmax(dim=1)
            adv_preds = model(adv_images).argmax(dim=1)

        changed_mask = clean_preds != adv_preds
        indices = torch.where(changed_mask)[0]

        for idx in indices.tolist():
            collected.append(
                {
                    "original": images[idx].detach().cpu(),
                    "adversarial": adv_images[idx].detach().cpu(),
                    "label": labels[idx].item(),
                    "clean_pred": clean_preds[idx].item(),
                    "adv_pred": adv_preds[idx].item(),
                }
            )
            if len(collected) >= num_examples:
                break
        if len(collected) >= num_examples:
            break

    if not collected:
        print("No prediction-changing adversarial examples found for current settings.")
        return None

    rows = len(collected)
    fig, axes = plt.subplots(rows, 3, figsize=(12, 3 * rows))
    if rows == 1:
        axes = [axes]

    for row, sample in enumerate(collected):
        original = sample["original"].permute(1, 2, 0).numpy()
        adversarial = sample["adversarial"].permute(1, 2, 0).numpy()
        delta = (adversarial - original) * 8.0 + 0.5
        delta = delta.clip(0.0, 1.0)

        axes[row][0].imshow(original)
        axes[row][0].set_title(
            f"Original\ntrue={CLASS_NAMES[sample['label']]}\npred={CLASS_NAMES[sample['clean_pred']]}"
        )
        axes[row][1].imshow(adversarial)
        axes[row][1].set_title(f"Adversarial\npred={CLASS_NAMES[sample['adv_pred']]}")
        axes[row][2].imshow(delta)
        axes[row][2].set_title("Perturbation (scaled)")

        for col in range(3):
            axes[row][col].axis("off")

    fig.suptitle(
        f"{model_name} under {attack_name} attack (epsilon={epsilon:.6f})",
        fontsize=12,
    )
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path
