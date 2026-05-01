import pandas as pd
import torch
from tqdm import tqdm

from src.attacks import fgsm_attack, pgd_attack
from src.config import CLASS_NAMES, DEVICE, FGSM_EPSILONS, NUM_CLASSES, PGD_ALPHA, PGD_EPSILONS, PGD_STEPS


def evaluate_on_clean(model, data_loader, desc="Clean Eval"):
    model.eval()
    total = 0
    correct = 0

    progress = tqdm(data_loader, desc=desc, leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for images, labels in progress:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            progress.set_postfix(acc=f"{100.0 * correct / max(total, 1):.2f}%")

    return 100.0 * correct / total


def evaluate_under_attack(model, data_loader, attack_name, epsilon, desc=None):
    model.eval()
    total = 0
    correct = 0

    if desc is None:
        desc = f"{attack_name} eps={epsilon:.6f}"
    progress = tqdm(data_loader, desc=desc, leave=False, dynamic_ncols=True)

    for images, labels in progress:
        if attack_name == "FGSM":
            adv_images = fgsm_attack(model, images, labels, epsilon, DEVICE)
        elif attack_name == "PGD":
            adv_images = pgd_attack(
                model,
                images,
                labels,
                epsilon=epsilon,
                alpha=PGD_ALPHA,
                steps=PGD_STEPS,
                device=DEVICE,
            )
        else:
            raise ValueError(f"Unsupported attack: {attack_name}")

        labels = labels.to(DEVICE, non_blocking=True)
        with torch.no_grad():
            outputs = model(adv_images)
            predictions = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()
        progress.set_postfix(acc=f"{100.0 * correct / max(total, 1):.2f}%")

    return 100.0 * correct / total


def evaluate_model(model, data_loader, model_name):
    rows = [
        {
            "model": model_name,
            "attack": "Clean",
            "epsilon": 0.0,
            "accuracy": evaluate_on_clean(model, data_loader, desc=f"{model_name} clean"),
        }
    ]

    for epsilon in FGSM_EPSILONS:
        if epsilon == 0.0:
            continue
        rows.append(
            {
                "model": model_name,
                "attack": "FGSM",
                "epsilon": epsilon,
                "accuracy": evaluate_under_attack(
                    model,
                    data_loader,
                    "FGSM",
                    epsilon,
                    desc=f"{model_name} FGSM eps={epsilon:.6f}",
                ),
            }
        )

    for epsilon in PGD_EPSILONS:
        rows.append(
            {
                "model": model_name,
                "attack": "PGD",
                "epsilon": epsilon,
                "accuracy": evaluate_under_attack(
                    model,
                    data_loader,
                    "PGD",
                    epsilon,
                    desc=f"{model_name} PGD eps={epsilon:.6f}",
                ),
            }
        )

    return pd.DataFrame(rows)


def evaluate_transfer_attack(
    source_model,
    target_model,
    data_loader,
    attack_name,
    epsilon,
    desc=None,
):
    source_model.eval()
    target_model.eval()
    total = 0
    correct = 0

    if desc is None:
        desc = f"Transfer {attack_name} eps={epsilon:.6f}"
    progress = tqdm(data_loader, desc=desc, leave=False, dynamic_ncols=True)

    for images, labels in progress:
        if attack_name == "FGSM":
            adv_images = fgsm_attack(source_model, images, labels, epsilon, DEVICE)
        elif attack_name == "PGD":
            adv_images = pgd_attack(
                source_model,
                images,
                labels,
                epsilon=epsilon,
                alpha=PGD_ALPHA,
                steps=PGD_STEPS,
                device=DEVICE,
            )
        else:
            raise ValueError(f"Unsupported attack: {attack_name}")

        labels = labels.to(DEVICE, non_blocking=True)
        with torch.no_grad():
            outputs = target_model(adv_images)
            predictions = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()
        progress.set_postfix(acc=f"{100.0 * correct / max(total, 1):.2f}%")

    return 100.0 * correct / total


def evaluate_transfer_matrix(models, data_loader):
    rows = []
    for source_name, source_model in models.items():
        for target_name, target_model in models.items():
            for epsilon in [e for e in FGSM_EPSILONS if e > 0]:
                rows.append(
                    {
                        "source_model": source_name,
                        "target_model": target_name,
                        "attack": "FGSM",
                        "epsilon": epsilon,
                        "accuracy": evaluate_transfer_attack(
                            source_model,
                            target_model,
                            data_loader,
                            "FGSM",
                            epsilon,
                            desc=f"Transfer {source_name}->{target_name} FGSM eps={epsilon:.6f}",
                        ),
                    }
                )
            for epsilon in PGD_EPSILONS:
                rows.append(
                    {
                        "source_model": source_name,
                        "target_model": target_name,
                        "attack": "PGD",
                        "epsilon": epsilon,
                        "accuracy": evaluate_transfer_attack(
                            source_model,
                            target_model,
                            data_loader,
                            "PGD",
                            epsilon,
                            desc=f"Transfer {source_name}->{target_name} PGD eps={epsilon:.6f}",
                        ),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_per_class_under_attack(model, data_loader, attack_name, epsilon, desc=None):
    model.eval()
    class_correct = [0 for _ in range(NUM_CLASSES)]
    class_total = [0 for _ in range(NUM_CLASSES)]

    if desc is None:
        desc = f"Per-class {attack_name} eps={epsilon:.6f}"
    progress = tqdm(data_loader, desc=desc, leave=False, dynamic_ncols=True)

    for images, labels in progress:
        if attack_name == "FGSM":
            adv_images = fgsm_attack(model, images, labels, epsilon, DEVICE)
        elif attack_name == "PGD":
            adv_images = pgd_attack(
                model,
                images,
                labels,
                epsilon=epsilon,
                alpha=PGD_ALPHA,
                steps=PGD_STEPS,
                device=DEVICE,
            )
        else:
            raise ValueError(f"Unsupported attack: {attack_name}")

        labels = labels.to(DEVICE, non_blocking=True)
        with torch.no_grad():
            outputs = model(adv_images)
            predictions = outputs.argmax(dim=1)

        for idx in range(labels.size(0)):
            label = labels[idx].item()
            class_total[label] += 1
            class_correct[label] += int(predictions[idx].item() == label)
        total_seen = sum(class_total)
        total_correct = sum(class_correct)
        progress.set_postfix(acc=f"{100.0 * total_correct / max(total_seen, 1):.2f}%")

    rows = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        acc = 100.0 * class_correct[class_idx] / max(class_total[class_idx], 1)
        rows.append(
            {
                "class_idx": class_idx,
                "class_name": class_name,
                "attack": attack_name,
                "epsilon": epsilon,
                "accuracy": acc,
            }
        )
    return rows


def evaluate_per_class(model, data_loader, model_name):
    rows = []
    for epsilon in [e for e in FGSM_EPSILONS if e > 0]:
        for row in evaluate_per_class_under_attack(
            model,
            data_loader,
            "FGSM",
            epsilon,
            desc=f"{model_name} per-class FGSM eps={epsilon:.6f}",
        ):
            row["model"] = model_name
            rows.append(row)
    for epsilon in PGD_EPSILONS:
        for row in evaluate_per_class_under_attack(
            model,
            data_loader,
            "PGD",
            epsilon,
            desc=f"{model_name} per-class PGD eps={epsilon:.6f}",
        ):
            row["model"] = model_name
            rows.append(row)
    return pd.DataFrame(rows)
