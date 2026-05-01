import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from tqdm import tqdm

from src.config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    TRADES_ALPHA,
    TRADES_BETA,
    TRADES_EPSILON,
    TRADES_MODEL_PATH,
    TRADES_TRAINING_PLOT_PATH,
    TRADES_TRAINING_RESULT_CSV_PATH,
    TRADES_STEPS,
    WEIGHT_DECAY,
)
from src.train import evaluate_clean
from src.utils import plot_trades_training_curves, save_model, save_results_to_csv


def _generate_trades_adversarial_examples(model, images):
    x_adv = images.detach() + 0.001 * torch.randn_like(images)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    with torch.no_grad():
        clean_logits = model(images)
        clean_probs = F.softmax(clean_logits, dim=1)

    for _ in range(TRADES_STEPS):
        x_adv.requires_grad_()
        logits_adv = model(x_adv)
        loss_kl = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            clean_probs,
            reduction="batchmean",
        )
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]

        x_adv = x_adv.detach() + TRADES_ALPHA * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, images - TRADES_EPSILON), images + TRADES_EPSILON)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach()


def train_trades_model(model, train_loader, test_loader):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    training_rows = []
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_natural = 0.0
        running_robust = 0.0

        progress = tqdm(
            train_loader,
            desc=f"TRADES Train Epoch {epoch + 1}/{EPOCHS}",
            leave=False,
            dynamic_ncols=True,
        )
        for batch_idx, (images, labels) in enumerate(progress, start=1):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            adv_images = _generate_trades_adversarial_examples(model, images)

            optimizer.zero_grad()
            logits_clean = model(images)
            logits_adv = model(adv_images)

            natural_loss = F.cross_entropy(logits_clean, labels)
            robust_loss = F.kl_div(
                F.log_softmax(logits_adv, dim=1),
                F.softmax(logits_clean.detach(), dim=1),
                reduction="batchmean",
            )
            loss = natural_loss + TRADES_BETA * robust_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_natural += natural_loss.item()
            running_robust += robust_loss.item()
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                nat=f"{running_natural / batch_idx:.4f}",
                rob=f"{running_robust / batch_idx:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        scheduler.step()
        clean_acc = evaluate_clean(model, test_loader)
        avg_loss = running_loss / len(train_loader)
        avg_natural = running_natural / len(train_loader)
        avg_robust = running_robust / len(train_loader)
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"loss={avg_loss:.4f} natural={avg_natural:.4f} robust={avg_robust:.4f} "
            f"clean_test_acc={clean_acc:.2f}%"
        )
        training_rows.append(
            {
                "epoch": epoch + 1,
                "total_loss": avg_loss,
                "natural_loss": avg_natural,
                "robust_loss": avg_robust,
                "clean_test_acc": clean_acc,
                "beta": TRADES_BETA,
                "epsilon": TRADES_EPSILON,
                "alpha": TRADES_ALPHA,
                "steps": TRADES_STEPS,
            }
        )

    save_model(model, TRADES_MODEL_PATH)
    training_df = pd.DataFrame(training_rows)
    save_results_to_csv(training_df, TRADES_TRAINING_RESULT_CSV_PATH)
    plot_trades_training_curves(training_df, TRADES_TRAINING_PLOT_PATH)
    print(f"Saved TRADES model to {TRADES_MODEL_PATH}")
    print(f"Saved TRADES training log to {TRADES_TRAINING_RESULT_CSV_PATH}")
    print(f"Saved TRADES training curves to {TRADES_TRAINING_PLOT_PATH}")
