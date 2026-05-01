import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from src.attacks import pgd_attack
from src.config import (
    ADV_MODEL_PATH,
    ADV_TRAIN_ALPHA,
    ADV_TRAIN_EPSILON,
    ADV_TRAIN_STEPS,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
)
from src.train import evaluate_clean
from src.utils import save_model


def train_adversarial_model(model, train_loader, test_loader):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"PGD Adv Train Epoch {epoch + 1}/{EPOCHS}",
            leave=False,
            dynamic_ncols=True,
        )
        for batch_idx, (images, labels) in enumerate(progress, start=1):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            adv_images = pgd_attack(
                model=model,
                images=images,
                labels=labels,
                epsilon=ADV_TRAIN_EPSILON,
                alpha=ADV_TRAIN_ALPHA,
                steps=ADV_TRAIN_STEPS,
                device=DEVICE,
            )

            optimizer.zero_grad()
            outputs = model(adv_images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(
                batch_loss=f"{loss.item():.4f}",
                avg_loss=f"{running_loss / batch_idx:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        scheduler.step()
        clean_acc = evaluate_clean(model, test_loader)
        avg_loss = running_loss / len(train_loader)
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"adv_loss={avg_loss:.4f} clean_test_acc={clean_acc:.2f}%"
        )

    save_model(model, ADV_MODEL_PATH)
    print(f"Saved adversarial model to {ADV_MODEL_PATH}")
