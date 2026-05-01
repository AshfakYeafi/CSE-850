import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from src.config import CLEAN_MODEL_PATH, DEVICE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY
from src.utils import save_model


def evaluate_clean(model, data_loader):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    return 100.0 * correct / total


def train_clean_model(model, train_loader, test_loader):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Train Epoch {epoch + 1}/{EPOCHS}",
            leave=False,
            dynamic_ncols=True,
        )
        for batch_idx, (images, labels) in enumerate(progress, start=1):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
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
            f"loss={avg_loss:.4f} clean_test_acc={clean_acc:.2f}%"
        )

    save_model(model, CLEAN_MODEL_PATH)
    print(f"Saved clean model to {CLEAN_MODEL_PATH}")
