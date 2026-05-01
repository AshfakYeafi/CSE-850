from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import BATCH_SIZE, DATA_DIR, DATASET_NAME, TEST_BATCH_SIZE, TRAIN_RESOLUTION


def _get_transforms_for_dataset(resolution: int, train: bool):
    if DATASET_NAME == "cifar10":
        # CIFAR-10 is 32x32. We still allow resizing for controlled resolution experiments.
        if train:
            return transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize(resolution),
                    transforms.ToTensor(),
                ]
            )
        return transforms.Compose([transforms.Resize(resolution), transforms.ToTensor()])

    if DATASET_NAME == "stl10":
        # STL-10 is 96x96 (but we still control resolution explicitly).
        if train:
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(resolution, scale=(0.85, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        return transforms.Compose([transforms.Resize(resolution), transforms.ToTensor()])

    raise ValueError(f"Unsupported DATASET_NAME: {DATASET_NAME}")


def get_data_loaders(train_resolution: int = None, eval_resolution: int = None):
    train_resolution = TRAIN_RESOLUTION if train_resolution is None else train_resolution
    eval_resolution = train_resolution if eval_resolution is None else eval_resolution

    train_transform = _get_transforms_for_dataset(train_resolution, train=True)
    test_transform = _get_transforms_for_dataset(eval_resolution, train=False)

    if DATASET_NAME == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=DATA_DIR,
            train=True,
            download=True,
            transform=train_transform,
        )
        test_dataset = datasets.CIFAR10(
            root=DATA_DIR,
            train=False,
            download=True,
            transform=test_transform,
        )
    elif DATASET_NAME == "stl10":
        # Use STL-10 train split for supervised training.
        train_dataset = datasets.STL10(
            root=DATA_DIR,
            split="train",
            download=True,
            transform=train_transform,
        )
        test_dataset = datasets.STL10(
            root=DATA_DIR,
            split="test",
            download=True,
            transform=test_transform,
        )
    else:
        raise ValueError(f"Unsupported DATASET_NAME: {DATASET_NAME}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, test_loader
