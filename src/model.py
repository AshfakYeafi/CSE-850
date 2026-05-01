from torchvision.models import resnet18

from src.config import NUM_CLASSES


def build_model():
    model = resnet18(weights=None, num_classes=NUM_CLASSES)
    return model
