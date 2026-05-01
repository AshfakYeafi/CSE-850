from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"
RESULT_DIR = OUTPUT_DIR / "results"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
BATCH_SIZE = 64
TEST_BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
EPOCHS = 50

# Dataset choice (switch to CIFAR-10 if needed)
DATASET_NAME = "stl10"  # "cifar10" or "stl10"

# Resizing for training/evaluation. STL10 images are 96x96 by default,
# but we explicitly resize so experiments are controlled/reproducible.
TRAIN_RESOLUTION = 96
# For the "novel finding" resolution-robustness evaluation.
EVAL_RESOLUTIONS = [96, 48]

FGSM_EPSILONS = [0.0, 2 / 255, 4 / 255, 8 / 255]
PGD_EPSILONS = [2 / 255, 4 / 255, 8 / 255]
PGD_ALPHA = 2 / 255
PGD_STEPS = 7
ADV_TRAIN_EPSILON = 8 / 255
ADV_TRAIN_ALPHA = 2 / 255
ADV_TRAIN_STEPS = 7
TRADES_EPSILON = 8 / 255
TRADES_ALPHA = 2 / 255
TRADES_STEPS = 10
TRADES_BETA = 6.0

NUM_CLASSES = 10

CLASS_NAMES_CIFAR10 = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

CLASS_NAMES_STL10 = (
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
)

CLASS_NAMES = CLASS_NAMES_STL10 if DATASET_NAME == "stl10" else CLASS_NAMES_CIFAR10

DATASET_TAG = DATASET_NAME

CLEAN_MODEL_PATH = MODEL_DIR / f"clean_resnet18_{DATASET_TAG}.pth"
ADV_MODEL_PATH = MODEL_DIR / f"adv_resnet18_{DATASET_TAG}.pth"
FGSM_MODEL_PATH = MODEL_DIR / f"fgsm_resnet18_{DATASET_TAG}.pth"
TRADES_MODEL_PATH = MODEL_DIR / f"trades_resnet18_{DATASET_TAG}.pth"
TRADES_TRAINING_RESULT_CSV_PATH = RESULT_DIR / f"trades_training_log_{DATASET_TAG}.csv"
TRADES_TRAINING_PLOT_PATH = PLOT_DIR / f"trades_training_curves_{DATASET_TAG}.png"

RESULT_CSV_PATH = RESULT_DIR / f"evaluation_results_{DATASET_TAG}.csv"
TRANSFER_RESULT_CSV_PATH = RESULT_DIR / "transfer_attack_results.csv"
TRANSFER_RESULT_CSV_PATH = RESULT_DIR / f"transfer_attack_results_{DATASET_TAG}.csv"
PER_CLASS_RESULT_CSV_PATH = RESULT_DIR / f"per_class_results_{DATASET_TAG}.csv"

PLOT_PATH = PLOT_DIR / f"robustness_plot_{DATASET_TAG}.png"
PER_CLASS_PLOT_PATH = PLOT_DIR / "per_class_robustness.png"
ATTACK_VIZ_DIR = PLOT_DIR / "attack_examples"

RESOLUTION_RESULT_CSV_PATH = RESULT_DIR / f"resolution_robustness_{DATASET_TAG}.csv"
RESOLUTION_PLOT_PATH = PLOT_DIR / f"resolution_robustness_{DATASET_TAG}.png"
