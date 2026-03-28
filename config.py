"""Training and data defaults — edit values here (no YAML/Hydra in v1)."""

DATASET_ROOT = "examples/sample_dataset"
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 1e-3
EPOCHS = 5
TRAIN_SPLIT = 0.8
SEED = 42
MODEL_NAME = "resnet18"
DEVICE = "auto"
PIN_MEMORY = True
SAVE_BEST_MODEL = True
CHECKPOINT_PATH = "best_model.pth"