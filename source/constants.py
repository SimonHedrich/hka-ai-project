import os

import torch

if os.environ.get("AM_I_IN_A_DOCKER_CONTAINER", False):
    PROJECT_ROOT = "/app"
    SOURCE_DIR = PROJECT_ROOT
else:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    SOURCE_DIR = os.path.join(PROJECT_ROOT, "source")

EXPERIMENT_NAME = "traffic_detection"
TRAIN_DATASET_NAME = "kitti"

DATASET_DIR = os.path.join(SOURCE_DIR, "datasets", TRAIN_DATASET_NAME)
TRAIN_IMAGE_DIR_PATH = os.path.join(DATASET_DIR, "images")
TRAIN_COCO_PATH = os.path.join(DATASET_DIR, "coco_train.json")
VALIDATION_IMAGE_DIR_PATH = os.path.join(DATASET_DIR, "images")
VALIDATION_COCO_PATH = os.path.join(DATASET_DIR, "coco_val.json")
EVALUATION_IMAGE_DIR_PATH = os.path.join(DATASET_DIR, "images")
EVALUATION_COCO_PATH = os.path.join(DATASET_DIR, "coco_test.json")

OUTPUT_DIR = os.path.join(SOURCE_DIR, "model_exports")

# Hyperparameters
EPOCH_COUNT = 4  # Number of epochs (>3, because of scheduler)
BATCH_SIZE = 64
TRAIN_BATCH_SIZE = BATCH_SIZE
VALIDATION_BATCH_SIZE = BATCH_SIZE
EVALUATION_BATCH_SIZE = BATCH_SIZE
CLASS_COUNT = 2
LEARNING_RATE = 0.0001
# From https://github.com/qfgaohao/pytorch-ssd/blob/master/train_ssd.py
MOMENTUM = 0.9
WEIGHT_DECAY = 0.5e-4
CONFIDENCE_THRESHOLD = 0.3  # Confidence threshold for predictions
SCHEDULER_GAMMER = 0.1
OPTIMIZER_MILESTONES = "80,100"
AUTO_STOP_THRESHOLD = 0.0001
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
