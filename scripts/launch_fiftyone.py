import os

import fiftyone


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SOURCE_DIR = os.path.join(PROJECT_ROOT, "source")

EXPERIMENT_NAME = "traffic_detection"
TRAIN_DATASET_NAME = "kitti"

DATASET_DIR = os.path.join(SOURCE_DIR, "datasets", TRAIN_DATASET_NAME)

DATASET_TYPE = "all"
# DATASET_TYPE = "train"
# DATASET_TYPE = "val"
# DATASET_TYPE = "test"

IMAGE_DIR_PATH = os.path.join(DATASET_DIR, "images")
print(f"{IMAGE_DIR_PATH=}")
COCO_FILE_PATH = os.path.join(DATASET_DIR, f"coco_{DATASET_TYPE}.json")
print(f"{COCO_FILE_PATH=}")

dataset = fiftyone.Dataset.from_dir(
    dataset_type=fiftyone.types.COCODetectionDataset,
    data_path=IMAGE_DIR_PATH,
    labels_path=COCO_FILE_PATH,
    include_id=True,
)
session = fiftyone.launch_app(dataset, port=8886)
session.wait()
