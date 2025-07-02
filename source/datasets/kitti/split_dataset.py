#!/usr/bin/env python3
import json
import os
import random

from traffic_detection_project.source.constants import DATASET_DIR

# —— Configuration constants ——
INPUT_JSON = os.path.join(DATASET_DIR, "kitti_original.json")
COCO_LABELS_PATH = os.path.join(DATASET_DIR, "..", "coco_labels_with_ids.json")
TRAIN_SIZE = 5000
VALIDATION_SIZE = 1200
TEST_SIZE = 1200
MAPPING_FROM_KITTI_TO_COCO = {
    "Car": "car",
    "Cyclist": "person",
    "Pedestrian": "person",
    "Person_sitting": "person",
    "Tram": "train",
    "Truck": "truck",
    "Van": "car",
}
RANDOM_SEED = 42  # Used for shuffling the images
# ————————————————————————

random.seed(RANDOM_SEED)


def load_json(path: str) -> dict:
    with open(path, "r") as file_handler:
        return json.load(file_handler)


def save_json(data: dict, path: str) -> None:
    with open(path, "w") as file_handler:
        json.dump(data, file_handler, indent=2)
        file_handler.write("\n")


def replace_categories_ids_with_coco_ids_and_filter(
    annotations: list[dict], kitti_categories_as_dict: dict[int, str], coco_categories: dict[int, str]
) -> list[dict]:
    filtered_annotations = []
    for annotation in annotations:
        category_name_of_annotation = kitti_categories_as_dict[annotation["category_id"]]
        if category_name_of_annotation in MAPPING_FROM_KITTI_TO_COCO:
            new_category_name = MAPPING_FROM_KITTI_TO_COCO[category_name_of_annotation]
            new_category_id = coco_categories[new_category_name]
            annotation["category_id"] = new_category_id
            filtered_annotations.append(annotation)
    return filtered_annotations


def split_coco() -> tuple[dict[str, list[dict]], dict[str, list[dict]], dict[str, list[dict]], dict[str, list[dict]]]:
    coco_data = load_json(INPUT_JSON)

    images = coco_data["images"]
    annotations = coco_data["annotations"]
    kitti_categories_as_list = coco_data["categories"]
    kitti_categories_as_dict = {category["id"]: category["name"] for category in kitti_categories_as_list}

    coco_categories_as_dict = load_json(COCO_LABELS_PATH)
    coco_categories_as_list = [
        {"id": category_id, "name": category_name, "supercategory": "none"}
        for category_name, category_id in coco_categories_as_dict.items()
    ]

    filtered_and_replaced_annotations = replace_categories_ids_with_coco_ids_and_filter(
        annotations, kitti_categories_as_dict, coco_categories_as_dict
    )

    random.shuffle(images)
    train_images = images[:TRAIN_SIZE]
    validation_images = images[TRAIN_SIZE : TRAIN_SIZE + VALIDATION_SIZE]
    test_images = images[TRAIN_SIZE + VALIDATION_SIZE : TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE]

    all_annotations = _extract_annotations_for_images(images, filtered_and_replaced_annotations)
    train_annotations = _extract_annotations_for_images(train_images, filtered_and_replaced_annotations)
    validation_annotations = _extract_annotations_for_images(validation_images, filtered_and_replaced_annotations)
    test_annotations = _extract_annotations_for_images(test_images, filtered_and_replaced_annotations)

    return (
        _make_coco(images, all_annotations, coco_categories_as_list),
        _make_coco(train_images, train_annotations, coco_categories_as_list),
        _make_coco(validation_images, validation_annotations, coco_categories_as_list),
        _make_coco(test_images, test_annotations, coco_categories_as_list),
    )


def _extract_annotations_for_images(images: list[dict], filtered_annotations: list[dict]) -> list[dict]:
    image_ids = {img["id"] for img in images}
    return [a for a in filtered_annotations if a["image_id"] in image_ids]


def _make_coco(
    images: list[dict], annotations: list[dict], categories: list[dict[str, str | int]]
) -> dict[str, list[dict]]:
    return {"images": images, "annotations": annotations, "categories": categories}


def main() -> None:
    all_coco_data, training_coco_data, validation_coco_data, testing_coco_data = split_coco()

    os.makedirs(DATASET_DIR, exist_ok=True)
    save_json(all_coco_data, os.path.join(DATASET_DIR, "coco_all.json"))
    save_json(training_coco_data, os.path.join(DATASET_DIR, "coco_train.json"))
    save_json(validation_coco_data, os.path.join(DATASET_DIR, "coco_val.json"))
    save_json(testing_coco_data, os.path.join(DATASET_DIR, "coco_test.json"))

    print(
        f"""Wrote splits to {DATASET_DIR}:
    coco_all.json   ({len(all_coco_data['images'])} images)
    coco_train.json ({len(training_coco_data['images'])} images)
    coco_val.json   ({len(validation_coco_data['images'])} images)
    coco_test.json  ({len(testing_coco_data['images'])} images)
    """
    )


if __name__ == "__main__":
    main()
