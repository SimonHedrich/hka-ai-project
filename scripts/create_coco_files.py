import json
import os

import cv2
from tqdm import tqdm

from traffic_detection_project.source.constants import DATASET_DIR


def convert_to_coco(root_dir, output_path, class_names):
    """
    Converts a dataset with image and label structure to COCO JSON format.

    Args:
        root_dir (str): Root directory containing 'images' and 'labels' subdirectories.
        output_path (str): Path to save the COCO JSON file.
        class_names (list): List of class names.
    """

    images_dir = os.path.join(root_dir, "images")
    labels_dir = os.path.join(root_dir, "labels")

    images = []
    annotations = []
    categories = []
    annotation_id = 1
    image_id = 1

    # Create categories
    for i, class_name in enumerate(class_names):
        categories.append({"id": i, "name": class_name})

    # Process each image
    for image_file in tqdm(os.listdir(images_dir)):
        if not image_file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image_name = os.path.splitext(image_file)[0]
        image_path = os.path.join(images_dir, image_file)
        label_file = os.path.join(labels_dir, image_name + ".txt")

        # Read image dimensions
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        height, width, _ = img.shape

        # Add image info
        images.append(
            {
                "id": image_id,
                "file_name": image_file,
                "width": width,
                "height": height,
            }
        )

        # Read annotations
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                for line in f:
                    class_id, center_x, center_y, w, h = map(float, line.split())

                    # Convert normalized coordinates to pixel values
                    x = (center_x - w / 2) * width
                    y = (center_y - h / 2) * height
                    w *= width
                    h *= height

                    # Add annotation
                    annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": int(class_id),
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "segmentation": [],
                            "iscrowd": 0,
                        }
                    )
                    annotation_id += 1
        image_id += 1

    # Create COCO JSON structure
    coco_output = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    # Save to JSON file
    with open(output_path, "w") as f:
        json.dump(coco_output, f, indent=4)


if __name__ == "__main__":
    class_names = ["bicycle", "bus", "car", "motorbike", "person"]

    for dataset_part in ("train", "valid", "test"):
        print(f"Convert '{dataset_part}'...")
        convert_to_coco(
            os.path.join(DATASET_DIR, dataset_part),
            os.path.join(DATASET_DIR, "coco", f"coco_{dataset_part}.json"),
            class_names,
        )

    print("COCO JSON conversion complete.")
