import numpy as np
import cv2
import torch

import json
import os
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

np.random.seed(3)


def get_predictor(sam2_checkpoint, model_cfg):
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor


def get_image(image_path):
    image_name, image_extension = os.path.splitext(os.path.basename(image_path))
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))
    return (image, image_name, image_extension)


def predict_single(predictor: SAM2ImagePredictor, image, input_box, point_coords=None, point_labels=None):
    predictor.set_image(image)
    input_box = np.array(input_box)
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=input_box[None, :],
        multimask_output=False,
    )
    return masks[0], scores[0]


def mask_morphology(mask, kernel_dim=5, morphology_iterations=1):
    assert kernel_dim % 2 == 1, "kernel dimension must be an odd number"
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_dim, kernel_dim))
    mask_dilated = cv2.dilate(mask, kernel, iterations=morphology_iterations)
    mask_eroded = cv2.erode(mask_dilated, kernel, iterations=morphology_iterations)
    return mask_eroded


def extract_object_with_transparency(image, mask):
    """Extract object from image using mask and create transparent background"""
    # Ensure mask is boolean
    mask_bool = mask.astype(bool)

    # Create RGBA image (RGB + Alpha channel)
    height, width = image.shape[:2]
    rgba_image = np.zeros((height, width, 4), dtype=np.uint8)

    # Copy RGB channels where mask is True
    rgba_image[mask_bool, :3] = image[mask_bool]

    # Set alpha channel: 255 where mask is True, 0 where False
    rgba_image[mask_bool, 3] = 255

    return rgba_image


def get_mask_bounding_box(mask):
    """Get the bounding box of the mask (non-zero pixels)"""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def crop_to_object(rgba_image, mask):
    """Crop the RGBA image to just the segmented object"""
    bbox = get_mask_bounding_box(mask)
    if bbox is None:
        return rgba_image

    rmin, rmax, cmin, cmax = bbox
    return rgba_image[rmin : rmax + 1, cmin : cmax + 1]


def main():
    # Configuration
    PROJECT_DIR = (
        "/home/debian/ai-lab/traffic_detection_project/source/data_augmentation"  # Adjust this path as needed
    )

    sam2_checkpoint = os.path.join(PROJECT_DIR, "sam2/checkpoints/sam2.1_hiera_large.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # Dataset paths
    IMAGE_DIR_PATH = os.path.join("/home/debian/ai-lab/traffic_detection_project/source/datasets/kitti/images")
    COCO_FILE_PATH = os.path.join(
        "/home/debian/ai-lab/traffic_detection_project/source/datasets/kitti/coco_train.json"
    )

    # Output directory for extracted objects
    EXTRACTED_OBJECTS_DIR = os.path.join(PROJECT_DIR, "extracted_objects")
    os.makedirs(EXTRACTED_OBJECTS_DIR, exist_ok=True)

    # SAM parameters
    BBOX_MARGIN = 0.02
    KERNEL_DIMENSION = 7
    MORPHOLOGY_ITERATIONS = 1

    # Initialize SAM predictor
    print("Loading SAM2 model...")
    predictor = get_predictor(sam2_checkpoint, model_cfg)

    # Load COCO data
    print("Loading COCO dataset...")
    with open(COCO_FILE_PATH, "r") as file:
        coco_data = json.load(file)

    # Create lookup dictionaries
    image_id2info = {img["id"]: img for img in coco_data["images"]}

    print(f"Processing {len(coco_data['annotations'])} annotations...")

    # Process each annotation
    for annotation in tqdm(coco_data["annotations"]):
        annotation_id = annotation["id"]
        image_id = annotation["image_id"]

        # Skip if output file already exists
        output_filename = f"object_{annotation_id}.png"
        output_path = os.path.join(EXTRACTED_OBJECTS_DIR, output_filename)
        if os.path.exists(output_path):
            continue

        # Get image info
        image_info = image_id2info[image_id]
        image_file_name = image_info["file_name"]
        image_width = image_info["width"]
        image_height = image_info["height"]

        # Load image
        try:
            image_path = os.path.join(IMAGE_DIR_PATH, image_file_name)
            image, _, _ = get_image(image_path)
        except Exception as e:
            print(f"Error loading image {image_file_name}: {e}")
            continue

        # Get bounding box in XYXY format with margin
        bbox_xywh = annotation["bbox"]
        bbox_xyxy = [
            max(0, int(bbox_xywh[0] - (bbox_xywh[2] * BBOX_MARGIN))),
            max(0, int(bbox_xywh[1] - (bbox_xywh[3] * BBOX_MARGIN))),
            min(image_width, int(bbox_xywh[0] + bbox_xywh[2] + (bbox_xywh[2] * BBOX_MARGIN))),
            min(image_height, int(bbox_xywh[1] + bbox_xywh[3] + (bbox_xywh[3] * BBOX_MARGIN))),
        ]

        # Create point prompts (center of object + background corners)
        center_x = int(bbox_xywh[0] + bbox_xywh[2] / 2)
        center_y = int(bbox_xywh[1] + bbox_xywh[3] / 2)
        object_points = [[center_x, center_y]]

        # Background points at corners of bounding box
        PIXEL_DISTANCE = 10
        background_points = [
            [bbox_xyxy[0] + PIXEL_DISTANCE, bbox_xyxy[1] + PIXEL_DISTANCE],  # top left
            [bbox_xyxy[2] - PIXEL_DISTANCE, bbox_xyxy[1] + PIXEL_DISTANCE],  # top right
            [bbox_xyxy[0] + PIXEL_DISTANCE, bbox_xyxy[3] - PIXEL_DISTANCE],  # bottom left
            [bbox_xyxy[2] - PIXEL_DISTANCE, bbox_xyxy[3] - PIXEL_DISTANCE],  # bottom right
        ]

        point_coords = np.array(object_points + background_points)
        BACKGROUND_LABEL, OBJECT_LABEL = 0, 1
        point_labels = np.array([OBJECT_LABEL] * len(object_points) + [BACKGROUND_LABEL] * len(background_points))

        try:
            # Run SAM prediction
            mask, score = predict_single(predictor, image, bbox_xyxy, point_coords, point_labels)

            # Apply morphological operations to clean up mask
            mask = mask_morphology(mask, kernel_dim=KERNEL_DIMENSION, morphology_iterations=MORPHOLOGY_ITERATIONS)

            # Extract object with transparent background
            rgba_object = extract_object_with_transparency(image, mask)

            # Crop to just the object
            cropped_object = crop_to_object(rgba_object, mask)

            # Save as PNG (supports transparency)
            pil_image = Image.fromarray(cropped_object, "RGBA")
            pil_image.save(output_path)

        except Exception as e:
            print(f"Error processing annotation {annotation_id}: {e}")
            continue

    print(f"Extraction complete! Objects saved to: {EXTRACTED_OBJECTS_DIR}")


if __name__ == "__main__":
    main()
