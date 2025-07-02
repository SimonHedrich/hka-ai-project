import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
from typing import List, Tuple, Dict, Any

def load_coco_json(json_path: str) -> Dict[str, Any]:
    """Load the original COCO JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def create_noise_image(width: int = 512, height: int = 512) -> Image.Image:
    """Create an image with random RGB noise."""
    noise_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(noise_array, 'RGB')

def check_overlap(bbox1: List[float], bbox2: List[float]) -> bool:
    """Check if two bounding boxes overlap."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or 
                y1 + h1 <= y2 or y2 + h2 <= y1)

def find_valid_position(
    img_width: int, 
    img_height: int, 
    obj_width: int, 
    obj_height: int, 
    existing_bboxes: List[List[float]], 
    max_attempts: int = 100
) -> Tuple[int, int] | None:
    """Find a valid position for an object that doesn't overlap with existing ones."""
    for _ in range(max_attempts):
        x = random.randint(0, img_width - obj_width)
        y = random.randint(0, img_height - obj_height)
        
        new_bbox = [x, y, obj_width, obj_height]
        
        # Check if this position overlaps with any existing bbox
        if not any(check_overlap(new_bbox, existing_bbox) for existing_bbox in existing_bboxes):
            return x, y
    
    return None

def create_synthetic_dataset(
    original_json_path: str,
    cutouts_dir: str,
    output_dir: str,
    num_images: int = 1,
    num_objects: int = 10,
    image_size: int = 512
):
    """Create multiple synthetic images with non-overlapping objects on noise background and saves COCO JSON."""
    
    # Load original COCO data
    original_coco = load_coco_json(original_json_path)
    
    # Get available object files
    available_objects = []
    for annotation in original_coco['annotations']:
        obj_path = os.path.join(cutouts_dir, f"object_{annotation['id']}.png")
        if os.path.exists(obj_path):
            available_objects.append(annotation)
    
    if len(available_objects) == 0:
        raise ValueError("No object cutout files found!")
    
    all_annotations = []
    all_images = []
    annotation_id_counter = 1
    
    for img_id in tqdm(range(1, num_images + 1)):
        # Create noise background
        background = create_noise_image(image_size, image_size)
        
        # Randomly select objects to place
        selected_objects = random.sample(
            available_objects, 
            min(num_objects, len(available_objects))
        )
        
        # Track placed bounding boxes and new annotations
        placed_bboxes = []
        new_annotations = []
        
        # Place objects on background
        for i, annotation in enumerate(selected_objects):
            obj_path = os.path.join(cutouts_dir, f"object_{annotation['id']}.png")
            
            try:
                # Load object image
                obj_img = Image.open(obj_path).convert('RGBA')
                obj_width, obj_height = obj_img.size
                
                # Skip if object is too large for the canvas
                if obj_width >= image_size or obj_height >= image_size:
                    print(f"Skipping object {annotation['id']} - too large")
                    continue
                
                # Find valid position
                position = find_valid_position(
                    image_size, image_size, obj_width, obj_height, placed_bboxes
                )
                
                if position is None:
                    print(f"Couldn't find valid position for object {annotation['id']}")
                    continue
                
                x, y = position
                
                # Paste object onto background
                background.paste(obj_img, (x, y), obj_img)
                
                # Create new annotation
                new_bbox = [x, y, obj_width, obj_height]
                placed_bboxes.append(new_bbox)
                
                new_annotation = {
                    "id": annotation_id_counter,  # Unique annotation ID
                    "image_id": img_id,  # Image ID
                    "category_id": annotation['category_id'],
                    "area": obj_width * obj_height,
                    "bbox": new_bbox,
                    "iscrowd": 0
                }
                new_annotations.append(new_annotation)
                all_annotations.append(new_annotation)
                annotation_id_counter += 1
                
            except Exception as e:
                print(f"Error processing object {annotation['id']}: {e}")
                continue
        
        # Save synthetic image
        image_name = f"synthetic_image_{img_id}.png"
        output_image_path = os.path.join(output_dir, image_name)
        background.save(output_image_path)
        
        all_images.append({
            "id": img_id,
            "width": image_size,
            "height": image_size,
            "file_name": image_name
        })
        
        print(f"Created synthetic image {img_id} with {len(new_annotations)} objects")
        print(f"Image saved to: {output_image_path}")
    
    # Create new COCO JSON
    new_coco = {
        "info": {
            "description": "Synthetic dataset with random noise background",
            "version": "1.0",
            "year": 2025
        },
        "licenses": original_coco.get('licenses', []),
        "images": all_images,
        "annotations": all_annotations,
        "categories": original_coco['categories']  # Copy original categories
    }
    
    # Save new COCO JSON
    output_json_path = os.path.join(output_dir, "synthetic_coco.json")
    with open(output_json_path, 'w') as f:
        json.dump(new_coco, f, indent=2)
    
    print(f"Annotations saved to: {output_json_path}")

# Example usage
if __name__ == "__main__":
    # Set your paths here
    original_json_path = "/Users/simonhedrich/HKA/1-AI-Lab/ai-lab/traffic_detection_project/source/datasets/kitti/coco_train.json"  # Path to your original COCO JSON
    cutouts_dir = "/Users/simonhedrich/HKA/1-AI-Lab/ai-lab/traffic_detection_project/source/data_augmentation/extracted_objects"  # Directory containing object_*.png files
    output_dir = "synthetic_dataset"  # Output directory for images and JSON
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    create_synthetic_dataset(
        original_json_path=original_json_path,
        cutouts_dir=cutouts_dir,
        output_dir=output_dir,
        num_images=10,  # Number of images to generate
        num_objects=15,  # Number of objects to place per image
        image_size=512
    )
