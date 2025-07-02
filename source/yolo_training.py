import torch  # Already in constants, but good for explicit check here

import constants as const
import os
import yaml
from datetime import datetime
from ultralytics import YOLO

MODEL_VARIANT = "yolov8n.pt"
IMG_SIZE = 640
PATIENCE_EARLY_STOPPING = 20  # You can adjust this
GENERATED_YAML_PATH = os.path.join(const.DATASET_DIR, "training_dataset.yaml")


def create_dataset_yaml(
    output_yaml_path,
    image_dir_path,  # const.TRAIN_IMAGE_DIR_PATH (shared for train/val images)
    train_coco_json_path,  # const.TRAIN_COCO_PATH
    val_coco_json_path,  # const.VALIDATION_COCO_PATH
    # test_coco_json_path=None, # Optional: const.EVALUATION_COCO_PATH
    # class_count=None, # Optional: const.CLASS_COUNT
    # class_names=None # Optional: list of class names
):
    """
    Creates the dataset.yaml file required by Ultralytics YOLO,
    configured for COCO JSON annotations.
    """
    # Validate paths
    if not os.path.isdir(image_dir_path):
        raise FileNotFoundError(f"Image directory not found: {image_dir_path}")
    if not os.path.isfile(train_coco_json_path):
        raise FileNotFoundError(f"Train COCO JSON file not found: {train_coco_json_path}")
    if not os.path.isfile(val_coco_json_path):
        raise FileNotFoundError(f"Validation COCO JSON file not found: {val_coco_json_path}")

    data_yaml = {
        # 'path' is the root directory for images if paths in COCO JSON are relative to it.
        # If COCO JSON has absolute paths or paths relative to JSON location,
        # this 'path' might be less critical but good for clarity.
        "path": os.path.abspath(image_dir_path),
        "train": os.path.abspath(train_coco_json_path),  # Path to train COCO JSON
        "val": os.path.abspath(val_coco_json_path),  # Path to val COCO JSON
    }

    # Optional: Add test set if provided
    # if test_coco_json_path:
    #     if not os.path.isfile(test_coco_json_path):
    #         print(f"Warning: Test COCO JSON file not found: {test_coco_json_path}")
    #     else:
    #         data_yaml["test"] = os.path.abspath(test_coco_json_path)

    # Ultralytics usually infers nc and names from COCO JSON.
    # Explicitly setting them can be done if needed:
    # if class_count is not None:
    #     data_yaml['nc'] = class_count
    # if class_names is not None:
    #     data_yaml['names'] = class_names
    # elif class_count is not None and class_names is None:
    #     print(f"Warning: nc={class_count} is set, but names are not. "
    #           "Consider providing class names for clarity.")
    #     # Create generic names if only count is provided
    #     # data_yaml['names'] = [f'class_{i}' for i in range(class_count)]

    with open(output_yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False, indent=2)
    print(f"Successfully created dataset YAML: {output_yaml_path}")
    print("YAML content:")
    print(yaml.dump(data_yaml, sort_keys=False, indent=2))
    return output_yaml_path


# --- 4. MAIN TRAINING SCRIPT ---
if __name__ == "__main__":
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Starting YOLO training script at {current_time_str}")
    print(f"Using Ultralytics YOLO with model: {MODEL_VARIANT}")
    print(f"Running on device: {const.DEVICE}")

    # Check if GPU is actually available if const.DEVICE is "cuda"
    if str(const.DEVICE) == "cuda":
        if not torch.cuda.is_available():
            print(
                "Warning: const.DEVICE is 'cuda' but CUDA is not available. "
                "Training will likely fall back to CPU or fail."
            )
        else:
            print(f"CUDA is available. Training on GPU: {torch.cuda.get_device_name(0)}")
    elif str(const.DEVICE) == "cpu":
        print("Training on CPU (this may be slow).")

    # Step 1: Create or verify the dataset.yaml file
    try:
        print("\n--- Creating dataset.yaml ---")
        dataset_yaml_file = create_dataset_yaml(
            output_yaml_path=GENERATED_YAML_PATH,
            image_dir_path=const.TRAIN_IMAGE_DIR_PATH,  # Shared image directory
            train_coco_json_path=const.TRAIN_COCO_PATH,
            val_coco_json_path=const.VALIDATION_COCO_PATH,
            # test_coco_json_path=const.EVALUATION_COCO_PATH, # Uncomment to include test set
            # class_count=const.CLASS_COUNT # Uncomment if you want to explicitly set nc
        )
    except FileNotFoundError as e:
        print(f"Error creating YAML: {e}")
        print("Please ensure your dataset paths in constants.py are correct.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during YAML creation: {e}")
        exit(1)

    # Step 2: Load the YOLO model
    try:
        print(f"\n--- Loading model '{MODEL_VARIANT}' ---")
        model = YOLO(MODEL_VARIANT)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Ensure the model variant is correct and you have internet " "access if it needs to be downloaded.")
        exit(1)

    # Step 3: Train the model
    # Construct run name for clarity in output folders
    run_name = (
        f"{MODEL_VARIANT.split('.')[0]}_{const.TRAIN_DATASET_NAME}_" f"e{const.EPOCH_COUNT}_b{const.TRAIN_BATCH_SIZE}"
    )

    print(f"\n--- Starting training for {const.EPOCH_COUNT} epochs ---")
    print(f"Project: {const.EXPERIMENT_NAME}, Run: {run_name}")
    print(f"Batch size: {const.TRAIN_BATCH_SIZE}, Image size: {IMG_SIZE}")
    print(f"Learning rate: {const.LEARNING_RATE}, Momentum: {const.MOMENTUM}")
    print(f"Weight decay: {const.WEIGHT_DECAY}")
    print(f"Results will be saved to: ./runs/train/{const.EXPERIMENT_NAME}/{run_name}")

    try:
        model.train(
            data=dataset_yaml_file,
            epochs=const.EPOCH_COUNT,
            batch=const.TRAIN_BATCH_SIZE,
            imgsz=IMG_SIZE,
            patience=PATIENCE_EARLY_STOPPING,
            project=const.EXPERIMENT_NAME,  # Main project folder
            name=run_name,  # Specific run folder
            exist_ok=True,  # Allow overwriting if run_name already exists
            device=str(const.DEVICE),  # Ensure it's a string 'cpu' or '0' etc.
            lr0=const.LEARNING_RATE,
            momentum=const.MOMENTUM,
            weight_decay=const.WEIGHT_DECAY,
            # Ultralytics uses 'gamma' for the LR scheduler (e.g. StepLR)
            # const.SCHEDULER_GAMMER might map to this if using a compatible scheduler
            # gamma=const.SCHEDULER_GAMMER, # Check Ultralytics docs for scheduler details
            # val=True, # Validation is enabled by default if 'val' is in dataset.yaml
            # workers= Set number of dataloader workers if needed
            # verbose=True, # For detailed logging
        )
        print("\nTraining completed successfully!")
        print(f"Trained model and results saved in " f"./runs/train/{const.EXPERIMENT_NAME}/{run_name}")
        print(
            f"The best model weights are typically saved as "
            f"./runs/train/{const.EXPERIMENT_NAME}/{run_name}/weights/best.pt"
        )

    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        print("Check your dataset, hyperparameters, and system resources " "(especially GPU memory if using CUDA).")
        import traceback

        traceback.print_exc()

    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nScript finished at {current_time_str}")
