# Traffic Detection Project

This repository contains code for a traffic detection project, including dataset preparation, training pipelines for various object detection models (Faster R-CNN, SSD-Lite, YOLOv8), data augmentation utilities, and dataset analysis scripts. MLflow is integrated for experiment tracking.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
  - [Prerequisites](#prerequisites)
  - [Docker Setup](#docker-setup)
  - [Local Setup (Optional)](#local-setup-optional)
- [Dataset Preparation](#dataset-preparation)
  - [Download Kaggle Dataset](#download-kaggle-dataset)
  - [Create COCO Files](#create-coco-files)
  - [Add IDs to COCO Labels](#add-ids-to-coco-labels)
- [Training Models](#training-models)
  - [Faster R-CNN](#faster-r-cnn)
  - [SSD-Lite](#ssd-lite)
  - [YOLOv8](#yolov8)
- [Dataset Analysis](#dataset-analysis)
- [Data Augmentation](#data-augmentation)
- [FiftyOne Visualization](#fiftyone-visualization)
- [MLflow Tracking](#mlflow-tracking)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Object Detection Models**: Implementations for Faster R-CNN, SSD-Lite, and YOLOv8.
- **COCO Dataset Integration**: Custom PyTorch `Dataset` for COCO annotations.
- **Automated Dataset Preparation**: Scripts to convert YOLO-style annotations to COCO format and add label IDs.
- **Data Augmentation**: Script for placing objects on synthetic backgrounds.
- **Dataset Analysis**: Jupyter notebook for exploring dataset statistics (category distribution, bounding box properties, co-occurrence).
- **MLflow Integration**: Comprehensive experiment tracking, logging metrics, parameters, and models.
- **Docker Support**: Containerized environment for reproducible development and training.
- **Makefile Automation**: Streamlined commands for building Docker images, running containers, and executing training.
- **Hydra Configuration**: Flexible configuration management for training parameters.

## Project Structure

```
.
├── Dockerfile
├── Makefile
├── requirements.txt
├── scripts/
│   ├── add_ids_to_coco_labels.py    # Adds numeric IDs to COCO labels
│   ├── create_coco_files.py        # Converts YOLO-style annotations to COCO JSON
│   └── launch_fiftyone.py          # Launches FiftyOne for dataset visualization
├── source/
│   ├── constants.py                # Global project constants and configurations
│   ├── dataset.py                  # Custom PyTorch COCODataset and Dataloader
│   ├── evaluation.py               # COCO evaluation metrics and MLflow logging
│   ├── Makefile                    # Internal Makefile for running training scripts
│   ├── model_faster_rcnn.py        # Faster R-CNN model definition and utilities
│   ├── model_ssd_lite.py           # SSD-Lite model definition and utilities
│   ├── run_training_faster_rcnn.py # Main script for Faster R-CNN training
│   ├── run_training_ssd_lite.py    # Main script for SSD-Lite training
│   ├── training_pipeline.py        # Generic training loop and pipeline
│   ├── yolo_training.py            # YOLOv8 training script
│   ├── configs/                    # Hydra configurations for models, optimizers, schedulers
│   │   ├── config.yaml
│   │   ├── optimizer_schema.py
│   │   ├── scheduler_schema.py
│   │   ├── optimizer/
│   │   └── scheduler/
│   ├── data_augmentation/
│   │   └── place_objects.py        # Script for synthetic data generation
│   ├── dataset_analysis/
│   │   └── dataset_analysis.ipynb  # Jupyter notebook for dataset exploration
│   └── helper/
│       └── instantiate_target.py   # Utility for Hydra instantiation
└── README.md
```

## Setup and Installation

### Prerequisites

- Git
- Docker (or Podman)
- Kaggle API token (for dataset download)

### Docker Setup

The recommended way to set up the environment is using Docker (or Podman).

1.  **Build the Docker Image**:
    Navigate to the root directory of the repository (`simonhedrich-hka-ai-project/`) and run:
    ```bash
    make build
    ```
    This command will build the Docker image named `ai-lab`.

2.  **Run the Docker Container**:
    To run the container with GPU support (recommended for training):
    ```bash
    make execute # This runs 'build' then 'run_gpu'
    # Or directly run with GPU after building:
    # make run_gpu
    ```
    To run the container without GPU support:
    ```bash
    make run
    ```
    These commands will start a detached container and then attach you to a bash session inside it. The `source` directory of your host machine will be mounted to `/app` inside the container.

    **Note on GPU**: Ensure your Docker/Podman installation is configured for GPU support (e.g., Docker Desktop with WSL2 on Windows, or NVIDIA Container Toolkit for Linux).

### Local Setup (Optional)

If you prefer to set up the environment locally without Docker:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/SimonHedrich/hka-ai-project.git
    cd hka-ai-project
    ```
2.  **Create a virtual environment and activate it**:
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Install Jupyter Kernel (for notebook)**:
    ```bash
    pip install ipykernel
    python -m ipykernel install --user --name=ai-lab --display-name="Python (AI Lab)"
    ```

## Dataset Preparation

The project expects the dataset to be in a specific structure and COCO JSON format.

### Download Kaggle Dataset

This project uses the `traffic-detection-project` dataset from Kaggle.

1.  **Set up Kaggle API**:
    Ensure you have your `kaggle.json` file in `~/.kaggle/` (or set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables).
2.  **Download the dataset**:
    From the project root directory, run:
    ```bash
    make download_dataset
    ```
    This will download `traffic-detection-project.zip`, unzip it into `source/datasets/kaggle/`, and remove the zip file.
    The unzipped dataset will have the structure: `source/datasets/kaggle/{train, valid, test}/{images, labels}/`.

### Create COCO Files

The raw dataset comes with YOLO-style `.txt` label files. The training pipelines require COCO JSON format.

Inside the Docker container (or locally):

1.  **Navigate to the `scripts` directory**:
    ```bash
    cd scripts
    ```
2.  **Run the conversion script**:
    ```bash
    python create_coco_files.py
    ```
    This script will generate `coco_train.json`, `coco_valid.json`, and `coco_test.json` files within `source/datasets/kaggle/coco/`.

### Add IDs to COCO Labels

This script creates a JSON mapping of COCO class names to integer IDs, which might be useful for consistent class ID management.

Inside the Docker container (or locally):

1.  **Navigate to the `scripts` directory**:
    ```bash
    cd scripts
    ```
2.  **Run the script**:
    ```bash
    python add_ids_to_coco_labels.py
    ```
    This generates `source/datasets/coco_labels_with_ids.json`.

## Training Models

The `constants.py` file defines various hyperparameters and paths. You might want to adjust these before training.

All training runs integrate with MLflow for tracking. Ensure MLflow is accessible (see [MLflow Tracking](#mlflow-tracking)).

### Faster R-CNN

To train the Faster R-CNN model:

Inside the Docker container:

1.  **Navigate to the `source` directory**:
    ```bash
    cd source
    ```
2.  **Run the training script**:
    ```bash
    make run_faster_rcnn # This calls 'nohup python3 run_training_faster_rcnn.py > output.log 2>&1 & tail -f output.log'
    ```
    Alternatively, for direct execution and to see logs in the foreground:
    ```bash
    python run_training_faster_rcnn.py
    ```
    Training progress and MLflow logs will be generated.

### SSD-Lite

To train the SSD-Lite model:

Inside the Docker container:

1.  **Navigate to the `source` directory**:
    ```bash
    cd source
    ```
2.  **Run the training script**:
    ```bash
    make run # This calls 'nohup python3 run_training_ssd_lite.py > output.log 2>&1 & tail -f output.log'
    ```
    Alternatively, for direct execution and to see logs in the foreground:
    ```bash
    python run_training_ssd_lite.py
    ```
    Training progress and MLflow logs will be generated.

### YOLOv8

To train a YOLOv8 model:

Inside the Docker container:

1.  **Navigate to the `source` directory**:
    ```bash
    cd source
    ```
2.  **Run the YOLO training script**:
    ```bash
    python yolo_training.py
    ```
    This script will automatically create the `dataset.yaml` required by YOLOv8 and start the training process. Results will be saved under `./runs/train/`.

## Dataset Analysis

To analyze the dataset characteristics, open the Jupyter notebook.

Inside the Docker container:

1.  **Start Jupyter Lab**:
    If not already running, you can start Jupyter Lab (you might need to install it first if using a minimal Docker image):
    ```bash
    pip install jupyterlab
    jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser
    ```
2.  **Access Jupyter Lab**:
    Open your web browser and navigate to `http://localhost:8888` (or the address provided by Jupyter Lab output, which might include a token).
3.  **Open the Notebook**:
    Navigate to `source/dataset_analysis/dataset_analysis.ipynb` and run all cells to generate plots and statistics.

## Data Augmentation

The `place_objects.py` script demonstrates how to create synthetic images by placing cut-out objects onto a noise background.

Before running:

-   You need "cutout" images of objects, typically extracted from an existing dataset. This script assumes these are named `object_ID.png` and located in the `extracted_objects` directory. You would need to create these yourself from your COCO dataset using an external tool or a custom script (e.g., using SAM to segment and extract objects).
-   Update `original_json_path`, `cutouts_dir`, and `output_dir` in `source/data_augmentation/place_objects.py` to match your paths.

Inside the Docker container (or locally):

1.  **Navigate to the `data_augmentation` directory**:
    ```bash
    cd source/data_augmentation
    ```
2.  **Run the script**:
    ```bash
    python place_objects.py
    ```
    This will generate synthetic images and a new COCO JSON file in the specified output directory.

## FiftyOne Visualization

FiftyOne is used for interactive visualization and exploration of the datasets.

Inside the Docker container (or locally):

1.  **Run the FiftyOne launch script**:
    ```bash
    python scripts/launch_fiftyone.py
    ```
    This will start the FiftyOne app and load the specified dataset (defaulting to the full COCO dataset). The script prints the URL to access the FiftyOne UI (usually `http://localhost:8886`).
2.  **Access FiftyOne UI**:
    Open your web browser and navigate to the printed URL.

## MLflow Tracking

MLflow is used to track experiments, parameters, metrics, and models.

1.  **Start MLflow Tracking Server**:
    The training scripts expect an MLflow tracking server to be running.
    If you don't have one running, you can start a local MLflow UI from the project root (outside the Docker container if you want persistent storage outside the container):
    ```bash
    mlflow ui --backend-store-uri file:/path/to/your/mlruns --host 0.0.0.0 --port 5000
    ```
    Replace `/path/to/your/mlruns` with a directory where MLflow should store its data.

2.  **Set MLFLOW_SERVER_URI**:
    The training scripts read the `MLFLOW_SERVER_URI` environment variable. Ensure this is set correctly in your environment (e.g., in a `.env` file that gets loaded by `dotenv`).
    For example, if running MLflow locally on port 5000:
    ```
    MLFLOW_SERVER_URI=http://localhost:5000
    ```
    This is configured to be loaded by `dotenv` in `run_training_faster_rcnn.py` and `run_training_ssd_lite.py`.

3.  **View Runs**:
    Once training starts, you can view the progress and results by navigating to the MLflow UI in your browser (e.g., `http://localhost:5000`).

## Configuration

The training scripts use [Hydra](https://hydra.cc/) for configuration management. Default configurations are defined in `source/configs/`.

- `source/configs/config.yaml`: The main configuration file, which composes other configurations.
- `source/configs/optimizer/`: Defines different optimizers (e.g., `sgd.yaml`).
- `source/configs/scheduler/`: Defines different learning rate schedulers (e.g., `multistep.yaml`).

You can override parameters from the command line, for example:
```bash
python source/run_training_ssd_lite.py epoch_count=10 optimizer.lr=0.0005
```

## License
MIT License
