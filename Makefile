# Image name
IMAGE_NAME = ai-lab

# Container name
CONTAINER_NAME = ai-lab

CONTAINER_CLI := $(shell if command -v podman >/dev/null 2>&1; then echo podman; else echo docker; fi)

# Volume mount path
OUTPUT_PATH = "${PWD}/source"

# GPU flag
GPU_FLAG = --gpus all

execute: build run_gpu

# Build the Docker image
build:
	$(CONTAINER_CLI) build -t $(IMAGE_NAME) .

# Run the Docker container
run_gpu: clean
	$(CONTAINER_CLI) run $(GPU_FLAG) --shm-size 16G -v $(OUTPUT_PATH):/app -d --privileged --name $(CONTAINER_NAME) $(IMAGE_NAME) tail -f /dev/null
	$(CONTAINER_CLI) exec -it $(CONTAINER_NAME) /bin/bash

run: clean
	$(CONTAINER_CLI) run --shm-size 16G -v $(OUTPUT_PATH):/app -d --privileged --name $(CONTAINER_NAME) $(IMAGE_NAME) tail -f /dev/null
	$(CONTAINER_CLI) exec -it $(CONTAINER_NAME) /bin/bash

# Remove the Docker container if it exists
clean:
	-@$(CONTAINER_CLI) rm -f $(CONTAINER_NAME) 2>/dev/null

# Remove the Docker image if needed
clean-image:
	-@$(CONTAINER_CLI) rmi $(IMAGE_NAME) 2>/dev/null


DATASET_NAME = traffic-detection-project
DATASET_ZIP = $(DATASET_NAME).zip
DATASET_URL = https://www.kaggle.com/api/v1/datasets/download/yusufberksardoan/$(DATASET_NAME)
DATA_DIR = source/datasets/kaggle

download_dataset: $(DATA_DIR)

$(DATA_DIR): $(DATASET_ZIP)
	mkdir -p $(DATA_DIR)
	unzip $(DATASET_ZIP) -d $(DATA_DIR)
	rm $(DATASET_ZIP)
	touch $@

$(DATASET_ZIP):
	curl -L -o $(DATASET_ZIP) "$(DATASET_URL)"
