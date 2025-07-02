import hydra
import logging
import os
import random
import traceback

import constants
import hydra
import mlflow
from dataset import COCODataset, Dataloader
from dotenv import load_dotenv
from model_faster_rcnn import faster_rcnn_model, model_optimizer, model_scheduler
from omegaconf import DictConfig
from training_pipeline import TrainingPipeline
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def training_run(cfg):
    detection_model, image_preprocessing = faster_rcnn_model(num_classes=constants.CLASS_COUNT)

    # TRAINING DATASET
    dataset_train = COCODataset(
        image_dir_path=constants.TRAIN_IMAGE_DIR_PATH,
        coco_file_path=constants.TRAIN_COCO_PATH,
        transform=image_preprocessing,
    )
    mlflow.log_param("TRAIN_COCO_PATH", constants.TRAIN_COCO_PATH)
    mlflow.log_param("TRAIN_IMAGE_COUNT", dataset_train.__len__())
    rand_train_img, rand_train_label = dataset_train[random.randint(0, len(dataset_train) - 1)]  # nosec B311
    print(f"random training label: {rand_train_label}")

    # VALIDATION DATASET
    dataset_valid = COCODataset(
        image_dir_path=constants.VALIDATION_IMAGE_DIR_PATH,
        coco_file_path=constants.VALIDATION_COCO_PATH,
        transform=image_preprocessing,
    )
    mlflow.log_param("VALIDATION_COCO_PATH", constants.VALIDATION_COCO_PATH)
    mlflow.log_param("VALIDATION_IMAGE_COUNT", dataset_valid.__len__())
    rand_val_img, rand_val_label = dataset_valid[random.randint(0, len(dataset_valid) - 1)]  # nosec B311
    # print(f"random validation label: {rand_val_label}")

    # EVALUATION DATASET
    dataset_eval = COCODataset(
        image_dir_path=constants.EVALUATION_IMAGE_DIR_PATH,
        coco_file_path=constants.EVALUATION_COCO_PATH,
        transform=image_preprocessing,
    )
    mlflow.log_param("EVALUATION_COCO_PATH", constants.EVALUATION_COCO_PATH)
    mlflow.log_param("EVALUATION_IMAGE_COUNT", dataset_eval.__len__())
    rand_eval_img, rand_eval_label = dataset_eval[random.randint(0, len(dataset_eval) - 1)]  # nosec B311
    # print(f"random evaluation label: {rand_eval_label}")

    # TRAINING DATALOADER
    dataloader_train = Dataloader(
        dataset=dataset_train,
        batch_size=constants.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    ).get_dataloader()
    mlflow.log_param("TRAIN_BATCH_SIZE", constants.TRAIN_BATCH_SIZE)
    # print(f"train dataloader inference: {inference(detection_model, dataloader_train)}")

    # VALIDATION DATALOADER
    dataloader_valid = Dataloader(
        dataset=dataset_valid,
        batch_size=constants.VALIDATION_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    ).get_dataloader()
    mlflow.log_param("VALIDATION_BATCH_SIZE", constants.VALIDATION_BATCH_SIZE)
    # print(f"validation dataloader inference: {inference(detection_model, dataloader_val)}")

    # EVALUATION DATALOADER
    dataloader_eval = Dataloader(
        dataset=dataset_eval,
        batch_size=constants.EVALUATION_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    ).get_dataloader()
    # print(f"evaluation dataloader inference: {inference(detection_model, dataloader_eval)}")
    mlflow.log_param("EVALUATION_BATCH_SIZE", constants.EVALUATION_BATCH_SIZE)

    optimizer = model_optimizer(cfg.optimizer, detection_model)
    scheduler = model_scheduler(cfg.scheduler, optimizer)

    model_name = f"model-{constants.EPOCH_COUNT}-{constants.TRAIN_DATASET_NAME}.pth"
    model_export_path = os.path.join(constants.OUTPUT_DIR, model_name)

    training_pipeline = TrainingPipeline(
        dataloader_train=dataloader_train,
        dataloader_valid=dataloader_valid,
        dataloader_eval=dataloader_eval,
        detection_model=detection_model,
        epoch_count=constants.EPOCH_COUNT,
        model_optimizer=optimizer,
        model_scheduler=scheduler,
        model_export_path=model_export_path,
    )
    training_pipeline.run_pipeline()


def setup_mlflow():
    MLFLOW_SERVER_URI = os.getenv("MLFLOW_SERVER_URI")
    mlflow.set_tracking_uri(MLFLOW_SERVER_URI)
    mlflow.enable_system_metrics_logging()
    logging.getLogger("mlflow.system_metrics.metrics.gpu_monitor").setLevel(logging.ERROR)  # Suppress GPU monitor logs
    mlflow.set_experiment(constants.EXPERIMENT_NAME)


@hydra.main(config_path="configs", config_name="config.yml", version_base=None)
def main(cfg: DictConfig):
    logger.debug("Loading environment variables...")
    load_dotenv()
    logger.debug("Setting up MLflow...")
    setup_mlflow()
    logger.info("Starting MLflow run...")
    with mlflow.start_run():
        try:
            training_run(cfg)
        except Exception as e:
            mlflow.log_param("Error", str(e))
            print(traceback.format_exc())
        finally:
            mlflow.end_run()

if __name__ == "__main__":
    main()
