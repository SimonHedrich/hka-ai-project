import hydra
import logging
import os
import random
import traceback
import constants as const
import mlflow

from dataset import COCODataset, Dataloader
from dotenv import load_dotenv
from model_ssd_lite import model_optimizer, model_scheduler, ssd_lite_model
from training_pipeline import TrainingPipeline
from omegaconf import OmegaConf, DictConfig
from configs.scheduler_schema import SchedulerConfig
from configs.optimizer_schema import OptimizerConfig
from pydantic import TypeAdapter

logger = logging.getLogger(__name__)
logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.INFO)
logging.getLogger("git").setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.basicConfig(level=logging.DEBUG)


def training_run(cfg):
    detection_model, image_preprocessing = ssd_lite_model()

    # TRAINING DATASET
    dataset_train = COCODataset(
        image_dir_path=const.TRAIN_IMAGE_DIR_PATH,
        coco_file_path=const.TRAIN_COCO_PATH,
        transform=image_preprocessing,
    )
    mlflow.log_param("TRAIN_COCO_PATH", const.TRAIN_COCO_PATH)
    mlflow.log_param("TRAIN_IMAGE_COUNT", dataset_train.__len__())
    rand_train_img, rand_train_label = dataset_train[random.randint(0, len(dataset_train) - 1)]  # nosec B311
    print(f"random training label: {rand_train_label}")

    # VALIDATION DATASET
    dataset_valid = COCODataset(
        image_dir_path=const.VALIDATION_IMAGE_DIR_PATH,
        coco_file_path=const.VALIDATION_COCO_PATH,
        transform=image_preprocessing,
    )
    mlflow.log_param("VALIDATION_COCO_PATH", const.VALIDATION_COCO_PATH)
    mlflow.log_param("VALIDATION_IMAGE_COUNT", dataset_valid.__len__())
    rand_val_img, rand_val_label = dataset_valid[random.randint(0, len(dataset_valid) - 1)]  # nosec B311
    # print(f"random validation label: {rand_val_label}")

    # EVALUATION DATASET
    dataset_eval = COCODataset(
        image_dir_path=const.EVALUATION_IMAGE_DIR_PATH,
        coco_file_path=const.EVALUATION_COCO_PATH,
        transform=image_preprocessing,
    )
    mlflow.log_param("EVALUATION_COCO_PATH", const.EVALUATION_COCO_PATH)
    mlflow.log_param("EVALUATION_IMAGE_COUNT", dataset_eval.__len__())
    rand_eval_img, rand_eval_label = dataset_eval[random.randint(0, len(dataset_eval) - 1)]  # nosec B311
    # print(f"random evaluation label: {rand_eval_label}")

    # TRAINING DATALOADER
    dataloader_train = Dataloader(
        dataset=dataset_train,
        batch_size=const.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    ).get_dataloader()
    mlflow.log_param("TRAIN_BATCH_SIZE", const.TRAIN_BATCH_SIZE)
    # print(f"train dataloader inference: {inference(detection_model, dataloader_train)}")

    # VALIDATION DATALOADER
    dataloader_valid = Dataloader(
        dataset=dataset_valid,
        batch_size=const.VALIDATION_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    ).get_dataloader()
    mlflow.log_param("VALIDATION_BATCH_SIZE", const.VALIDATION_BATCH_SIZE)
    # print(f"validation dataloader inference: {inference(detection_model, dataloader_val)}")

    # EVALUATION DATALOADER
    dataloader_eval = Dataloader(
        dataset=dataset_eval,
        batch_size=const.EVALUATION_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    ).get_dataloader()
    # print(f"evaluation dataloader inference: {inference(detection_model, dataloader_eval)}")
    mlflow.log_param("EVALUATION_BATCH_SIZE", const.EVALUATION_BATCH_SIZE)

    optimizer = model_optimizer(cfg.optimizer, detection_model)
    scheduler = model_scheduler(cfg.scheduler, optimizer)

    model_name = f"model-{const.EPOCH_COUNT}-{const.TRAIN_DATASET_NAME}.pth"
    model_export_path = os.path.join(const.OUTPUT_DIR, model_name)

    training_pipeline = TrainingPipeline(
        dataloader_train=dataloader_train,
        dataloader_valid=dataloader_valid,
        dataloader_eval=dataloader_eval,
        detection_model=detection_model,
        epoch_count=const.EPOCH_COUNT,
        model_optimizer=optimizer,
        model_scheduler=scheduler,
        model_export_path=model_export_path,
        auto_stop_threshold=const.AUTO_STOP_THRESHOLD,
    )
    training_pipeline.run_pipeline()


def setup_mlflow():
    MLFLOW_SERVER_URI = os.getenv("MLFLOW_SERVER_URI")
    mlflow.set_tracking_uri(MLFLOW_SERVER_URI)
    mlflow.enable_system_metrics_logging()
    logging.getLogger("mlflow.system_metrics.metrics.gpu_monitor").setLevel(logging.ERROR)  # Suppress GPU monitor logs
    mlflow.set_experiment(const.EXPERIMENT_NAME)



@hydra.main(config_path=os.path.join(os.path.dirname(__file__),"configs"), config_name="config.yaml", version_base=None)
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
