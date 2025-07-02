import logging
import os
import time

import matplotlib.pyplot as plt
import mlflow
import torch
import torch.optim.optimizer
from constants import DEVICE
from evaluation import eval_log_mlflow, evaluate
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(
        self,
        dataloader_train,
        dataloader_valid,
        dataloader_eval,
        detection_model,
        epoch_count,
        model_optimizer,
        model_scheduler: torch.optim.lr_scheduler.MultiStepLR,
        model_export_path,
        auto_stop_threshold,
    ):
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid
        self.dataloader_eval = dataloader_eval
        self.detection_model = detection_model
        self.epoch_count = epoch_count
        self.model_optimizer = model_optimizer
        self.model_scheduler = model_scheduler
        self.model_export_path = model_export_path
        self.auto_stop_threshold = auto_stop_threshold

    def run_pipeline(self):
        train_loss_list = []
        validation_loss_list = []
        self.detection_model.train()  # set model in training mode
        mlflow.log_param("EPOCH_COUNT", self.epoch_count)
        keep_training = True
        last_loss = 100
        epoch = 0
        while keep_training:
            epoch += 1
            epoch_training_start_time = time.time()
            logger.info(f"Starting training for epoch {(epoch)} ...")
            N = len(self.dataloader_train.dataset)
            current_train_loss = 0
            # train loop
            for images, targets in tqdm(self.dataloader_train, desc="train"):
                # move data to device and build the right input format for our model
                images = list(image.to(DEVICE) for image in images)
                targets = [
                    {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets
                ]

                loss_dict = self.detection_model(images, targets)
                losses = sum(loss for loss in loss_dict.values())  # <class 'torch.Tensor'>
                self.model_optimizer.zero_grad()
                losses.backward()
                self.model_optimizer.step()

                current_train_loss += losses
            train_loss_list.append(current_train_loss / N)
            mlflow.log_metric("train loss", train_loss_list[-1], step=epoch)

            epoch_training_end_time = time.time()
            epoch_training_duration = epoch_training_end_time - epoch_training_start_time
            mlflow.log_metric("training time / epoch", epoch_training_duration, step=epoch)
            logger.info(
                f"Epoch {(epoch)}: train_loss: {train_loss_list[-1]:.5f} (t_train: {epoch_training_duration:5.2f}s)"
            )

            # validation loop
            epoch_validation_start_time = time.time()
            N = len(self.dataloader_valid.dataset)
            current_validation_loss = 0
            with torch.no_grad():
                for images, targets in tqdm(self.dataloader_valid, desc="val"):
                    images = list(image.to(DEVICE) for image in images)
                    targets = [
                        {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets
                    ]
                    loss_dict = self.detection_model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    current_validation_loss += losses
            validation_loss_list.append(current_validation_loss / N)

            output_info = f"Epoch {(epoch)}: "
            mlflow.log_metric("Validation loss", validation_loss_list[-1], step=epoch)
            output_info += f"val_loss: {validation_loss_list[-1]:.5f}"
            epoch_validation_end_time = time.time()
            epoch_validation_duration = epoch_validation_end_time - epoch_validation_start_time
            mlflow.log_metric("validation time / epoch", epoch_validation_duration, step=epoch)
            if self.model_scheduler:
                new_learning_rate = self.model_scheduler.get_last_lr()[0]
                mlflow.log_metric("learning rate", new_learning_rate, step=epoch)
                output_info += f" - l_rate: {new_learning_rate:.6f}"
                self.model_scheduler.step()
            loss_difference = last_loss - current_validation_loss
            output_info += f" - loss_delta: {loss_difference:.5f}"
            last_loss = current_validation_loss
            keep_training = epoch < self.epoch_count or loss_difference < self.auto_stop_threshold
            output_info += f" (t_val: {epoch_validation_duration:5.2f}s)"
            logger.info(output_info)

        logger.info(f"Training finished with {epoch} epochs")
        coco_eval = evaluate(self.detection_model, self.dataloader_eval)
        eval_log_mlflow(coco_eval)

        model_export_dir = os.path.dirname(self.model_export_path)
        model_name = os.path.basename(model_export_dir)
        if not os.path.exists(model_export_dir):
            os.mkdir(model_export_dir)
        # torch.save(
        #     self.detection_model, self.model_export_path
        # )
        mlflow.pytorch.log_model(pytorch_model=self.detection_model, artifact_path=model_name)

        # plot losses
        train_loss = [x.cpu().detach().numpy() for x in train_loss_list]
        validation_loss = [x.cpu().detach().numpy() for x in validation_loss_list]

        plt.plot(train_loss, "-o", label="train loss")
        plt.plot(validation_loss, "-o", label="validation loss")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        mlflow.log_figure(plt.gcf(), artifact_file="loss_graph.png")
