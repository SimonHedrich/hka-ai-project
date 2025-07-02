import constants as const
import mlflow
import torch
import torchvision

from configs.scheduler_schema import SchedulerConfig
from configs.optimizer_schema import OptimizerConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pydantic import TypeAdapter
from helper import instantiate_target

def inference(model, data_loader):
    X, y = next(iter(data_loader))  # get a validation batch

    model.eval()  # set the model in evaluation mode
    with torch.no_grad():  # do not compute gradients
        X = [x.to(const.DEVICE) for x in X]  # move images to device
        return model(X)  # model forward pass


def ssd_lite_model():
    """return model and preprocessing transform"""
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    )
    model.to(const.DEVICE)
    preprocess = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1.transforms()
    return model, preprocess


def model_optimizer(cfg_optimizer, model):
    params = [p for p in model.parameters() if p.requires_grad]

    cfg_dict_optimizer = OmegaConf.to_container(cfg_optimizer, resolve=True)
    optimizer_config = TypeAdapter(OptimizerConfig).validate_python(cfg_dict_optimizer)
    optimizer_config.params = params
    print(type(optimizer_config))

    mlflow.log_param("start_learning_rate", const.LEARNING_RATE)
    mlflow.log_param("momentum", const.MOMENTUM)
    mlflow.log_param("weight_decay", const.WEIGHT_DECAY)

    return instantiate_target.instantiate_from_target(optimizer_config)


def model_scheduler(cfg_scheduler, optimizer):
    cfg_dict_scheduler = OmegaConf.to_container(cfg_scheduler, resolve=True)
    scheduler_config = TypeAdapter(SchedulerConfig).validate_python(cfg_dict_scheduler)
    scheduler_config.optimizer = optimizer
    print(type(scheduler_config.optimizer))
    print(scheduler_config)
    return instantiate_target.instantiate_from_target(scheduler_config)

