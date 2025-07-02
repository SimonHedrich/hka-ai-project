import constants
import mlflow
import torch
import torchvision
from configs.scheduler_schema import SchedulerConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

from configs.scheduler_schema import SchedulerConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

def inference(model, data_loader):
    X, y = next(iter(data_loader))  # get a validation batch

    model.eval()  # set the model in evaluation mode
    with torch.no_grad():  # do not compute gradients
        X = [x.to(constants.DEVICE) for x in X]  # move images to device
        return model(X)  # model forward pass


def faster_rcnn_model(num_classes):
    """return model and preprocessing transform"""
    # other faster rcnn models: https://pytorch.org/vision/stable/models/faster_rcnn.html
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    )
    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(
        in_features=model.roi_heads.box_predictor.cls_score.in_features,
        out_features=num_classes,
        bias=True,
    )
    model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(
        in_features=model.roi_heads.box_predictor.bbox_pred.in_features,
        out_features=num_classes * 4,
        bias=True,
    )
    model.to(constants.DEVICE)
    preprocess = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms()
    return model, preprocess


def model_optimizer(model):
    params = [p for p in model.parameters() if p.requires_grad]
    model_optimizer = torch.optim.SGD(
        params,
        lr=constants.LEARNING_RATE,
        momentum=constants.MOMENTUM,
        weight_decay=constants.WEIGHT_DECAY,
    )
    mlflow.log_param("start_learning_rate", constants.LEARNING_RATE)
    mlflow.log_param("momentum", constants.MOMENTUM)
    mlflow.log_param("weight_decay", constants.WEIGHT_DECAY)
    return model_optimizer


def model_scheduler(cfg: SchedulerConfig, optimizer):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["optimizer"] = optimizer
    return instantiate(cfg_dict)
