import mlflow
import torch
from constants import DEVICE
from dataset import Dataloader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


@torch.no_grad()
def evaluate(model, data_loader: Dataloader):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model.eval()

    coco_gt = COCO()
    coco_dataset = data_loader.dataset.coco_data
    image_ids = [image_data["id"] for image_data in coco_dataset["images"]]
    coco_gt.dataset = coco_dataset
    coco_gt.createIndex()

    results = []
    image_count = 0
    for images, _ in data_loader:
        images = list(img.to(DEVICE) for img in images)
        outputs = model(images)

        for idx, output in enumerate(outputs):
            image_id = image_ids[image_count]
            image_count += 1

            # Extract data from the model output
            boxes = output["boxes"].detach().cpu().numpy()
            labels = output["labels"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()

            for box, label, score in zip(boxes, labels, scores):
                x_min, y_min, x_max, y_max = box
                bbox = [
                    float(x_min),
                    float(y_min),
                    float(x_max - x_min),  # Width
                    float(y_max - y_min),  # Height
                ]
                result = {
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": bbox,
                    "score": float(score),
                }
                results.append(result)

    coco_dt = coco_gt.loadRes(results)

    # running evaluation
    ann_type = "bbox"
    coco_eval = COCOeval(coco_gt, coco_dt, ann_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    torch.set_num_threads(n_threads)
    return coco_eval


def eval_log_mlflow(coco_eval: COCOeval):
    ap_iou_50_95 = coco_eval.stats[0]
    ap_iou_50 = coco_eval.stats[1]
    ap_iou_75 = coco_eval.stats[2]
    ap_small = coco_eval.stats[3]
    ap_medium = coco_eval.stats[4]
    ap_large = coco_eval.stats[5]
    ar_max_1 = coco_eval.stats[6]
    ar_max_10 = coco_eval.stats[7]
    ar_max_100 = coco_eval.stats[8]
    ar_small = coco_eval.stats[9]
    ar_medium = coco_eval.stats[10]
    ar_large = coco_eval.stats[11]

    mlflow.log_table(
        {
            "evaluation metric": [
                "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
                "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]",
                "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]",
                "Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
                "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
                "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]",
                "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]",
                "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]",
                "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
                "Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
                "Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
                "Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]",
            ],
            "value": [
                ap_iou_50_95,
                ap_iou_50,
                ap_iou_75,
                ap_small,
                ap_medium,
                ap_large,
                ar_max_1,
                ar_max_10,
                ar_max_100,
                ar_small,
                ar_medium,
                ar_large,
            ],
        },
        artifact_file="coco_evaluation.json",
    )

    mlflow.log_text(
        (
            f"Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {ap_iou_50_95:2.3f}\n"
            f"Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {ap_iou_50:2.3f}\n"
            f"Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {ap_iou_75:2.3f}\n"
            f"Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {ap_small:2.3f}\n"
            f"Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {ap_medium:2.3f}\n"
            f"Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {ap_large:2.3f}\n"
            f"Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {ar_max_1:2.3f}\n"
            f"Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {ar_max_10:2.3f}\n"
            f"Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {ar_max_100:2.3f}\n"
            f"Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {ar_small:2.3f}\n"
            f"Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {ar_medium:2.3f}\n"
            f"Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {ar_large:2.3f}"
        ),
        artifact_file="coco_evaluation.txt",
    )

    # mlflow.log_metric("AP - IoU_0.50/0.95 - All", ap_iou_50_95)
    # mlflow.log_metric("AP - IoU_0.50 - All", ap_iou_50)
    # mlflow.log_metric("AP - IoU_0.75 - All", ap_iou_75)
    # mlflow.log_metric("AP - IoU_0.50/0.95 - Small", ap_small)
    # mlflow.log_metric("AP - IoU_0.50/0.95 - Medium", ap_medium)
    # mlflow.log_metric("AP - IoU_0.50/0.95 - Large", ap_large)
    # mlflow.log_metric("AR - MaxDets_1", ar_max_1)
    # mlflow.log_metric("AR - MaxDets_10", ar_max_10)
    # mlflow.log_metric("AR - MaxDets_100", ar_max_100)
    # mlflow.log_metric("AR - MaxDets_100 - Small", ar_small)
    # mlflow.log_metric("AR - MaxDets_100 - Medium", ar_medium)
    # mlflow.log_metric("AR - MaxDets_100 - Large", ar_large)
