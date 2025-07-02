import json
import os
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset


class COCODataset(Dataset):  # , FileSystemDatasetSource):
    """PyTorch dataset for COCO annotations."""

    # adapted from https://github.com/pytorch/vision/issues/2720

    def __init__(self, image_dir_path, coco_file_path, transform=None):
        """Load COCO annotation data."""
        self.image_dir_path = image_dir_path
        self.coco_file_path = coco_file_path
        self.transform = transform

        # load the COCO annotations json
        with open(coco_file_path) as file_obj:
            self.coco_data = json.load(file_obj)

        # put all of the annos into a dict where keys are image IDs to speed up retrieval
        latest_annot_id = 0
        self.image_id_to_annos = defaultdict(list)
        for anno in self.coco_data["annotations"]:
            image_id = anno["image_id"]
            self.image_id_to_annos[image_id] += [anno]
            latest_annot_id = anno["id"]

        # fill all images without annotation
        for image in self.coco_data["images"]:
            image_id = image["id"]
            if not self.image_id_to_annos[image_id]:
                width = image["width"]
                height = image["height"]
                latest_annot_id += 1
                self.image_id_to_annos[image_id] = [
                    {
                        "id": latest_annot_id,
                        "image_id": image_id,
                        "category_id": 0,
                        "bbox": [0, 0, width, height],
                        "area": 0,
                        "iscrowd": 0,
                    }
                ]

    def __len__(self):
        return len(self.coco_data["images"])

    def __getitem__(self, index):
        """Return tuple of image and labels as torch tensors."""
        image_data = self.coco_data["images"][index]
        image_id = image_data["id"]
        image_path = os.path.join(self.image_dir_path, image_data["file_name"])
        image = Image.open(image_path).convert("RGB")

        annos = self.image_id_to_annos[image_id]
        anno_data = {
            "boxes": [],
            "labels": [],
            "area": [],
            "iscrowd": [],
        }
        for anno in annos:
            coco_bbox = anno["bbox"]
            left = coco_bbox[0]
            top = coco_bbox[1]
            right = coco_bbox[0] + coco_bbox[2]
            bottom = coco_bbox[1] + coco_bbox[3]
            area = coco_bbox[2] * coco_bbox[3]
            anno_data["boxes"].append([left, top, right, bottom])
            anno_data["labels"].append(anno["category_id"])
            anno_data["area"].append(area)
            if "iscrowd" in anno:
                anno_data["iscrowd"].append(anno["iscrowd"])

        target = {
            "boxes": torch.as_tensor(anno_data["boxes"], dtype=torch.float32),
            "labels": torch.as_tensor(anno_data["labels"], dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
            "area": torch.as_tensor(anno_data["area"], dtype=torch.float32),
            "iscrowd": torch.as_tensor(anno_data["iscrowd"], dtype=torch.int64),
        }

        if self.transform is not None:
            image = self.transform(image)

        return image, target


def collate(batch):
    """return tuple data"""
    return tuple(zip(*batch))


class Dataloader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        collate_fn=collate,
    ):
        self.dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    def get_dataloader(self):
        return self.dataloader
