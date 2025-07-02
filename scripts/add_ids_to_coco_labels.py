import json
import os

from traffic_detection_project.source.constants import SOURCE_DIR

INPUT_FILE = os.path.join(SOURCE_DIR, "datasets", "coco_labels.txt")
OUTPUT_FILE = os.path.join(SOURCE_DIR, "datasets", "coco_labels_with_ids.json")

mapping = {}
with open(INPUT_FILE, "r", encoding="utf-8") as input_file_handler:
    for idx, line in enumerate(input_file_handler, start=1):
        mapping[line.rstrip("\n")] = idx

with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file_handler:
    json.dump(mapping, output_file_handler, indent=4)
