"""Convert VPGNet format to COCO format for instance segmentation.

Usage:
    python vpg2COCO.py --input <VPGNet data path> --output <output path of gathered metrics to be stored>
    Only arrows/stop-line/across-walk are considered as instance segmentation, other categories are ignored for now.
    Images without annotations are filted out since normally usually there are more negative samples than positive samples.
    Segmentation saved in polygon format for each instance. To save RLE format segmentation, please refer to :https://github.com/nightrome/cocostuffapi/blob/master/PythonAPI/cocostuff/pngToCocoResultDemo.py

Note:
    1. This file is modified from yolov5 repo

"""

import os.path as osp
import argparse
import json
import os
import argparse
import scipy.io
import numpy as np
import cv2
from tqdm import tqdm

import sys
# add parent path to sys.path to import utils
sys.path.append(osp.abspath(osp.join(__file__, "../../")))
from utils.yolov5seg2yolov5det import *


# read the .txt fiels of yolov5 segmentation format and convert it to yolov5 detection format
def yolov5seg2det(input_folder, output_folder):
    output_folder = osp.join(output_folder, "labels_det")
    # make folder for output_folder
    if not osp.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        if file.endswith(".txt"):
            file_name = file[:-4]
            with open(osp.join(input_folder, file), "r") as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
                lines = [seg_to_bbox(line) for line in lines]
                with open(osp.join(output_folder, file_name + ".txt"), "w") as f:
                    for line in lines:
                        f.write(line + "\n")

if __name__ == "__main__":
    input_folder = "/home/yi/dev/VEXUS/RM_VPG_yolo/test/labels"
    output_folder = "/home/yi/dev/VEXUS/RM_VPG_yolo/test"
    yolov5seg2det(input_folder, output_folder)