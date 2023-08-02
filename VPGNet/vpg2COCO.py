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
from utils.mask2coco import *

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert VPGNet format to COCO format for instance segmentation"
    )

    parser.add_argument(
        "--output", type=str, help="output path of gathered metrics to be stored"
    )
    parser.add_argument("--input", type=str, help="VPGNet data path")
    args = parser.parse_args()
    return args


# Label ids of the dataset
category_ids = {
    "stop-line": 9,
    "arrow-left": 10,
    "arrow-right": 11,
    "arrow-go-straight": 12,
    "arrow-u-turn": 13,
    "across-walk": 15,
}

id_to_category = {
    9: "stop-line",
    10: "arrow-left",
    11: "arrow-right",
    12: "arrow-go-straight",
    13: "arrow-u-turn",
    15: "across-walk",
}

# TODO: change to use unified distributed HSV format
# Define which colors match which categories in the images
category_colors = {
    "(255, 0, 0)": 9,  # stop-line
    "(255, 255, 0)": 10,  # arrow-left
    "(128, 0, 255)": 11,  # arrow-right
    "(255, 128, 0)": 12,  # arrow-go-straight
    "(0, 0, 255)": 13,  # arrow-u-turn
    "(0, 255, 0)": 15,  # across-walk
}

def convert_vpgnet_to_coco(dataSetDir, out_path, iscrowd=0):
    # load VPGNet orignal images and annos
    listOfFiles = list()
    listOfFilenames = list()
    for dirpath, dirnames, filenames in os.walk(dataSetDir):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
        listOfFilenames += [file for file in filenames]
    files = list(zip(listOfFiles, listOfFilenames))
    sorted(files)

    # set VPGNet train/val split to 9:1
    np.random.seed(42)
    rnd = np.random.rand(1, len(files))
    train_idx, test_idx = (rnd <= 0.9).nonzero()[1], (rnd > 0.9).nonzero()[1]
    # count scene scenarios numbers in train/test
    scene_count = {
        "train_scene_1": 0,
        "train_scene_2": 0,
        "train_scene_3": 0,
        "train_scene_4": 0,
        "test_scene_1": 0,
        "test_scene_2": 0,
        "test_scene_3": 0,
        "test_scene_4": 0,
    }
    category_image_count = {
        "train_stop-line": 0,
        "train_arrow-left": 0,
        "train_arrow-right": 0,
        "train_arrow-go-straight": 0,
        "train_arrow-u-turn": 0,
        "train_across-walk": 0,
        "test_stop-line": 0,
        "test_arrow-left": 0,
        "test_arrow-right": 0,
        "test_arrow-go-straight": 0,
        "test_arrow-u-turn": 0,
        "test_across-walk": 0,
    }
    category_instance_count = {
        "train_stop-line": 0,
        "train_arrow-left": 0,
        "train_arrow-right": 0,
        "train_arrow-go-straight": 0,
        "train_arrow-u-turn": 0,
        "train_across-walk": 0,
        "test_stop-line": 0,
        "test_arrow-left": 0,
        "test_arrow-right": 0,
        "test_arrow-go-straight": 0,
        "test_arrow-u-turn": 0,
        "test_across-walk": 0,
    }
    multipoly_category = set()
    multipoly_img_count = 0
    annotation_id = 0
    image_id = 0
    annotations = []
    images = []
    for split in [("train", train_idx), ("test", test_idx)]:
        # create folder for RGB images of VPGNet
        out_image_path = os.path.join(out_path, split[0] + "/images")
        if not os.path.exists(out_image_path):
            os.makedirs(out_image_path)
        # create coco_format for train and test split
        coco_format = get_coco_json_format()
        # create anno section
        coco_format["categories"] = create_category_annotation(category_ids)
        for img_idx in tqdm(split[1]):
            # if img_idx > 50:
            #     continue
            data = scipy.io.loadmat(files[img_idx][0])
            rgb_seg_vp = data["rgb_seg_vp"]
            rgb = rgb_seg_vp[:, :, 0:3]
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            seg = rgb_seg_vp[:, :, 3]
            # skip mask that does not conatin needed categories
            contain_categories = np.unique(seg).tolist()
            # check if contain categories listed by id_to_category.keys()
            maskedAll = list(set(contain_categories) & set(id_to_category.keys()))
            if len(maskedAll) == 0:
                continue

            scene = files[img_idx][0].split("/")[-3]
            scene_count[split[0] + "_" + scene] += 1
            w, h, _ = rgb.shape
            # save RGB image
            cv2.imwrite(os.path.join(out_image_path, f"{image_id:06d}" + ".jpg"), rgb)
            # create image info, scene info are inserted here!!!!!!
            image_anno = create_image_annotation(
                # files[img_idx][1], w, h, f"{image_id:06d}", scene
                f"{image_id:06d}" + ".jpg", w, h, f"{image_id:06d}", scene
            )
            images.append(image_anno)

            # according to (https://cocodataset.org/#format-data) iscrowd=0 instance segmentation, iscrows=1 is semantic segmentation?
            if iscrowd == 0:
                # find all instances for each category, n
                multipoly_marker = 0
                for category_id in maskedAll:
                    # here equvilant to create_sub_mask() function
                    # update category count in train/test split
                    category_image_count[
                        split[0] + "_" + id_to_category[category_id]
                    ] += 1
                    mask = (np.uint8(seg == category_id)).tolist()
                    # pad mask in both direction with 1, cause measure.findcontour doesnt work for bleeding edge
                    mask = np.pad(
                        mask, ((1, 1), (1, 1)), "constant", constant_values=0
                    ).tolist()
                    # TODO: paddings are subtracted inside "create_sub_mask_annotation() function
                    polygons, segmentations, is_multipoly = create_sub_mask_annotation(
                        mask
                    )

                    category_instance_count[
                        split[0] + "_" + id_to_category[category_id]
                    ] += len(polygons)

                    if is_multipoly:
                        multipoly_category.add(id_to_category[category_id])
                        # TODO
                        if multipoly_marker == 0:
                            multipoly_img_count += 1
                            multipoly_marker = 1

                    for i in range(len(polygons)):
                        # # Cleaner to recalculate this variable
                        # segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]

                        annotation = create_annotation_format(
                            polygons[i],
                            segmentations[i],
                            f"{image_id:06d}",
                            category_id,
                            annotation_id,
                        )

                        annotations.append(annotation)
                        annotation_id += 1
            else:
                # TODO
                raise Exception("iscrowd==1 not implemented yet")

            image_id += 1

        coco_format["images"], coco_format["annotations"], annotation_cnt = (
            images,
            annotations,
            annotation_id,
        )
        out_file = os.path.join(out_path, f"{split[0]}.json")
        with open(out_file, "w") as f:
            json.dump(coco_format, f)
        print(
            "Created %d annotations for %s images in folder: %s"
            % (annotation_cnt, split[0], out_path)
        )

    print("----------------------")
    print(f"scene count: \n {scene_count} \n")
    print(f"Image count for each category: \n {category_image_count} \n")
    print(f"Instance count for each categoriy: \n {category_instance_count} \n")
    print(f"categories have multipoly: \n {multipoly_category}")
    print(f"Total multipoly count: \n {multipoly_img_count}")


if __name__ == "__main__":
    args = parse_args()
    convert_vpgnet_to_coco(args.input, args.output, 0)
