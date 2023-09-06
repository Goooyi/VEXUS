# %%
"""_summary_

Merget 2 datasets in YOLOv5 Fomat by
first align the index-names between 2 datasets
then save the images and labels in another directory

"""

import random
import os
import shutil
import yaml
import fileinput


# TODO parser
# TODO naming
ccSource_1 = "/code/dataset/RM_VPG_yoloDet"
ccSource_2 = "/code/yolov5/dataset/TRoM_yoloDetFO"

# load yaml file from ccSource_1 and ccSource_2
with open(os.path.join(ccSource_1, "dataset.yaml")) as f:
    yaml1 = yaml.safe_load(f)
with open(os.path.join(ccSource_2, "dataset.yaml")) as f:
    yaml2 = yaml.safe_load(f)

# %%
# TODO: avoid copy
# take ccSource_1 yolo dataset as baseline, ccSource_2's filed and index-name are mapped to ccSource_1(mannually check index-naming)
# copy all files and directories from ccSource_1 to outPath
outPath = "/code/dataset/RM_VPG_TRoM_merge_yoloDet"
# clear output path
if os.path.exists(outPath):
    shutil.rmtree(outPath)
print("copy first dataset to outPath")
shutil.copytree(ccSource_1, outPath, dirs_exist_ok=True)

# %%
# manually check index-names remapping
# TODO: choose the dataset with the most number of categories as base and check for duplicated namings
name2id = {
    "stop-line": "0",
    "arrow-left": "1",
    "arrow-right": "2",
    "arrow-go-straight": "3",
    "arrow-u-turn": "4",
    "across-walk": "5",
    "straight-plus-left": "6",
    "straight-plus-right": "7",
    "left-and-right": "8",
    "straight-plus-left" : "9",
    "straight-plus-right" : "10",
    "left-and-right" : "11"
}

id2name = {v: k for k, v in name2id.items()}


# delete the dataset.yaml in outPath

if os.path.exists(os.path.join(outPath, "dataset.yaml")):
    os.remove(os.path.join(outPath, "dataset.yaml"))

# create a new dataset.yaml in outPath
with open(os.path.join(outPath, "dataset.yaml"), "w") as f:
    f.write("names: \n")
    newMapping = [f"  {id}: {name}\n" for name, id in name2id.items()]
    mapppingStr = "".join(newMapping)
    f.write(mapppingStr)
    f.write(f"nc: {len(name2id)}\n")
    f.write(f"path: ../{outPath.split('/')[-1]}\n")
    f.write(f"train: ./images/train/\n")
    f.write(f"val: ./images/val/\n")
    # TODO test diff from val
    f.write(f"test: ./images/val/\n")

# %%
# correct the index-name mapping for the base
print("Correct the idx of base index")
for path, dirs, files in os.walk(os.path.join(outPath, "labels")):
    for file in files:
        # correct the label for each line
        newTextContent = ""
        with open(os.path.join(path, file), "r") as f:
            lines = f.readlines()
            # replate the first interger of each line with correct label index
            for line in lines:
                newTextContent += line.replace(line.split()[0], name2id[yaml1["names"][int(line.split()[0])]], 1) + "\n"

        with open(os.path.join(path, file), "w") as f:
            # clear the .txt file and replace it whith newTextContent
            f.write(newTextContent)


# %%
# correct the index-name mapping for another dataset
print("Copy 2nd dataset and Correct the idx of the second dataset")
for path, dirs, files in os.walk(os.path.join(ccSource_2, "labels")):
    for file in files:
        # correct the label for each line
        newTextContent = ""
        with open(os.path.join(path, file), "r") as f:
            lines = f.readlines()
            # replate the first interger of each line with correct label index
            for line in lines:
                newTextContent += line.replace(line.split()[0], name2id[yaml2["names"][int(line.split()[0])]], 1) + "\n"

        # check the total number of files in target image folder
        maxFileName = sorted(os.listdir(os.path.join(outPath,"labels",path.split("/")[-1])))[-1]
        newFileName = f"{int(maxFileName.split('.')[0])+1:06d}.txt"
        # save dataset2 labels
        with open(os.path.join(outPath, "labels", path.split("/")[-1], newFileName), "w") as f:
            # clear the .txt file and replace it whith newTextContent
            f.write(newTextContent)
        # save dataset2 images
        sourceImage = os.path.join(path.replace("labels", "images"), file.split(".")[0] + ".jpg")
        targetImage = os.path.join(outPath, f"images/{path.split('/')[-1]}/{newFileName.split('.txt')[0]}.jpg")
        # copy image
        shutil.copy(sourceImage, targetImage)
# %%
