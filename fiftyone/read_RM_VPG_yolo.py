import fiftyone as fo


name = "RM_VPG_yolo"
# data_path = "D:\\Dataset\\RM_VPG_yoloDet"
# labels_path = "D:\\Dataset\\RM_VPG_yoloDet\\train.json"

dataset_dir = "/code/dataset/RM_VPG_TRoM_merge_yoloDet"
# dataset_dir = "/code/dataset//RM_VPG_yoloDet"

# The splits to load
splits = ["val"]

# Load the dataset, using tags to mark the samples in each split
dataset = fo.Dataset(name)
for split in splits:
    dataset.add_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        split=split,
        tags=split,
)


session = fo.launch_app(dataset)

session.wait()
