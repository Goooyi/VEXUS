import fiftyone as fo


name = "RM_VPG_yolo"
# data_path = "D:\\Dataset\\RM_VPG_yoloDet"
# labels_path = "D:\\Dataset\\RM_VPG_yoloDet\\train.json"

dataset_dir = "D:\\Dataset\\RM_VPG_yoloDet"

# # Import dataset by explicitly providing paths to the source media and labels
# dataset = fo.Dataset.from_dir(
#     dataset_type=fo.types.COCODetectionDataset,
#     data_path=data_path,
#     labels_path=labels_path,
#     name=name,
# )

# The splits to load
splits = ["train"]

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
