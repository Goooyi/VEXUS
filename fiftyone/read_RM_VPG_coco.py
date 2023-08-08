import fiftyone as fo


name = "RM_VPG_yolo_test"
data_path = "D:\\Dataset\\RM_VPG_COCO"
labels_path = "D:\\Dataset\\RM_VPG_COCO\\test.json"
split = labels_path.split("\\")[-1].split(".")[0]


# Import dataset by explicitly providing paths to the source media and labels
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=data_path,
    labels_path=labels_path,
    name=name,
    # fields = 'detections'
)

# Export the dataset
export_dir = "D:\\Dataset\\RM_VPG_yoloDetFO\\"
dataset.export(
    export_dir=export_dir,
    dataset_type=fo.types.YOLOv5Dataset,
    classes=[
        "stop-line",
        "arrow-left",
        "arrow-right",
        "arrow-go-straight",
        "arrow-u-turn",
        "across-walk",
    ],
    split=split,
)

# session = fo.launch_app(dataset)

# session.wait()
