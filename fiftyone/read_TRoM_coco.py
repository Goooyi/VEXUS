import fiftyone as fo


name = "TRoM_coco"
data_path = "/home/yi/dataset/TRoM_coco"
labels_path = "/home/yi/dataset/TRoM_coco/val.json"
split = labels_path.split("/")[-1].split(".")[0]


# Import dataset by explicitly providing paths to the source media and labels
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=data_path,
    labels_path=labels_path,
    name=name,
    # fields = 'detections'
)

# Export the dataset
export_dir = "/home/yi/dataset/TRoM_yoloDetFO/"
dataset.export(
    export_dir=export_dir,
    # data_path=export_dir,
    dataset_type=fo.types.YOLOv5Dataset,
    classes=[
        "stop-line",
        "arrow-left",
        "arrow-right",
        "arrow-go-straight",
        "across-walk",
        "straight-plus-left",
        "straight-plus-right",
        "left-and-right",
    ],
    split=split,
)

session = fo.launch_app(dataset)

session.wait()
