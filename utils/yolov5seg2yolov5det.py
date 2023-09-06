def seg_to_bbox(seg_info):
    # Example input: 5 0.046875 0.369141 0.0644531 0.384766 0.0800781 0.402344 ...
    class_id, *points = seg_info.split()
    points = [float(p) for p in points]
    x_min, y_min, x_max, y_max = min(points[0::2]), min(points[1::2]), max(points[0::2]), max(points[1::2])
    width, height = x_max - x_min, y_max - y_min
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    bbox_info = f"{int(class_id)} {x_center} {y_center} {width} {height}"
    return bbox_info