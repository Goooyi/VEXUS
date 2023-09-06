import cv2
import numpy as np
import glob

img_array = []
for filename in glob.glob('/code/inference_file/tfl_test_res/concat0/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


# TODO video len?
out = cv2.VideoWriter('/code/inference_file/tfl_test_res/concat0.avi',cv2.VideoWriter_fourcc(*'DIVX'), 1, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()