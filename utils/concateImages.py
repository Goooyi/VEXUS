import os
import cv2
import numpy as np

dir1 = '/code/inference_file/python_n008-2018-08-28-16-16-48-0400__CAM_FRONT'
dir2 = '/code/inference_file/tfl_test_res/resolverDump'
dir3 = '/code/inference_file/tfl_test_res/trackedDump'

save_dir = '/code/inference_file/tfl_test_res/concat0'

image_names = os.listdir(dir1)

for name in image_names:
    print("img1 path: ", os.path.join(dir1, name))
    print("img2 path: ", os.path.join(dir2, name))
    print("img3 path: ", os.path.join(dir3, name))

    img1 = cv2.imread(os.path.join(dir1, name))
    img2 = cv2.imread(os.path.join(dir2, name))
    img3 = cv2.imread(os.path.join(dir3, name))

    print("img1 shape: ", img1.shape)
    print("img2 shape: ", img2.shape)
    print("img3 shape: ", img3.shape)

    concat_img = np.concatenate((img1, img2, img3), axis=0)

    cv2.imwrite(os.path.join(save_dir, name), concat_img)