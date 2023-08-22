# %%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# %%
listOfLabelFiles = list()
listOfLabelFilenames = list()
listOfImageFiles = list()
listOfImageFilenames = list()

for dirpath, dirnames, filenames in os.walk("/home/yi/dataset/TRoM/val/gt"):
    listOfLabelFiles += [os.path.join(dirpath, file) for file in filenames]
    listOfLabelFilenames += [file for file in filenames]

for dirpath, dirnames, filenames in os.walk("/home/yi/dataset/TRoM/val/image"):
    listOfImageFiles += [os.path.join(dirpath, file) for file in filenames]
    listOfImageFilenames += [file for file in filenames]

labelfiles = list(zip(listOfLabelFiles, listOfLabelFilenames))
imagefiles = list(zip(listOfImageFiles, listOfImageFilenames))
sorted(labelfiles)
sorted(imagefiles)

labelImg = cv2.imread(labelfiles[0][0])
imageImg = cv2.imread(imagefiles[0][0])

plt.xticks([]), plt.yticks([])
plt.imshow(labelImg)
plt.imshow(imageImg)
plt.show()

# %%
print(labelImg.shape)
print(np.unique(labelImg))
# %%
