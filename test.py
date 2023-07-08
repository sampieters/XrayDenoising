# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 15:57:24 2023

@author: Ben
"""

import numpy as np
from PIL import Image
from phantominator import shepp_logan
import matplotlib.pyplot as plt

def imwrite(matrix, path):
    l = Image.fromarray(matrix)
    l.save(path)

# Load the image
flatfield = Image.open("input/duplicate_testing/real_0/dbeer_5_5_0192.tif")
darkfield = Image.open("input/duplicate_testing/real_0/dbeer_5_5_0001.tif")

phantom = shepp_logan(np.shape(flatfield))

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(flatfield)
plt.subplot(1, 2, 2)
plt.imshow(phantom)


flatfield = np.array(flatfield)
flatfield = flatfield / (2 ** 16 - 1)

darkfield = np.array(darkfield)
darkfield = darkfield / (2 ** 16 - 1)

image_bad = np.exp(-phantom) * flatfield + darkfield
#image_bad = phantom * flatfield
#scaleOutputImages = [0, 2]
#image_bad = (image_bad - scaleOutputImages[0]) / (scaleOutputImages[1] - scaleOutputImages[0])


image_bad = np.round((2 ** 16 - 1) * image_bad).astype(np.uint16)
plt.figure()
plt.imshow(image_bad)

imwrite(phantom, "phantom.tif")
imwrite(image_bad, "input/duplicate_testing/real_0/dbeer_5_5_0321.tif")
