from PIL import Image as im
import numpy as np

def imread(path):
    image = im.open(path)
    image = np.asarray(image)
    image = image.astype(np.double)
    return image

def imwrite(matrix, type, path):
    matrix = type(np.round((np.iinfo(type).max * matrix)))
    image = im.fromarray(matrix)
    image.save(path)
