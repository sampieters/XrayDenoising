import os
import numpy as np
from PIL import Image as im

#################################################
# PARAMETERS
#################################################

# Directory with raw dark fields, flat fields and projections in .tif format
readDIR = './input/'
# Directory for the output files
outDIR = './output/'

# file names
prefixProj =         'dbeer_5_5_'   # prefix of the original projections
outPrefixFFC =       'FFC'          # prefix of the CONVENTIONAL flat field corrected projections
prefixFlat =         'dbeer_5_5_'   # prefix of the flat fields
prefixDark =         'dbeer_5_5_'   # prefix of the dark fields
numType =            '04d'         # number type used in image names
fileFormat =         '.tif'         # image format

nrDark =             20             # number of dark fields
firstDark =          1              # image number of first dark field
nrWhitePrior =       300            # number of white (flat) fields BEFORE acquiring the projections
firstWhitePrior =    21             # image number of first prior flat field
nrWhitePost =        300            # number of white (flat) fields AFTER acquiring the projections
firstWhitePost =     572            # image number of first post flat field
nrProj =             50        	    # number of acquired projections
firstProj =          321            # image number of first projection

# options output images
scaleOutputImages =  [0, 2]         # output images are scaled between these values

#################################################
# CODE
#################################################
def imread(path):
    image = im.open(path)
    return np.asarray(image)

def imwrite(matrix, path):
    im.fromarray(matrix).save(path)

if __name__ == '__main__':
    # Make the output directories
    if not os.path.exists(outDIR):
        os.mkdir(outDIR)

    # Get a list of all projection indices and get the dimensions of a .tif image (because they are all the same,
    # get the dimensions of the first image)
    nrImage = np.arange(firstProj, firstProj+nrProj)
    print("Load dark and flat fields")
    tmp = imread(readDIR + prefixProj + f'{1:{numType}}' + fileFormat)
    dims = tmp.shape

    # Make an m*n*p matrix to store all the dark fields and get the mean value
    print("Load dark fields...")
    dark = np.zeros((nrDark, dims[0], dims[1]))
    for i in range(firstDark - 1, firstDark + nrDark - 1):
        dark[:][:][i] = imread(readDIR + prefixProj + f'{i+1:{numType}}' + fileFormat)
    # Get the mean (sum of all elements divided by the dimensions)
    meanDarkfield = np.mean(dark, 0)

    print("Load one flat field...")
    f_j = imread(readDIR + prefixFlat + f'{firstWhitePrior:{numType}}' + fileFormat)

    print("Get the \"perfect\" image...")
    n_j = imread("./output/DFFC0321.tif")
    print("Calculate noisy image...")

    p_j = n_j * (f_j - meanDarkfield) + meanDarkfield
    p_j[p_j < 0] = 0
    p_j = -np.log(p_j)
    p_j[np.isinf(p_j)] = 10 ** 5
    p_j = (p_j - scaleOutputImages[0]) / (scaleOutputImages[1] - scaleOutputImages[0])
    #p_j = (p_j - scaleOutputImages[1]) / (scaleOutputImages[0] - scaleOutputImages[1])
    p_j = np.round((2 ** 16 - 1) * p_j).astype(np.uint16)
    imwrite(p_j, "./simulate_noisy.tif")