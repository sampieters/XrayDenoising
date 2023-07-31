import os
import numpy as np
from PIL import Image as im

#################################################
# PARAMETERS
#################################################

# Directory with raw dark fields, flat fields and projections in .tif format
readDIR = '../../input/real/'
# Directory for the output files
outDIR = '../../output/FFC/Python/'

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

def ConventionalFlatFieldCorrection():
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

    # Make an m*n*p matrix to store all the flat fields and get the mean value
    # The algorithm combines the prior and post flat fields and gets the mean from this combination
    print('Load white fields...')
    flat = np.zeros((nrWhitePrior + nrWhitePost, dims[0] * dims[1]))
    k = 0
    for i in range(firstWhitePrior, firstWhitePrior + nrWhitePrior):
        tmp = imread(readDIR + prefixFlat + f'{i:{numType}}' + fileFormat) - meanDarkfield
        flat[:][k] = np.reshape(tmp, -1, 'F') - np.reshape(meanDarkfield, -1, 'F')
        k += 1
    for i in range(firstWhitePost, firstWhitePost + nrWhitePost):
        tmp = imread(readDIR + prefixFlat + f'{i:{numType}}' + fileFormat) - meanDarkfield
        flat[:][k] = np.reshape(tmp, -1, 'F') - np.reshape(meanDarkfield, -1, 'F')
        k += 1
    mn = np.mean(flat, 0)

    # subtract mean flat field
    N, M = flat.shape
    Data = (flat - np.tile(mn, (N, 1))).transpose()
    eig0 = np.reshape(mn, dims, order='F')
    del dark, flat, Data

    for i in range(1, len(nrImage)+1):
        print(f'Conventional FFC: {str(i)}/{str(len(nrImage))}...')
        # Load projection
        projection = imread(readDIR + prefixProj + f'{nrImage[i-1]:{numType}}' + fileFormat)
        # used to be np.divide instead of '/', check if this does the same
        tmp = (np.squeeze(projection) - meanDarkfield) / eig0

        tmp[tmp < 0] = 0
        tmp = -np.log(tmp)
        # TODO: maybe there is a one line fix but not sure yet
        tmp[tmp < 0] = 0
        tmp[np.isinf(tmp)] = 10 ** 5

        tmp = (tmp - scaleOutputImages[0]) / (scaleOutputImages[1] - scaleOutputImages[0])
        tmp = np.uint16(np.round((2 ** 16 - 1) * tmp))
        imwrite(tmp, outDIR + outPrefixFFC + f'{nrImage[i - 1]:{numType}}' + fileFormat)