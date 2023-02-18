import os
import bm3d
import condTVmean
import numpy as np
import parallelAnalysis
from PIL import Image as im

#################################################
# PARAMETERS
#################################################

# Directory with raw dark fields, flat fields and projections in .tif format
readDIR = '../../input/'
# Directory for the output files
outDIR = '../../output/'
# Directory where the DYNAMIC flat field corrected projections are saved
outDIRDFFC = './DFFC/'
# Directory where the CONVENTIONAL flat field corrected projections are saved
outDIRFFC = './FFC/'

# file names
prefixProj =         'dbeer_5_5_'   # prefix of the original projections
outPrefixDFFC =      'DFFC'         # prefix of the DYNAMIC flat field corrected projections
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

# algorithm parameters
downsample =         2              # amount of downsampling during dynamic flat field estimation (integer between 1 and 20)
nrPArepetions =      10             # number of parallel analysis repetitions

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
    if not os.path.exists(outDIRFFC):
        os.mkdir(outDIRDFFC)
    if not os.path.exists(outDIR):
        os.mkdir(outDIR)
    if not os.path.exists(outDIRFFC):
        os.mkdir(outDIRFFC)

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
    Data = flat - np.tile(mn, (N, 1))
    Data = Data.transpose()
    del dark, flat

    print("Parallel Analysis:")
    V1, D1, nrEigenflatfields = parallelAnalysis.parallelAnalysis(Data, nrPArepetions)
    print(f"{str(nrEigenflatfields)} eigen flat fields selected.")

    eig0 = np.reshape(mn, dims, order='F')
    EigenFlatfields = np.zeros((nrEigenflatfields+1, eig0.shape[0], eig0.shape[1]))
    EigenFlatfields[:][:][0] = eig0
    # TODO: There is happening something wrong in these two loops somewhere
    # TODO: In the first loop the matrix signs is inverted
    for i in range(0, nrEigenflatfields):
        print(f"HELP {N-i}")
        whut = np.reshape(np.matmul(Data, V1[:, N-i-1]), dims, order='F')
        EigenFlatfields[:][:][i+1] = whut
    del Data

    print("Filter eigen flat fields...")
    filteredEigenFlatfields = np.zeros((nrEigenflatfields+1, dims[0], dims[1]))

    for i in range(1, nrEigenflatfields+1):
        print(f'Filter eigen flat field {str(i)}')
        min = np.min(EigenFlatfields[:][:][i])
        max = np.max(EigenFlatfields[:][:][i])
        tmp = (EigenFlatfields[:][:][i] - min) / (max - min)
        tmp2 = bm3d.bm3d(tmp, 1)
        filteredEigenFlatfields[:][:][i] = (tmp2 * max) - min + min

    meanVector = np.zeros(len(nrImage))

    for i in range(1, len(nrImage)+1):
        print(f'Conventional FFC: {str(i)}/{str(len(nrImage))}...')
        # Load projection
        projection = imread(readDIR + prefixProj + f'{nrImage[i-1]:{numType}}' + fileFormat)

        tmp = np.divide((np.squeeze(projection) - meanDarkfield), EigenFlatfields[:][:][0])
        meanVector[i-1] = np.mean(tmp[:])

        # TODO: Seems redundant
        for x in range(0, tmp.shape[0]):
            for y in range(0, tmp.shape[1]):
                if tmp[x][y] < 0:
                    tmp[x][y] = 0

        tmp = -np.log(tmp)

        #TODO: after the log it fixes the bug but before it does nothing
        for x in range(0, tmp.shape[0]):
            for y in range(0, tmp.shape[1]):
                if tmp[x][y] < 0:
                    tmp[x][y] = 0

        # TODO: Seems redundant, the provided dataset has no case for this
        hulp = np.isinf(tmp)
        for x in range(0, tmp.shape[0]):
            for y in range(0, tmp.shape[1]):
                if hulp[x][y] is True:
                    tmp[x][y] = 10 ** 5

        tmp = (tmp - scaleOutputImages[0]) / (scaleOutputImages[1] - scaleOutputImages[0])
        tmp = np.round((2 ** 16 - 1) * tmp).astype(np.uint16)
        imwrite(tmp, outDIRFFC + outPrefixFFC + f'{nrImage[i - 1]:{numType}}' + fileFormat)
    xArray = np.zeros((len(nrImage), nrEigenflatfields))
    for i in range(1, len(nrImage)+1):
        print(f"Estimation projection {str(i)}/{str(len(nrImage))}...")
        projection = imread(readDIR + prefixProj + f'{nrImage[i - 1]:{numType}}' + fileFormat)

        x = condTVmean.condTVmean(projection, EigenFlatfields[:][:][0], filteredEigenFlatfields[:][:][1:(1+nrEigenflatfields)], meanDarkfield, np.zeros((1, nrEigenflatfields)), downsample)
        xArray[:][i-1] = x

        FFeff = np.zeros((meanDarkfield.shape[0], meanDarkfield.shape[1]))
        for j in range(0, nrEigenflatfields):
            FFeff = FFeff + x[j] * filteredEigenFlatfields[:][:][j+1]

        tmp = np.divide((np.squeeze(projection) - meanDarkfield), (EigenFlatfields[:][:][0] + FFeff))
        tmp = tmp / np.mean(tmp[:]) * meanVector[i-1]

        # TODO: Seems redundant
        for x in range(0, tmp.shape[0]):
            for y in range(0, tmp.shape[1]):
                if tmp[x][y] < 0:
                    tmp[x][y] = 0

        tmp = -np.log(tmp)

        # TODO: Seems redundant
        hulp = np.isinf(tmp)
        for x in range(0, tmp.shape[0]):
            for y in range(0, tmp.shape[1]):
                if hulp[x][y] is True:
                    tmp[x][y] = 10 ** 5

        tmp = (tmp - scaleOutputImages[0]) / (scaleOutputImages[1] - scaleOutputImages[0])
        tmp = np.round((2 ** 16 - 1) * tmp).astype(np.uint16)
        imwrite(tmp, outDIRDFFC + outPrefixDFFC + f'{nrImage[i - 1]:{numType}}' + fileFormat)
