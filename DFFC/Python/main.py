import bm3d
import condTVmean
import numpy as np
import parallelAnalysis
from PIL import Image as im

#################################################
# PARAMETERS
#################################################

# Directory with raw dark fields, flat fields and projections in .tif format
readDIR = '../../input/duplicate_testing/real_0/'
# Directory for the output files
outDIR = '../../output/DFFC/'

# file names
prefixProj =         'dbeer_5_5_'   # prefix of the original projections
outPrefixDFFC =      'DFFC'         # prefix of the dynamic flat field corrected projections
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
    for i in range(0, nrEigenflatfields):
        EigenFlatfields[:][:][i+1] = np.reshape(Data @ V1[:, N-i-1], dims, order='F')
    del Data

    print("Filter eigen flat fields...")
    filteredEigenFlatfields = np.zeros((nrEigenflatfields+1, dims[0], dims[1]))

    for i in range(1, nrEigenflatfields+1):
        print(f'Filter eigen flat field {str(i)}')
        min = np.min(EigenFlatfields[:][:][i])
        max = np.max(EigenFlatfields[:][:][i])
        tmp = (EigenFlatfields[:][:][i] - min) / (max - min)
        tmp2 = bm3d.bm3d(tmp, 25/255)
        filteredEigenFlatfields[:][:][i] = (tmp2 * (max - min)) + min

    meanVector = np.zeros(len(nrImage))
    xArray = np.zeros((len(nrImage), nrEigenflatfields))
    for i in range(1, len(nrImage)+1):
        print(f"Estimation projection {str(i)}/{str(len(nrImage))}...")
        projection = imread(readDIR + prefixProj + f'{nrImage[i - 1]:{numType}}' + fileFormat)
        tmp = np.divide((np.squeeze(projection) - meanDarkfield), EigenFlatfields[:][:][0])
        meanVector[i-1] = np.mean(tmp[:])

        x = condTVmean.condTVmean(projection, EigenFlatfields[:][:][0], filteredEigenFlatfields[:][:][1:(1+nrEigenflatfields)], meanDarkfield, np.zeros((1, nrEigenflatfields)), downsample)
        xArray[:][i-1] = x

        FFeff = np.zeros((meanDarkfield.shape[0], meanDarkfield.shape[1]))
        for j in range(0, nrEigenflatfields):
            FFeff = FFeff + x[j] * filteredEigenFlatfields[:][:][j+1]

        tmp = np.divide((np.squeeze(projection) - meanDarkfield), (EigenFlatfields[:][:][0] + FFeff))
        tmp = (tmp / np.mean(tmp[:])) * meanVector[i-1]

        tmp[tmp < 0] = 0
        tmp = -np.log(tmp)
        tmp[np.isinf(tmp)] = 10 ** 5

        tmp = (tmp - scaleOutputImages[0]) / (scaleOutputImages[1] - scaleOutputImages[0])
        tmp = np.round((2 ** 16 - 1) * tmp).astype(np.uint16)
        imwrite(tmp, outDIR + outPrefixDFFC + f'{nrImage[i - 1]:{numType}}' + fileFormat)
