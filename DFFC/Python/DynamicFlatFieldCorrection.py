import bm3d
from DFFC.Python.condTVmean import condTVmean
import DFFC.Python.parallelAnalysis as parallelAnalysis
from Utils.Utils import *

def DynamicFlatFieldCorrection(param):
    # Get a list of all projection indices and get the dimensions of a .tif image (because they are all the same,
    # get the dimensions of the first image)
    nrImage = np.arange(param["firstProj"], param["firstProj"] + param["nrProj"])
    print("Load dark and flat fields")
    dims = param['size']

    # Make an m*n*p matrix to store all the dark fields and get the mean value (sum of all elements divided by the first dimension)
    print("Load dark fields...")
    dark = np.zeros((param["nrDark"], dims[0], dims[1]))
    for i in range(param["nrDark"]):
        dark[:][:][i] = imread(param["inDir"] + param["prefixProj"] + f'{param["firstDark"] + i:{param["numType"]}}' + param["fileFormat"])
    meanDarkfield = np.mean(dark, 0)

    # Make an m*n*p matrix to store all the flat fields (prior and post) and get the mean value
    print('Load white fields...')
    flat = np.zeros((param["nrWhitePrior"] + param["nrWhitePost"], dims[0] * dims[1]))
    k = 0
    for i in range(param["nrWhitePrior"]):
        tmp = imread(param["inDir"] + param["prefixFlat"] + f'{param["firstWhitePrior"] + i:{param["numType"]}}' + param["fileFormat"]) - meanDarkfield
        flat[:][k] = np.reshape(tmp, -1, 'F') - np.reshape(meanDarkfield, -1, 'F')
        k += 1
    for i in range(param["nrWhitePost"]):
        tmp = imread(param["inDir"] + param["prefixFlat"] + f'{param["firstWhitePost"] + i:{param["numType"]}}' + param["fileFormat"]) - meanDarkfield
        flat[:][k] = np.reshape(tmp, -1, 'F') - np.reshape(meanDarkfield, -1, 'F')
        k += 1
    mn = np.mean(flat, 0)

    # subtract mean flat field
    N, M = flat.shape
    Data = flat - np.tile(mn, (N, 1))
    Data = Data.transpose()
    del dark, flat

    print("Parallel Analysis:")
    V1, D1, nrEigenflatfields = parallelAnalysis.parallelAnalysis(Data, param["DFFC"]["nrPArepetitions"])

    print(f"{nrEigenflatfields} eigen flat fields selected.")
    eig0 = np.reshape(mn, dims, order='F')
    EigenFlatfields = np.zeros((nrEigenflatfields+1, eig0.shape[0], eig0.shape[1]))
    EigenFlatfields[:][:][0] = eig0
    for i in range(0, nrEigenflatfields):
        EigenFlatfields[:][:][i+1] = np.reshape(np.dot(Data, V1[:, N-i-1]), dims, order='F')
    del Data

    print("Filter eigen flat fields...")
    filteredEigenFlatfields = np.zeros((nrEigenflatfields+1, dims[0], dims[1]))
    for i in range(1, nrEigenflatfields+1):
        print(f'Filter eigen flat field {str(i)}')
        min = np.min(EigenFlatfields[:][:][i])
        max = np.max(EigenFlatfields[:][:][i])
        tmp = (EigenFlatfields[:][:][i] - min) / (max - min)
        imwrite(tmp, param["bit"], param["DFFC"]["outDir"] + 'eigenflatfields/eigenflatfield_' + f'{i:{param["numType"]}}' + param["fileFormat"])
        tmp2 = bm3d.bm3d(tmp, 25/255)
        filteredEigenFlatfields[:][:][i] = (tmp2 * (max - min)) + min
    meanVector = np.zeros(len(nrImage))
    xArray = np.zeros((len(nrImage), nrEigenflatfields))
    for i in range(1, len(nrImage)+1):
        print(f"Estimation projection {str(i)}/{str(len(nrImage))}...")
        projection = imread(param["inDir"] + param["prefixProj"] + f'{nrImage[i - 1]:{param["numType"]}}' + param["fileFormat"])
        tmp = np.divide((np.squeeze(projection) - meanDarkfield), EigenFlatfields[:][:][0])
        meanVector[i-1] = np.mean(tmp[:])

        x = condTVmean(projection, EigenFlatfields[:][:][0], filteredEigenFlatfields[:][:][1:(1+nrEigenflatfields)], meanDarkfield, np.zeros((1, nrEigenflatfields)), param["DFFC"]["downsample"])
        xArray[:][i-1] = x

        FFeff = np.zeros((meanDarkfield.shape[0], meanDarkfield.shape[1]))
        for j in range(0, nrEigenflatfields):
            FFeff = FFeff + x[j] * filteredEigenFlatfields[:][:][j+1]

        tmp = np.divide((np.squeeze(projection) - meanDarkfield), (EigenFlatfields[:][:][0] + FFeff))
        tmp = (tmp / np.mean(tmp[:])) * meanVector[i-1]
        tmp[tmp < 0] = 0
        tmp = np.clip(-np.log(tmp), a_min=0, a_max=10 ** 5)
        tmp = (tmp - param["scale"][0]) / (param["scale"][1] - param["scale"][0])
        imwrite(tmp, param["bit"], param["DFFC"]["outDir"] + param["DFFC"]["outPrefix"] + f'{nrImage[i - 1]:{param["numType"]}}' + param["fileFormat"])
