import os
from Utils.Utils import *

def ConventionalFlatFieldCorrection(param):
    # Make the output directories
    if not os.path.exists(param["FFC"]["outDir"]):
        os.mkdir(param["FFC"]["outDir"])

    # Get a list of all projection indices and get the dimensions of a .tif image (because they are all the same,
    # get the dimensions of the first image)
    nrImage = np.arange(param["firstProj"], param["firstProj"] + param["nrProj"])
    print("Load dark and flat fields")
    dims = param["size"]

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
    eig0 = np.reshape(mn, dims, order='F')
    del dark, flat

    for i in range(1, len(nrImage)+1):
        print(f'Conventional FFC: {str(i)}/{str(len(nrImage))}...')
        # Load projection
        projection = imread(param["inDir"] + param["prefixProj"] + f'{nrImage[i-1]:{param["numType"]}}' + param["fileFormat"])
        tmp = (np.squeeze(projection) - meanDarkfield) / eig0
        tmp[tmp < 0] = 0
        tmp = np.clip(-np.log(tmp), a_min=0, a_max=10 ** 5)
        tmp = (tmp - param["scale"][0]) / (param["scale"][1] - param["scale"][0])
        imwrite(tmp, param["bit"], param["FFC"]["outDir"] + param["FFC"]["outPrefix"] + f'{nrImage[i - 1]:{param["numType"]}}' + param["fileFormat"])
