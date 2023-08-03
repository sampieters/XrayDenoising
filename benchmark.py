import FFC.Python.ConventionalFlatFieldCorrection as FFC
import DFFC.Python.DynamicFlatFieldCorrection as DFFC

from PIL import Image as im
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor

benchmark_DIR = "./input/SimulatedFFCs/"
FFC_DIR = "./output/FFC/Python/"
DFFC_DIR = "./output/DFFC/Python/"

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

def imread(path):
    image = im.open(path)
    image = np.asarray(image).astype(np.double)
    image = image / (2 ** 16 - 1)
    return image

def check():
    for i in range(nrProj):
        check1 = imread(benchmark_DIR + "perfect0/" + prefixProj + f"{firstProj + i:{numType}}" + fileFormat)
        check2 = imread(FFC_DIR + outPrefixFFC + f"{firstProj + i:{numType}}" + fileFormat)

        totensor = ToTensor()
        check1 = totensor(check1)
        check2 = totensor(check2)

        mse_loss = F.mse_loss(check1, check2)
        print(f"MSE Loss: {mse_loss}")

    pass


if __name__ == "__main__":
    #idk = ['FFC', 'DFFC', 'AI']
    idk = ['FFC']

    if idk.count('FFC') == 1:
        FFC.ConventionalFlatFieldCorrection()
        check()



    if idk.count('DFFC') == 1:
        DFFC.DynamicFlatFieldCorrection()
    if idk.count('AI') == 1:
        pass
        #train()
        #test("./input/real/0")



