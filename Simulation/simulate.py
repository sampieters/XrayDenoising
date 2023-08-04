from PIL import Image as im
import numpy as np
import Machine
import os

# image size
image_size = (256, 1248)
# How many different clean images there are
variations = 10
# How many variations/different flatfields/noisy projections per clean image there are
img_per_dir = 30

out_dir = "../input/simulated/perfect/"
output_dir = "../input/simulated/training/"
# Directory with raw dark fields, flat fields and projections in .tif format
readDIR = './input/real/0/'
# Directory for the output files
outDIR = './output/'

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

num_lines = 150
line_thickness = 1
curve_factor = 0.9
gradient_factor = 2.0


def imread(path):
    image = im.open(path)
    image = np.asarray(image).astype(np.longdouble)
    image = image / (2 ** 16 - 1)
    return image


def imwrite(matrix, path):
    matrix = np.round(((2 ** 16 - 1) * matrix)).astype(np.uint16)
    image = im.fromarray(matrix)
    image.save(path)



def make_simulations():
    machine = Machine.XXX()
    dark = machine.generate_darkfields()
    flat = machine.generate_flatfields()
    for directory in range(variations):
        if not os.path.exists(out_dir + f'{directory}'):
            os.mkdir(out_dir + f'{directory}')
        if not os.path.exists(output_dir + f'{directory}'):
            os.mkdir(output_dir + f'{directory}')
        d_j = dark[0]
        for unique in range(img_per_dir):
            n_j = machine.generate_projections()

            f_j = flat[directory * img_per_dir + unique]
            p_j = np.exp(-n_j) * (f_j - d_j) + d_j
            imwrite(n_j, out_dir + f'{directory}/' + prefixProj + f'{unique}' + fileFormat)
            imwrite(p_j, output_dir + f'{directory}/' + prefixProj + f'{unique}' + fileFormat)
        print(f"Generated {img_per_dir} perfect/noisy images in directory {directory}")

def make_FFC_dataset():
    writeDIR = './input/SimulatedFFCs/'

    machine = Machine.XXX()
    # Load all dark fields
    darkfields = machine.generate_darkfields()
    # Load all flat fields
    flatfields = machine.generate_flatfields()

    # Make the simulated noisy projections
    clean = np.zeros((nrProj + 1, image_size[0], image_size[1]))
    projections = np.zeros((nrProj + 1, image_size[0], image_size[1]))
    for i in range(nrProj + 1):
        d_j = darkfields[np.random.randint(0, nrDark)]
        f_j = flatfields[np.random.randint(0, nrWhitePrior)]
        n_j = machine.generate_projections()
        clean[i] = n_j
        projections[i] = np.exp(-n_j) * (f_j - d_j) + d_j

    # Write everything to a file
    for i in range(nrProj):
        imwrite(clean[i], writeDIR + 'perfect0/' + prefixDark + f'{firstProj + i:{numType}}' + fileFormat)

    for i in range(nrDark):
        imwrite(darkfields[i], writeDIR + 'noisy0/' + prefixDark + f'{firstDark + i:{numType}}' + fileFormat)

    for i in range(nrWhitePrior):
        imwrite(flatfields[i], writeDIR + 'noisy0/' + prefixFlat + f'{firstWhitePrior + i:{numType}}' + fileFormat)

    for i in range(nrProj):
        imwrite(projections[i], writeDIR + 'noisy0/' + prefixProj + f'{firstProj + i:{numType}}' + fileFormat)
