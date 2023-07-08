import numpy as np
import os
from PIL import Image as im
from phantominator import shepp_logan, dynamic, mr_ellipsoid_parameters

# Directory with raw dark fields, flat fields and projections in .tif format
readDIR = './input/duplicate_testing/real_0/'
# Directory for the output files
outDIR = './output/'

# file names
prefixProj =         'dbeer_5_5_'   # prefix of the original projections
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

in_dir = "./input/pngs/"
out_dir = "./input/perfect/"
output_dir = "input/training/"
amount = 5
generated = 5
variations = 10
type = '{0:04d}'


def imread(path):
    image = im.open(path)
    return np.asarray(image)


def imwrite(matrix, path):
    l = im.fromarray(matrix)
    l.save(path)


def png_to_tif(input_path, output_path):
    # Load the png file
    image = imread(input_path)

    # Convert to uint16 and scale values between 0 and 65535
    image = image.astype('uint16')
    image = ((image / image.max()) * (2 ** 16 - 1)).astype('uint16')
    if image.shape[2] == 3:
        image = image[:, :, 0]
    # Save as tiff file
    imwrite(image, output_path)


def simulate_noisy(clean, out_path):
    # Get a list of all projection indices and get the dimensions of a .tif image (because they are all the same,
    # get the dimensions of the first image)
    print("Get the \"perfect\" image...")
    n_j = imread(clean)
    n_j = n_j / (2 ** 16 - 1)
    dims = n_j.shape

    # Make an m*n*p matrix to store all the dark fields and get the mean value
    print("Load dark fields...")
    dark = np.zeros((nrDark, dims[0], dims[1]))
    for i in range(firstDark - 1, firstDark + nrDark - 1):
        dark[:][:][i] = imread(readDIR + prefixProj + f'{i + 1:{numType}}' + fileFormat)
    # Get the mean (sum of all elements divided by the dimensions)
    meanDarkfield = np.mean(dark, 0)

    print("Load all possible flat fields...")
    # This loop is merging the perfect projection with one prior white field
    print("Generated training image (based on prior white fields)...")
    for i in range(nrWhitePrior):
        f_j = imread(readDIR + prefixFlat + f'{firstWhitePrior + i:{numType}}' + fileFormat)
        p_j = n_j * (f_j - meanDarkfield) + meanDarkfield
        p_j = np.round(p_j).astype(np.uint16)
        imwrite(p_j, out_path + f"noisy_{i}.tif")

    # This loop is merging the perfect projection with one post white field
    print("Generated training image (based on post white fields)...")
    for i in range(nrWhitePost):
        f_j = imread(readDIR + prefixFlat + f'{firstWhitePost + i:{numType}}' + fileFormat)
        f_j = f_j / (2 ** 16 - 1)
        meanDarkfield = meanDarkfield / (2 ** 16 - 1)
        # n_j is values between 0 and 1
        p_j = n_j * f_j
        p_j = np.round(p_j * (2 ** 16 - 1)).astype(np.uint16)
        imwrite(p_j, out_path + f"noisy_{nrWhitePrior + i}.tif")


# Loop over all pngs and convert them to tiff files
for i in range(0, amount):
    if not os.path.exists(out_dir + f'{i}'):
        os.mkdir(out_dir + f'{i}')
    for j in range(0, 600):
        png_to_tif(in_dir + f'perfect_{type.format(i)}.png', out_dir + f'{i}/perfect_{j}.tif')


# Also generate some images if not enough
ph = shepp_logan((generated + 2, 256, 1248))
ph = ph[1:-1, :, :]
for i in range(amount, amount + generated):
    slice = ph[:][:][i - amount]
    slice = np.round(((2 ** 16 - 1) * slice)).astype(np.uint16)
    if not os.path.exists(out_dir + f'{i}'):
        os.mkdir(out_dir + f'{i}')
    for j in range(0, 600):
        imwrite(slice, out_dir + f'{i}/' + f'perfect_{j}.tif')

# Generate training images
for i in range(0, amount + generated):
    if not os.path.exists(output_dir + f'{i}'):
        os.mkdir(output_dir + f'{i}')
    simulate_noisy(out_dir + f'{i}/perfect_0.tif', output_dir + f'{i}/')

