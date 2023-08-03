from phantominator import shepp_logan
from PIL import Image as im
from skimage import draw
import numpy as np
import Noise
import os

# image size
image_size = (256, 1248)
# How many different clean images there are
variations = 10
# How many variations/different flatfields/noisy projections per clean image there are
img_per_dir = 30

out_dir = "input/simulated_one/perfect/"
output_dir = "input/simulated_one/training/"
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

def imread(path):
    image = im.open(path)
    image = np.asarray(image).astype(np.longdouble)
    image = image / (2 ** 16 - 1)
    return image


def imwrite(matrix, path):
    matrix = np.round(((2 ** 16 - 1) * matrix)).astype(np.uint16)
    image = im.fromarray(matrix)
    image.save(path)


def load_darkfields(amount):
    # Make an m*n*p matrix to store all the dark fields and get the mean value
    print("Load all dark fields...")
    dark = np.zeros((nrDark + 1, image_size[0], image_size[1]))
    for i in range(firstDark, firstDark + nrDark):
        dark[:][:][i] = imread(readDIR + prefixProj + f'{i:{numType}}' + fileFormat)
    return dark


def load_flatfields(amount):
    print("Load all flat fields...")
    flat = np.zeros((nrWhitePrior + 1, image_size[0], image_size[1]))
    for i in range(nrWhitePrior + 1):
        flat[:][:][i] = imread(readDIR + prefixFlat + f'{firstWhitePrior + i:{numType}}' + fileFormat)
    return flat


# AMOUNT = VARIATIONS
def generate_clean_variations(amount):
    print("generate clean projection...")
    # Generate clean images (delete the first and last one because shepp_logan() returns black images)
    ph = shepp_logan((amount + 2, image_size[0], image_size[1]))
    ph = ph[1:-1]

    for directory in range(amount):
        slice = ph[:][:][directory]
        if not os.path.exists(out_dir + f'{directory}'):
            os.mkdir(out_dir + f'{directory}')
        for unique in range(img_per_dir):
            imwrite(slice, out_dir + f'{directory}/' + prefixProj + f'{unique}' + fileFormat)
    return ph


gradient_factor = 2.0


def create_xray_image():
    x = np.arange(image_size[1])
    y = np.arange(image_size[0])
    X, Y = np.meshgrid(x, y)

    # Generate a gradient representing the attenuation of X-rays
    gradient = gradient_factor * np.sqrt((X - image_size[1] / 2) ** 2 + (Y - image_size[0] / 2) ** 2)
    gradient = gradient.astype(np.uint16)

    # Combine the gradient and noise to create the X-ray image-like matrix
    xray_image = gradient
    xray_image = xray_image / np.max(xray_image)
    return xray_image


def create_solid_image(value):
    return np.full(image_size, value)

def get_circle_coordinates(radius, middle_point):
    """
    Get all coordinates that form a circle with the given radius and middle point.

    Parameters:
        radius (int): Radius of the circle.
        middle_point (tuple): Tuple (x, y) representing the middle point of the circle.

    Returns:
        list: List of (x, y) tuples representing the coordinates of the circle pixels.
    """
    x0, y0 = middle_point
    coordinates = []

    x = radius
    y = 0
    decision_over = 1 - x

    while x >= y:
        coordinates.append((x + x0, y + y0))
        coordinates.append((-x + x0, y + y0))
        coordinates.append((x + x0, -y + y0))
        coordinates.append((-x + x0, -y + y0))

        coordinates.append((y + x0, x + y0))
        coordinates.append((-y + x0, x + y0))
        coordinates.append((y + x0, -x + y0))
        coordinates.append((-y + x0, -x + y0))

        y += 1
        if decision_over <= 0:
            decision_over += 2 * y + 1
        else:
            x -= 1
            decision_over += 2 * (y - x) + 1
    return coordinates


def draw_circle(im, coor, radius):
    l = get_circle_coordinates(radius, coor)
    for coordinate in l:
        if coordinate[0] >= 0 and coordinate[1] >= 0 and coordinate[0] < image_size[0] and coordinate[1] < image_size[1]:
            color = np.random.randint(15000, 40000)
            im[coordinate[0], coordinate[1]] = (im[coordinate[0], coordinate[1]] + color / (2 ** 16 - 1)) / 2
    return im


def draw_circles(im):
    for i in range(200):
        x = np.random.randint(0, image_size[0])
        y = np.random.randint(0, image_size[1])
        radius = np.random.randint(5, 500)
        im = draw_circle(im, (x, y), radius)
    return im




num_lines = 150
line_thickness = 1
curve_factor = 0.9


def create_lines_image():
    # Create a black background
    lines_image = np.zeros(image_size)

    # Generate random curved lines
    for _ in range(num_lines):
        x0, y0 = np.random.randint(0, image_size[1]), np.random.randint(0, image_size[0])
        x1, y1 = np.random.randint(0, image_size[1]), np.random.randint(0, image_size[0])
        x2, y2 = np.random.randint(0, image_size[1]), np.random.randint(0, image_size[0])
        row, column = draw.bezier_curve(y0, x0, y1, x1, y2, x2, 3)
        lines_image[row, column] = np.random.random()
    return lines_image


def make_simulations():
    #clean = generate_clean_variations(variations)
    dark = load_darkfields(variations * img_per_dir)
    flat = load_flatfields(variations * img_per_dir)
    for directory in range(variations):
        if not os.path.exists(out_dir + f'{directory}'):
            os.mkdir(out_dir + f'{directory}')
        if not os.path.exists(output_dir + f'{directory}'):
            os.mkdir(output_dir + f'{directory}')
        #n_j = clean[directory]
        #x_ray = create_xray_image()

        #lines = create_lines_image()
        #n_j = (n_j + x_ray + lines)/3

        #TODO: Generate more darkfields and set this in the second loop
        d_j = dark[0]
        for unique in range(img_per_dir):
            bkg = np.random.randint(0, 6000)
            #x_ray = create_solid_image(bkg / (2 ** 16 - 1))
            #x_ray = draw_circles(x_ray)
            #n_j = (clean[directory] + x_ray) / 2
            n_j = Noise.generate_ridged_multifractal_noise(image_size[1], image_size[0], 50.0, 6, 0.5, 2.0, None)

            f_j = flat[directory * img_per_dir + unique]
            p_j = np.exp(-n_j) * (f_j - d_j) + d_j
            imwrite(n_j, out_dir + f'{directory}/' + prefixProj + f'{unique}' + fileFormat)
            imwrite(p_j, output_dir + f'{directory}/' + prefixProj + f'{unique}' + fileFormat)
        print(f"Generated {img_per_dir} perfect/noisy images in directory {directory}")

def make_FFC_dataset():
    writeDIR = './input/SimulatedFFCs/'

    # Load all dark fields
    darkfields = load_darkfields(nrDark + 1)

    # Load all prior flat fields (not the post flat fields)
    flatfields = load_flatfields(nrWhitePrior + 1)

    # Make the simulated noisy projections
    clean = np.zeros((nrProj + 1, image_size[0], image_size[1]))
    projections = np.zeros((nrProj + 1, image_size[0], image_size[1]))
    for i in range(nrProj + 1):
        d_j = darkfields[np.random.randint(0, nrDark)]
        f_j = flatfields[np.random.randint(0, nrWhitePrior)]
        n_j = Noise.generate_ridged_multifractal_noise(image_size[1], image_size[0], 50.0, 6, 0.5, 2.0, None)
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



make_FFC_dataset()