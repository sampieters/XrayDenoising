from PIL import Image as im
import numpy as np
import noise
import phantominator

def imread(path):
    """
    This function opens an image file located at the given file path, converts it into a NumPy array, and ensures that
    the pixel values are represented as double-precision floating-point numbers. It is commonly used for reading image
    data for further processing.

    :param path: A string containing the file path to the image.
    :return: image: The image data as a NumPy array of double precision.
    """
    image = im.open(path)
    image = np.asarray(image)
    image = image.astype(np.double)
    return image

def imwrite(matrix, type, path):
    """
    This function takes a NumPy array, converts it to the specified data type, and saves it as an image file at the
    specified path. The data type `dtype` should match the intended bit depth of the image

    :param matrix: The NumPy array representing the image data.
    :param type: The data type to use for image pixel values (e.g., np.uint8 for 8-bit images).
    :param path: The file path where the image should be saved.
    """
    matrix = type(np.round((np.iinfo(type).max * matrix)))
    image = im.fromarray(matrix)
    image.save(path)


def generate_perlin_noise_2d(size, scale=50.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=None):
    """

    :param size:
    :param scale:
    :param octaves:
    :param persistence:
    :param lacunarity:
    :param seed:
    :return:
    """
    height, width = size
    if seed is None:
        seed = np.random.randint(0, 100)

    # Generate the Perlin noise using the noise library
    perlin_noise = np.zeros((height, width))
    for x in range(width):
        for y in range(height):
            perlin_noise[y][x] = noise.pnoise2(x/scale, y/scale, octaves=octaves, persistence=persistence,
                                               lacunarity=lacunarity, repeatx=width, repeaty=height, base=seed)
    perlin_noise = (perlin_noise - np.min(perlin_noise)) / (np.max(perlin_noise) - np.min(perlin_noise))
    return perlin_noise

def generate_ridged_multifractal_noise(size, scale=50.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=None):
    height, width = size
    # Generate a grid of 2D noise
    perlin_noise = generate_perlin_noise_2d(size, scale, octaves, persistence, lacunarity, seed)
    # Apply ridge function
    perlin_noise = np.abs(perlin_noise)

    # Normalize to 0-1 range
    perlin_noise = (perlin_noise - np.min(perlin_noise)) / (np.max(perlin_noise) - np.min(perlin_noise))
    return perlin_noise

def generate_background(size):
    background_value = np.random.randint(0, 5000)
    background_value = background_value / (2 ** 16 - 1)
    matrix = np.full(size, background_value)
    return matrix

def generate_phantoms(size):
    size = (size[0], size[1])
    help = phantominator.shepp_logan(size)
    return help

def generate_shape_matrix(size):
    shapes = 50
    matrix = np.zeros(size)

    for shape in range(shapes):
        shape_type = "rectangle"
        x = np.random.randint(0, size[1])
        y = np.random.randint(0, size[0])
        gray_value = np.random.uniform(0, 1)
        shape_np = generate_single_shape(shape_type, size, gray_value)
        matrix = (matrix + shape_np) / 2

    return matrix

def generate_single_shape(shape_type, size, gray_value):
    if shape_type == "rectangle":
        shape_tensor = np.full(size, gray_value)
    elif shape_type == "ellipse":
        shape_tensor = np.full(size, gray_value)
    # Add more shape types as needed
    return shape_tensor
