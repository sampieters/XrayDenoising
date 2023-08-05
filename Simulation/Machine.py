import noise
from Utils.Utils import *

class Machine:
    def __init__(self):
        pass

    def generate_darkfields(self):
        pass

    def generate_flatfields(self):
        pass

    def generate_projections(self):
        pass


class XXX(Machine):
    def __init__(self):
        super().__init__()
        self.directory = './input/real/'
        self.prefixDark = 'dbeer_5_5_'      # prefix of the dark fields
        self.firstDark = 1                  # image number of first dark field
        self.nrDark = 20                    # number of dark fields

        self.prefixFlat = 'dbeer_5_5_'      # prefix of the flat fields
        self.firstFlat = 21                 # image number of first prior flat field
        self.nrFlat = 300                   # number of white (flat) fields BEFORE acquiring the projections

        self.prefixProj = 'dbeer_5_5_'      # prefix of the original projections
        self.firstProj = 321                # image number of first projection
        self.nrProj = 300                   # number of acquired projections

        self.size = (256, 1248)             # the dimensions of the input projections
        self.numType = '04d'                # number type used in image names
        self.fileFormat = '.tif'            # image format
        self.prefixOut = 'FFC'              # prefix of the CONVENTIONAL flat field corrected projections

    def generate_darkfields(self):
        # Make an m*n*p matrix to store all the dark fields and get the mean value
        print("Load all dark fields...")
        dark = np.zeros((self.nrDark + 1, self.size[0], self.size[1]))
        for i in range(self.nrDark + 1):
            dark[:][:][i] = imread(self.directory + self.prefixProj + f'{self.firstDark + i:{self.numType}}' + self.fileFormat)
            dark[:][:][i] = dark[:][:][i] / (2 ** 16 - 1)
        return dark

    def generate_flatfields(self):
        print("Load all flat fields...")
        flat = np.zeros((self.nrFlat + 1, self.size[0], self.size[1]))
        for i in range(self.nrFlat + 1):
            flat[:][:][i] = imread(self.directory + self.prefixFlat + f'{self.firstFlat + i:{self.numType}}' + self.fileFormat)
            flat[:][:][i] = flat[:][:][i] / (2 ** 16 - 1)
        return flat

    def generate_projections(self):
        print("Simulating projections (ridged multifractal noise)...")
        projections = np.zeros((self.nrProj + 1, self.size[0], self.size[1]))
        for proj in range(self.nrProj):
            # Generate a grid of 2D noise
            perlin_noise = generate_perlin_noise_2d(self.size[1], self.size[0], scale=50.0,
                                                    octaves=6, persistence=0.5, lacunarity=2.0, seed=None)
            # Apply ridge function
            perlin_noise = np.abs(perlin_noise)

            # Normalize to 0-1 range
            perlin_noise = (perlin_noise - np.min(perlin_noise)) / (np.max(perlin_noise) - np.min(perlin_noise))
            projections[proj] = perlin_noise
        return projections



def generate_perlin_noise_2d(width, height, scale, octaves, persistence, lacunarity, seed):
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

def generate_ridged_multifractal_noise(width, height, scale, octaves, persistence, lacunarity, seed):
    # Generate a grid of 2D noise
    perlin_noise = generate_perlin_noise_2d(width, height, scale, octaves, persistence, lacunarity, seed)
    # Apply ridge function
    perlin_noise = np.abs(perlin_noise)

    # Normalize to 0-1 range
    perlin_noise = (perlin_noise - np.min(perlin_noise)) / (np.max(perlin_noise) - np.min(perlin_noise))
    return perlin_noise

def generate_turbulence_noise(width, height, scale, octaves, persistence, lacunarity, seed):
    turbulence_noise = np.zeros((height, width))

    for octave in range(octaves):
        octave_scale = scale * (lacunarity ** octave)
        octave_amplitude = persistence ** octave

        perlin_noise = generate_perlin_noise_2d(width, height, octave_scale, 1, 0.5, 2.0, seed)
        turbulence_noise += octave_amplitude * np.abs(perlin_noise)

    return turbulence_noise
