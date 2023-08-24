
from Utils.Utils import *

class Machine:
    def __init__(self):
        pass

    def generate_darkfields(self, amount):
        pass

    def generate_flatfields(self, amount):
        pass

    def generate_projections(self, amount):
        pass


class XXX(Machine):
    def __init__(self):
        super().__init__()
        self.directory = './input/real/'
        self.prefixDark = 'dbeer_5_5_'      # prefix of the dark fields
        self.firstDark = 1                  # image number of first dark field
        self.nrDark = 20                    # number of dark fields

        self.prefixFlat = 'dbeer_5_5_'      # prefix of the flat fields
        self.firstWhitePrior = 21           # image number of first prior flat field
        self.nrWhitePrior = 300             # number of white (flat) fields BEFORE acquiring the projections

        self.prefixProj = 'dbeer_5_5_'      # prefix of the original projections
        self.firstProj = 321                # image number of first projection
        self.nrProj = 300                   # number of acquired projections

        self.firstWhitePost = 572           # image number of first post flat field
        self.nrWhitePost = 300              # number of white (flat) fields AFTER acquiring the projections

        self.size = (256, 1248)             # the dimensions of the input projections
        self.numType = '04d'                # number type used in image names
        self.fileFormat = '.tif'            # image format
        self.prefixOut = 'FFC'              # prefix of the CONVENTIONAL flat field corrected projections

    def generate_darkfields(self, amount):
        # Make an m*n*p matrix to store all the dark fields and get the mean value
        print("Load all dark fields...")
        dark = np.zeros((self.nrDark, self.size[0], self.size[1]))
        for i in range(self.nrDark):
            dark[:][:][i] = imread(self.directory + self.prefixProj + f'{self.firstDark + i:{self.numType}}' + self.fileFormat)
            dark[:][:][i] = dark[:][:][i] / (2 ** 16 - 1)
        return dark

    def generate_flatfields(self, amount):
        print("Load all flat fields...")
        flat = np.zeros((self.nrWhitePrior + self.nrWhitePost, self.size[0], self.size[1]))
        for i in range(self.nrWhitePrior):
            flat[:][:][i] = imread(self.directory + self.prefixFlat + f'{self.firstWhitePrior + i:{self.numType}}' + self.fileFormat)
            flat[:][:][i] = flat[:][:][i] / (2 ** 16 - 1)
        for i in range(self.nrWhitePost):
            flat[:][:][self.nrWhitePrior + i] = imread(self.directory + self.prefixFlat + f'{self.firstWhitePost + i:{self.numType}}' + self.fileFormat)
            flat[:][:][self.nrWhitePrior + i] = flat[:][:][self.nrWhitePrior + i] / (2 ** 16 - 1)
        return flat

    def generate_projections(self, amount):
        print("Simulating projections (ridged multifractal noise)...")
        projections = np.zeros((amount, self.size[0], self.size[1]))
        for proj in range(amount):
            # Generate a grid of 2D noise
            perlin_noise = generate_perlin_noise_2d(self.size, scale=50.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=None)
            # Apply ridge function
            perlin_noise = np.abs(perlin_noise)

            # Normalize to 0-1 range
            perlin_noise = (perlin_noise - np.min(perlin_noise)) / (np.max(perlin_noise) - np.min(perlin_noise))
            projections[proj] = perlin_noise
        return projections


class XYZ(Machine):
    def __init__(self):
        super().__init__()
        self.directory = './input/real/'

        self.nrDark = 300                   # number of dark fields

        self.prefixFlat = 'dbeer_5_5_'      # prefix of the flat fields
        self.firstFlat = 21                 # image number of first prior flat field
        self.nrFlat = 300                   # number of white (flat) fields BEFORE acquiring the projections

        self.nrProj = 300                   # number of acquired projections

        self.size = (256, 1248)             # the dimensions of the input projections
        self.numType = '04d'                # number type used in image names
        self.fileFormat = '.tif'            # image format
        self.prefixOut = 'FFC'              # prefix of the CONVENTIONAL flat field corrected projections

    def generate_darkfields(self, amount):
        # Make an m*n*p matrix to store all the dark fields and get the mean value
        print("Load all dark fields...")
        dark = np.zeros((self.nrDark, self.size[0], self.size[1]))
        for i in range(self.nrDark):
            dark[:][:][i] = np.random.randint(500, size=self.size)
            dark[:][:][i] = dark[:][:][i] / (2 ** 16 - 1)
        return dark

    def generate_flatfields(self, amount):
        print("Load all flat fields...")
        flat = np.zeros((self.nrFlat + 1, self.size[0], self.size[1]))
        for i in range(self.nrFlat + 1):
            flat[:][:][i] = imread(self.directory + self.prefixFlat + f'{self.firstFlat + i:{self.numType}}' + self.fileFormat)
            flat[:][:][i] = flat[:][:][i] / (2 ** 16 - 1)
        return flat

    def generate_projections(self, amount):
        variations = 1
        print("Simulating projections...")
        projections = np.zeros((self.nrProj + 1, self.size[0], self.size[1]))
        projection_list = [generate_perlin_noise_2d,
                           #generate_background,
                           #generate_phantoms,
                           #generate_shape_matrix,
                           ]

        help = int(self.nrProj / variations)
        for var in range(variations):
            for proj in range(help):
                projections[var * help + proj] = projection_list[var](self.size)
        return projections

class HELP(Machine):
    def __init__(self):
        super().__init__()
        self.directory = './input/real/'

        self.prefixDark = 'dbeer_5_5_'      # prefix of the flat fields
        self.firstDark = 1                  # image number of first dark field
        self.nrDark = 300                   # number of dark fields

        self.prefixFlatPrior = 'dbeer_5_5_' # prefix of the flat fields
        self.firstFlatPrior = 21            # image number of first prior flat field
        self.nrFlatPrior = 300              # number of white (flat) fields BEFORE acquiring the projections

        self.firstFlatPost = 572            # image number of first post flat field
        self.nrFlatPost = 300               # number of white (flat) fields AFTER acquiring the projections

        self.firsProj = 321
        self.nrProj = 50                    # number of acquired projections

        self.size = (256, 1248)             # the dimensions of the input projections
        self.numType = '04d'                # number type used in image names
        self.fileFormat = '.tif'            # image format
        self.prefixOut = 'FFC'              # prefix of the CONVENTIONAL flat field corrected projections


    def generate_darkfields(self, amount):
        # Number of real images to combine for each simulated image
        num_real_images_to_combine = 3  # Adjust as needed

        # Make an m*n*p matrix to store all the dark fields and get the mean value
        print("Load all dark fields...")
        dark = np.zeros((self.nrDark, self.size[0], self.size[1]))
        for i in range(self.nrDark):
            dark[:][:][i] = imread(self.directory + self.prefixDark + f'{self.firstDark + i:{self.numType}}' + self.fileFormat)
            dark[:][:][i] = dark[:][:][i] / (2 ** 16 - 1)

        return_dark = np.zeros((amount, self.size[0], self.size[1]))

        # Generate simulated images
        for i in range(amount):
            # Randomly select real images to combine
            selected_indices = np.random.choice(len(dark), size=num_real_images_to_combine, replace=False)
            selected_real_images = [dark[i] for i in selected_indices]

            # Combine selected real images (e.g., using average)
            return_dark[i] = np.mean(selected_real_images, axis=0)

        return return_dark

    def generate_flatfields(self, amount):
        # Number of real images to combine for each simulated image
        num_real_images_to_combine = 3

        print("Load all flat fields...")
        flat = np.zeros((self.nrFlatPrior, self.size[0], self.size[1]))
        for i in range(self.nrFlatPrior):
            flat[:][:][i] = imread(self.directory + self.prefixFlatPrior + f'{self.firstFlatPrior + i+1:{self.numType}}' + self.fileFormat)
            flat[:][:][i] = flat[:][:][i] / (2 ** 16 - 1)

        return_flat = np.zeros((amount, self.size[0], self.size[1]))

        # Generate simulated images
        for i in range(amount):
            # Randomly select real images to combine
            selected_indices = np.random.choice(len(flat), size=num_real_images_to_combine, replace=False)
            selected_real_images = [flat[i] for i in selected_indices]

            # Combine selected real images (e.g., using average)
            return_flat[i] = np.mean(selected_real_images, axis=0)

        return return_flat

    def generate_projections(self, amount):
        print("Simulating projections...")
        projections = np.zeros((amount, self.size[0], self.size[1]))
        projection_list = [generate_perlin_noise_2d,
                           #generate_background,
                           #generate_phantoms,
                           #generate_shape_matrix,
                           ]

        for i in range(amount):
            projections[i] = generate_perlin_noise_2d(self.size)
        return projections

class PHANTOM(Machine):
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

    def generate_darkfields(self, amount):
        # Make an m*n*p matrix to store all the dark fields and get the mean value
        print("Load all dark fields...")
        dark = np.zeros((self.nrDark + 1, self.size[0], self.size[1]))
        for i in range(self.nrDark + 1):
            dark[:][:][i] = imread(self.directory + self.prefixProj + f'{self.firstDark + i:{self.numType}}' + self.fileFormat)
            dark[:][:][i] = dark[:][:][i] / (2 ** 16 - 1)
        return dark

    def generate_flatfields(self, amount):
        print("Load all flat fields...")
        flat = np.zeros((self.nrFlat + 1, self.size[0], self.size[1]))
        for i in range(self.nrFlat + 1):
            flat[:][:][i] = imread(self.directory + self.prefixFlat + f'{self.firstFlat + i:{self.numType}}' + self.fileFormat)
            flat[:][:][i] = flat[:][:][i] / (2 ** 16 - 1)
        return flat

    def generate_projections(self, amount):
        print("Simulating projections (ridged multifractal noise)...")
        size = (amount, self.size[0], self.size[1])
        phantom = phantominator.shepp_logan(size)
        return phantom