import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im

# Directory with raw dark fields, flat fields and projections in .tif format
readDIR = '../input/'
# Directory for the output files
outDIR = '../output/'

# file names
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

# options output images
scaleOutputImages =  [0, 2]         # output images are scaled between these values

def imread(path):
    image = im.open(path)
    return np.asarray(image)

def imwrite(matrix, path):
    im.fromarray(matrix).save(path)

class Data:
    def __init__(self):
        # how many samples per batch to load
        self.batch_size = 3
        # Create training and test dataloaders
        # TODO: https://stackoverflow.com/questions/53998282/how-does-the-number-of-workers-parameter-in-pytorch-dataloader-actually-work
        self.num_workers = 0
    def simulate_noisy(self):
        # Get a list of all projection indices and get the dimensions of a .tif image (because they are all the same,
        # get the dimensions of the first image)
        print("Get the \"perfect\" image...")

        # Open the PNG image
        png_image = im.open('../AEinput/clean2.png')
        # Convert the image to grayscale
        grayscale_image = png_image.convert('L')
        # Convert the grayscale image to 16-bit
        converted_image = grayscale_image.convert('I;16')
        # Save the image as a TIFF file
        l = np.asarray(converted_image)
        l = np.round(l).astype(np.uint16)
        imwrite(l, "../AEinput/clean2.tif")

        n_j = imread("../AEinput/clean2.tif")
        dims = n_j.shape

        # Make an m*n*p matrix to store all the dark fields and get the mean value
        print("Load dark fields...")
        dark = np.zeros((nrDark, dims[0], dims[1]))
        for i in range(firstDark - 1, firstDark + nrDark - 1):
            dark[:][:][i] = imread(readDIR + prefixProj + f'{i + 1:{numType}}' + fileFormat)
        # Get the mean (sum of all elements divided by the dimensions)
        meanDarkfield = np.mean(dark, 0)

        print("Load one flat field...")
        f_j = imread(readDIR + prefixFlat + f'{firstWhitePrior:{numType}}' + fileFormat)

        print("Calculate noisy image...")
        p_j = n_j * (f_j - meanDarkfield) + meanDarkfield
        p_j[p_j < 0] = 0
        p_j = -np.log(p_j)
        p_j[np.isinf(p_j)] = 10 ** 5
        p_j = (p_j - scaleOutputImages[0]) / (scaleOutputImages[1] - scaleOutputImages[0])
        # p_j = (p_j - scaleOutputImages[1]) / (scaleOutputImages[0] - scaleOutputImages[1])
        p_j = np.round((2 ** 16 - 1) * p_j).astype(np.uint16)
        imwrite(p_j, "./simulate_noisy.tif")

    def read_from_link(self):
        # convert data (PIL image or ndarray) to torch.FloatTensor
        # It is a multi dimensional array with a lot of extras
        # Needs to be a list to make it iterable (for e.g. visualize())
        transform = transforms.Compose([transforms.ToTensor()])
        # TODO: https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html use other transform functions to generate more data
        # Use of compose here is to Compose several transforms together. If not necessary, remove the compose

        # load the training and test datasets. root: The filesystem path to where we want the data to go.
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        # prepare data loaders
        # TODO: shuffle=True to randomize the order
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, num_workers=self.num_workers)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, num_workers=self.num_workers)

    def read_from_folder(self):
        transform = transforms.Compose([transforms.ToTensor()])

        trainset = torchvision.datasets.ImageFolder(root='../AEinput', transform=transform)
        testset = torchvision.datasets.ImageFolder(root='../AEinput', transform=transform)

        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, num_workers=self.num_workers)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, num_workers=self.num_workers)

    def visualize(self):
        # get some random training images
        dataiter = iter(self.trainloader)
        images, labels = next(dataiter)

        # show images
        img = torchvision.utils.make_grid(images)
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg))
        plt.show()

test = Data()
test.simulate_noisy()