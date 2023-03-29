import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im

def imread(path):
    image = im.open(path)
    return np.asarray(image)

def imwrite(matrix, path):
    im.fromarray(matrix).save(path)

class Data:
    def __init__(self):
        # how many samples per batch to load
        self.batch_size = 64
        # Create training and test dataloaders
        # TODO: https://stackoverflow.com/questions/53998282/how-does-the-number-of-workers-parameter-in-pytorch-dataloader-actually-work
        self.num_workers = 0

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

    def read_from_folder(self, folder):
        transform = transforms.Compose([transforms.ToTensor()])

        trainset = torchvision.datasets.ImageFolder(root=folder, transform=transform)
        testset = torchvision.datasets.ImageFolder(root=folder, transform=transform)

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