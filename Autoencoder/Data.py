import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor
import torchvision.transforms.functional as F
from CustomImageFolder import CustomImageFolder

def imread(path):
    image = im.open(path)
    return np.asarray(image)

def imwrite(matrix, path):
    im.fromarray(matrix).save(path)

class Data:
    def __init__(self):
        # how many samples per batch to load
        self.batch_size = 75
        # Create training and test dataloaders
        # TODO: https://stackoverflow.com/questions/53998282/how-does-the-number-of-workers-parameter-in-pytorch-dataloader-actually-work
        self.num_workers = 0

    def read_from_folder(self, noisy, perfect):
        transform = transforms.Compose([
            transforms.Lambda(lambda img: np.asarray(img) / (2 ** 16 - 1)),
            transforms.Lambda(lambda img: torch.from_numpy(img)),
        ])


        train_noisy_set = CustomImageFolder(root_dir=noisy, transform=transform)
        train_perfect_set = CustomImageFolder(root_dir=perfect, transform=transform)

        testset = CustomImageFolder(root_dir="../input/testing", transform=transform)

        #image, label = train_perfect_set[600]
        #m = torch.max(image)
        #image = np.squeeze(image.numpy())
        #im.fromarray(image).save("../test.tif")



        self.train_noisy_loader = torch.utils.data.DataLoader(train_noisy_set, batch_size=self.batch_size, num_workers=self.num_workers)
        self.train_perfect_loader = torch.utils.data.DataLoader(train_perfect_set, batch_size=self.batch_size, num_workers=self.num_workers)

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