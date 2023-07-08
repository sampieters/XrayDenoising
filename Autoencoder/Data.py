from CustomImageFolder import CustomImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from PIL import Image as im
import numpy as np
import torch

def imread(path):
    image = im.open(path)
    return np.asarray(image)

def imwrite(matrix, path):
    im.fromarray(matrix).save(path)

class Data:
    def __init__(self):
        # how many samples per batch to load
        self.batch_size = 32
        # Create training and test dataloaders
        # TODO: https://stackoverflow.com/questions/53998282/how-does-the-number-of-workers-parameter-in-pytorch-dataloader-actually-work
        self.num_workers = 0

    def read_from_folder(self, root):
        transform = transforms.Compose([
            transforms.Lambda(lambda img: np.asarray(img) / (2 ** 16 - 1)),
            transforms.Lambda(lambda img: torch.from_numpy(img)),
            transforms.Lambda(lambda img: img.to(torch.float32)),
            transforms.Lambda(lambda img: img.unsqueeze(0))
        ])
        dataset = CustomImageFolder(root_dir=root, transform=transform)
        return dataset

    def random_split(self, noisy_dataset, perf_dataset, lengths=0.75):
        # Define training and validation set sizes
        train_size = int(lengths * len(noisy_dataset))

        # Define random samplers for each subset
        train_sampler = SubsetRandomSampler(range(train_size))
        val_sampler = SubsetRandomSampler(range(train_size, len(noisy_dataset)))

        train_sampler = None
        val_sampler = None

        # Define data loaders for each subset
        noisy_train_loader = DataLoader(noisy_dataset, batch_size=self.batch_size, sampler=train_sampler)
        perf_train_loader = DataLoader(perf_dataset, batch_size=self.batch_size, sampler=train_sampler)

        noisy_val_loader = DataLoader(noisy_dataset, batch_size=self.batch_size, sampler=val_sampler)
        perf_val_loader = DataLoader(perf_dataset, batch_size=self.batch_size, sampler=val_sampler)
        return noisy_train_loader, perf_train_loader, noisy_val_loader, perf_val_loader