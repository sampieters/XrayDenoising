from torch.utils.data import DataLoader, random_split
from CustomImageFolder import CustomImageFolder
import torchvision.transforms as transforms
from PIL import Image as im
import numpy as np
import torch


def imwrite(matrix, path):
    image = im.fromarray(matrix)
    image.save(path)


class Data:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def read_from_folder(self, noisy_root, perf_root=None, data_augmentation=False):
        transform = transforms.Compose([
            transforms.Lambda(lambda img: np.asarray(img) / (2 ** 16 - 1)),
            transforms.Lambda(lambda img: torch.from_numpy(img)),
            transforms.Lambda(lambda img: img.to(torch.float32)),
            transforms.Lambda(lambda img: img.unsqueeze(0))
        ])

        if data_augmentation:
            # Define the data augmentation transformations
            augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            ])
            transform = transforms.Compose([transform, augmentation])

        dataset = CustomImageFolder(noisy_root=noisy_root, perf_root=perf_root, transform=transform)
        return dataset

    def rand_split(self, dataset, training_perc, validation_perc, test_perc=0.0):
        train_size = int(training_perc * len(dataset))
        val_size = int(validation_perc * len(dataset))
        test_size = int(test_perc * len(dataset))
        train_set, valid_set, test_set = random_split(dataset, [train_size, val_size, test_size])

        train_loader = DataLoader(train_set, batch_size=self.batch_size)
        val_loader = DataLoader(valid_set, batch_size=self.batch_size)
        test_loader = DataLoader(test_set, batch_size=self.batch_size)
        return train_loader, val_loader, test_loader
