import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageFolder(Dataset):
    def __init__(self, noisy_root, perf_root, transform=None):
        self.transform = transform

        self.noisy_root_dir = noisy_root
        self.noisy_classes = sorted(os.listdir(self.noisy_root_dir))
        self.noisy_class_to_idx = {self.noisy_classes[i]: i for i in range(len(self.noisy_classes))}
        self.noisy = []

        self.perfect_root_dir = perf_root
        self.perfect_classes = []
        self.perfect_class_to_idx = {}
        if perf_root is not None:
            self.perfect_classes = sorted(os.listdir(self.perfect_root_dir))
            self.perfect_class_to_idx = {self.perfect_classes[i]: i for i in range(len(self.perfect_classes))}
        self.perfect = []

        # Check files in the root directory
        for fname in sorted(os.listdir(noisy_root)):
            # Only consider files, not directories
            if os.path.isfile(os.path.join(noisy_root, fname)):
                path = os.path.join(noisy_root, fname)
                self.noisy.append(path)

        # Check files in the root directory
        if perf_root is not None:
            for fname in sorted(os.listdir(perf_root)):
                # Only consider files, not directories
                if os.path.isfile(os.path.join(perf_root, fname)):
                    path = os.path.join(perf_root, fname)
                    self.perfect.append(path)

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, index):
        path = self.noisy[index]
        with open(path, 'rb') as f:
            noisy_img = Image.open(f)
            if self.transform is not None:
                noisy_img = self.transform(noisy_img)

        perf_img = torch.empty(0, 0)
        if self.perfect_root_dir is not None:
            path = self.perfect[index]
            with open(path, 'rb') as f:
                perf_img = Image.open(f)
                if self.transform is not None:
                    perf_img = self.transform(perf_img)
        return noisy_img, perf_img

    def showitem(self, index):
        noisy_img = self.noisy[index]
        perf_img = self.perfect[index]
        img = Image.open(noisy_img)
        img2 = Image.open(perf_img)
        img.show()
        img2.show()
