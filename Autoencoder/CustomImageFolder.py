import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class CustomImageFolder(Dataset):
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

        for target in sorted(self.noisy_class_to_idx.keys()):
            target_path = os.path.join(self.noisy_root_dir, target)
            for root, _, fnames in sorted(os.walk(target_path)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    self.noisy.append(path)

        for target in sorted(self.perfect_class_to_idx.keys()):
            target_path = os.path.join(self.perfect_root_dir, target)
            for root, _, fnames in sorted(os.walk(target_path)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
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
