import os
from PIL import Image
from torch.utils.data import Dataset


class CustomImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.images = []
        for target in sorted(self.class_to_idx.keys()):
            target_path = os.path.join(self.root_dir, target)
            for root, _, fnames in sorted(os.walk(target_path)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    self.images.append((path, self.class_to_idx[target]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path, label = self.images[index]
        with open(path, 'rb') as f:
            img = Image.open(f)
            if self.transform is not None:
                img = self.transform(img)
        return img, label
