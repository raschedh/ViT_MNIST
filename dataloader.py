import os
import random
import torch
from glob import glob
from math import ceil, sqrt
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.io import read_image

class CompositeMNISTDataset(Dataset):
    def __init__(self, path: str, output_size: int, min_digits: int, max_digits: int, mode: str):
        
        assert mode in {"train", "test"}

        self.output_size = output_size
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.path = path
        self.mode = mode

        if self.mode == "test":
            # Load pre-generated test set
            print(f"Loading fixed test set from: {self.path}")
            data = torch.load(self.path)
            self.test_images = data["images"]
            self.test_labels = data["labels"]
            self.num_samples = len(self.test_labels)
        
        else:
            # Load digit images
            self.paths = glob(os.path.join(path, "**", "*.png"), recursive=True)
            self.labels = [int(os.path.basename(os.path.dirname(p))) for p in self.paths]
            self.images = [read_image(p)[0].float() / 255.0 for p in self.paths]
            self.img_size = self.images[0].shape
            self.resize = T.Resize((output_size, output_size))

    def __len__(self):
        return self.num_samples if self.mode == "test" else int(1e9)

    def __getitem__(self, idx):
        if self.mode == "test":
            return self.test_images[idx], self.test_labels[idx]
        else:
            return self._generate_sample()

    def _generate_sample(self):
        num_digits = random.randint(self.min_digits, self.max_digits)
        indices = random.sample(range(len(self.images)), num_digits)
        selected_imgs = [self.images[i] for i in indices]
        selected_labels = [self.labels[i] for i in indices]

        grid_cols = ceil(sqrt(num_digits))
        grid_rows = ceil(num_digits / grid_cols)
        h, w = self.img_size
        grid = torch.ones((grid_rows * h, grid_cols * w))

        for i, digit_img in enumerate(selected_imgs):
            row, col = divmod(i, grid_cols)
            y0, x0 = row * h, col * w
            grid[y0:y0 + h, x0:x0 + w] = digit_img

        resized = self.resize(grid.unsqueeze(0))
        label_sequence = ["<s>"] + [str(l) for l in selected_labels] + ["<eos>"]
        return resized, label_sequence