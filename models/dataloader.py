import os
import random
import torch
from glob import glob
from math import ceil, sqrt
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.io import read_image
from torch.nn.utils.rnn import pad_sequence

VOCAB = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "<pad>": 10, "<s>": 11, "<eos>": 12}
IDX_TO_TOKEN = {idx: token for token, idx in VOCAB.items()}

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    label_tensors = [torch.tensor([VOCAB[token] for token in seq], dtype=torch.long) for seq in labels]
    labels_padded = pad_sequence(label_tensors, batch_first=True, padding_value=VOCAB["<pad>"])
    return images, labels_padded


class CompositeMNISTDataset(Dataset):
    def __init__(self, path: str, output_size: int, mode: str, min_digits: int, max_digits: int, train_samples: int = None):
        
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
            self.train_samples = train_samples

            # some common augmentations for each image, can add more here
            # self.augment = T.Compose([
            #     T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            #     T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            #     T.RandomAdjustSharpness(sharpness_factor=2),
            # ])

    def __len__(self):
        return self.num_samples if self.mode == "test" else self.train_samples

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
        grid = torch.zeros((grid_rows * h, grid_cols * w)) # black background

        for i, digit_img in enumerate(selected_imgs):
            row, col = divmod(i, grid_cols)
            y0, x0 = row * h, col * w
            # Augment per digit (convert to 1xHxW tensor before applying) 
            grid[y0:y0 + h, x0:x0 + w] = digit_img             
            # grid[y0:y0 + h, x0:x0 + w] = self.augment(digit_img.unsqueeze(0)).squeeze(0)

        # resized = self.resize(grid.unsqueeze(0))
        # resized = torch.clamp((resized - resized.min()) / (resized.max() - resized.min() + 1e-8), min=0.0)  # Normalize

        resized = self.resize(grid.unsqueeze(0))
        label_sequence = ["<s>"] + [str(l) for l in selected_labels] + ["<eos>"]
        return resized, label_sequence