import os
import numpy as np
from glob import glob
from PIL import Image
import random
import torch
from math import ceil, sqrt
import torchvision.transforms as T
from torchvision.io import read_image

# ----------------- LOAD THE MNIST DATASET -------------------------------
def load_mnist(path: str):
    train_dir = os.path.join(path, "train")
    test_dir = os.path.join(path, "test")

    train_paths = glob(os.path.join(train_dir, "**", "*.png"), recursive=True)
    test_paths = glob(os.path.join(test_dir, "**", "*.png"), recursive=True)

    x_train = [np.array(Image.open(p).convert("L")) for p in train_paths]
    y_train = [int(os.path.basename(os.path.dirname(p))) for p in train_paths]

    x_test = [np.array(Image.open(p).convert("L")) for p in test_paths]
    y_test = [int(os.path.basename(os.path.dirname(p))) for p in test_paths]

    return x_train, y_train, x_test, y_test


# ------------------ CREATES FIXED TEST DATASET FOR EVALUATION -------------

def generate_fixed_mnist_grid_data(mnist_path: str, num_samples: int, output_file: str,
                        output_size: int = 224, min_digits: int = 2, max_digits: int = 16):

    # Load all MNIST digit images and labels
    paths = glob(os.path.join(mnist_path, "**", "*.png"), recursive=True)
    labels = [int(os.path.basename(os.path.dirname(p))) for p in paths]
    images = [read_image(p)[0].float() / 255.0 for p in paths]  # (H, W), grayscale
    img_size = images[0].shape  # (H, W)
    resize = T.Resize((output_size, output_size))

    images_out = []
    labels_out = []

    for i in range(num_samples):

        print(f"Generating {i}/{num_samples}", end="\r")
        num_digits = random.randint(min_digits, max_digits)
        indices = random.sample(range(len(images)), num_digits)
        selected_imgs = [images[i] for i in indices]
        selected_labels = [labels[i] for i in indices]

        grid_cols = ceil(sqrt(num_digits))
        grid_rows = ceil(num_digits / grid_cols)
        h, w = img_size
        grid = torch.ones((grid_rows * h, grid_cols * w))  # white background

        for j, digit_img in enumerate(selected_imgs):
            row, col = divmod(j, grid_cols)
            y0, x0 = row * h, col * w
            grid[y0:y0 + h, x0:x0 + w] = digit_img

        resized = resize(grid.unsqueeze(0))  # shape: (1, H, W)
        label_sequence = ["<s>"] + [str(l) for l in selected_labels] + ["<eos>"]

        images_out.append(resized)
        labels_out.append(label_sequence)


    torch.save({"images": images_out, "labels": labels_out}, output_file)
    print(f"\n Test data saved to {output_file}")


if __name__ == "__main__":
    # create a fixed test dataset, the train set will be generated on the fly using the dataloader
    generate_fixed_mnist_grid_data(
        mnist_path="MNIST_dataset/test",
        num_samples=100000,
        output_file="composite_test_data.pt",
        output_size=224,
        min_digits=2,
        max_digits=16
    )