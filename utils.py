import os
import numpy as np
from glob import glob
from PIL import Image

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