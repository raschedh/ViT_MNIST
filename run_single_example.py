import torch
import os
import random
from math import ceil, sqrt
from glob import glob
from torchvision.io import read_image
import torchvision.transforms as T
from vit_enc_dec import VisionTransformer
from dataloader import VOCAB, IDX_TO_TOKEN


def generate_grid_sample(num_digits: int, image_paths: list[str], output_size: int = 224):
    # Collect all image paths

    # Randomly choose indices of images
    indices = random.sample(range(len(image_paths)), num_digits)
    selected_paths = [image_paths[i] for i in indices]

    # Read selected images and labels
    selected_imgs = [read_image(p)[0].float() / 255.0 for p in selected_paths]  # Grayscale (H, W)
    selected_labels = [int(os.path.basename(os.path.dirname(p))) for p in selected_paths]

    img_size = selected_imgs[0].shape  # (H, W)
    resize = T.Resize((output_size, output_size))

    # Create an empty grid canvas
    grid_cols = ceil(sqrt(num_digits))
    grid_rows = ceil(num_digits / grid_cols)
    h, w = img_size
    grid = torch.zeros((grid_rows * h, grid_cols * w))  # Black background

    for j, digit_img in enumerate(selected_imgs):
        row, col = divmod(j, grid_cols)
        y0, x0 = row * h, col * w
        grid[y0:y0 + h, x0:x0 + w] = digit_img

    resized = resize(grid.unsqueeze(0))  # shape: (1, H, W)
    label_sequence = ["<s>"] + [str(l) for l in selected_labels] + ["<eos>"]

    return resized, label_sequence



if __name__ == "__main__":

    MNIST_PATH = "MNIST_dataset/test"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_paths = glob(os.path.join(MNIST_PATH, "**", "*.png"), recursive=True)

    # Model
    model = VisionTransformer(image_shape=(224, 224),
                              patch_size=28,
                              channels=1,
                              embed_dim=64,
                              vocab_size=len(VOCAB), 
                              encoder_layers=8,
                              decoder_layers=8,
                              attention_heads=2)

    model.load_state_dict(torch.load("models_grid_data/vit_mnist_20250625_155804/best_model.pth"))
    model.to(DEVICE)
    model.eval()

    sample_image, labels = generate_grid_sample(num_digits=4, image_paths=image_paths)

    correct_tokens = 0
    total_tokens = len(labels) - 2

    sample_image = sample_image.to(DEVICE)
    predicted_labels = []

    # Start with the <s> token
    decoder_input = torch.tensor([[VOCAB["<s>"]]], device=DEVICE)  # shape: (1, 1)

    with torch.no_grad():
        for _ in range(20):  # Max decode steps to prevent infinite loop
            logits, _ = model(sample_image, decoder_input)  # logits: (1, seq_len, vocab_size)
            next_token_logits = logits[:, -1, :]  # get last token's logits
            prediction = torch.argmax(next_token_logits, dim=-1)  # shape: (1,)

            predicted_labels.append(prediction.item())

            if prediction.item() == VOCAB["<eos>"]:
                break

            # Append to decoder input
            decoder_input = torch.cat([decoder_input, prediction.unsqueeze(0)], dim=1)

    # Convert string labels to vocab indices
    target_label_ids = [VOCAB[token] for token in labels[1:-1]]  # exclude <s> and <eos>
    predicted_label_ids = predicted_labels[:-1]  # exclude <eos>

    correct_tokens = sum(p == t for p, t in zip(predicted_label_ids, target_label_ids))
    total_tokens = len(target_label_ids)
    test_token_acc = correct_tokens / total_tokens

    print("Predicted tokens:", [IDX_TO_TOKEN[i] for i in predicted_labels])
    print("True tokens:", labels[1:-1])
    print(f"Token Accuracy: {test_token_acc:.2%}")


