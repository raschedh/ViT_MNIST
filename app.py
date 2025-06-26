import streamlit as st
import torch
import os
import random
from math import ceil, sqrt
from glob import glob
from torchvision.io import read_image
import torchvision.transforms as T
from vit_enc_dec import VisionTransformer
from dataloader import VOCAB, IDX_TO_TOKEN
from PIL import Image
import matplotlib.pyplot as plt

# Load model once
@st.cache_resource
def load_model():
    model = VisionTransformer(image_shape=(224, 224),
                              patch_size=28,
                              channels=1,
                              embed_dim=64,
                              vocab_size=len(VOCAB),
                              encoder_layers=8,
                              decoder_layers=8,
                              attention_heads=2)

    model.load_state_dict(torch.load("best_model.pth"))
    model.to(DEVICE)
    model.eval()
    return model

def generate_grid_sample(num_digits: int, image_paths: list[str], output_size: int = 224):
    indices = random.sample(range(len(image_paths)), num_digits)
    selected_paths = [image_paths[i] for i in indices]

    selected_imgs = [read_image(p)[0].float() / 255.0 for p in selected_paths]  # Grayscale (H, W)
    selected_labels = [int(os.path.basename(os.path.dirname(p))) for p in selected_paths]

    img_size = selected_imgs[0].shape
    resize = T.Resize((output_size, output_size))

    grid_cols = ceil(sqrt(num_digits))
    grid_rows = ceil(num_digits / grid_cols)
    h, w = img_size
    grid = torch.zeros((grid_rows * h, grid_cols * w))

    for j, digit_img in enumerate(selected_imgs):
        row, col = divmod(j, grid_cols)
        y0, x0 = row * h, col * w
        grid[y0:y0 + h, x0:x0 + w] = digit_img

    resized = resize(grid.unsqueeze(0))
    label_sequence = ["<s>"] + [str(l) for l in selected_labels] + ["<eos>"]

    return resized, label_sequence, grid

# --- Main Streamlit App ---
st.title("Digit Grid Prediction App (ViT)")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MNIST_PATH = "MNIST_dataset/test"
image_paths = glob(os.path.join(MNIST_PATH, "**", "*.png"), recursive=True)
model = load_model()

if st.button("Generate Random Grid and Predict"):
    num_digits = random.randint(2, 16)
    sample_image, labels, raw_grid = generate_grid_sample(num_digits, image_paths)

    # Display the image grid
    st.subheader(f"Input Grid of {num_digits} Digits")
    fig, ax = plt.subplots()
    ax.imshow(raw_grid, cmap="gray")
    ax.axis("off")
    st.pyplot(fig)

    sample_image = sample_image.to(DEVICE)
    decoder_input = torch.tensor([[VOCAB["<s>"]]], device=DEVICE)

    predicted_labels = []

    with torch.no_grad():
        for _ in range(20):
            logits, _ = model(sample_image, decoder_input)
            next_token_logits = logits[:, -1, :]
            prediction = torch.argmax(next_token_logits, dim=-1)
            predicted_labels.append(prediction.item())
            if prediction.item() == VOCAB["<eos>"]:
                predicted_labels = predicted_labels[:-1]
                break
            decoder_input = torch.cat([decoder_input, prediction.unsqueeze(0)], dim=1)

    target_label_ids = [VOCAB[token] for token in labels[1:-1]]
    predicted_label_ids = predicted_labels[:-1]

    correct_tokens = sum(p == t for p, t in zip(predicted_label_ids, target_label_ids))
    total_tokens = len(target_label_ids)
    test_token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0

    st.subheader("Prediction Results")
    st.write("**Predicted Tokens:**", [IDX_TO_TOKEN[i] for i in predicted_labels])
    st.write("**Ground Truth:**", labels[1:-1])
    st.write(f"**Token Accuracy:** {test_token_acc:.2%}")
