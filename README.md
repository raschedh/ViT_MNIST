# Vision Transformer on the MNIST Dataset

A custom implementation of Vision Transformer (ViT) for digit classification, extended to encoder-decoder modeling for digit sequence recognition.

---

## 🚀 Overview

This project reimplements the Vision Transformer architecture from the original [ViT paper](https://arxiv.org/abs/2010.11929), trained on the MNIST dataset for handwritten digit recognition. In addition to a standard ViT encoder, we extend the model into an encoder-decoder architecture inspired by the Transformer model from *Attention Is All You Need*.

The encoder-decoder model processes a grid of digit images to predict sequences — similar to OCR-style tasks — and is useful for structured multi-digit input scenarios.

### 🔧 Architecture Notes

- **Encoder**: Based on the original ViT paper, using *pre-layer norm* with skip connections.
- **Decoder**: Follows *post-layer norm* as in the original Transformer paper.
- This implementation mixes both approaches. Feel free to change. 

---

## Training

To train the models:

```bash
# For standard ViT encoder on MNIST
python models/vit_enc.py

# For encoder-decoder on digit grids
python models/vit_enc_dec.py
```
---

## Streamlit

To run the streamlit app:

```bash
docker build -t vit-mnist-app .
docker run -p 8501:8501 vit-mnist-app



