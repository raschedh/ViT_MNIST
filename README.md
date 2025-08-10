# Vision Transformer on the MNIST Dataset

A custom implementation of Vision Transformer (ViT) for digit classification, extended to encoder-decoder modeling for digit sequence recognition.

---

## 🚀 Overview

This project reimplements the Vision Transformer architecture from the original [ViT paper](https://arxiv.org/abs/2010.11929), trained on the MNIST dataset for handwritten digit recognition. In addition to a standard ViT encoder, the model is extended into an encoder-decoder architecture inspired by the Transformer model from *Attention Is All You Need*. The encoder-decoder model processes a grid of digit images to predict sequences. Both implementations are coded from scratch (no AI - you will have to trust me) following the original papers.

### 🔧 Architecture Notes

- **Encoder**: Based on the original ViT paper, using *pre-layer norm* with skip connections.
- **Decoder**: Follows *post-layer norm* as in the original Transformer paper.
- This repo mixes both approaches for the encoder-decoder. Feel free to change. 

---

## Training

Create the val/test data: 
```bash
python models/utils.py
```

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
streamlit run app.py
```

Alternatively, the Dockerfile is standalone:

```bash
docker build -t vit-mnist-app .
docker run -p 8501:8501 vit-mnist-app



