import torch 
import torch.nn as nn
from typing import Tuple
from torch.utils.data import TensorDataset, DataLoader
from utils import load_mnist
from torch import Tensor 
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from datetime import datetime
from dataloader import CompositeMNISTDataset


# ----------------- DECODER CLASS (a single decoder layer) AND DECODER STACK CLASS (multiple stacked) -------------------------------
class Decoder(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 attention_heads: int,
                 scale: int):

        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layer_norm3 = nn.LayerNorm(embed_dim)

        self.masked_attention = MultiHeadAttention(attention_heads, embed_dim)
        self.attention = MultiHeadAttention(attention_heads, embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, scale * embed_dim),
            nn.GELU(),
            nn.Linear(scale * embed_dim, embed_dim)
        )

    def forward(self, x: Tensor, encoder_output: Tensor):
        
        x = self.masked_attention(x,x,x, mask = True) + x
        x = self.layer_norm1(x)

        x = self.attention(x, encoder_output,encoder_output) + x
        x = self.layer_norm2(x)

        x = self.feed_forward(x) + x
        x = self.layer_norm3(x)

        return x

class DecoderStack(nn.Module):
    def __init__(self, num_layers: int, embed_dim: int, attention_heads: int, scale: int):
        super().__init__()
        self.layers = nn.ModuleList([
            Decoder(embed_dim, attention_heads, scale)
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor, encoder_output: Tensor):
        for layer in self.layers:
            x = layer(x, encoder_output)
        return x

# ----------------- ENCODER CLASS (a single encoder layer) AND ENCODER STACK CLASS (multiple stacked) -------------------------------
class Encoder(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 attention_heads: int,
                 scale: int):

        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim) 

        self.attention = MultiHeadAttention(attention_heads, embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, scale * embed_dim),
            nn.GELU(),
            nn.Linear(scale * embed_dim, embed_dim)
        )

    def forward(self, x: Tensor):

        # patches are of shape (B, N, D)
        x_skip = x

        # it goes through layer norm
        x = self.layer_norm1(x)

        # we apply multiheaded attention which should return (B,N,D)
        x = self.attention(x, x, x)

        # we add skip connection
        x = x_skip + x 

        x_skip = x

        # we norm again
        x = self.layer_norm2(x)

        # we pass through the MLP 
        x = self.feed_forward(x)

        # we have another skip conection 
        x = x_skip + x

        return x

class EncoderStack(nn.Module):
    def __init__(self, num_layers: int, embed_dim: int, attention_heads: int, scale: int):
        super().__init__()
        self.layers = nn.ModuleList([
            Encoder(embed_dim, attention_heads, scale)
            for _ in range(num_layers)
        ])
    
    def forward(self, x: Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


# ----------------- MULTI-HEAD ATTENTION CLASS -------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, embed_dim: int):
        super().__init__()

        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)

        self.n_heads = n_heads
        self.embed_dim = embed_dim
        assert self.embed_dim % self.n_heads == 0, "embed_dim must be divisible by n_heads"

        self.D_per_head = self.embed_dim // self.n_heads
        self.linear_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, mask: bool = False):    
        
        B, N, _ = queries.shape
 
        # these are of shape (B, N, D)
        Q = self.linear_q(queries)
        K = self.linear_k(keys)
        V = self.linear_v(values)

        # reshape them from (B, N, D) to (B, N, heads, D_per_head) and then permute to get (B, heads, N, Dh)
        Q = Q.reshape(B, N, self.n_heads, self.D_per_head).permute(0,2,1,3)
        K = K.reshape(B, N, self.n_heads, self.D_per_head).permute(0,2,1,3)
        V = V.reshape(B, N, self.n_heads, self.D_per_head).permute(0,2,1,3)
                                
        # multiply queries and keys so we have (B, heads, N, N)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.D_per_head ** 0.5)

        # We need to add an upper triangular matrix of -inf (above the diagonal),
        # so that softmax assigns zero probability to future tokens.
        # The diagonal and lower triangle are allowed (not masked).
        #
        # For example, for a 3×3 sequence, the mask looks like:
        # [[ 0, -inf, -inf],
        #  [ 0,    0, -inf],
        #  [ 0,    0,    0]]
        # by default the mask is False
        if mask:
            mask_matrix = torch.triu(torch.ones(N, N), diagonal=1)
            mask_matrix = mask_matrix.masked_fill(mask_matrix == 1, float("-inf"))
            mask_matrix = mask_matrix.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, N, N)
            attention_scores = attention_scores + mask_matrix # torch takes care of broadcasting to (B, heads, N, N)
            
        # softmax across the last dim to get a prob. distribution, if mask the -inf turn to 0
        attn_weights = torch.softmax(attention_scores, dim=-1)

        # attn_weights is of shape (B, heads, N, N) and V is of shape (B, heads, N, Dh) and  and the output needs to be (B, heads, N, Dh)
        attention = torch.matmul(attn_weights, V)

        # permute so we have (B, N, heads, D)
        attention = attention.permute(0,2,1,3)

        # aggregate across heads to get (B, N, embed_dim (total D))
        attention = attention.reshape(B, N, self.embed_dim)

        # pass through one last linear layer
        attention = self.linear_layer(attention)

        return attention


# ----------------- POSITIONAL EMBEDDING CLASS -------------------------------
class PositionEmbedding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int):

        super().__init__()
        
        pe = torch.zeros(max_len, embed_dim)
        
        positions = torch.arange(0, max_len, step=1).float().unsqueeze(1)
        power = 10000 ** (torch.arange(0, embed_dim, step=2) / embed_dim) # this is the 10000^(2i/D) part in the orignal paper
        
        pe[:, 0::2] = torch.sin(positions / power)
        pe[:, 1::2] = torch.cos(positions / power)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor):
        return x + self.pe[:x.size(1), :]


# ----------------- VISION TRANSFORMER CLASS -------------------------------  
class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_shape: Tuple[int, int],  # (H, W)
        patch_size: int,
        channels: int,
        embed_dim: int,
        vocab: dict, 
        num_classes: int,
        encoder_layers: int,
        decoder_layers: int, 
        attention_heads: int
    ):
        super().__init__()
        # we need to do some maths here to deal with non divisble cases
        self.patch_size = patch_size 
        self.patches_along_height = image_shape[0] // self.patch_size
        self.patches_along_width = image_shape[1] // self.patch_size

        self.channels = channels
        self.embed_dim = embed_dim
        self.N = self.patches_along_height * self.patches_along_width

        # linear layer to project the flattend patches across channels to D dimensional vectors for downstream encoder blocks
        # we need to pad with zeros to make it exactly divisible
        self.encoder_embeddings = nn.Linear(self.patch_size **2 * self.channels, self.embed_dim)
        self.vocab = vocab 
        # we use the same embed dim as encoder projection but it doesn't have to be
        self.decoder_embeddings = nn.Embedding(num_embeddings=len(self.vocab), embedding_dim=self.embed_dim)

        self.decoder_layers = DecoderStack(num_layers=decoder_layers,
                                           embed_dim=self.embed_dim,
                                           attention_heads=attention_heads,
                                           scale=4)

        self.encoder_layers = EncoderStack(num_layers=encoder_layers, 
                                           embed_dim=self.embed_dim, 
                                           attention_heads=attention_heads, 
                                           scale=4)

        self.encoder_pos_embedding  = PositionEmbedding(embed_dim=self.embed_dim, max_len= self.N)
        self.decoder_pos_embedding = PositionEmbedding(embed_dim=self.embed_dim, max_len=len(self.vocab))

        self.linear = nn.Linear(embed_dim, num_classes)
        self.softmax = nn.Softmax(num_classes)

    def forward(self, image: Tensor, targets: Tensor):

        # split the image into N patches, it may be 3 dimensional
        # first we reshape into blocks so we get (Y blocks of height, P, X blocks along width, P, C)
        batch_size = image.shape[0]
        image = image.reshape(batch_size, self.patches_along_height, self.patch_size, self.patches_along_width, self.patch_size, self.channels)

        # we permute to get the group the patches to get a grid of patches to get (batch_size, Y blocks of height, X blocks along width, P, P, C)
        image = image.permute(0, 1, 3, 2, 4, 5)

        # we reshape to (batch_size, N, P, P, C) to go from a group of patches to a "list" of 2D patches
        image = image.reshape(batch_size, self.N, self.patch_size, self.patch_size, self.channels)

        # we flatten to (batch_size, N , P * P * C) to get a list of flattened patches
        image_patches = image.reshape(batch_size, self.N, self.patch_size * self.patch_size * self.channels)  # (batch_size,, N, P²·C)

        # pass them through the linear projection  - output is (batch_size, N, D)
        projected_patches = self.encoder_embeddings(image_patches)
        # inject patch class embedding and then positional embedding
        projected_patches_wpe = self.encoder_pos_embedding(projected_patches)

        # send the projected patches through the encoder, the encoder preserves the embed_dim 
        # the patches are of shape (B, N, D) and the output is also (B, N, D) after the encoder as transformer encoder does not change shape        
        encoder_output = self.encoder_layers(projected_patches_wpe)

        target_embeddings = self.decoder_embeddings(targets) # (B, T, D)
        target_embeddings_wpe = self.decoder_pos_embedding(target_embeddings)  # (B, T, D)
        decoder_output = self.decoder_layers(target_embeddings_wpe, encoder_output)

        class_logits = self.linear(decoder_output)
        class_probs = self.softmax(class_logits)

        return class_probs

if __name__ == "__main__":

    EPOCHS = 15
    BATCH_SIZE = 32
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Timestamped directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs_composite", f"vit_mnist_{timestamp}")
    model_dir = os.path.join("models_composite", f"vit_mnist_{timestamp}")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # TensorBoard writer setup
    writer = SummaryWriter(log_dir=run_dir)

    # Create the dataset instance (adjust path as needed)
    train_dataset = CompositeMNISTDataset(
        path="MNIST_dataset/train",      # Path to train images, generated on the fly
        output_size=224,
        min_digits=4,
        max_digits=16,
        mode="train"
    )

    test_dataset = CompositeMNISTDataset(
        path="composite_test_data.pt",      # Path to fixed test data
        output_size=224,
        min_digits=4,
        max_digits=16,
        mode="test"
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    vocab = { "<s>": 0, "<eos>": 1, "0": 2, "1": 3, "2": 4, "3": 5, "4": 6, "5": 7, "6": 8, "7": 9, "8": 10, "9": 11}
    
    # Model
    model = VisionTransformer(image_shape=(28, 28),
                              patch_size=14,
                              channels=1,
                              embed_dim=32,
                              vocab=vocab, 
                              num_classes=10,
                              encoder_layers=1,
                              decoder_layers=1,
                              attention_heads=1)

    model.to(DEVICE)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_test_loss = float('inf')
    best_model_path = os.path.join(model_dir, "best_model.pth")

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False)
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item())

        train_accuracy = 100 * correct / total
        avg_train_loss = total_loss / len(train_loader)

        # Evaluate on test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)

        # Logging to TensorBoard
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
        writer.add_scalar("Loss/Test", avg_test_loss, epoch)

        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Loss: {avg_test_loss:.4f}")

        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Best model saved at epoch {epoch+1} with Test Loss: {best_test_loss:.4f}")

    writer.close()