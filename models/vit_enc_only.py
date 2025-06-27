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

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor):    
        
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
        
        # APPLY SOFTMAX AFTER CONCAT
        # softmax across the last dim to get a prob. distribution
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
        num_classes: int,
        encoder_layers: int,
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
        # linear layer to project the flattend patches across channels to D dimensional vectors 
        # we need to pad with zeros to make it exactly divisible
        self.encoder_embed = nn.Linear(self.patch_size **2 * self.channels, self.embed_dim)

        self.encoder_layers = EncoderStack(num_layers=encoder_layers, 
                                           embed_dim=self.embed_dim, 
                                           attention_heads=attention_heads, 
                                           scale=4)

        self.position_embedding  = PositionEmbedding(embed_dim=self.embed_dim, max_len= self.N + 1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.classification = nn.Linear(embed_dim, num_classes)
    
    def patch(self, image: Tensor):
        # split the image into N patches, it may be 3 dimensional
        # first we reshape into blocks so we get (Y blocks of height, P, X blocks along width, P, C)
        batch_size = image.shape[0]
        patches = image.reshape(batch_size, self.patches_along_height, self.patch_size, self.patches_along_width, self.patch_size, self.channels)

        # we permute (to group the patches) to get a grid of patches (batch_size, Y blocks of height, X blocks along width, P, P, C)
        patches = patches.permute(0, 1, 3, 2, 4, 5)

        # we reshape to (batch_size, N, P, P, C) to go from a group of patches to a "list" of 2D patches
        patches = patches.reshape(batch_size, self.N, self.patch_size, self.patch_size, self.channels)

        # we flatten to (batch_size, N , P * P * C) to get a list of flattened patches
        patches = patches.reshape(batch_size, self.N, self.patch_size * self.patch_size * self.channels)  # (batch_size,, N, P²·C)

        return patches
    
    def encode(self, patches: Tensor):
        # pass them through the linear projection  - output is (batch_size, N, D)
        embedded_patches = self.encoder_embed(patches)

        # inject patch class embedding and then positional embedding
        cls_token = self.cls_token.expand(embedded_patches.shape[0], 1, self.embed_dim)  # shape: (B, 1, D)
        embedded_patches = torch.concat([cls_token, embedded_patches], dim=1)

        embedded_patches = self.position_embedding(embedded_patches)
        # send the embedded patches through the encoder, the encoder preserves the embed_dim 
        # the patches are of shape (B, N+1, D) and the output is also (B, N+1, D) after the encoder as transformer encoder does not change shape        
        encoder_out = self.encoder_layers(embedded_patches)

        return encoder_out
    
    def forward(self, image: Tensor):

        patches = self.patch(image)
        encoder_output = self.encode(patches)
        # extract the cls token from the encoded patches and pass this through a single linear layer
        cls_token = encoder_output[:, 0, :] 
        classes = self.classification(cls_token)
        return classes

if __name__ == "__main__":

    EPOCHS = 15
    BATCH_SIZE = 32
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Timestamped directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("training_runs/saved_runs", f"vit_mnist_{timestamp}")
    model_dir = os.path.join("training_runs/saved_models", f"vit_mnist_{timestamp}")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # TensorBoard writer setup
    writer = SummaryWriter(log_dir=run_dir)

    # Load MNIST dataset
    x_train, y_train, x_test, y_test = load_mnist("MNIST_dataset")

    # Convert to tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Create datasets and loaders
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Model
    model = VisionTransformer(image_shape=(28, 28),
                              patch_size=14,
                              channels=1,
                              embed_dim=32,
                              num_classes=10,
                              encoder_layers=1,
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