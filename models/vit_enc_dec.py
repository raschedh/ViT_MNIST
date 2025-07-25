import torch 
import torch.nn as nn
from typing import Tuple
from torch.utils.data import DataLoader
from torch import Tensor 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from datetime import datetime
from models.dataloader import CompositeMNISTDataset, collate_fn, VOCAB, IDX_TO_TOKEN
from torch.optim.lr_scheduler import ReduceLROnPlateau

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        
        x = self.masked_attention(x,x,x, apply_mask=True) + x
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

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, apply_mask: bool = False):    
        
        B, N_q, _ = queries.shape
        N_k = keys.shape[1]
        N_v = values.shape[1] 
        # for sake of showing we have N_v seperate 
        # but it is same as N_k otherwise later (see later) matmul doesn't work
 
        # these are of shape (B, N_{q,k,v}, D)
        Q = self.linear_q(queries)
        K = self.linear_k(keys)
        V = self.linear_v(values)

        # reshape them from (B, N_{q,k,v}, D) to (B, N_{q,k,v}, heads, Dh) and then permute to (B, heads, N_{q,k,v}, Dh)
        Q = Q.reshape(B, N_q, self.n_heads, self.D_per_head).permute(0,2,1,3)
        K = K.reshape(B, N_k, self.n_heads, self.D_per_head).permute(0,2,1,3)
        V = V.reshape(B, N_v, self.n_heads, self.D_per_head).permute(0,2,1,3)
                                
        # multiply queries and keys so we have 
        # Q = (B, heads, N_q, Dh)
        # K.transpose(-2,-1) = (B, heads, Dh, N_k)
        # such that Q @ K.T = (B, heads, N_q, N_k)
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

        # note when we do masked attention N_q = N_k so it's square, 
        # we do not need a mask for cross attention when N_q != N_k is not true necessarily 

        if apply_mask:
            mask = torch.triu(torch.ones(N_q, N_q), diagonal=1).to(DEVICE)
            mask = mask.masked_fill(mask == 1, float("-inf"))
            mask = mask.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, N, N)
            attention_scores = attention_scores + mask # torch takes care of broadcasting to (B, heads, N, N)
            
        # softmax across the last dim to get a prob. distribution, if mask the -inf turn to 0
        attn_weights = torch.softmax(attention_scores, dim=-1)

        # attn_weights = (B, heads, N_q, N_k)
        # V = (B, heads, N_v, Dh) 
        # attn_weight @ V = (B, heads, N_q, Dh) as N_k == N_v otherwise it doesn't work especially for cross attention
        # The decoder queries are "paying attention" to encoder keys.
        # Cross attention then selects which of the encoder values are most important. We essentially select most important features
        # from the encoder output.
        attention = torch.matmul(attn_weights, V)

        # permute so we have (B, heads, N_q, Dh) --> (B, N_q, heads, Dh)
        attention = attention.permute(0,2,1,3)

        # aggregate across heads to get (B, N, D)
        attention = attention.reshape(B, N_q, self.embed_dim)

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
        vocab_size: int, 
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
        self.encoder_embed = nn.Linear(self.patch_size **2 * self.channels, self.embed_dim)
        self.vocab_size = vocab_size 
        # we use the same embed dim as encoder projection but it doesn't have to be
        self.decoder_embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim)

        self.decoder_layers = DecoderStack(num_layers=decoder_layers,
                                           embed_dim=self.embed_dim,
                                           attention_heads=attention_heads,
                                           scale=4)

        self.encoder_layers = EncoderStack(num_layers=encoder_layers, 
                                           embed_dim=self.embed_dim, 
                                           attention_heads=attention_heads, 
                                           scale=4)

        self.encoder_pos_embed  = PositionEmbedding(embed_dim=self.embed_dim, max_len= self.N)
        self.decoder_pos_embed = PositionEmbedding(embed_dim=self.embed_dim, max_len=20)
        # the labels will only have a max of 16 numbers

        self.linear = nn.Linear(embed_dim, self.vocab_size)
        self.dropout = nn.Dropout(0.1)

    def patch(self, image: Tensor):

        # split the image into N patches, it may be 3 dimensional
        # first we reshape into blocks so we get (Y blocks of height, P, X blocks along width, P, C)
        batch_size = image.shape[0]
        patches = image.reshape(batch_size, self.patches_along_height, self.patch_size, self.patches_along_width, self.patch_size, self.channels)

        # we permute to get a grid of patches (batch_size, Y blocks of height, X blocks along width, P, P, C)
        patches = patches.permute(0, 1, 3, 2, 4, 5)

        # we reshape to (batch_size, N, P, P, C) to go from a group of patches to a "list" of 2D patches
        patches = patches.reshape(batch_size, self.N, self.patch_size, self.patch_size, self.channels)

        # we flatten to (batch_size, N , P * P * C) to get a list of flattened patches
        patches = patches.reshape(batch_size, self.N, self.patch_size * self.patch_size * self.channels)  # (batch_size,, N, P²·C)
        return patches

    def encode(self, patches: Tensor):

        # pass them through the linear projection  - output is (batch_size, N, D)
        patches = self.encoder_embed(patches)
        # inject patch class embedding and then positional embedding
        patches = self.encoder_pos_embed(patches)
        patches = self.dropout(patches)  # dropout

        # send the projected patches through the encoder, the encoder preserves the embed_dim 
        # the patches are of shape (B, N, D) and the output is also (B, N, D) after the encoder as transformer encoder does not change shape        
        encoder_out = self.encoder_layers(patches)
        encoder_out = self.dropout(encoder_out)

        return encoder_out

    def decode(self, decoder_input: Tensor, encoder_out: Tensor):

        decoder_embed = self.decoder_embed(decoder_input) # (B, T, D)
        decoder_embed = self.decoder_pos_embed(decoder_embed)  # (B, T, D)
        decoder_embed = self.dropout(decoder_embed)   #  decoder embedding dropout
        decoder_output = self.decoder_layers(decoder_embed, encoder_out)

        return decoder_output

    def forward(self, image: Tensor, decoder_input: Tensor):

        patches = self.patch(image)
        encoder_out = self.encode(patches)
        decoder_output = self.decode(decoder_input, encoder_out)
        class_logits = self.linear(decoder_output)        
        class_probs = torch.softmax(class_logits, dim=-1)  # for prob distribution visualisation

        return class_logits, class_probs

if __name__ == "__main__":

    EPOCHS = 50
    BATCH_SIZE = 32
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Timestamped directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("training_runs/saved_runs_grid_data", f"vit_mnist_{timestamp}")
    model_dir = os.path.join("training_runs/saved_models_grid_data", f"vit_mnist_{timestamp}")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # TensorBoard writer setup
    writer = SummaryWriter(log_dir=run_dir)

    # Create the dataset instance (adjust path as needed)
    train_dataset = CompositeMNISTDataset(
        path="MNIST_dataset/train",      # Path to train images, generated on the fly
        output_size=224,
        mode="train",
        min_digits=2,
        max_digits=16,
        train_samples=100000 # length of samples per epoch
    )

    test_dataset = CompositeMNISTDataset(
        path="composite_test_data.pt",      # Path to fixed test data
        output_size=224,
        mode="test",
        min_digits=None,
        max_digits=None
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Model
    model = VisionTransformer(image_shape=(224, 224),
                              patch_size=28,
                              channels=1,
                              embed_dim=64,
                              vocab_size=len(VOCAB), 
                              encoder_layers=8,
                              decoder_layers=8,
                              attention_heads=2)

    # model.load_state_dict(torch.load("models_grid_data/vit_mnist_20250625_172227/best_model.pth"))

    print(f"Number of trainable params {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model.to(DEVICE)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=VOCAB["<pad>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    best_test_loss = float('inf')
    best_model_path = os.path.join(model_dir, "best_model.pth")

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False)

        train_loss = 0
        correct_tokens = 0
        total_tokens = 0
        sequence_correct = 0
        total_sequences = 0

        for images, targets in train_bar:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            decoder_input = targets[:, :-1]
            decoder_target = targets[:, 1:]


            logits, probs = model(images, decoder_input)
            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = decoder_target.reshape(-1)

            loss = criterion(logits_flat, target_flat)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute accuracy
            predictions = torch.argmax(logits, dim=-1)  # (B, T)
            mask = (decoder_target != VOCAB["<pad>"])

            correct_tokens += (predictions == decoder_target)[mask].sum().item()
            total_tokens += mask.sum().item()

            sequence_correct += ((predictions == decoder_target) | ~mask).all(dim=1).sum().item()
            total_sequences += decoder_target.size(0)

            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        train_token_acc = correct_tokens / total_tokens
        train_seq_acc = sequence_correct / total_sequences
        avg_train_loss = train_loss / len(train_loader)

        # Evaluate on test set
        model.eval()
        test_loss = 0
        correct_tokens = 0
        total_tokens = 0
        sequence_correct = 0
        total_sequences = 0

        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                decoder_input = targets[:, :-1]
                decoder_target = targets[:, 1:]

                logits, probs = model(images, decoder_input)
                logits_flat = logits.reshape(-1, logits.size(-1))
                target_flat = decoder_target.reshape(-1)

                loss = criterion(logits_flat, target_flat)
                test_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1)
                mask = (decoder_target != VOCAB["<pad>"])

                correct_tokens += (predictions == decoder_target)[mask].sum().item()
                total_tokens += mask.sum().item()

                sequence_correct += ((predictions == decoder_target) | ~mask).all(dim=1).sum().item()
                total_sequences += decoder_target.size(0)

        avg_test_loss = test_loss / len(test_loader)
        test_token_acc = correct_tokens / total_tokens
        test_seq_acc = sequence_correct / total_sequences

        # Scheduler step
        scheduler.step(avg_test_loss)

        # Logging to TensorBoard
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Test", avg_test_loss, epoch)
        writer.add_scalar("Accuracy/Train_Token", train_token_acc, epoch)
        writer.add_scalar("Accuracy/Test_Token", test_token_acc, epoch)
        writer.add_scalar("Accuracy/Train_Sequence", train_seq_acc, epoch)
        writer.add_scalar("Accuracy/Test_Sequence", test_seq_acc, epoch)

        print(f"Epoch [{epoch+1}/{EPOCHS}], "
              f"Train Loss: {avg_train_loss:.4f}, Token Acc: {train_token_acc:.4f}, Seq Acc: {train_seq_acc:.4f}, "
              f"Test Loss: {avg_test_loss:.4f}, Token Acc: {test_token_acc:.4f}, Seq Acc: {test_seq_acc:.4f}")

        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Best model saved at epoch {epoch+1} with Test Loss: {best_test_loss:.4f}")

    writer.close()