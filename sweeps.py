import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from dataloader import CompositeMNISTDataset, collate_fn, VOCAB
from vit_enc_dec import VisionTransformer


def main(config=None):
    with wandb.init(config=config):
        config = wandb.config

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        EPOCHS = 10
        criterion = nn.CrossEntropyLoss(ignore_index=VOCAB["<pad>"])

        # Move datasets here so they exist inside sweep runs
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

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=collate_fn)

        model = VisionTransformer(
            image_shape=(224, 224),
            patch_size=config.patch_size,
            channels=1,
            embed_dim=config.embed_dim,
            vocab_size=len(VOCAB),
            encoder_layers=config.encoder_layers,
            decoder_layers=config.decoder_layers,
            attention_heads=config.attention_heads
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            correct_tokens, total_tokens = 0, 0
            sequence_correct, total_sequences = 0, 0

            for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/10"):
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                decoder_input = targets[:, :-1]
                decoder_target = targets[:, 1:]

                logits, _ = model(images, decoder_input)
                loss = criterion(logits.reshape(-1, logits.size(-1)), decoder_target.reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                mask = (decoder_target != VOCAB["<pad>"])

                correct_tokens += (predictions == decoder_target)[mask].sum().item()
                total_tokens += mask.sum().item()
                sequence_correct += ((predictions == decoder_target) | ~mask).all(dim=1).sum().item()
                total_sequences += decoder_target.size(0)

            avg_train_loss = train_loss / len(train_loader)
            train_token_acc = correct_tokens / total_tokens
            train_seq_acc = sequence_correct / total_sequences

            # --- Evaluation ---
            model.eval()
            test_loss = 0
            correct_tokens, total_tokens = 0, 0
            sequence_correct, total_sequences = 0, 0

            with torch.no_grad():
                for images, targets in test_loader:
                    images, targets = images.to(DEVICE), targets.to(DEVICE)
                    decoder_input = targets[:, :-1]
                    decoder_target = targets[:, 1:]

                    logits, _ = model(images, decoder_input)
                    loss = criterion(logits.reshape(-1, logits.size(-1)), decoder_target.reshape(-1))
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

            scheduler.step(avg_test_loss)

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "test_loss": avg_test_loss,
                "train_token_acc": train_token_acc,
                "test_token_acc": test_token_acc,
                "train_seq_acc": train_seq_acc,
                "test_seq_acc": test_seq_acc
            })

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", type=int)
    parser.add_argument("--embed_dim", type=int)
    parser.add_argument("--encoder_layers", type=int)
    parser.add_argument("--decoder_layers", type=int)
    parser.add_argument("--attention_heads", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)

    args = parser.parse_args()
    wandb.init(config=vars(args))
    main(vars(args))
