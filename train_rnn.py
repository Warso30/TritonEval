import argparse
import flag_gems
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from utils.logger import StatsLogger
from datasets import load_dataset
from collections import Counter
from torch.utils.data import DataLoader


def tokenize(text):
    return text.lower().split()


def collate_batch(batch, stoi, max_length):
    texts, labels = [], []
    pad_idx = stoi["<pad>"]
    unk_idx = stoi["<unk>"]
    for example in batch:
        tokens = tokenize(example["text"])
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens = tokens + ["<pad>"] * (max_length - len(tokens))
        ids = [stoi.get(t, unk_idx) for t in tokens]
        texts.append(torch.tensor(ids, dtype=torch.long))
        labels.append(example["label"])
    texts = torch.stack(texts)
    labels = torch.tensor(labels, dtype=torch.long)
    return texts, labels


class TextRNNClassifier(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, hidden_dim, output_dim, rnn_type, num_layers
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if rnn_type == "RNN":
            self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True)
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        hidden_last = hidden[-1]
        return self.fc(hidden_last)


def get_dataloaders(batch_size, num_workers, max_length, max_vocab_size):
    # load imdb
    raw = load_dataset("imdb")
    train_data = raw["train"]
    test_data = raw["test"]

    counter = Counter()
    for example in train_data:
        counter.update(tokenize(example["text"]))
    most_common = [tok for tok, _ in counter.most_common(max_vocab_size)]
    itos = ["<pad>", "<unk>"] + most_common
    stoi = {tok: idx for idx, tok in enumerate(itos)}
    vocab_size = len(itos)

    collator = lambda batch: collate_batch(batch, stoi, max_length)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader, vocab_size


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    for texts, labels in tqdm.tqdm(loader, desc="Train", leave=False):
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        start_evt.record()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        end_evt.record()
        torch.cuda.synchronize()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        stats_logger.log_round(start_evt.elapsed_time(end_evt) / 1000, loss.item())
    return total_loss / total_samples


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in tqdm.tqdm(loader, desc="Val  ", leave=False):
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train(
    model, train_loader, val_loader, criterion, optimizer, device, epochs, save_path
):
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        stats_logger.log_epoch(train_loss, val_loss, val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(
                f"Validation accuracy improved to {val_acc:.4f}, saved to {save_path}"
            )
        else:
            print(f"No improvement from {best_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark RNN/LSTM on IMDB")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count())
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-vocab-size", type=int, default=25000)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument(
        "--rnn-type", type=str, choices=["RNN", "LSTM", "GRU"], default="LSTM"
    )
    parser.add_argument("--enable-flaggems", action="store_true")
    parser.add_argument("--stat-dir", type=str, default="stats")
    parser.add_argument("--stat-name", type=str, default=None)
    parser.add_argument("--model-save-name", type=str, default="best_text_rnn.pth")
    args = parser.parse_args()

    torch.manual_seed(42)
    global stats_logger
    stats_logger = StatsLogger(args.stat_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading IMDB via HuggingFace Datasets...")
    train_loader, val_loader, vocab_size = get_dataloaders(
        args.batch_size, args.num_workers, args.max_length, args.max_vocab_size
    )

    model = TextRNNClassifier(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        output_dim=2,
        rnn_type=args.rnn_type,
        num_layers=args.num_layers,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    save_path = os.path.join(args.stat_dir, args.model_save_name)
    print(f"Best model will be saved to: {save_path}")

    start_time = datetime.now()
    print(f"Starting training at {start_time}")
    if args.enable_flaggems and torch.cuda.is_available():
        with flag_gems.use_gems():
            train(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                device,
                args.epochs,
                save_path,
            )
    else:
        train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            args.epochs,
            save_path,
        )
    end_time = datetime.now()
    print(f"Training finished at {end_time}. Duration: {end_time - start_time}")
    
    if args.enable_flaggems:
        autotuner = os.getenv("TRITON_AUTOTUNE", "default")
    else:
        autotuner = "no"
    if args.stat_name:
        stats_logger.save(
            args.stat_name
            if args.stat_name.endswith(".json")
            else f"{args.stat_name}.json"
        )
    else:
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_logger.save(f"{args.rnn_type}_{autotuner}_{date_str}.json")


if __name__ == "__main__":
    main()
