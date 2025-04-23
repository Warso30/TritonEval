import argparse
import flag_gems
import os
import torch
import torchvision
import tqdm

flag_gems.enable()  # Enable FlagGems for accelerated Triton-backed kernels :contentReference[oaicite:2]{index=2}


def get_dataloaders(batch_size=4, num_workers=os.cpu_count()):
    mean_values = (0.5, 0.5, 0.5)
    std_values = (0.5, 0.5, 0.5)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Pad(96),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean_values, std_values),
        ]
    )
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return trainloader, testloader


def build_model(num_classes):
    """Load pretrained VGG16 and replace classifier head."""
    model = torchvision.models.vgg16(
        weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
    )  # VGG16 with ImageNet-1K weights :contentReference[oaicite:3]{index=3}
    # Freeze all feature extractor layers
    for param in model.features.parameters():
        param.requires_grad = False  # start by training only the classifier :contentReference[oaicite:4]{index=4}
    # Replace the classifier to output `num_classes` categories
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in tqdm.tqdm(loader, desc="Train", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm.tqdm(loader, desc="Val  ", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return val_loss / len(loader.dataset), correct / total


def main():
    parser = argparse.ArgumentParser(description="Fine-tune VGG16 with FlagGems")
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Path to dataset root (with train/ and val/ subfolders)",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training and validation",
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--num-classes", type=int, default=10, help="Number of target classes"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders(args.batch_size)
    model = build_model(args.num_classes).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4
    )

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:02d}  Train Loss: {train_loss:.4f}  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_vgg16_flaggems.pth")


if __name__ == "__main__":
    main()
