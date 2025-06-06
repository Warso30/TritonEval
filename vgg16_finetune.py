import argparse
import flag_gems
import os
from datetime import datetime
import torch
import torchvision
import tqdm
from utils.logger import StatsLogger, redirect_stdout
from utils.log_analyzer import analyze_log


def get_dataloaders(batch_size, num_workers):
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
    start_timer = torch.cuda.Event(enable_timing=True)
    end_timer = torch.cuda.Event(enable_timing=True)
    round_num = 1
    for inputs, targets in tqdm.tqdm(loader, desc="Train", leave=False):
        print(f"round: {round_num}")
        round_num += 1
        start_timer.record()
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        end_timer.record()
        torch.cuda.synchronize()
        round_loss = loss.item() * inputs.size(0)
        running_loss += round_loss
        stats_logger.log_round(start_timer.elapsed_time(end_timer) / 1000, round_loss)
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


def train(
    model, train_loader, val_loader, criterion, optimizer, device, epochs, log_path
):
    best_acc = 0.0

    with redirect_stdout(log_path):
        for epoch in range(1, epochs + 1):
            print(f"epoch: {epoch}")
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            stats_logger.log_epoch(train_loss, val_loss, val_acc)

            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "best_vgg16_flaggems.pth")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune VGG16")
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training and validation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count(),
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--enable-flaggems", action="store_true", help="Enable FlagGems"
    )
    parser.add_argument(
        "--stat-dir", type=str, default="stats", help="Directory path for statistics"
    )
    parser.add_argument(
        "--stat-name", type=str, default=None, help="File name to store statistics"
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="stats/log.json",
        help="File path to the log file",
    )
    parser.add_argument(
        "--ana-path",
        type=str,
        default="stats/ana.json",
        help="File path to save the analysis result",
    )
    args = parser.parse_args()

    global stats_logger
    stats_logger = StatsLogger(args.stat_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders(args.batch_size, args.num_workers)
    model = build_model(10).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4
    )

    try:
        if args.enable_flaggems:
            with flag_gems.use_gems():
                train(
                    model,
                    train_loader,
                    val_loader,
                    criterion,
                    optimizer,
                    device,
                    args.epochs,
                    args.log_path,
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
                args.log_path,
            )
    finally:
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
            date: str = datetime.now().strftime("%m_%d_%H_%M_%S")
            stats_logger.save(f"vgg16_{autotuner}_autotuner_{date}.json")
        analyze_log(args.log_path, args.ana_path)


if __name__ == "__main__":
    main()
