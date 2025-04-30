import argparse
import flag_gems
import os
from datetime import datetime
import torch
import torchvision
import tqdm
from utils.logger import StatsLogger

import torch.nn as nn
import torch.optim as optim
from models import *


def get_dataloaders(batch_size, num_workers):
    """数据加载：CIFAR10，随机裁剪、翻转、标准化"""
    transform_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )
    transform_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return trainloader, testloader


def train_one_epoch(model, loader, criterion, optimizer, device):
    """单轮训练，记录每步耗时和累计损失"""
    model.train()
    running_loss = 0.0
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    for inputs, targets in tqdm.tqdm(loader, desc="Train", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        start_evt.record()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        end_evt.record()
        torch.cuda.synchronize()
        batch_loss = loss.item() * inputs.size(0)
        running_loss += batch_loss
        stats_logger.log_round(start_evt.elapsed_time(end_evt) / 1000, batch_loss)
    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    """验证，计算损失和准确率"""
    model.eval()
    val_loss = 0.0
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in tqdm.tqdm(loader, desc="Val  ", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    return val_loss / len(loader.dataset), correct / total


def train(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    """完整训练流程：多轮训练 + 验证 + 最佳模型保存"""
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        stats_logger.log_epoch(train_loss, val_loss, val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_cifar_flaggems.pth")


def main():
    parser = argparse.ArgumentParser(description="Benchmark CIFAR10 with FlagGems")
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training and validation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count(),
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--enable-flaggems",
        action="store_true",
        help="Enable FlagGems operator optimization",
    )
    parser.add_argument(
        "--stat-dir", type=str, default="stats", help="Directory path for statistics"
    )
    parser.add_argument(
        "--stat-name", type=str, default=None, help="File name to store statistics"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="VGG16",
        choices=[
            "VGG16",
            "ResNet18",
            "PreActResNet18",
            "GoogLeNet",
            "DenseNet121",
            "ResNeXt29_2x64d",
            "MobileNet",
            "MobileNetV2",
            "DPN92",
            "ShuffleNetG2",
            "SENet18",
            "ShuffleNetV2",
            "EfficientNetB0",
            "RegNetX_200MF",
            "SimpleDLA",
        ],
        help="Which architecture to train (default: %(default)s)",
    )
    args = parser.parse_args()

    global stats_logger
    stats_logger = StatsLogger(args.stat_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders(args.batch_size, args.num_workers)

    # 构建模型（VGG16），并行 + 冻结部分层 :contentReference[oaicite:7]{index=7}
    MODEL_MAP = {
        "VGG16": lambda: VGG("VGG16"),
        "ResNet18": ResNet18,
        "PreActResNet18": PreActResNet18,
        "GoogLeNet": GoogLeNet,
        "DenseNet121": DenseNet121,
        "ResNeXt29_2x64d": ResNeXt29_2x64d,
        "MobileNet": MobileNet,
        "MobileNetV2": MobileNetV2,
        "DPN92": DPN92,
        "ShuffleNetG2": ShuffleNetG2,
        "SENet18": SENet18,
        "ShuffleNetV2": lambda: ShuffleNetV2(1),
        "EfficientNetB0": EfficientNetB0,
        "RegNetX_200MF": RegNetX_200MF,
        "SimpleDLA": SimpleDLA,
    }
    model = MODEL_MAP[args.model]().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
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
            )
    finally:
        # 结果保存
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


if __name__ == "__main__":
    main()
