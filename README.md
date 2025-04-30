# TritonEval
This repo is used to evaluate the performance of triton

## Setup

### Create Virtual Environment

```bash
python3 -m venv .venv --prompt eval
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```
Based on our experience, Triton could be built faster with LLVM toolchain, so you can provide environment variables to enable this option `TRITON_BUILD_WITH_CLANG_LLD` like
```bash
TRITON_BUILD_WITH_CLANG_LLD=1 pip install -r requirements.txt
```

## Code Style
Run
```bash
black *.py
```
to format all codes before commit.

# CIFAR-10 Training
Run
```bash
TRITON_AUTOTUNE=<optimal_choice> python train_cifar10.py --enable-flaggems --model=<model_name>
```
### Parameters:
* `TRITON_AUTOTUNE=<optimal_choice>`: default, ...
* `--model=<model_name>`: choices=[
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
        ]
