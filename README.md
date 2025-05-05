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
* `TRITON_AUTOTUNE=<optimal_choice>`: default, epsilon, stepwise
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

# Translation Training
Run
```bash
TRITON_AUTOTUNE=<optimal_choice> python finetune_translation.py --model t5-small --epochs 3 --batch-size 8 --dataset-name iwslt2017 --dataset-config iwslt2017-en-de --source-lang en --target-lang de --enable-flaggems
```
### Parameters:
* `TRITON_AUTOTUNE=<optimal_choice>`: default, epsilon, stepwise
* `--model=<model_name>`: choices=[
            "t5-small",
        ]

# Visualization

```bash
python utils/plotter.py [OPTIONS]
```
All options are **optional**. If you omit an option, its corresponding plot (or smoothing) will simply be skipped or use built-in defaults.

### Options

| Option                 | Argument             | Description                                                                               |
|------------------------|----------------------|-------------------------------------------------------------------------------------------|
| `--default <FILE>`     | Path to JSON file    | Plot the **default autotuner** data series from `<FILE>`.                                           |
| `--stepwise <FILE>`    | Path to JSON file    | Plot the **stepwise autotuner** data series from `<FILE>`.                                          |
| `--epsilon <FILE>`     | Path to JSON file    | Plot the **epsilon autotuner** data series from `<FILE>`.                                           |
| `--avg-len <INT>`      | Integer (e.g. 300)   | Apply a running‐average filter of length `<INT>` to each plotted series.                  |
| `--step-size <INT>`    | Integer (e.g. 100)   | When filtering, jump the window by `<INT>` samples between consecutive averages.          |

### How It Works

- **Running‐average filter** (`--avg-len`): smooths each series by averaging over a sliding window of the given length.
- **Step size** (`--step-size`): controls how far the window moves between consecutive averages. A step size smaller than `avg-len` gives overlapping windows; equal to `avg-len` gives non-overlapping blocks.

### Example
   ```bash
   python utils/plotter.py --default stats/default.json --stepwise stats/stepwise.json --epsilon stats/epsilon.json --avg-len=100 --step-size=100
   ```