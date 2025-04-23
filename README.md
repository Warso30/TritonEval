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
