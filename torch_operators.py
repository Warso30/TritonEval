from tqdm import tqdm
import torch
import flag_gems
import argparse
import json
from datetime import datetime
import random
import os


random.seed(42)


def benchmark_torch_ops(op_func, iters=1000):
    res = []
    start_timer = torch.cuda.Event(enable_timing=True)
    end_timer = torch.cuda.Event(enable_timing=True)
    for _ in tqdm(range(iters)):
        start_timer.record()
        op_func()
        end_timer.record()
        torch.cuda.synchronize()
        res.append(start_timer.elapsed_time(end_timer))
    return res


def random_mm(device):
    m = random.randint(64, 2048)
    k = random.randint(64, 2048)
    n = random.randint(64, 2048)
    A = torch.randn(m, k, device=device)
    B = torch.randn(k, n, device=device)
    return lambda: A @ B


def random_conv2d(device):
    B = random.randint(1, 32)
    C_in = random.choice([1, 3, 16, 32, 64, 128, 256])
    C_out = random.choice([1, 3, 16, 32, 64, 128, 256])
    H = random.randint(64, 4096)
    W = random.randint(64, 4096)
    K = random.choice([1, 3, 5, 7])
    conv = torch.nn.Conv2d(C_in, C_out, K).to(device)
    x = torch.randn(B, C_in, H, W, device=device)
    return lambda: conv(x)


def random_attention(device):
    seq_len = random.randint(32, 2048)
    batch_size = random.randint(1, 32)
    embed_dim = random.choice([64, 128, 256, 512, 768, 1024, 2048])
    num_heads = random.choice([1, 2, 4, 8, 16])
    attn = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads).to(
        device
    )
    Q = torch.randn(seq_len, batch_size, embed_dim, device=device)
    K = torch.randn(seq_len, batch_size, embed_dim, device=device)
    V = torch.randn(seq_len, batch_size, embed_dim, device=device)
    return lambda: attn(Q, K, V)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-iters",
        type=int,
        default=1000,
        help="Number of iterations for each operator",
    )
    parser.add_argument(
        "--stat-dir", type=str, default="stats", help="Directory path for statistics"
    )
    parser.add_argument(
        "--stat-name", type=str, default=None, help="File name to store statistics"
    )
    args = parser.parse_args()
    autotuner = os.getenv("TRITON_AUTOTUNE", "default")
    device = torch.device("cuda")
    with flag_gems.use_gems():
        mm_time = benchmark_torch_ops(random_mm(device), args.num_iters)
        conv_time = benchmark_torch_ops(random_conv2d(device), args.num_iters)
        attn_time = benchmark_torch_ops(random_attention(device), args.num_iters)
    stats = {"mm": mm_time, "conv": conv_time, "attn": attn_time}
    os.makedirs(args.stat_dir, exist_ok=True)
    date: str = datetime.now().strftime("%m_%d_%H_%M_%S")
    file_name = (
        args.stat_name if args.stat_name else f"torchops_{autotuner}_{date}.json"
    )
    file_path = os.path.join(args.stat_dir, file_name)
    with open(file_path, "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
