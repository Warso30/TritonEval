import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, List, Dict


def read_stats(path: Optional[str]) -> list:
    if not path:
        return []
    with open(path, "r") as f:
        stats = json.load(f)
    return stats["round_stats"]


def adjust_yvalues(
    values: List[float],
    threshold: float = 0.15,
    max: float = 100,
    lower_ratio: float = 0.6,
):
    y_vals = np.array(values, dtype=float)
    adj_vals = np.zeros_like(y_vals)
    adj_vals[y_vals <= threshold] = (
        y_vals[y_vals <= threshold] / threshold
    ) * lower_ratio
    if max > threshold:
        adj_vals[y_vals > threshold] = lower_ratio + (
            y_vals[y_vals > threshold] - threshold
        ) / (max - threshold) * (1 - lower_ratio)
    return adj_vals


def apply_avg_filter(data: List[float], avg_len: int, step_size: int) -> List[float]:
    avg_len = max(1, avg_len)
    step_size = max(1, step_size)
    n = len(data)
    out = []
    i = -(avg_len - 1)
    while i < n:
        start = max(i, 0)
        end = min(i + avg_len, n)
        window = data[start:end]
        if window:
            out.append(sum(window) / len(window))
        i += step_size
    return out


def plot(stats: Dict[str, List[float]], save: bool, avg_len: int, step_size: int):
    threshold = 0.1
    max_val = 60
    yticks = np.arange(0, threshold + 0.01, step=0.02)
    if max_val > threshold:
        high_yticks = np.linspace(threshold, max_val, num=5)[1:].astype(np.int16)
        yticks = np.concatenate((yticks, high_yticks))
    adj_yticks = adjust_yvalues(yticks, threshold, max_val)

    fig_durations, ax_durations = plt.subplots()
    fig_losses, ax_losses = plt.subplots()
    for stat_type in stats:
        stat_data, stat_marker = stats[stat_type]
        if stat_data:
            durations = [s["duration"] for s in stat_data]
            durations = durations[:1] + apply_avg_filter(
                durations[1:], avg_len, step_size
            )
            durations_steps = list(range(1, len(durations) + 1))
            losses = [s["loss"] for s in stat_data]
            losses_steps = list(range(1, len(losses) + 1))
            durations = adjust_yvalues(durations, threshold, max_val)
            ax_durations.plot(
                durations_steps,
                durations,
                label=stat_type,
                markevery=[0],
                marker=stat_marker,
            )
            ax_durations.legend()
            ax_losses.plot(losses_steps, losses, label=stat_type)
            ax_losses.legend()

    ax_durations.set_yticks(adj_yticks, labels=[f"{t:.2f}" for t in yticks])
    ax_durations.set_xlabel("Step")
    ax_durations.set_ylabel("Time")
    ax_durations.set_title("Time for Each Step")

    ax_losses.set_xlabel("Step")
    ax_losses.set_ylabel("Loss")
    ax_losses.set_title("Loss for Each Step")

    if save:
        os.makedirs("plots", exist_ok=True)
        date: str = datetime.now().strftime("%m_%d_%H_%M_%S")
        fig_durations.savefig(f"plots/time_{date}")
        fig_losses.savefig(f"plots/loss_{date}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot finetune stat")
    parser.add_argument(
        "--none",
        type=str,
        help="JSON file containing stats for finetune without Autotuner",
    )
    parser.add_argument(
        "--default",
        type=str,
        help="JSON file containing stats for finetune with Autotuner",
    )
    parser.add_argument(
        "--stepwise",
        type=str,
        help="JSON file containing stats for finetune with StepwiseAutotuner",
    )
    parser.add_argument(
        "--epsilon",
        type=str,
        help="JSON file containing stats for finetune with EpsilonAutotuner",
    )
    parser.add_argument(
        "--confidence",
        type=str,
        help="JSON file containing stats for finetune with ConfidenceAutotuner",
    )
    parser.add_argument(
        "--avg-len",
        type=int,
        default=10,
        help="The length of the moving average filter",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=1,
        help="The length of the moving average filter",
    )
    parser.add_argument("--no-save", action="store_true", help="Save the plot")
    args = parser.parse_args()

    stats = {
        "none": (read_stats(args.none), "o"),
        "default": (read_stats(args.default), "*"),
        "stepwise": (read_stats(args.stepwise), "X"),
        "epsilon": (read_stats(args.epsilon), "D"),
        "confidence": (read_stats(args.confidence), "s"),
    }

    plot(stats, not args.no_save, args.avg_len, args.step_size)


if __name__ == "__main__":
    main()
