import os
import math
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


def find_yrange(data: List[float]):
    first_max = 0
    second_max = 0
    min = 10**9
    for val in data:
        if val > first_max:
            second_max = first_max
            first_max = val
        elif val > second_max:
            second_max = val
        if val < min:
            min = val
    return first_max, second_max, min


def adjust_yvalues(
    values: List[float],
    threshold: float = 0.15,
    max: float = 100,
    lower_ratio: float = 0.8,
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
    ymax, ymid, ymin = 0, 0, 10**9
    for stat_type in stats:
        stat_data, _ = stats[stat_type]
        if stat_data:
            stat_data = [s["duration"] for s in stat_data]
            curr_max, curr_mid, curr_min = find_yrange(stat_data)
            ymax = max(ymax, curr_max)
            ymid = max(ymid, curr_mid)
            ymin = min(ymin, curr_min)
    ymax, ymid, ymin = (
        math.ceil(ymax * 100) / 100,
        math.ceil(ymid * 100) / 100,
        math.floor(ymin * 100) / 100,
    )
    yticks = np.arange(ymin, ymid + 0.002, step=0.005)
    high_yticks = np.linspace(ymid, ymax, num=5)[1:].astype(np.float32)
    yticks = np.concatenate((yticks, high_yticks))
    adj_yticks = adjust_yvalues(yticks, ymid, ymax)

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
            durations = adjust_yvalues(durations, ymid, ymax)
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
    ax_durations.set_yticks(adj_yticks, labels=[f"{t:.3f}" for t in yticks])
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
