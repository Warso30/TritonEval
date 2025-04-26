import os
import json
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, List, Dict


def read_stats(path: Optional[str]) -> list:
    if not path:
        return []
    with open(path, "r") as f:
        stats = json.load(f)
    return stats["round_stats"]


def plot(stats: Dict[str, List[float]], save: bool):
    fig_durations, ax_durations = plt.subplots()
    fig_losses, ax_losses = plt.subplots()
    for stat_type in stats:
        if stats[stat_type]:
            steps = list(range(1, len(stats[stat_type]) + 1))
            durations = [s["duration"] for s in stats[stat_type]]
            losses = [s["loss"] for s in stats[stat_type]]
            ax_durations.plot(steps, durations, label=stat_type)
            ax_durations.legend()
            ax_losses.plot(steps, losses, label=stat_type)
            ax_losses.legend()

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
    parser.add_argument("--no-save", action="store_true", help="Save the plot")
    args = parser.parse_args()

    stats = {
        "none": read_stats(args.none),
        "default": read_stats(args.default),
        "stepwise": read_stats(args.stepwise),
        "epsilon": read_stats(args.epsilon),
        "confidence": read_stats(args.confidence),
    }

    plot(stats, not args.no_save)


if __name__ == "__main__":
    main()
