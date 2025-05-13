import numpy as np
import pandas as pd
import seaborn as sns
from brokenaxes import brokenaxes
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

import argparse
import json


def apply_avg_filter(data, size):
    low_idx, high_idx = 0, 0
    sum = 0
    filtered_data = []
    for v in data:
        if high_idx - low_idx + 1 > size:
            sum -= data[low_idx]
            low_idx += 1
        sum += v
        filtered_data.append(sum / (high_idx - low_idx + 1))
        high_idx += 1
    return filtered_data


def preprocess(mode, json_path):
    with open(json_path, "r") as f:
        stats = json.load(f)
    for stat in stats:
        stats[stat] = pd.DataFrame(
            {
                "iteration": np.arange(1, len(stats[stat]) + 1),
                "time": stats[stat],
                "mode": mode,
            }
        )
    return stats


parser = argparse.ArgumentParser()
parser.add_argument(
    "--default",
    type=str,
    help="JSON file containing stats for finetuning with DefaultAutotuner",
)
parser.add_argument(
    "--epsilon",
    type=str,
    help="JSON file containing stats for finetuning with EpsilonAutotuner",
)
parser.add_argument(
    "--stepwise",
    type=str,
    help="JSON file containing stats for finetuning with StepwiseAutotuner",
)
args = parser.parse_args()

default_stats = preprocess("default", args.default)
epsilon_stats = preprocess("epsilon", args.epsilon)
stepwise_stats = preprocess("stepwise", args.stepwise)


dfs = []
ops = []
for op in default_stats:
    ops.append(op)
    dfs.append(pd.concat([default_stats[op], epsilon_stats[op], stepwise_stats[op]]))


sns.set_theme(
    context="paper",  # smaller labels ideal for publications
    font="serif",
    font_scale=1.2,  # slightly larger than default
)
for i, df in enumerate(dfs):
    fig = plt.figure(figsize=(8, 5))
    if ops[i] == "mm":
        bax = brokenaxes(ylims=((0, 0.2), (50, 100)), hspace=0.1, height_ratios=(1, 2))
        bax_high, bax_low = bax.axs
        bax_low.set_ylim(0.1, 0.2)
        bax_high.set_ylim(50, 100)
        bax.set_title(
            f"Torch Matmul Operator Time Comparison on Different Autotuner Modes"
        )
        bax_low.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
        bax_low.yaxis.set_minor_locator(mticker.MultipleLocator(0.01))
    elif ops[i] == "conv":
        bax = brokenaxes(ylims=((0, 5), (100, 150)), hspace=0.1, height_ratios=(1, 2))
        bax_high, bax_low = bax.axs
        bax_low.set_ylim(3.8, 4.6)
        bax_high.set_ylim(100, 150)
        bax.set_title(
            f"Torch Conv2d Operator Time Comparison on Different Autotuner Modes"
        )
        bax_low.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
        bax_low.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    elif ops[i] == "attn":
        bax = brokenaxes(ylims=((20, 40), (250, 300)), hspace=0.1, height_ratios=(1, 2))
        bax_high, bax_low = bax.axs
        bax_low.set_ylim(18, 30)
        bax_high.set_ylim(250, 300)
        bax.set_title(
            f"Torch MultiheadAttention Operator Time Comparison on Different Autotuner Modes"
        )
        bax_low.yaxis.set_major_locator(mticker.MultipleLocator(5))
        bax_low.yaxis.set_minor_locator(mticker.MultipleLocator(1))

    for ax in (bax_low, bax_high):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        ax.xaxis.set_minor_locator(plt.NullLocator())

    sns.lineplot(data=df, x="iteration", y="time", hue="mode", ax=bax_low, legend=False)
    sns.lineplot(data=df, x="iteration", y="time", hue="mode", ax=bax_high)

    for line in bax_high.lines:
        ymin, ymax = bax_high.get_ylim()
        label = line.get_label()
        xdata, ydata = line.get_xdata(), line.get_ydata()
        if len(xdata) == 0:
            continue
        x0, y0 = xdata[0], ydata[0]
        color = line.get_color()
        if y0 > ymax:
            bax_high.scatter(
                x0,
                ymax,
                marker="o",
                s=80,
                color=color,
                edgecolor="black",
                linewidth=1.0,
                zorder=5,
            )
            if ops[i] == "mm":
                x_text = x0 * 80
                y_text = ymax * 0.92
            elif ops[i] == "conv":
                x_text = x0 * 50
                y_text = ymax * 0.95
            elif ops[i] == "attn":
                x_text = x0 * 70
                y_text = ymax * 0.97
            bax_high.text(
                x_text, y_text, f"{y0:.2f}", ha="center", va="bottom", fontsize=9
            )
        else:
            bax_high.scatter(
                x0,
                y0,
                marker="o",
                s=80,
                color=color,
                edgecolor="black",
                linewidth=1.0,
                zorder=5,
            )

    bax_low.set_xlabel("Iteration")
    bax_low.set_ylabel("Time(ms)")
    bax_high.legend(title="Autotuner Mode", loc="upper right")

plt.show()
