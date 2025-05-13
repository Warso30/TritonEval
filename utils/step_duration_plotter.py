import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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


def preprocess(type_name, json_path):
    with open(json_path, "r") as f:
        stats = json.load(f)
    durations = [s["duration"] for s in stats["round_stats"]]
    durations = apply_avg_filter(durations, 40)
    stat = pd.DataFrame(
        {
            "steps": np.arange(1, len(durations) + 1),
            "time": durations,
            "type_name": type_name,
        }
    )
    return stat


def get_best_config_indices(file_path):
    with open(file_path, "r") as f:
        stat = json.load(f)
    best_config_indices = set()
    idx = 0
    for epoch in stat:
        for step in epoch:
            for kernel in step.values():
                if type(kernel) is dict and "best_config" in kernel:
                    best_config_indices.add(idx)
            idx += 1
    return best_config_indices


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
parser.add_argument(
    "--best-config",
    type=str,
    default=None,
    help="JSON file containing best config info for Stepwise Autotuner",
)
args = parser.parse_args()

df = pd.concat(
    [
        preprocess("default", args.default),
        preprocess("epsilon", args.epsilon),
        preprocess("stepwise", args.stepwise),
    ],
    ignore_index=True,
)

sns.set_theme(
    context="paper",  # smaller labels ideal for publications
    font="serif",
    font_scale=1.2,  # slightly larger than default
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))
sns.lineplot(data=df, x="steps", y="time", hue="type_name", ax=ax)

for line in ax.lines:
    label = line.get_label()
    xdata, ydata = line.get_xdata(), line.get_ydata()
    if len(xdata) == 0:
        continue
    x0, y0 = xdata[0], ydata[0]
    color = line.get_color()

    ax.scatter(
        x0,
        y0,
        marker="o",
        s=80,
        color=color,
        edgecolor="black",
        linewidth=1.0,
        zorder=5,
    )

if args.best_config:
    ymin, ymax = ax.get_ylim()
    ymid = 35
    ax.set_ylim(-5, ymax)
    best_config_indices = get_best_config_indices(args.best_config)
    for i, index in enumerate(best_config_indices):
        ax.vlines(
            index, ymin, ymid, color="black", linestyle=":", linewidth=1, zorder=0
        )
        if i == 0:
            ax.scatter(
                [index],
                [ymid],
                marker="*",
                s=50,
                zorder=5,
                color="forestgreen",
                label="best config",
            )
        else:
            ax.scatter([index], [ymid], marker="*", s=50, zorder=5, color="forestgreen")

axins = inset_axes(ax, width="40%", height="60%", loc="upper right")
sns.lineplot(data=df, x="steps", y="time", hue="type_name", ax=axins, legend=False)
axins.set_ylim(0.2, 0.6)
axins.set_xlim(1000, 3000)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
ax.set_xlabel("Number of Steps")
ax.set_ylabel("Time per Step(ms)")
ax.set_title("T5 Finetuning Time Comparison on Different Autotuner Modes")
ax.legend(title="Autotuner Mode", loc="upper center")
axins.set_xlabel("Number of Steps")
axins.set_ylabel("Time per Step(ms)")
plt.show()
