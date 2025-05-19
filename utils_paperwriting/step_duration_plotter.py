import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import argparse
import json
import math


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
    durations = apply_avg_filter(durations, int(0.01 * len(durations)))
    durations = [math.log(d) for d in durations]
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
parser.add_argument(
    "--MODEL_NAME",
    type=str,
    default=None,
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


max_step = df["steps"].max()
df = df[df["steps"] < max_step]


mapping = {"default": "Default", "stepwise": "Stepwise", "epsilon": "Epsilon"}
df["type_name"] = df["type_name"].map(mapping)
hue_order = ["Default", "Stepwise", "Epsilon"]

sns.set_theme(
    context="paper",  # smaller labels ideal for publications
    style="white",
    font="serif",
    font_scale=1.9,  # slightly larger than default
)

fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(data=df, hue="type_name", x="steps", y="time", hue_order=hue_order, ax=ax)

ax.tick_params(
    axis="both", 
    # which='major', 
    direction="out",
    length=3,
    width=1,
    # top=True,
    # right=True,
    bottom=True,
    left=True,
)

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
        s=50,
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


N_zoom = 300
zoom_df = df[df["steps"] >= df["steps"].max() - N_zoom]
axins = inset_axes(ax, width="70%", height="50%", loc="upper right", borderpad=0.3)
sns.lineplot(
    data=zoom_df,
    x="steps",
    y="time",
    hue="type_name",
    hue_order=hue_order,
    ax=axins,
    legend=False,
)

x0, x1 = zoom_df["steps"].min(), zoom_df["steps"].max()
y0, y1 = zoom_df["time"].min(), zoom_df["time"].max()
axins.set_xlim(x0, x1)
axins.set_ylim(y0, y1)
axins.tick_params(labelsize=10)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

axins.tick_params(
    axis="both", 
    # which='major',   
    direction="out", 
    length=3, 
    width=1,
    # top=True, 
    # right=True,
    bottom=True,
    left=True,
)

axins.set_xlabel("")
axins.set_ylabel("")


# axins.tick_params(
#     axis="both",     
#     which="both",   
#     bottom=False,    
#     left=False,      
#     labelbottom=False,
#     labelleft=False,
# )
axins.grid(
    True, 
    which="major",
    linestyle="--", 
    color="gray",
    linewidth=0.5,
)

ax.set_xlabel("Steps")
ax.set_ylabel("Time(ms) per Step (log scale)")

model_name = args.MODEL_NAME
ax.legend(
    loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=True, fontsize=14
)

plt.tight_layout()
plt.savefig(f"plots/{args.MODEL_NAME}_step_time.png", dpi=600, bbox_inches="tight")
plt.savefig(f"plots/{args.MODEL_NAME}_step_time.svg", format="svg", bbox_inches="tight")
plt.close(fig)
