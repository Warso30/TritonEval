import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("transformer.csv")
pivot = df.pivot(index="model_name", columns="type_name", values="scaled_time")
pivot = pivot.reindex(columns=["default", "stepwise", "epsilon"], fill_value=0)

models = ["t5-small", "llama"]
pivot = pivot.loc[models]

x = np.arange(len(models)) * 0.7
width = 0.2
offsets = np.array([-width, 0, width])

colors = ["white", "#7FB6D0", "#4C72B0"]
hatches = ["///", None, None]
labels = ["Default", "Stepwise", "Epsilon"]
values = pivot[["default", "stepwise", "epsilon"]].to_numpy()

fig, ax = plt.subplots(figsize=(5, 4))

for i in range(3):
    bars = ax.bar(
        x + offsets[i],
        values[:, i],
        width,
        label=labels[i],
        color=colors[i],
        edgecolor="black",
        hatch=hatches[i] or "",
    )

    if i > 0:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01, 
                f"{h * 100:.1f}%", 
                ha="center",
                va="bottom",
                fontsize=10,
            )

ax.set_xticks(x)
ax.set_xticklabels(["t5-small", "Llama-2-7b"])

ax.set_ylabel("Ratio to Default Autotuner")

ax.set_ylim(0, 1)
ax.set_yticks(np.linspace(0, 1, 6))
ax.yaxis.grid(True, linestyle="--", alpha=0.7)

ax.legend(loc="upper right", ncol=1, frameon=True, fontsize=9)

plt.tight_layout()
plt.savefig("endtoend_transformer.png", dpi=600, bbox_inches="tight")
plt.savefig("endtoend_transformer.svg", bbox_inches="tight")
plt.close(fig)
