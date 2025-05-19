import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("bar_data.csv")
pivot = df.pivot(index="model_name", columns="type_name", values="scaled_time")

desired_order = [
    "VGG16",
    "GoogLeNet",
    "DPN92",
    "DenseNet121",
    "EfficientNetB0",
    "MobileNet",
    "MobileNetV2",
    "ResNet18",
    "PreActResNet18",
    "RegNetX",
    "ResNeXt29",
    "SENet18",
    "ShuffleNetG2",
    "ShuffleNetV2",
    "SimpleDLA",
    "RNN",
    "LSTM",
]

pivot = pivot.reindex(
    index=desired_order, columns=["default", "stepwise", "epsilon"], fill_value=0
)

models = pivot.index.tolist()
x = np.arange(len(models))
width = 0.2
offsets = np.array([-width, 0, width])

labels = ["Default", "Stepwise", "Epsilon"]
colors = ["white", "#7FB6D0", "#4C72B0"]
hatches = ["///", None, None]
values = pivot[["default", "stepwise", "epsilon"]].to_numpy()

fig, ax = plt.subplots(figsize=(8, 4))

for i in range(3):
    ax.bar(
        x + offsets[i],
        values[:, i],
        width,
        label=labels[i],
        color=colors[i],
        edgecolor="black",
        hatch=hatches[i] or "",
    )

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha="right")

ax.set_ylabel("Ratio to Default Autotuner")

ax.set_ylim(0, 1)
ax.set_yticks(np.linspace(0, 1, 6))
ax.yaxis.grid(True, linestyle="--", alpha=0.7)

ax.legend(
    loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=True, fontsize=9
)

plt.tight_layout()
plt.savefig("endtoend_cnn.png", dpi=600, bbox_inches="tight")
plt.savefig("endtoend_cnn.svg", bbox_inches="tight")
plt.close(fig)
