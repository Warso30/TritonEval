import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def calculate_total_time(stats):
    total_time = 0
    for stat in stats:
        total_time += stat["duration"]
    return total_time


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dir",
    type=str,
    required=True,
    help="The directory that contains the json data files.",
)
args = parser.parse_args()
data_dir = Path(args.data_dir)

records = []
for json_file in data_dir.glob("*.json"):
    model_name, type_name, _ = json_file.stem.split("_", 2)
    with open(json_file, "r") as f:
        data = json.load(f)
    total_time = calculate_total_time(data["round_stats"])
    records.append(
        {"model_name": model_name, "type_name": type_name, "total_time": total_time}
    )
    if len(records) % 3 == 0:
        records[-1], records[-2] = records[-2], records[-1]
df = pd.DataFrame(records)

default_total_times = df[df.type_name == "default"].set_index("model_name")[
    "total_time"
]
df["scaled_time"] = df.apply(
    lambda r: r["total_time"] / default_total_times[r["model_name"]], axis=1
)

plt.figure(figsize=(12, 5))
sns.set_theme(style="darkgrid")
sns.barplot(
    data=df, x="model_name", y="scaled_time", hue="type_name", palette="dark", alpha=0.8
)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Ratio to Default Autotuner")
plt.xlabel("Model")
plt.title("End-to-End Training Time on Different Autotuner Modes")
plt.legend(title="Autotuner Mode")
# plt.tight_layout()
plt.show()
