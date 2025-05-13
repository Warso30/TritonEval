import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse


def preprocess(type_name, json_path):
    with open(json_path, "r") as f:
        stats = json.load(f)
    tokens_per_sec = []
    total_time = 0
    for num_tokens, s in enumerate(stats["round_stats"]):
        total_time += s["duration"]
        # tokens_per_sec.append(1000 * (num_tokens + 1) / total_time)
        tokens_per_sec.append(1000 / s["duration"])
    stat = pd.DataFrame(
        {
            "num_tokens": np.arange(1, len(tokens_per_sec) + 1),
            "tokens_per_sec": tokens_per_sec,
            "type_name": type_name,
        }
    )
    return stat


parser = argparse.ArgumentParser()
parser.add_argument(
    "--default",
    type=str,
    help="JSON file containing stats for inference with DefaultAutotuner",
)
parser.add_argument(
    "--epsilon",
    type=str,
    help="JSON file containing stats for inference with EpsilonAutotuner",
)
parser.add_argument(
    "--stepwise",
    type=str,
    help="JSON file containing stats for inference with StepwiseAutotuner",
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
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="num_tokens",
    y="tokens_per_sec",
    hue="type_name",
)
plt.xlabel("Number of Generated Tokens")
plt.ylabel("Tokens per Second")
plt.title("Llama Inference Speed on Different Autotuner Modes")
plt.legend(title="Autotuner Mode")
plt.tight_layout()
plt.show()
