import pandas as pd
from io import StringIO


df = pd.read_csv("bar_data_all.csv")
pivot = df.pivot_table(
    index="model_name", columns="type_name", values="total_time", aggfunc="first"
)
pivot = pivot.reindex(columns=["default", "stepwise", "epsilon"]).round(3)

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
    "t5-small",
    "llama",
]
pivot = pivot.reindex(index=desired_order)

for model, row in pivot.iterrows():
    vals = row.to_dict()
    min_val = min(vals.values())
    formatted = []
    for strat in ["default", "stepwise", "epsilon"]:
        s = f"{vals[strat]:.2f}"
        if vals[strat] == min_val:
            s = f"\\textbf{{{s}}}"
        formatted.append(s)
    print(f"{model} & {formatted[0]} & {formatted[1]} & {formatted[2]} \\\\")
