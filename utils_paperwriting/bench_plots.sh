#!/bin/bash

folder="/Users/kangqiwang/Desktop/various/TritonEval/stats/cnn_stats"

# 提取模型名（第一个下划线前）
models=$(ls ${folder}/*.json | xargs -n1 basename | sed -E 's/^([^_]*)_.*$/\1/' | sort | uniq)

for model in $models; do
    if [[ "$model" == "VGG16" ]]; then
        continue  # Skip VGG16
    fi
    # Skip models with "default" in their name
    default_file=$(ls ${folder}/${model}_default_*.json 2>/dev/null | head -n 1)
    epsilon_file=$(ls ${folder}/${model}_epsilon_*.json 2>/dev/null | head -n 1)
    stepwise_file=$(ls ${folder}/${model}_stepwise_*.json 2>/dev/null | head -n 1)

    if [[ -n "$default_file" && -n "$epsilon_file" && -n "$stepwise_file" ]]; then
        echo "Running for model: $model"
        python step_duration_plotter.py \
            --MODEL_NAME "$model" \
            --default "$default_file" \
            --epsilon "$epsilon_file" \
            --stepwise "$stepwise_file"
    else
        echo "Skipping $model: missing one of default/epsilon/stepwise"
    fi
done
