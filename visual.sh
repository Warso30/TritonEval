#!/usr/bin/env bash
set -euo pipefail

models=(
  VGG16 ResNet18 PreActResNet18 GoogLeNet DenseNet121
  ResNeXt29_2x64d MobileNet MobileNetV2 DPN92 ShuffleNetG2
  SENet18 ShuffleNetV2 EfficientNetB0 RegNetX_200MF SimpleDLA
  t5-small RNN LSTM
)

for model in "${models[@]}"; do
  echo "=== model=$model ==="
  file_default=$(ls stats/${model}_default_*.json)
  file_stepwise=$(ls stats/${model}_stepwise_*.json)
  file_epsilon=$(ls stats/${model}_epsilon_*.json)
  python utils/plotter.py \
    --default $file_default\
    --stepwise $file_stepwise\
    --epsilon $file_epsilon \
    --step-size 10 \
    --save-file-name $model
  echo
done

echo "All visualizations finished."
