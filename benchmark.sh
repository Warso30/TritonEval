#!/usr/bin/env bash
set -euo pipefail

# TRITON_AUTOTUNE
autotune_values=(stepwise epsilon default)

# CIFAR-10
cifar_models=(
  VGG16 ResNet18 PreActResNet18 GoogLeNet DenseNet121
  ResNeXt29_2x64d MobileNet MobileNetV2 DPN92 ShuffleNetG2
  SENet18 ShuffleNetV2 EfficientNetB0 RegNetX_200MF SimpleDLA
)

# translation
translation_models=(  
  t5-small
)

# rnn
rnn_types=(
  RNN
  LSTM
)

########################
# 2. Translation Training
########################
for tune in "${autotune_values[@]}"; do
  for model in "${translation_models[@]}"; do
    echo "=== Translation | TRITON_AUTOTUNE=$tune | model=$model ==="
    rm -rf ~/.triton/cache && TRITON_AUTOTUNE="$tune" TRITON_PRINT_AUTOTUNING=1 TRITON_ENABLE_RUNTIME_MEASUREMENT=1 \
      python finetune_translation.py \
      --enable-flaggems \
      --model="$model" \
      --epochs 1 \
      --batch-size 64 \
      --dataset-name iwslt2017 \
      --dataset-config iwslt2017-en-de \
      --source-lang en \
      --target-lang de \
      --log-path stats/${model}_${tune}_log.txt \
      --ana-path stats/${model}_${tune}_ana.txt
    echo
  done
done

# ########################
# # 3. Text Classification Training
# ########################
# for tune in "${autotune_values[@]}"; do
#   for rnn in "${rnn_types[@]}"; do
#     echo "=== Text Classification | TRITON_AUTOTUNE=$tune | rnn-type=$rnn ==="
#     rm -rf ~/.triton/cache && TRITON_AUTOTUNE="$tune" \
#       python train_rnn.py \
#       --enable-flaggems \
#       --rnn-type "$rnn" \
#       --epochs 1
#     echo
#   done
# done

# ########################
# # 1. CIFAR-10 Training
# ########################
# for tune in "${autotune_values[@]}"; do
#   for model in "${cifar_models[@]}"; do
#     echo "=== CIFAR-10 | TRITON_AUTOTUNE=$tune | model=$model ==="
#     rm -rf ~/.triton/cache && TRITON_AUTOTUNE="$tune" \
#       python train_cifar10.py \
#       --enable-flaggems \
#       --model="$model" \
#       --epochs 1
#     echo
#   done
# done

echo "All experiments finished."
