#!/usr/bin/env bash


STAT_DIR="/Users/kangqiwang/Desktop/various/TritonEval/stats/stats/llama-1000tokens"


PY_SCRIPT="utils/infer_plotter.py"


mkdir -p plots

for LEN in 64 128 256 512; do
  echo "=== processing promptlength=${LEN} ==="


  JSON_DEFAULT=$(ls "${STAT_DIR}"/llama_default_promptlength${LEN}_*.json)
  JSON_EPSILON=$(ls "${STAT_DIR}"/llama_epsilon_promptlength${LEN}_*.json)
  JSON_STEPWISE=$(ls "${STAT_DIR}"/llama_stepwise_promptlength${LEN}_*.json)


  python3 "${PY_SCRIPT}" \
    --default  "${JSON_DEFAULT}" \
    --epsilon  "${JSON_EPSILON}" \
    --stepwise "${JSON_STEPWISE}" \
    --len ${LEN}

  echo "Finish promptlength=${LEN}"
  echo
done

