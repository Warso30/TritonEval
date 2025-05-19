#!/usr/bin/env bash
set -euo pipefail

# TRITON_AUTOTUNE
autotune_values=(stepwise epsilon default)

for tune in "${autotune_values[@]}"; do
    echo "=== TRITON_AUTOTUNE=$tune ==="
    rm -rf ~/.triton/cache && TRITON_AUTOTUNE="$tune" python torch_operators.py --num-iters 100000
    echo
done

echo "All experiments finished."
