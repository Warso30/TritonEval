source /etc/network_turbo
export HUGGINGFACE_HUB_CACHE="/root/autodl-tmp"
# rm -rf ~/.triton/cache && TRITON_AUTOTUNE=stepwise python test_lan_generation.py --enable-flaggems --access-token ... --prompt_tokens 64
# rm -rf ~/.triton/cache && TRITON_AUTOTUNE=stepwise python test_lan_generation.py --enable-flaggems --access-token ... --prompt_tokens 128
# rm -rf ~/.triton/cache && TRITON_AUTOTUNE=stepwise python test_lan_generation.py --enable-flaggems --access-token ... --prompt_tokens 256
# rm -rf ~/.triton/cache && TRITON_AUTOTUNE=stepwise python test_lan_generation.py --enable-flaggems --access-token ... --prompt_tokens 512

# rm -rf ~/.triton/cache && TRITON_AUTOTUNE=epsilon python test_lan_generation.py --enable-flaggems --access-token ... --prompt_tokens 64
# rm -rf ~/.triton/cache && TRITON_AUTOTUNE=epsilon python test_lan_generation.py --enable-flaggems --access-token ... --prompt_tokens 128
# rm -rf ~/.triton/cache && TRITON_AUTOTUNE=epsilon python test_lan_generation.py --enable-flaggems --access-token ... --prompt_tokens 256
# rm -rf ~/.triton/cache && TRITON_AUTOTUNE=epsilon python test_lan_generation.py --enable-flaggems --access-token ... --prompt_tokens 512

# rm -rf ~/.triton/cache && TRITON_AUTOTUNE=default python test_lan_generation.py --enable-flaggems --access-token ... --prompt_tokens 64
# rm -rf ~/.triton/cache && TRITON_AUTOTUNE=default python test_lan_generation.py --enable-flaggems --access-token ... --prompt_tokens 128
# rm -rf ~/.triton/cache && TRITON_AUTOTUNE=default python test_lan_generation.py --enable-flaggems --access-token ... --prompt_tokens 256
# rm -rf ~/.triton/cache && TRITON_AUTOTUNE=default python test_lan_generation.py --enable-flaggems --access-token ... --prompt_tokens 512

# rm -rf ~/.triton/cache && TRITON_PRINT_AUTOTUNING=1 TRITON_ENABLE_RUNTIME_MEASUREMENT=1 TRITON_AUTOTUNE=stepwise python test_lan_generation.py \
#     --enable-flaggems --access-token ... --prompt_tokens 256 \
#     --log-path stats/llama_stepwise_256_log.txt \
#     --ana-path stats/llama_stepwise_256_ana.txt

# rm -rf ~/.triton/cache && TRITON_PRINT_AUTOTUNING=1 TRITON_ENABLE_RUNTIME_MEASUREMENT=1 TRITON_AUTOTUNE=epsilon python test_lan_generation.py \
#     --enable-flaggems --access-token ... --prompt_tokens 256 \
#     --log-path stats/llama_epsilon_256_log.txt \
#     --ana-path stats/llama_epsilon_256_ana.txt

# rm -rf ~/.triton/cache && TRITON_PRINT_AUTOTUNING=1 TRITON_ENABLE_RUNTIME_MEASUREMENT=1 TRITON_AUTOTUNE=stepwise python test_lan_generation.py \
#     --enable-flaggems --access-token ... --prompt_tokens 512 \
#     --log-path stats/llama_stepwise_512_log.txt \
#     --ana-path stats/llama_stepwise_512_ana.txt

# rm -rf ~/.triton/cache && TRITON_PRINT_AUTOTUNING=1 TRITON_ENABLE_RUNTIME_MEASUREMENT=1 TRITON_AUTOTUNE=epsilon python test_lan_generation.py \
#     --enable-flaggems --access-token ... --prompt_tokens 512 \
#     --log-path stats/llama_epsilon_512_log.txt \
#     --ana-path stats/llama_epsilon_512_ana.txt

# rm -rf ~/.triton/cache && TRITON_PRINT_AUTOTUNING=1 TRITON_ENABLE_RUNTIME_MEASUREMENT=1 TRITON_AUTOTUNE=default python test_lan_generation.py \
#     --enable-flaggems --access-token ... --prompt_tokens 256 \
#     --log-path stats/llama_default_256_log.txt \
#     --ana-path stats/llama_default_256_ana.txt

# rm -rf ~/.triton/cache && TRITON_PRINT_AUTOTUNING=1 TRITON_ENABLE_RUNTIME_MEASUREMENT=1 TRITON_AUTOTUNE=default python test_lan_generation.py \
#     --enable-flaggems --access-token ... --prompt_tokens 512 \
#     --log-path stats/llama_default_512_log.txt \
#     --ana-path stats/llama_default_512_ana.txt

rm -rf ~/.triton/cache && TRITON_PRINT_AUTOTUNING=1 TRITON_ENABLE_RUNTIME_MEASUREMENT=1 TRITON_AUTOTUNE=stepwise python test_lan_generation.py \
    --enable-flaggems --access-token ... --prompt_tokens 64 \
    --log-path stats/llama_stepwise_64_log.txt \
    --ana-path stats/llama_stepwise_64_ana.txt

rm -rf ~/.triton/cache && TRITON_PRINT_AUTOTUNING=1 TRITON_ENABLE_RUNTIME_MEASUREMENT=1 TRITON_AUTOTUNE=epsilon python test_lan_generation.py \
    --enable-flaggems --access-token ... --prompt_tokens 64 \
    --log-path stats/llama_epsilon_64_log.txt \
    --ana-path stats/llama_epsilon_64_ana.txt

rm -rf ~/.triton/cache && TRITON_PRINT_AUTOTUNING=1 TRITON_ENABLE_RUNTIME_MEASUREMENT=1 TRITON_AUTOTUNE=stepwise python test_lan_generation.py \
    --enable-flaggems --access-token ... --prompt_tokens 128 \
    --log-path stats/llama_stepwise_128_log.txt \
    --ana-path stats/llama_stepwise_128_ana.txt

rm -rf ~/.triton/cache && TRITON_PRINT_AUTOTUNING=1 TRITON_ENABLE_RUNTIME_MEASUREMENT=1 TRITON_AUTOTUNE=epsilon python test_lan_generation.py \
    --enable-flaggems --access-token ... --prompt_tokens 128 \
    --log-path stats/llama_epsilon_128_log.txt \
    --ana-path stats/llama_epsilon_128_ana.txt
