import argparse
import json
import os
import time
from datetime import datetime
import torch
import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.logger import StatsLogger, redirect_stdout
from utils.log_analyzer import analyze_log

import flag_gems


# Function to get the correct device
def get_device():
    if flag_gems.device is not None:
        print(f"Using flag_gems.device: {flag_gems.device}")
        return torch.device(flag_gems.device)
    elif torch.cuda.is_available():
        print("CUDA is available. Using CUDA device.")
        return torch.device("cuda")
    else:
        print("CUDA not available. Using CPU.")
        return torch.device("cpu")


def generate_and_time_tokens(
    model,
    tokenizer,
    prompt_text,
    device,
    max_new_tokens=1000,
    use_flaggems_optimizer=False,
):
    """
    Generates tokens one by one and records the time for each.
    """
    model.eval()  # Ensure model is in evaluation mode

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=model.config.max_position_embeddings - max_new_tokens - 5,
    )  # Reserve space for new tokens
    print(model.config.max_position_embeddings)
    input_ids = inputs.input_ids.to(device)
    original_prompt_length = input_ids.shape[1]

    # # Warm-up run (optional but good for stable measurements)
    # # This is especially important for the first generation call
    # if device.type == "cuda":
    #     print("Performing a warm-up generation...")
    #     with torch.no_grad():
    #         _ = model.generate(
    #             input_ids,
    #             max_new_tokens=2,
    #             num_beams=1,
    #             do_sample=False,
    #             pad_token_id=tokenizer.eos_token_id,
    #             temperature=1.0,
    #             top_p=1.0,
    #         )
    #     torch.cuda.synchronize()
    #     print("Warm-up complete.")

    generated_ids = input_ids
    token_timings = []
    generated_token_texts = []

    # CUDA events for precise timing
    start_event = (
        torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    )
    end_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    context_manager = (
        flag_gems.use_gems() if use_flaggems_optimizer else torch.no_grad()
    )

    print(f"\nGenerating tokens for prompt (first 50 chars): '{prompt_text[:50]}...'")
    with context_manager:  # Handles both flag_gems and torch.no_grad()
        round_num = 1
        for i in range(max_new_tokens):
            print(f"round: {round_num}")
            round_num += 1
            current_input_ids = generated_ids
            print(f"Current input IDs shape: {current_input_ids.shape}")
            if device.type == "cuda" and start_event:
                start_event.record()
            else:
                cpu_start_time = time.perf_counter()

            # Generate only one new token
            # Important: set pad_token_id to avoid warnings if the model doesn't have one set by default.
            # For Llama, eos_token_id is typically used.
            outputs = model.generate(
                current_input_ids,
                max_new_tokens=1,
                num_beams=1,  # Greedy decoding for simplest speed test
                do_sample=False,  # Disable sampling
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,  # Explicitly set EOS for stopping
                temperature=1.0,
                top_p=1.0,
            )

            if device.type == "cuda" and end_event:
                end_event.record()
                torch.cuda.synchronize()  # Wait for the operation to complete
                token_time_ms = start_event.elapsed_time(end_event)
            else:
                cpu_end_time = time.perf_counter()
                token_time_ms = (cpu_end_time - cpu_start_time) * 1000.0

            # Ensure outputs is not empty and has the expected shape
            if outputs.shape[0] == 0 or outputs.shape[1] <= current_input_ids.shape[1]:
                print(
                    f"Warning: model.generate did not produce a new token at step {i+1}. Stopping."
                )
                # This might happen if current_input_ids already ends with EOS and max_new_tokens=1
                # or if there's an issue with the generation process.
                if (
                    outputs.shape[0] > 0
                    and outputs.shape[1] == current_input_ids.shape[1]
                    and outputs[0, -1].item() == tokenizer.eos_token_id
                ):
                    print(
                        f"Token {i+1}: EOS token was the input's last token. Stopping."
                    )
                break

            new_token_id = outputs[0, -1].item()  # Get the last token ID

            # Check if it's an EOS token
            if new_token_id == tokenizer.eos_token_id:
                print(
                    f"Token {i+1}/{max_new_tokens}: EOS token generated. (Continuing generation as requested)"
                )

            token_timings.append(token_time_ms / 1000.0)  # Convert to seconds
            new_token_text = tokenizer.decode(new_token_id)
            generated_token_texts.append(new_token_text)
            print(
                f"Token {i+1}/{max_new_tokens}: ID={new_token_id}, Text='{new_token_text}', Time={token_timings[-1]:.4f}s"
            )
            stats_logger.log_round(token_time_ms / 1000.0, i)  # Log per-step loss
            # Append the new token to the generated sequence for the next iteration
            generated_ids = torch.cat(
                [generated_ids, outputs[:, -1].unsqueeze(-1)], dim=-1
            )

            # Safety break if sequence gets too long (shouldn't be an issue with max_new_tokens)
            if (
                generated_ids.shape[1] >= model.config.max_position_embeddings - 1
            ):  # -1 for safety
                print("Warning: Reached maximum model sequence length.")
                break

    full_generated_text = tokenizer.decode(
        generated_ids[0, original_prompt_length:], skip_special_tokens=True
    )
    return token_timings, generated_token_texts, full_generated_text


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Llama-2 token generation speed."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="sharpbai/Llama-2-7b-hf",  # User's preferred model
        # default="meta-llama/Llama-2-7b-hf", # Official model, requires access
        help="Hugging Face model identifier or path to local model.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="Name of the dataset from Hugging Face datasets library.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Configuration of the dataset (e.g., 'wikitext-2-raw-v1').",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",  # 'test' or 'validation'
        help="Dataset split to use (e.g., 'test', 'validation').",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples from the dataset to test.",
    )
    parser.add_argument(
        "--prompt_tokens",  # Renamed from prompt_length for clarity
        type=int,
        default=128,  # Number of tokens for the prompt
        help="Number of initial tokens from the dataset sample to use as prompt.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1000,
        help="Maximum number of new tokens to generate for each prompt.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="stats/llama_bench_results",
        help="Directory to save benchmark results.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help="Filename for the benchmark results JSON. Defaults to a generated name.",
    )
    parser.add_argument(
        "--enable-flaggems",
        action="store_true",
        help="Enable FlagGems operator optimization if available.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (e.g. for dataset sampling).",
    )
    parser.add_argument(
        "--access-token",
        type=str,
        default=None,
        help="Hugging Face auth token if needed for the model.",
    )
    parser.add_argument(
        "--stat-dir", type=str, default="stats", help="Directory path for statistics"
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="stats/log.json",
        help="File path to the log file",
    )
    parser.add_argument(
        "--ana-path",
        type=str,
        default="stats/ana.json",
        help="File path to save the analysis result",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    global stats_logger
    stats_logger = StatsLogger(args.stat_dir)

    device = get_device()

    if args.enable_flaggems:
        use_flaggems_optimizer = True
        print("FlagGems optimizations will be used.")
    else:
        use_flaggems_optimizer = False
        print("FlagGems optimizations disabled by command line flag.")

    # --- Load Model and Tokenizer ---
    print(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, token=args.access_token
    )
    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad token. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        # This is crucial for `model.generate` if padding is ever implicitly needed,
        # and for consistency, although our one-by-one generation might not strictly need it.
    # Configure pad_token_id in model config as well if needed by generate
    # model_config.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=(
            torch.float16 if device.type == "cuda" else torch.float32
        ),  # FP16 for GPU, FP32 for CPU
        token=args.access_token,
    )
    model.generation_config.top_p = 1.0
    model.generation_config.do_sample = False
    model.generation_config.temperature = 1.0
    model.to(device)
    model.eval()  # Set to evaluation mode

    # --- Load Dataset ---
    print(
        f"Loading dataset: {args.dataset_name} ({args.dataset_config}), split: {args.dataset_split}"
    )
    try:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.dataset_split,
            token=args.access_token,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(
            "Please ensure the dataset name, config, and split are correct, and you have internet access / necessary auth."
        )
        return

    # Filter out very short texts and select samples
    dataset = dataset.filter(
        lambda example: len(example["text"].strip()) >= args.prompt_tokens
    )  # Ensure text is somewhat substantial
    if len(dataset) < args.num_samples:
        print(
            f"Warning: Requested {args.num_samples} samples, but dataset split '{args.dataset_split}' (after filtering) only has {len(dataset)} samples. Using all available."
        )
        args.num_samples = len(dataset)

    if args.num_samples == 0:
        print("No suitable samples found in the dataset. Exiting.")
        return

    selected_indices = list(
        range(args.num_samples)
    )  # Take the first N samples after shuffling (if dataset is shuffled)
    # Or use a random sampler if dataset is very large.
    # For wikitext, just taking the first N is fine for benchmarking.

    # --- Benchmarking Loop ---
    all_results = []
    total_tokens_generated_overall = 0
    total_time_spent_generating_overall = 0.0

    try:
        with redirect_stdout(args.log_path):
            for i in tqdm.tqdm(range(args.num_samples), desc="Processing Samples"):
                print(f"epoch: {i}")
                sample_idx = selected_indices[i]
                sample_text = dataset[sample_idx]["text"]

                # Prepare prompt: take the first `args.prompt_tokens` tokens
                # We tokenize, then truncate, then decode back to text to ensure a clean prompt.
                # This is more robust than character slicing.
                prompt_input_ids = tokenizer(
                    sample_text, return_tensors="pt", truncation=False
                ).input_ids[0]
                if len(prompt_input_ids) > args.prompt_tokens:
                    prompt_input_ids = prompt_input_ids[: args.prompt_tokens]

                # If prompt is too short after initial text processing, skip or error
                if len(prompt_input_ids) < 10:  # Arbitrary minimum prompt length
                    print(
                        f"Sample {sample_idx} resulted in a very short prompt ({len(prompt_input_ids)} tokens). Skipping."
                    )
                    continue

                prompt_text_for_model = tokenizer.decode(
                    prompt_input_ids, skip_special_tokens=True
                )

                sample_result = {
                    "sample_index": sample_idx,
                    "prompt_text_preview": prompt_text_for_model[:100]
                    + "...",  # Store a preview
                    "prompt_length_tokens": len(prompt_input_ids),
                    "requested_max_new_tokens": args.max_new_tokens,
                    "token_generation_times_s": [],
                    "generated_tokens": [],
                    "full_generated_sequence_text": "",
                }

                try:
                    token_timings, generated_tokens_text, full_gen_text = (
                        generate_and_time_tokens(
                            model,
                            tokenizer,
                            prompt_text_for_model,
                            device,
                            args.max_new_tokens,
                            use_flaggems_optimizer,
                        )
                    )
                    sample_result["token_generation_times_s"] = token_timings
                    sample_result["generated_tokens"] = generated_tokens_text
                    sample_result["num_tokens_generated"] = len(token_timings)
                    sample_result["total_generation_time_s"] = sum(token_timings)
                    sample_result["full_generated_sequence_text"] = full_gen_text
                    if token_timings:
                        sample_result["avg_time_per_token_s"] = sum(
                            token_timings
                        ) / len(token_timings)
                        sample_result["tokens_per_second"] = (
                            len(token_timings) / sum(token_timings)
                            if sum(token_timings) > 0
                            else 0
                        )

                    total_tokens_generated_overall += len(token_timings)
                    total_time_spent_generating_overall += sum(token_timings)

                except Exception as e:
                    print(f"Error during generation for sample {sample_idx}: {e}")
                    sample_result["error"] = str(e)

                all_results.append(sample_result)
                print("-" * 50)
                stats_logger.log_epoch(0, 0, 0)

            # --- Save Results ---
            os.makedirs(args.output_dir, exist_ok=True)
            if args.output_filename is None:
                model_short_name = args.model_name_or_path.split("/")[-1]
                flaggems_str = "flaggems" if use_flaggems_optimizer else "noflaggems"
                date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{model_short_name}_{flaggems_str}_{args.dataset_config}_{date_str}.json"
            else:
                output_filename = args.output_filename

            output_path = os.path.join(args.output_dir, output_filename)

            summary_stats = {
                "model_name": args.model_name_or_path,
                "dataset_name": args.dataset_name,
                "dataset_config": args.dataset_config,
                "dataset_split": args.dataset_split,
                "num_samples_processed": len(all_results),
                "prompt_tokens_target": args.prompt_tokens,
                "max_new_tokens_target": args.max_new_tokens,
                "flaggems_enabled_attempted": args.enable_flaggems,
                "flaggems_actually_used": use_flaggems_optimizer
                and device.type != "cpu",
                "device": str(device),
                "total_tokens_generated_across_samples": total_tokens_generated_overall,
                "total_time_spent_generating_across_samples_s": total_time_spent_generating_overall,
            }
            if (
                total_tokens_generated_overall > 0
                and total_time_spent_generating_overall > 0
            ):
                summary_stats["overall_avg_time_per_token_s"] = (
                    total_time_spent_generating_overall / total_tokens_generated_overall
                )
                summary_stats["overall_tokens_per_second"] = (
                    total_tokens_generated_overall / total_time_spent_generating_overall
                )
            else:
                summary_stats["overall_avg_time_per_token_s"] = 0
                summary_stats["overall_tokens_per_second"] = 0

            final_output = {
                "summary_stats": summary_stats,
                "detailed_results": all_results,
            }

            with open(output_path, "w") as f:
                json.dump(final_output, f, indent=4)

            print(f"\nBenchmark complete. Results saved to: {output_path}")
            if summary_stats["overall_tokens_per_second"] > 0:
                print(
                    f"Overall average tokens per second: {summary_stats['overall_tokens_per_second']:.2f}"
                )
                print(
                    f"Overall average time per token: {summary_stats['overall_avg_time_per_token_s']:.4f}s"
                )
    finally:
        # --- Statistics Saving ---
        print("Saving final statistics...")
        if args.enable_flaggems and torch.cuda.is_available():
            autotuner = os.getenv("TRITON_AUTOTUNE", "default")
            run_mode = "flaggems_enabled"
        else:
            autotuner = "no"
            run_mode = "flaggems_disabled"

        # Create a more descriptive default name
        model_short_name = "llama"
        date_str: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_short_name}_{autotuner}_promptlength{args.prompt_tokens}_{date_str}.json"

        stats_logger.save(filename)
        analyze_log(args.log_path, args.ana_path)


if __name__ == "__main__":
    main()
