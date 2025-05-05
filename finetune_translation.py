import argparse
import flag_gems
import os
from datetime import datetime
import torch
import tqdm
from utils.logger import StatsLogger

import torch.nn as nn

# Use AdamW, common for transformers
from torch.optim import AdamW

# Hugging Face Libraries
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    set_seed,
)
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_dataloaders(
    model_name,
    dataset_name,
    dataset_config,
    source_lang,
    target_lang,
    max_length,
    batch_size,
    num_workers,
):
    """数据加载：使用Hugging Face datasets和tokenizer"""
    print(f"Loading dataset '{dataset_name}' ({dataset_config})")
    raw_datasets = load_dataset(dataset_name, dataset_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Source Lang: {source_lang}, Target Lang: {target_lang}")

    def preprocess_function(examples):
        # Ensure examples['translation'] exists and is a list of dicts
        if not isinstance(examples.get("translation"), list):
            raise ValueError(
                f"Expected 'translation' field to be a list in dataset '{dataset_name}'. Check dataset structure."
            )
        if not all(
            isinstance(item, dict) and source_lang in item and target_lang in item
            for item in examples["translation"]
        ):
            raise ValueError(
                f"Each item in 'translation' list must be a dict with keys '{source_lang}' and '{target_lang}'."
            )

        inputs = [ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]

        # Tokenize inputs
        model_inputs = tokenizer(
            inputs,
            max_length=max_length,
            truncation=True,
            padding=False,  # Padding handled by collator
        )

        # Tokenize targets (labels)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=max_length,
                truncation=True,
                padding=False,  # Padding handled by collator
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Preprocessing dataset...")
    # Ensure 'train' and 'validation' splits exist
    required_splits = ["train", "validation"]
    for split in required_splits:
        if split not in raw_datasets:
            # Try common alternatives like 'test' for validation if 'validation' is missing
            if split == "validation" and "test" in raw_datasets:
                print(
                    "Warning: 'validation' split not found, using 'test' split instead."
                )
                raw_datasets["validation"] = raw_datasets["test"]
            else:
                raise KeyError(
                    f"Dataset '{dataset_name}' is missing the required split: '{split}'"
                )

    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets[
            "train"
        ].column_names,  # Remove original text columns
    )

    # Data collator handles dynamic padding
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,  # Improves GPU transfer speed if using CUDA
    )
    val_loader = DataLoader(
        eval_dataset,
        shuffle=False,  # No need to shuffle validation data
        collate_fn=data_collator,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, tokenizer


def train_one_epoch(model, loader, optimizer, device):
    """单轮训练 (adapted for Hugging Face models)"""
    model.train()
    running_loss = 0.0
    total_samples = 0
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    for batch in tqdm.tqdm(loader, desc="Train", leave=False):
        # Move batch to device (DataCollator usually returns dict of tensors)
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        start_evt.record()
        outputs = model(
            **batch
        )  # Models typically compute loss internally if labels are provided
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        end_evt.record()
        torch.cuda.synchronize()  # Wait for GPU operations to complete for timing

        batch_size = batch["input_ids"].size(0)
        batch_loss = loss.item() * batch_size  # Scale loss by batch size
        running_loss += batch_loss
        total_samples += batch_size
        stats_logger.log_round(
            start_evt.elapsed_time(end_evt) / 1000, loss.item()
        )  # Log per-step loss

    return running_loss / total_samples if total_samples > 0 else 0.0


def validate(model, loader, device):
    """验证 (adapted for Hugging Face models)"""
    model.eval()
    val_loss = 0.0
    total_correct = 0
    total_tokens = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Val  ", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            batch_size = batch["input_ids"].size(0)
            val_loss += loss.item() * batch_size
            total_samples += batch_size

            # Calculate token-level accuracy (ignoring padding tokens)
            labels = batch["labels"]
            predictions = torch.argmax(logits, dim=-1)

            # Mask where labels are not padding (-100 is common convention)
            active_tokens = labels != -100
            correct_tokens = (predictions == labels) & active_tokens

            total_correct += correct_tokens.sum().item()
            total_tokens += active_tokens.sum().item()

    avg_loss = val_loss / total_samples if total_samples > 0 else 0.0
    # Accuracy is token-level accuracy. BLEU/ROUGE are better translation metrics.
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, accuracy


def train(model, train_loader, val_loader, optimizer, device, epochs, model_save_path):
    """完整训练流程 (adapted)"""
    best_val_metric = float("inf")  # Use validation loss for saving best model
    # Or use float('-inf') if using accuracy/BLEU

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, device)

        stats_logger.log_epoch(train_loss, val_loss, val_acc)  # Log full epoch stats

        # Save best model based on validation loss (lower is better)
        if val_loss < best_val_metric:
            print(
                f"Validation loss improved ({best_val_metric:.4f} --> {val_loss:.4f}). Saving model..."
            )
            best_val_metric = val_loss
            # Save the model state dict and potentially tokenizer/config if needed
            # For Hugging Face models, saving the full model is often easier
            # model.save_pretrained(os.path.dirname(model_save_path))
            # Or just the state_dict like the original script:
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved to {model_save_path}")
        else:
            print(f"Validation loss did not improve from {best_val_metric:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Translation Models with FlagGems"
    )
    # --- Training Arguments ---
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (adjust based on GPU memory)",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate for AdamW optimizer"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # --- Model Arguments ---
    parser.add_argument(
        "--model",
        type=str,
        default="t5-small",  # Example seq2seq model
        help="Hugging Face model identifier (e.g., t5-small, facebook/bart-base)",
    )

    # --- Dataset Arguments ---
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="iwslt2017",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="iwslt2017-en-de",
        help="Dataset configuration (e.g., language pair)",
    )
    parser.add_argument(
        "--source-lang",
        type=str,
        default="en",
        help="Source language identifier in the dataset",
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="de",
        help="Target language identifier in the dataset",
    )
    parser.add_argument(
        "--max-length", type=int, default=128, help="Max sequence length for tokenizer"
    )

    # --- System Arguments ---
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(4, os.cpu_count()),
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--enable-flaggems",
        action="store_true",
        help="Enable FlagGems operator optimization",
    )
    parser.add_argument(
        "--stat-dir", type=str, default="stats", help="Directory path for statistics"
    )
    parser.add_argument(
        "--stat-name",
        type=str,
        default=None,
        help="File name to store statistics (JSON)",
    )
    parser.add_argument(
        "--model-save-name",
        type=str,
        default="best_translation_model.pth",
        help="File name for saving the best model state dict",
    )

    args = parser.parse_args()

    # --- Setup ---
    set_seed(args.seed)  # For reproducibility
    global stats_logger
    stats_logger = StatsLogger(args.stat_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if not torch.cuda.is_available():
        print(
            "Warning: CUDA not available, running on CPU. FlagGems typically targets GPU optimization."
        )

    # --- Data ---
    print("Loading data...")
    try:
        train_loader, val_loader, tokenizer = get_dataloaders(
            model_name=args.model,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    except (KeyError, ValueError) as e:
        print(f"Error loading or processing dataset: {e}")
        print(
            "Please check dataset name, config, and language keys (--source-lang, --target-lang)."
        )
        return  # Exit if data loading fails

    # --- Model ---
    print(f"Loading model: {args.model}")
    # Use AutoModelForSeq2SeqLM for translation tasks
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    if args.enable_flaggems:
        print("Attempting to disable attention dropout for FlagGems compatibility...")
        if hasattr(model.config, "attention_dropout"):
            print(f"Original attention_dropout: {model.config.attention_dropout}")
            model.config.attention_dropout = 0.0
            print(f"Set attention_dropout to: {model.config.attention_dropout}")
            # Re-apply config modifications to the model instance if necessary
            # (Often not needed as layers read from config, but good practice)
            # This part can be tricky and model-specific.
            # For BART, modifying the config *might* be enough.
            # You might need to iterate through model.model.encoder.layers and
            # model.model.decoder.layers to set dropout rates directly if
            # modifying the config doesn't propagate correctly.
            # Example (conceptual - verify attribute names for BART):
            # for layer in model.model.encoder.layers:
            #     layer.self_attn.dropout = 0.0
            # for layer in model.model.decoder.layers:
            #     layer.self_attn.dropout = 0.0
            #     layer.encoder_attn.dropout = 0.0
        else:
            print(
                "Warning: Model config does not have 'attention_dropout' attribute. Cannot disable automatically."
            )

    model = model.to(device)

    # --- Optimizer ---
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # --- Training ---
    model_save_path = os.path.join(args.stat_dir, args.model_save_name)
    print(f"Best model will be saved to: {model_save_path}")

    try:
        start_time = datetime.now()
        print(f"Starting training at {start_time}")
        if args.enable_flaggems:
            if not torch.cuda.is_available():
                print(
                    "Warning: FlagGems requested but CUDA is not available. Running without FlagGems."
                )
                train(
                    model,
                    train_loader,
                    val_loader,
                    optimizer,
                    device,
                    args.epochs,
                    model_save_path,
                )
            else:
                print("Running training with FlagGems enabled...")
                with flag_gems.use_gems():
                    train(
                        model,
                        train_loader,
                        val_loader,
                        optimizer,
                        device,
                        args.epochs,
                        model_save_path,
                    )
        else:
            print("Running training without FlagGems...")
            train(
                model,
                train_loader,
                val_loader,
                optimizer,
                device,
                args.epochs,
                model_save_path,
            )

        end_time = datetime.now()
        print(f"Training finished at {end_time}. Duration: {end_time - start_time}")

    finally:
        # --- Statistics Saving ---
        print("Saving final statistics...")
        if args.enable_flaggems and torch.cuda.is_available():
            autotuner = os.getenv("TRITON_AUTOTUNE", "default")
            run_mode = "flaggems_enabled"
        else:
            autotuner = "no"
            run_mode = "flaggems_disabled"
        if args.stat_name:
            filename = (
                args.stat_name
                if args.stat_name.endswith(".json")
                else f"{args.stat_name}.json"
            )
        else:
            # Create a more descriptive default name
            model_short_name = args.model.split("/")[
                -1
            ]  # Get last part of HF identifier
            date_str: str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_short_name}_{autotuner}_{date_str}.json"

        stats_logger.save(filename)


if __name__ == "__main__":
    main()

    # Example Usage:
    # Basic T5-small run:
    # python finetune_translation.py --model t5-small --epochs 3 --batch-size 8 --dataset-name iwslt2017 --dataset-config iwslt2017-en-de --source-lang en --target-lang de
    #
    # Run with FlagGems enabled:
    # python finetune_translation.py --model t5-small --epochs 3 --batch-size 8 --dataset-name iwslt2017 --dataset-config iwslt2017-en-de --source-lang en --target-lang de --enable-flaggems
    #
    # Run with BART:
    # python finetune_translation.py --model facebook/bart-base --epochs 3 --batch-size 8 --dataset-name iwslt2017 --dataset-config iwslt2017-en-fr --source-lang en --target-lang fr --enable-flaggems
