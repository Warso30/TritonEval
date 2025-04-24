#!/usr/bin/env python3
"""
Fine-tune Llama3.2-1b on the Stanford Alpaca dataset using LoRA (4-bit quantization)
Requirements:
    pip install transformers datasets accelerate peft bitsandbytes
"""
import argparse
import flag_gems
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class TimerCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.start_timer = torch.cuda.Event(enable_timing=True)
        self.end_timer = torch.cuda.Event(enable_timing=True)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.start_timer.record()

    def on_epoch_end(self, args, state, control, **kwargs):
        self.end_timer.record()
        torch.cuda.synchronize()
        print(
            f"Epoch {int(state.epoch)} finished in {self.start_timer.elapsed_time(self.end_timer) / 1000}s"
        )


def format_batch(batch, tokenizer):
    """
    Tokenize and format a batch of Alpaca examples into input_ids and labels.
    """
    inputs, labels = [], []
    for instr, inp, out in zip(batch["instruction"], batch["input"], batch["output"]):
        if inp:
            prompt = f"### Instruction:\n{instr}\n### Input:\n{inp}\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instr}\n### Response:\n"
        full = prompt + out
        tokenized = tokenizer(
            full, truncation=True, max_length=512, padding="max_length"
        )
        inputs.append(tokenized["input_ids"])
        labels.append(tokenized["input_ids"])
    return {
        "input_ids": inputs,
        "attention_mask": [
            [1 if id != tokenizer.pad_token_id else 0 for id in seq] for seq in inputs
        ],
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune llama3.2-1b")
    parser.add_argument(
        "--access-token", required=True, help="Huggingface access token"
    )
    parser.add_argument(
        "--enable-flaggems", action="store_true", help="Enable FlagGems"
    )
    args = parser.parse_args()
    model_name = "meta-llama/Llama-3.2-1B"
    output_dir = "finetuned_models"

    dataset = load_dataset("tatsu-lab/alpaca")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=args.access_token)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=args.access_token,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    tokenized = dataset["train"].map(
        format_batch,
        batched=True,
        remove_columns=dataset["train"].column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        fp16=False,
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[TimerCallback()],
    )

    if args.enable_flaggems:
        print("Enabled FlagGems")
        with flag_gems.use_gems():
            trainer.train()
    else:
        trainer.train()

    model.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
