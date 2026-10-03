"""Optional QLoRA fine-tuning for Stage 5 structured JSON generation.

This is intentionally optional. It trains only on grounded prompt/answer pairs
created by build_answer_sft_data.py and saves a PEFT adapter.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out_dir", default="adapters/qwen25-15b-json-lora")
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Optional QLoRA training requires a CUDA GPU.")

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        quantization_config=quant,
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)

    ds = load_dataset("json", data_files={"train": args.train_jsonl})["train"]

    def render(example):
        messages = [
            {"role": "system", "content": "Return valid JSON only. Stay grounded in supplied metadata."},
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]
        if hasattr(tok, "apply_chat_template"):
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            text = messages[0]["content"] + "\n\n" + messages[1]["content"] + "\n\n" + messages[2]["content"]
        enc = tok(text, truncation=True, max_length=args.max_length)
        enc["labels"] = list(enc["input_ids"])
        return enc

    tokenized = ds.map(render, remove_columns=ds.column_names)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    train_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=1,
        save_strategy="epoch",
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )
    trainer.train()
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"saved LoRA adapter to {args.out_dir}")


if __name__ == "__main__":
    main()
