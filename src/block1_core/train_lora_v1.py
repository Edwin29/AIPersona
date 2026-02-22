import os
import json
import argparse
from pathlib import Path

import yaml
from datasets import Dataset

# Unsloth (VL)
from unsloth import FastVisionModel  # Qwen3-VL notebook style :contentReference[oaicite:4]{index=4}
from trl import SFTTrainer
from transformers import TrainingArguments


def read_messages_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" not in obj:
                raise ValueError("Each JSONL row must contain 'messages'.")
            rows.append(obj)
    return rows


def messages_to_text(sample: dict) -> dict:
    # Qwen 계열은 chat template가 있는 경우가 많아서, 가능하면 apply_chat_template를 씀.
    # 하지만 dataset 쪽에서는 일단 단순 stringify 형태로 넣고,
    # trainer 단계에서 tokenizer chat template로 변환되도록 처리할 수도 있음.
    # 여기서는 최소 구현으로 user/assistant turn들을 텍스트로 직렬화.
    msgs = sample["messages"]
    parts = []
    for m in msgs:
        role = m["role"]
        content = m["content"]
        parts.append(f"{role.upper()}: {content}")
    sample["text"] = "\n".join(parts)
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/block1_train.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    dataset_path = cfg["paths"]["dataset_jsonl"]
    out_dir = Path(cfg["paths"]["output_adapter_dir"])
    logs_dir = Path(cfg["paths"]["logs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load dataset
    raw = read_messages_jsonl(dataset_path)
    ds = Dataset.from_list(raw).map(messages_to_text)

    # 2) Load model
    model_id = cfg["model"]["train_model_id"]
    max_seq_length = int(cfg["model"]["max_seq_length"])
    load_in_4bit = bool(cfg["model"]["load_in_4bit"])

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = model_id,
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
    )

    # 3) Attach LoRA
    lora_cfg = cfg["lora"]
    model = FastVisionModel.get_peft_model(
        model,
        # 용량
        r = int(lora_cfg["r"]),   
        target_modules = list(lora_cfg["target_modules"]),
        lora_alpha = int(lora_cfg["alpha"]),
        lora_dropout = float(lora_cfg["dropout"]),
        bias = str(lora_cfg["bias"]),
        use_gradient_checkpointing = "unsloth",
        random_state = int(cfg["train"]["seed"]),
    )

    # 4) Train
    tcfg = cfg["train"]
    train_args = TrainingArguments(
        output_dir = str(logs_dir / cfg["project"]["run_name"]),
        num_train_epochs = float(tcfg["epochs"]),
        per_device_train_batch_size = int(tcfg["per_device_train_batch_size"]),
        gradient_accumulation_steps = int(tcfg["gradient_accumulation_steps"]),
        learning_rate = float(tcfg["learning_rate"]),
        warmup_steps = int(tcfg["warmup_steps"]),
        logging_steps = int(tcfg["logging_steps"]),
        save_steps = int(tcfg["save_steps"]),
        seed = int(tcfg["seed"]),
        fp16 = True,
        report_to = [],
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = ds,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        args = train_args,
        packing = False,
    )

    trainer.train()

    # 5) Save adapter (LoRA cartridge)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    print(f"\n Saved LoRA adapter to: {out_dir}")


if __name__ == "__main__":
    main()