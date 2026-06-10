"""
Cost-capped Modal fine-tune harness for Twi ASR.

This script only runs when called explicitly:

    modal run finetune/finetune_mms_twi.py --max-train-samples 128 --max-eval-samples 32

For a real run, pass a Hub repo and enable push:

    modal run finetune/finetune_mms_twi.py \
      --output-repo teckedd/mms-akan-ani-kese-v1 \
      --max-train-samples 12000 \
      --max-eval-samples 1200 \
      --num-train-epochs 3 \
      --push-to-hub

To add a small held-out YouVersion Akan robustness slice:

    modal run finetune/finetune_mms_twi.py \
      --include-youversion \
      --youversion-mode eval \
      --max-youversion-samples 200

Requires a Modal secret named `huggingface-token` with `HF_TOKEN`.
"""

from __future__ import annotations

import modal


APP_NAME = "ani-kese-finetune"
DATASET_ID = "ghananlpcommunity/twi-speech-text-multispeaker-16k"
DATASET_CONFIG = "default"
YOUVERSION_DATASET_ID = "AfriSpeech/youversion-african-speech"
YOUVERSION_CONFIG = "Akan_aka"
BASE_MODEL = "facebook/mms-1b-all"
TARGET_LANG = "aka"


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "accelerate>=0.30",
        "datasets[audio]>=2.19",
        "evaluate>=0.4",
        "huggingface_hub>=0.24",
        "jiwer>=3.0",
        "librosa>=0.10",
        "numpy<2",
        "soundfile>=0.12",
        "torch>=2.2",
        "transformers>=4.44",
    )
)

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("ani-kese-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 5,
    scaledown_window=10,
    volumes={"/cache": cache_volume},
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def finetune(
    output_repo: str,
    max_train_samples: int = 128,
    max_eval_samples: int = 32,
    num_train_epochs: float = 1.0,
    per_device_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-4,
    include_youversion: bool = False,
    youversion_mode: str = "eval",
    max_youversion_samples: int = 200,
    push_to_hub: bool = False,
) -> dict:
    import os
    import re
    import inspect
    import unicodedata
    from dataclasses import dataclass
    from typing import Any

    import evaluate
    import numpy as np
    import torch
    from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset
    from transformers import AutoProcessor, Trainer, TrainingArguments, Wav2Vec2ForCTC

    if push_to_hub and not output_repo:
        raise ValueError("output_repo is required when push_to_hub=True")

    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface/transformers"
    os.environ["HF_DATASETS_CACHE"] = "/cache/huggingface/datasets"

    hf_token = os.environ.get("HF_TOKEN")
    if push_to_hub and not hf_token:
        raise ValueError("HF_TOKEN is required to push the fine-tuned model")
    if youversion_mode not in {"train", "eval"}:
        raise ValueError("youversion_mode must be 'train' or 'eval'")

    def normalize_text(value: str) -> str:
        text = unicodedata.normalize("NFKC", value or "").lower()
        text = re.sub(r"[\u2018\u2019`´]", "'", text)
        text = re.sub(r"[\u201c\u201d]", '"', text)
        text = re.sub(r"[^a-zà-ÿɔɛŋɲáéíóúýàèìòùâêîôûäëïöüãẽĩõũñç'\\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    print(f"Loading {DATASET_ID} ({DATASET_CONFIG}/train)")
    raw = load_dataset(DATASET_ID, DATASET_CONFIG, split="train")
    raw = raw.cast_column("audio", Audio(sampling_rate=16000))
    raw = raw.filter(lambda row: 0.25 <= float(row["duration"]) <= 10.0 and bool(normalize_text(row["text"])))
    split = raw.train_test_split(test_size=0.1, seed=42)
    ds = DatasetDict({"train": split["train"], "eval": split["test"]})

    if max_train_samples > 0:
        ds["train"] = ds["train"].select(range(min(max_train_samples, len(ds["train"]))))
    if max_eval_samples > 0:
        ds["eval"] = ds["eval"].select(range(min(max_eval_samples, len(ds["eval"]))))

    print(f"Prepared rows: train={len(ds['train'])}, eval={len(ds['eval'])}")

    youversion_rows = 0
    if include_youversion:
        print(f"Loading supplemental {YOUVERSION_DATASET_ID} ({YOUVERSION_CONFIG}/train)")
        yv = load_dataset(YOUVERSION_DATASET_ID, YOUVERSION_CONFIG, split="train")
        yv = yv.cast_column("audio", Audio(sampling_rate=16000))
        yv = yv.filter(
            lambda row: 0.5 <= float(row["duration"]) <= 15.0 and bool(normalize_text(row["text"]))
        )
        if max_youversion_samples > 0:
            yv = yv.select(range(min(max_youversion_samples, len(yv))))
        youversion_rows = len(yv)
        if youversion_mode == "train":
            ds["train"] = concatenate_datasets([ds["train"], yv])
        else:
            ds["eval"] = concatenate_datasets([ds["eval"], yv])
        print(
            f"YouVersion rows added: {youversion_rows} to {youversion_mode}. "
            f"Now train={len(ds['train'])}, eval={len(ds['eval'])}"
        )

    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "set_target_lang"):
        processor.tokenizer.set_target_lang(TARGET_LANG)

    model = Wav2Vec2ForCTC.from_pretrained(
        BASE_MODEL,
        target_lang=TARGET_LANG,
        ignore_mismatched_sizes=True,
    )
    if hasattr(model, "freeze_base_model"):
        model.freeze_base_model()
    if hasattr(model, "load_adapter"):
        model.load_adapter(TARGET_LANG)

    def prepare(batch: dict[str, Any]) -> dict[str, Any]:
        audio = batch["audio"]
        batch["input_values"] = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_values[0]
        labels = processor(text=normalize_text(batch["text"])).input_ids
        batch["labels"] = labels
        return batch

    ds = ds.map(
        prepare,
        remove_columns=ds["train"].column_names,
        desc="Preparing CTC features",
    )

    @dataclass
    class DataCollatorCTCWithPadding:
        processor: Any
        padding: bool | str = True

        def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            batch = self.processor.pad(
                input_features,
                padding=self.padding,
                return_tensors="pt",
            )
            try:
                labels_batch = self.processor.pad(
                    labels=label_features,
                    padding=self.padding,
                    return_tensors="pt",
                )
            except TypeError:
                with self.processor.as_target_processor():
                    labels_batch = self.processor.pad(
                        label_features,
                        padding=self.padding,
                        return_tensors="pt",
                    )
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels
            return batch

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = [normalize_text(text) for text in processor.batch_decode(pred_ids)]
        label_str = [normalize_text(text) for text in processor.batch_decode(pred.label_ids, group_tokens=False)]
        return {
            "wer": wer_metric.compute(predictions=pred_str, references=label_str),
            "cer": cer_metric.compute(predictions=pred_str, references=label_str),
        }

    training_arg_values = {
        "output_dir": "/cache/mms-twi-checkpoints",
        "per_device_train_batch_size": per_device_batch_size,
        "per_device_eval_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "warmup_steps": 0,
        "fp16": False,
        "fp16_full_eval": False,
        "eval_strategy": "epoch",
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "wer",
        "greater_is_better": False,
        "gradient_checkpointing": True,
        "group_by_length": True,
        "dataloader_num_workers": 2,
        "logging_steps": 25,
        "report_to": "none",
        "push_to_hub": push_to_hub,
        "hub_model_id": output_repo or None,
        "hub_token": hf_token,
    }
    accepted_training_args = set(inspect.signature(TrainingArguments.__init__).parameters)
    training_args = TrainingArguments(
        **{key: value for key, value in training_arg_values.items() if key in accepted_training_args}
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        processing_class=processor,
        data_collator=DataCollatorCTCWithPadding(processor=processor),
        compute_metrics=compute_metrics,
    )

    baseline_metrics = trainer.evaluate(metric_key_prefix="baseline")
    print(f"Baseline metrics: {baseline_metrics}")
    train_result = trainer.train()
    print(f"Train metrics: {train_result.metrics}")
    eval_metrics = trainer.evaluate(metric_key_prefix="eval")
    print(f"Eval metrics: {eval_metrics}")

    if push_to_hub:
        trainer.push_to_hub(
            commit_message="Fine-tune MMS Twi ASR for Ani Kɛse",
            tags=["ani-kese", "akan", "asr", "ghana", "mms", "twi"],
        )
        processor.push_to_hub(output_repo, token=hf_token)

    cache_volume.commit()
    return {
        "dataset": DATASET_ID,
        "base_model": BASE_MODEL,
        "target_language": TARGET_LANG,
        "train_rows": len(ds["train"]),
        "eval_rows": len(ds["eval"]),
        "youversion_included": include_youversion,
        "youversion_mode": youversion_mode if include_youversion else None,
        "youversion_rows": youversion_rows,
        "baseline_metrics": baseline_metrics,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "pushed_to_hub": push_to_hub,
        "output_repo": output_repo,
    }


@app.local_entrypoint()
def main(
    output_repo: str = "",
    max_train_samples: int = 128,
    max_eval_samples: int = 32,
    num_train_epochs: float = 1.0,
    per_device_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-4,
    include_youversion: bool = False,
    youversion_mode: str = "eval",
    max_youversion_samples: int = 200,
    push_to_hub: bool = False,
):
    result = finetune.remote(
        output_repo=output_repo,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        num_train_epochs=num_train_epochs,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        include_youversion=include_youversion,
        youversion_mode=youversion_mode,
        max_youversion_samples=max_youversion_samples,
        push_to_hub=push_to_hub,
    )
    print(result)
