"""
Cost-capped Modal fine-tune harness for Ani Kese Akan/Twi ASR.

Default behavior is intentionally conservative:

- train only the MMS language adapter/head, not the full 1B model
- keep primary Twi eval separate from YouVersion robustness eval
- early-stop on primary WER
- refuse to push to Hub unless primary WER beats the baseline gate

Smoke:

    modal run finetune/finetune_mms_twi.py \
      --output-repo teckedd/mms-akan-ani-kese-v1 \
      --max-train-samples 256 \
      --max-eval-samples 64 \
      --num-train-epochs 1 \
      --eval-steps 25 \
      --save-steps 25

Publish-gated run:

    modal run finetune/finetune_mms_twi.py \
      --output-repo teckedd/mms-akan-ani-kese-v1 \
      --max-train-samples 12000 \
      --max-eval-samples 1200 \
      --include-youversion \
      --youversion-mode eval \
      --max-youversion-samples 200 \
      --num-train-epochs 3 \
      --push-to-hub

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
        "safetensors>=0.4",
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
    learning_rate: float = 3e-5,
    warmup_ratio: float = 0.1,
    eval_steps: int = 250,
    save_steps: int = 250,
    early_stopping_patience: int = 2,
    min_wer_delta: float = 0.005,
    include_youversion: bool = False,
    youversion_mode: str = "eval",
    max_youversion_samples: int = 200,
    reinit_adapters: bool = False,
    push_to_hub: bool = False,
    force_push: bool = False,
) -> dict:
    import inspect
    import os
    import re
    import unicodedata
    from dataclasses import dataclass
    from typing import Any

    import evaluate
    import numpy as np
    import torch
    from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset
    from safetensors.torch import save_file as safe_save_file
    from transformers import (
        AutoProcessor,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        Wav2Vec2ForCTC,
    )
    from transformers.models.wav2vec2.modeling_wav2vec2 import WAV2VEC2_ADAPTER_SAFE_FILE

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
    if eval_steps <= 0 or save_steps <= 0:
        raise ValueError("eval_steps and save_steps must be positive")
    if save_steps != eval_steps:
        print("save_steps differs from eval_steps; early stopping waits for save checkpoints.")

    def normalize_text(value: str) -> str:
        text = unicodedata.normalize("NFKC", value or "").lower()
        text = re.sub(r"[\u2018\u2019`´]", "'", text)
        text = re.sub(r"[\u201c\u201d]", '"', text)
        text = re.sub(r"[^a-zà-ÿɔɛŋɲáéíóúýàèìòùâêîôûäëïöüãẽĩõũñç'\\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def valid_raw_row(
        row: dict[str, Any],
        min_duration: float,
        max_duration: float,
        max_text_chars: int,
        max_chars_per_second: float,
    ) -> bool:
        duration = float(row["duration"])
        text = normalize_text(row["text"])
        if not (min_duration <= duration <= max_duration):
            return False
        if not (2 <= len(text) <= max_text_chars):
            return False
        return (len(text) / max(duration, 0.1)) <= max_chars_per_second

    print(f"Loading primary dataset {DATASET_ID} ({DATASET_CONFIG}/train)")
    primary = load_dataset(DATASET_ID, DATASET_CONFIG, split="train")
    primary = primary.cast_column("audio", Audio(sampling_rate=16000))
    primary = primary.filter(
        lambda row: valid_raw_row(
            row,
            min_duration=0.5,
            max_duration=8.0,
            max_text_chars=150,
            max_chars_per_second=24.0,
        ),
        desc="Filtering primary audio/text rows",
    )
    split = primary.train_test_split(test_size=0.1, seed=42)
    ds = DatasetDict({"train": split["train"], "eval_primary": split["test"]})

    if max_train_samples > 0:
        ds["train"] = ds["train"].select(range(min(max_train_samples, len(ds["train"]))))
    if max_eval_samples > 0:
        ds["eval_primary"] = ds["eval_primary"].select(
            range(min(max_eval_samples, len(ds["eval_primary"])))
        )

    youversion_rows = 0
    if include_youversion:
        print(f"Loading supplemental {YOUVERSION_DATASET_ID} ({YOUVERSION_CONFIG}/train)")
        yv = load_dataset(YOUVERSION_DATASET_ID, YOUVERSION_CONFIG, split="train")
        yv = yv.cast_column("audio", Audio(sampling_rate=16000))
        yv = yv.filter(
            lambda row: valid_raw_row(
                row,
                min_duration=0.75,
                max_duration=12.0,
                max_text_chars=220,
                max_chars_per_second=24.0,
            ),
            desc="Filtering YouVersion audio/text rows",
        )
        if max_youversion_samples > 0:
            yv = yv.select(range(min(max_youversion_samples, len(yv))))
        youversion_rows = len(yv)
        if youversion_mode == "train":
            ds["train"] = concatenate_datasets([ds["train"], yv])
        else:
            ds["eval_youversion"] = yv
        print(
            f"YouVersion rows added: {youversion_rows} to {youversion_mode}. "
            f"train={len(ds['train'])}, primary_eval={len(ds['eval_primary'])}"
        )

    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "set_target_lang"):
        processor.tokenizer.set_target_lang(TARGET_LANG)

    model = Wav2Vec2ForCTC.from_pretrained(
        BASE_MODEL,
        target_lang=TARGET_LANG,
        ignore_mismatched_sizes=True,
        ctc_zero_infinity=True,
        ctc_loss_reduction="mean",
    )
    if hasattr(model, "load_adapter"):
        model.load_adapter(TARGET_LANG)
    if reinit_adapters and hasattr(model, "init_adapter_layers"):
        model.init_adapter_layers()
    if hasattr(model, "freeze_base_model"):
        model.freeze_base_model()

    adapter_weights = model._get_adapters() if hasattr(model, "_get_adapters") else {}
    for param in adapter_weights.values():
        param.requires_grad = True

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())
    print(
        "Trainable params: "
        f"{trainable_params:,}/{total_params:,} ({trainable_params / max(total_params, 1):.2%})"
    )
    print(f"Adapter tensors marked trainable: {len(adapter_weights)}")

    def ctc_input_frames(input_length: int) -> int:
        frames = model._get_feat_extract_output_lengths(torch.tensor([input_length]))
        return int(frames.item())

    def prepare(batch: dict[str, Any]) -> dict[str, Any]:
        audio = batch["audio"]
        input_values = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_values[0]
        reference_text = normalize_text(batch["text"])
        labels = processor(text=reference_text).input_ids
        batch["input_values"] = input_values
        batch["input_length"] = len(input_values)
        batch["input_frames"] = ctc_input_frames(len(input_values))
        batch["labels"] = labels
        batch["label_length"] = len(labels)
        batch["reference_text"] = reference_text
        return batch

    prepared = DatasetDict()
    for name, dataset in ds.items():
        prepared[name] = dataset.map(
            prepare,
            remove_columns=dataset.column_names,
            desc=f"Preparing CTC features for {name}",
        )
        before = len(prepared[name])
        prepared[name] = prepared[name].filter(
            lambda row: row["label_length"] < row["input_frames"],
            desc=f"Filtering CTC-invalid rows for {name}",
        )
        dropped = before - len(prepared[name])
        if dropped:
            print(f"Dropped {dropped} CTC-invalid rows from {name}")
    ds = prepared

    print(
        "Prepared rows: "
        f"train={len(ds['train'])}, primary_eval={len(ds['eval_primary'])}, "
        f"youversion_eval={len(ds['eval_youversion']) if 'eval_youversion' in ds else 0}"
    )
    if not len(ds["train"]) or not len(ds["eval_primary"]):
        raise ValueError("Training and primary eval sets must be non-empty after filtering")

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        label_ids = np.array(pred.label_ids, copy=True)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = [normalize_text(text) for text in processor.batch_decode(pred_ids)]
        label_str = [
            normalize_text(text)
            for text in processor.batch_decode(label_ids, group_tokens=False)
        ]
        return {
            "wer": wer_metric.compute(predictions=pred_str, references=label_str),
            "cer": cer_metric.compute(predictions=pred_str, references=label_str),
        }

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

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    training_arg_values = {
        "output_dir": "/cache/mms-twi-checkpoints",
        "per_device_train_batch_size": per_device_batch_size,
        "per_device_eval_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "fp16": False,
        "fp16_full_eval": False,
        "eval_strategy": "steps",
        "evaluation_strategy": "steps",
        "eval_steps": eval_steps,
        "save_strategy": "steps",
        "save_steps": save_steps,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_wer",
        "greater_is_better": False,
        "gradient_checkpointing": False,
        "max_grad_norm": 0.5,
        "group_by_length": True,
        "dataloader_num_workers": 2,
        "logging_steps": max(1, min(25, eval_steps)),
        "report_to": "none",
        "remove_unused_columns": False,
        "push_to_hub": False,
        "hub_model_id": output_repo or None,
        "hub_token": hf_token,
    }
    accepted_training_args = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "eval_strategy" in accepted_training_args:
        training_arg_values.pop("evaluation_strategy", None)
    else:
        training_arg_values.pop("eval_strategy", None)
    training_args = TrainingArguments(
        **{key: value for key, value in training_arg_values.items() if key in accepted_training_args}
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["eval_primary"],
        processing_class=processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    def sample_predictions(dataset, count: int = 5) -> list[dict[str, str]]:
        sample_count = min(count, len(dataset))
        if sample_count == 0:
            return []
        rows = [dataset[i] for i in range(sample_count)]
        batch = data_collator(rows)
        device = trainer.model.device
        batch = {key: value.to(device) for key, value in batch.items()}
        trainer.model.eval()
        with torch.no_grad():
            logits = trainer.model(**batch).logits
        pred_ids = torch.argmax(logits, dim=-1)
        predictions = [normalize_text(text) for text in processor.batch_decode(pred_ids)]
        return [
            {"reference": rows[index]["reference_text"], "prediction": predictions[index]}
            for index in range(sample_count)
        ]

    baseline_primary = trainer.evaluate(
        eval_dataset=ds["eval_primary"],
        metric_key_prefix="baseline_primary",
    )
    baseline_youversion = None
    if "eval_youversion" in ds:
        baseline_youversion = trainer.evaluate(
            eval_dataset=ds["eval_youversion"],
            metric_key_prefix="baseline_youversion",
        )
    baseline_samples = sample_predictions(ds["eval_primary"])
    print(f"Baseline primary metrics: {baseline_primary}")
    if baseline_youversion:
        print(f"Baseline YouVersion metrics: {baseline_youversion}")
    print(f"Baseline samples: {baseline_samples}")

    if early_stopping_patience > 0:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
    train_result = trainer.train()
    trainer.callback_handler.callbacks = [
        callback
        for callback in trainer.callback_handler.callbacks
        if not isinstance(callback, EarlyStoppingCallback)
    ]
    print(f"Train metrics: {train_result.metrics}")

    final_primary = trainer.evaluate(
        eval_dataset=ds["eval_primary"],
        metric_key_prefix="final_primary",
    )
    final_youversion = None
    if "eval_youversion" in ds:
        final_youversion = trainer.evaluate(
            eval_dataset=ds["eval_youversion"],
            metric_key_prefix="final_youversion",
        )
    final_samples = sample_predictions(ds["eval_primary"])
    print(f"Final primary metrics: {final_primary}")
    if final_youversion:
        print(f"Final YouVersion metrics: {final_youversion}")
    print(f"Final samples: {final_samples}")

    if adapter_weights:
        adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE.format(TARGET_LANG)
        adapter_path = os.path.join(training_args.output_dir, adapter_file)
        safe_save_file(model._get_adapters(), adapter_path, metadata={"format": "pt"})
        print(f"Saved adapter weights to {adapter_path}")

    baseline_wer = float(baseline_primary["baseline_primary_wer"])
    final_wer = float(final_primary["final_primary_wer"])
    wer_improvement = baseline_wer - final_wer
    push_allowed = force_push or (wer_improvement >= min_wer_delta)
    push_reason = (
        "force_push enabled"
        if force_push
        else f"primary WER improved by {wer_improvement:.4f}; required {min_wer_delta:.4f}"
    )
    if not push_allowed:
        push_reason = (
            f"not pushed: primary WER changed from {baseline_wer:.4f} to {final_wer:.4f}; "
            f"required improvement is {min_wer_delta:.4f}"
        )

    pushed = False
    if push_to_hub and push_allowed:
        trainer.push_to_hub(
            commit_message="Fine-tune MMS Akan ASR for Ani Kese",
            tags=["ani-kese", "akan", "asr", "ghana", "mms", "twi"],
        )
        processor.push_to_hub(output_repo, token=hf_token)
        pushed = True
    elif push_to_hub:
        print(push_reason)

    cache_volume.commit()
    return {
        "dataset": DATASET_ID,
        "base_model": BASE_MODEL,
        "target_language": TARGET_LANG,
        "train_rows": len(ds["train"]),
        "primary_eval_rows": len(ds["eval_primary"]),
        "youversion_eval_rows": len(ds["eval_youversion"]) if "eval_youversion" in ds else 0,
        "youversion_included": include_youversion,
        "youversion_mode": youversion_mode if include_youversion else None,
        "youversion_rows": youversion_rows,
        "reinit_adapters": reinit_adapters,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "learning_rate": learning_rate,
        "baseline_primary_metrics": baseline_primary,
        "baseline_youversion_metrics": baseline_youversion,
        "train_metrics": train_result.metrics,
        "final_primary_metrics": final_primary,
        "final_youversion_metrics": final_youversion,
        "baseline_samples": baseline_samples,
        "final_samples": final_samples,
        "wer_improvement": wer_improvement,
        "min_wer_delta": min_wer_delta,
        "push_allowed": push_allowed,
        "push_reason": push_reason,
        "pushed_to_hub": pushed,
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
    learning_rate: float = 3e-5,
    warmup_ratio: float = 0.1,
    eval_steps: int = 250,
    save_steps: int = 250,
    early_stopping_patience: int = 2,
    min_wer_delta: float = 0.005,
    include_youversion: bool = False,
    youversion_mode: str = "eval",
    max_youversion_samples: int = 200,
    reinit_adapters: bool = False,
    push_to_hub: bool = False,
    force_push: bool = False,
    background: bool = False,
):
    kwargs = dict(
        output_repo=output_repo,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        num_train_epochs=num_train_epochs,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        eval_steps=eval_steps,
        save_steps=save_steps,
        early_stopping_patience=early_stopping_patience,
        min_wer_delta=min_wer_delta,
        include_youversion=include_youversion,
        youversion_mode=youversion_mode,
        max_youversion_samples=max_youversion_samples,
        reinit_adapters=reinit_adapters,
        push_to_hub=push_to_hub,
        force_push=force_push,
    )
    if background:
        call = finetune.spawn(**kwargs)
        print(
            {
                "status": "spawned",
                "function_call_id": getattr(call, "object_id", None),
                "output_repo": output_repo,
                "push_to_hub": push_to_hub,
            }
        )
        return
    result = finetune.remote(**kwargs)
    print(result)
