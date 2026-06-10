# Ani Kɛse ASR Fine-Tune Pipeline

This pipeline is for improving Akan/Twi speech recognition without burning Modal credit blindly. It is intentionally staged: dataset audit, smoke run, baseline eval, fine-tune, final eval, Hub push, cleanup, then Space testing.

## Goal

Produce the best practical WER improvement for Ani Kɛse family check-ins while preserving honest fallback behavior. If a transcript is weak, the product should still mark the response `needs_review` rather than feeding bad text into concern scoring.

The training harness is baseline-gated. It will not push a model to Hub unless the trained adapter improves primary Twi WER over the base MMS adapter by at least `--min-wer-delta`.

## Current Training Target

- Base model: `facebook/mms-1b-all`
- MMS adapter language: `aka`
- Output repo: `teckedd/mms-akan-ani-kese-v1`
- Primary metric: normalized WER on the held-out GhanaNLP Twi split
- Secondary metric: normalized CER
- Robustness metric: separate YouVersion Akan WER/CER, never used for checkpoint selection

`aka` is used because MMS ASR rejected `twi` in earlier tests and accepted Akan as `aka`.

## Datasets

### Dataset 1: Main Twi ASR

- Hub: `ghananlpcommunity/twi-speech-text-multispeaker-16k`
- Config/split: `default/train`
- Live rows checked: `15,560`
- Columns: `audio`, `text`, `duration`
- Duration: `0.039s` to `10.0s`, mean `2.20s`
- Use: primary training and primary held-out eval

### Dataset 2: YouVersion Akan

- Hub: `AfriSpeech/youversion-african-speech`
- Config/split: `Akan_aka/train`
- Live rows checked: `2,180`
- Columns: `id`, `language`, `text`, `duration`, `source_file`, `audio`
- Duration: `0.1s` to `29.9s`, mean `6.05s`
- Domain: Bible/YouVersion read speech
- Use first as supplemental eval, not dominant training data

## Small Things That Matter

- Use a fixed seed (`42`) and a fixed held-out split.
- Normalize references and predictions the same way before WER/CER.
- Report both WER and CER because Twi spelling variation can make word-level metrics harsh.
- Filter tiny clips and very long clips before training.
- Keep domain mix controlled. Family-care check-ins should dominate.
- Run a small smoke job before a full paid job.
- Compare against the baseline model on the exact same primary eval split.
- Keep supplemental/domain-shift evals separate from the primary model-selection metric.
- Do not push if primary WER does not beat the baseline gate.
- Keep low-confidence ASR as `needs_review` in the product.

## Run Stages

### 1. Smoke Job

Purpose: confirm dataset loading, audio casting, labels, collator, baseline eval, training loop, and eval metrics.

```bash
modal run finetune/finetune_mms_twi.py \
  --output-repo teckedd/mms-akan-ani-kese-v1 \
  --max-train-samples 256 \
  --max-eval-samples 64 \
  --num-train-epochs 1 \
  --eval-steps 25 \
  --save-steps 25
```

Expected output:

- `baseline_primary_wer`
- `baseline_primary_cer`
- `final_primary_wer`
- `final_primary_cer`
- sample baseline/final predictions
- no Hub push

### 2. Supplemental Eval Smoke

Purpose: verify the model can evaluate against YouVersion Akan without mixing that domain into training.

```bash
modal run finetune/finetune_mms_twi.py \
  --output-repo teckedd/mms-akan-ani-kese-v1 \
  --max-train-samples 256 \
  --max-eval-samples 64 \
  --include-youversion \
  --youversion-mode eval \
  --max-youversion-samples 200 \
  --num-train-epochs 1 \
  --eval-steps 25 \
  --save-steps 25
```

### 3. Full Cost-Capped Run

Purpose: train the publishable model.

```bash
modal run finetune/finetune_mms_twi.py \
  --output-repo teckedd/mms-akan-ani-kese-v1 \
  --max-train-samples 12000 \
  --max-eval-samples 1200 \
  --include-youversion \
  --youversion-mode eval \
  --max-youversion-samples 200 \
  --num-train-epochs 3 \
  --per-device-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --learning-rate 3e-5 \
  --warmup-ratio 0.1 \
  --eval-steps 250 \
  --save-steps 250 \
  --early-stopping-patience 2 \
  --min-wer-delta 0.005 \
  --push-to-hub
```

Do not use `youversion-mode train` unless the eval-only path shows value.

## Evaluation Gates

The fine-tune is accepted only if:

- The smoke job finishes with valid WER/CER.
- Full-run normalized WER improves over baseline on the same primary eval split.
- CER does not regress materially when WER improves.
- YouVersion robustness WER/CER is reported separately and does not hide primary regression.
- The model can be loaded from Hub by the ASR eval Space.
- A real voice sample in the testing Space produces a usable transcript.

## Cleanup

After every Modal run:

```bash
modal app stop ani-kese-finetune --yes
modal app list
```

Expected state:

- `ani-kese-finetune`: `stopped`
- active tasks: `0`

## Testing Space Update

After the model is pushed:

1. Add `teckedd/mms-akan-ani-kese-v1` to the ASR evaluation Space model choices.
2. Run the same real Twi/Fante voice clips through:
   - `facebook/mms-1b-all`
   - `teckedd/whisper_small-waxal_akan-asr-v1`
   - `GiftMark/akan-whisper-model`
   - `teckedd/mms-akan-ani-kese-v1`
3. Record WER/CER where references exist.
4. Keep community voting for meaning preservation where references are unavailable.

## Sources Checked

- Hugging Face Transformers ASR task docs: https://huggingface.co/docs/transformers/en/tasks/asr
- Hugging Face low-resource XLSR/Wav2Vec2 fine-tuning guide: https://huggingface.co/blog/fine-tune-xlsr-wav2vec2
- Hugging Face Audio Course ASR evaluation: https://huggingface.co/learn/audio-course/en/chapter5/evaluation
- Hugging Face Evaluate WER implementation: https://github.com/huggingface/evaluate/blob/main/metrics/wer/wer.py
