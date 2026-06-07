ASR_CONFIG = {
    "primary": {
        "label": "MMS-1B-all",
        "model_id": "facebook/mms-1b-all",
        "type": "mms",
        "parameters_b": 1.0,
    },
    "fine_tuned": {
        "label": "Adwuma Pa Akan Whisper fine-tune",
        "model_id": "teckedd/whisper_small-waxal_akan-asr-v1",
        "type": "whisper",
        "parameters_b": 0.2,
    },
    "fallback": {
        "label": "GiftMark Akan Whisper",
        "model_id": "GiftMark/akan-whisper-model",
        "type": "whisper",
        "parameters_b": 0.2,
    },
    "default_language": "twi",
    "supported_languages": {
        "Twi": "aka",
        "Fante": "aka",
        "English": "eng",
    },
    "display_languages": {
        "twi": "Twi",
        "fat": "Fante",
        "eng": "English",
    },
    "confidence_threshold": 0.4,
}

LLM_CONFIG = {
    "label": "Qwen2.5-7B-Instruct",
    "model_id": "Qwen/Qwen2.5-7B-Instruct",
    "parameters_b": 7.0,
    "max_new_tokens": 384,
    "temperature": 0.2,
}

TRANSLATION_CONFIG = {
    "label": "Twi-English NLLB",
    "model_id": "ninte/twi-en-nllb-v2",
    "parameters_b": 0.6,
    "source_language_map": {
        "twi": "twi_Latn",
        "fat": "twi_Latn",
        "aka": "twi_Latn",
        "eng": "eng_Latn",
    },
    "target_language": "eng_Latn",
}

TTS_CONFIG = {
    "label": "MMS TTS",
    "model_id": "facebook/mms-tts-aka",
    "parameters_b": 1.2,
    "language_map": {
        "twi": "aka",
        "fat": "aka",
        "eng": "eng",
    },
    "model_map": {
        "aka": "facebook/mms-tts-aka",
        "eng": "facebook/mms-tts-eng",
    },
}


def total_parameter_budget_b() -> float:
    return (
        ASR_CONFIG["primary"]["parameters_b"]
        + TRANSLATION_CONFIG["parameters_b"]
        + LLM_CONFIG["parameters_b"]
        + TTS_CONFIG["parameters_b"]
    )
