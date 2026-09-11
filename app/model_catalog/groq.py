from typing import Final


CHAT: Final = "chat"
TEXT_TO_SPEECH: Final = "text_to_speech"
SPEECH_TO_TEXT: Final = "speech_to_text"
MODERATION: Final = "moderation"
COMPOUND: Final = "compound"


GROQ_MODEL_CAPABILITIES: dict[str, set[str]] = {
    "canopylabs/orpheus-v1-english": {
        TEXT_TO_SPEECH,
    },
    "canopylabs/orpheus-arabic-saudi": {
        TEXT_TO_SPEECH,
    },
    "whisper-large-v3": {
        SPEECH_TO_TEXT,
    },
    "whisper-large-v3-turbo": {
        SPEECH_TO_TEXT,
    },
    "meta-llama/llama-prompt-guard-2-86m": {
        MODERATION,
    },
    "meta-llama/llama-prompt-guard-2-22m": {
        MODERATION,
    },
    "groq/compound": {
        COMPOUND,
    },
    "groq/compound-mini": {
        COMPOUND,
    },
}

def get_groq_model_capabilities(model_id: str) -> set[str]:
    return GROQ_MODEL_CAPABILITIES.get(model_id, {CHAT})

def is_groq_chat_model(model_id: str) -> bool:
    return CHAT in get_groq_model_capabilities(model_id)