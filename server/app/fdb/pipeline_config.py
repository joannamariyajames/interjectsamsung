"""The benchmark pipeline's settings, in one place.

The agent (``runner``), the text replay and the run collector all read the
configuration through here, so a result file always records the model and voice
that actually ran - not a default from another pipeline.
"""

from __future__ import annotations

import os
from typing import Any

GROQ_PROVIDERS = {"groq", "groq_cascaded", "cascaded_groq"}
NVIDIA_PROVIDERS = {"nvidia", "nvidia_cascaded", "cascaded_nvidia"}
CASCADED_PROVIDERS = GROQ_PROVIDERS | NVIDIA_PROVIDERS

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_DEFAULT_MODEL = "openai/gpt-oss-20b"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"


def provider() -> str:
    return os.getenv("LK_PROVIDER", "gemini3_8").strip().lower()


def nvidia_model() -> str:
    return os.getenv("FDB_LLM_MODEL", NVIDIA_DEFAULT_MODEL)


def groq_model() -> str:
    return os.getenv("GROQ_LLM_MODEL", GROQ_DEFAULT_MODEL)


def llm_model() -> str:
    """The primary model: NVIDIA's for ``nvidia``, Groq's otherwise."""
    return nvidia_model() if provider() in NVIDIA_PROVIDERS else groq_model()


def fallback_enabled() -> bool:
    """``FDB_LLM_FALLBACK=1``: the other provider's model answers when the primary fails.

    ``nvidia`` -> NVIDIA first, Groq when NVIDIA stalls or errors (the benchmark:
    the bulk of the requests stay off the Groq free quota). ``groq`` -> Groq
    first, NVIDIA once Groq's rate limit is reached (demos). The one-command
    script turns it on; without it a single model answers, as before.
    """
    return provider() in CASCADED_PROVIDERS and os.getenv("FDB_LLM_FALLBACK", "0") == "1"


def fallback_attempt_timeout() -> float:
    """Seconds an attempt may go without a response chunk before the next model is tried."""
    return float(os.getenv("FDB_LLM_ATTEMPT_TIMEOUT", "10"))


def llm_order() -> list[str]:
    """The models in the order they are tried (one entry without a fallback)."""
    nvidia = f"nvidia {nvidia_model()}"
    groq = f"groq {groq_model()}"
    first, second = (nvidia, groq) if provider() in NVIDIA_PROVIDERS else (groq, nvidia)
    return [first, second] if fallback_enabled() else [first]


def tts_choice() -> str:
    default = "orpheus" if provider() in GROQ_PROVIDERS else "piper"
    return os.getenv("FDB_TTS", default).strip().lower()


def describe() -> dict[str, Any]:
    """What a run used, for ``config.json`` and per-scenario result files."""
    p = provider()
    if p not in CASCADED_PROVIDERS:
        return {"lk_provider": p}
    nvidia = p in NVIDIA_PROVIDERS
    model = llm_model()
    desc: dict[str, Any] = {
        "lk_provider": p,
        "stt": f"groq {os.getenv('GROQ_STT_MODEL', 'whisper-large-v3-turbo')}",
        "llm_model": model,
        "llm_endpoint": os.getenv("FDB_LLM_BASE_URL", NVIDIA_BASE_URL) if nvidia else "https://api.groq.com/openai/v1",
        "llm_temperature": float(os.getenv("FDB_LLM_TEMPERATURE" if nvidia else "GROQ_LLM_TEMPERATURE", "0")),
    }
    desc["llm_fallback"] = (
        {"order": llm_order(), "attempt_timeout_s": fallback_attempt_timeout()} if fallback_enabled() else "off"
    )
    if nvidia:
        desc["llm_seed"] = int(os.getenv("FDB_LLM_SEED", "7"))
    if "gpt-oss" in model:
        desc["llm_reasoning_effort"] = os.getenv("FDB_REASONING_EFFORT", "low") if nvidia else "low"
    desc["spoken_ack"] = (
        {"text": os.getenv("FDB_ACK_TEXT", "Sure, one moment."), "lead_in_s": float(os.getenv("FDB_ACK_DELAY_S", "1.0"))}
        if os.getenv("FDB_ACK", "0") == "1"
        else "off"
    )
    if tts_choice() == "orpheus":
        desc["tts"] = f"groq {os.getenv('GROQ_TTS_MODEL', 'canopylabs/orpheus-v1-english')}"
    else:
        from app.fdb.local_tts import VOICE_NAME  # noqa: PLC0415 - avoids importing LiveKit for the Groq voice

        desc["tts"] = f"piper {VOICE_NAME} (local)"
    return desc
