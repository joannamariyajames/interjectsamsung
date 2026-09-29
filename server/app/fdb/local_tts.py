"""Free, offline text-to-speech for the benchmark agent.

The hosted voice (Groq Orpheus) allows about 100 requests a day on a free
account, far short of a 100-recording run, so the default benchmark pipeline
speaks with Piper instead: a small ONNX voice that runs on the CPU, needs no key
or account, and has no quota. On a laptop it renders a short sentence in well
under a tenth of a second once warm, so it adds little to response latency.

The voice is LJSpeech (public-domain dataset) from the rhasspy/piper-voices
release, pinned by revision and SHA-256 so every run speaks with the same model.
``python -m app.fdb.local_tts --download`` fetches it ahead of a run; the agent
also fetches it on first use if it is missing.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import urllib.request
from pathlib import Path
from typing import Any

from livekit.agents import APIConnectOptions, tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

VOICE_NAME = "en_US-ljspeech-medium"
VOICE_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ljspeech/medium"
VOICE_FILES = {
    f"{VOICE_NAME}.onnx": "6f52a751e2349abe7a76735eb09dc1875298c77ea2342ffd2fef79ff81b87f22",
    f"{VOICE_NAME}.onnx.json": "141d612cc0a95ed7efc1ca936b845c2364967f2e9217c5dbfcf69fc4d6c65860",
}
DEFAULT_VOICE_DIR = Path(__file__).resolve().parents[3] / ".voices"


def voice_dir() -> Path:
    return Path(os.environ.get("FDB_TTS_VOICE_DIR") or DEFAULT_VOICE_DIR)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_voice(directory: Path | None = None) -> Path:
    """Download the pinned voice if needed and return the path of its model file."""
    directory = directory or voice_dir()
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in VOICE_FILES.items():
        target = directory / name
        if target.exists() and _sha256(target) == expected:
            continue
        partial = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(f"{VOICE_BASE_URL}/{name}", partial)
        actual = _sha256(partial)
        if actual != expected:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"{name}: checksum {actual} does not match the pinned {expected}")
        partial.replace(target)
    return directory / f"{VOICE_NAME}.onnx"


class PiperTTS(tts.TTS):
    """A LiveKit TTS that renders each sentence locally with Piper.

    Synthesis runs in a worker thread, so the conversation's event loop - and
    with it barge-in and tool calls - never waits on the CPU.
    """

    def __init__(self, *, model_path: str | Path | None = None, voice: Any = None) -> None:
        if voice is None:
            from piper import PiperVoice  # imported here: only this pipeline needs it

            voice = PiperVoice.load(str(model_path or ensure_voice()))
            # The first synthesis pays for ONNX start-up (seconds); do it now,
            # not in the middle of the first answer.
            for _ in voice.synthesize("Ready."):
                pass
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=int(voice.config.sample_rate),
            num_channels=1,
        )
        self._voice = voice

    @property
    def model(self) -> str:
        return VOICE_NAME

    @property
    def provider(self) -> str:
        return "Piper (local)"

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "_PiperStream":
        return _PiperStream(tts=self, input_text=text, conn_options=conn_options)

    def render(self, text: str) -> list[bytes]:
        """16-bit mono PCM for ``text``, one chunk per sentence Piper produced."""
        return [chunk.audio_int16_bytes for chunk in self._voice.synthesize(text)]


class _PiperStream(tts.ChunkedStream):
    def __init__(self, *, tts: PiperTTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._piper = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._piper.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )
        for pcm in await asyncio.to_thread(self._piper.render, self._input_text):
            output_emitter.push(pcm)
        output_emitter.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the pinned Piper voice used by the benchmark agent.")
    parser.add_argument("--download", action="store_true", help="download and verify the voice files")
    parser.add_argument("--dir", type=Path, default=None, help=f"voice directory (default {DEFAULT_VOICE_DIR})")
    args = parser.parse_args()
    if args.download:
        print(ensure_voice(args.dir))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
