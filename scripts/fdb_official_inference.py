"""Run the official FDB-v3 batch inference (run_tool_benchmark_all_released.py).

The official script is imported and its main() called unchanged, with the same
command-line arguments. One accommodation: its ASR loader moves the NeMo model
to CUDA unconditionally (``model.cuda()``), which fails on a machine without an
NVIDIA GPU. When - and only when - no CUDA device is available, that loader is
replaced by one that keeps the model on the CPU. On a GPU machine (the official
evaluation setup) nothing is changed.

    cd Full-Duplex-Bench/v3 && python <this file> --provider <label> [--root_dir ...] [--force]
"""

from __future__ import annotations

import sys
from pathlib import Path

V3 = Path.cwd()
if not (V3 / "run_tool_benchmark_all_released.py").exists():
    sys.exit("run from Full-Duplex-Bench/v3")
sys.path.insert(0, str(V3))

import run_tool_benchmark as rtb  # noqa: E402
import run_tool_benchmark_all_released as released  # noqa: E402


def _load_asr_model_cpu():
    import nemo.collections.asr as nemo_asr

    print("🔊 Loading ASR model (no CUDA device: running on CPU)...")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=rtb.ASR_MODEL_NAME)
    print("✅ ASR model loaded")
    return model


try:
    import torch

    cuda = torch.cuda.is_available()
except ImportError:
    cuda = False

if not cuda:
    rtb.load_asr_model = _load_asr_model_cpu
    released.load_asr_model = _load_asr_model_cpu

released.main()
