from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure tests run deterministically with the mock provider by default,
# avoiding live external API calls during testing unless explicitly configured.
if "GEMINI_API_KEY" in os.environ and not os.environ.get("USE_REAL_GEMINI_TESTS"):
    os.environ.pop("GEMINI_API_KEY", None)
