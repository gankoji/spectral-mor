#!/usr/bin/env python3
"""
Phase D systems-style timing entry point. Delegates to ``compressed_inference_harness``
with the same CLI; use ``--max-new-tokens`` and ``--prefill-repeats`` for benchmark runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from compressed_inference_harness import main

if __name__ == "__main__":
    raise SystemExit(main())
