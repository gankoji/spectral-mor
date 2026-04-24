"""Tests for prompt loading helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from inference_eval import load_prompts_for_eval


class TestLoadPrompts(unittest.TestCase):
    def test_prompts_file_overrides_text(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "p.txt"
            p.write_text("line one\n\nline two\n", encoding="utf-8")
            out = load_prompts_for_eval(
                text="ignored",
                prompts_file=p,
                use_default_prompt_set=False,
                default_prompts_path=Path(td) / "missing.txt",
            )
            self.assertEqual(out, ["line one", "line two"])

    def test_fallback_text(self) -> None:
        out = load_prompts_for_eval(
            text="only this",
            prompts_file=None,
            use_default_prompt_set=False,
            default_prompts_path=Path("/nonexistent/default"),
        )
        self.assertEqual(out, ["only this"])


if __name__ == "__main__":
    unittest.main()
