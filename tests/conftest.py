"""Shared test fixtures: imports, a stub LLM, and a tiny evaluator."""

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The whole backend is imported as `services.*`, so only the repo root needs to
# be importable — never `services/` itself, which would load a second copy.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Point the API at a throwaway database. Without this the tests silently use
# `data/rus.db`, which only works when that file happens to exist.
_TEST_DB_DIR = Path(tempfile.mkdtemp(prefix="rus-tests-"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TEST_DB_DIR / 'test.db'}")


# ── stub LLM ─────────────────────────────────────────────────────────────────

from services.worker.llm import LLMResponse  # noqa: E402


class FakeLLM:
    """Duck-typed stand-in for LLMClient — no network, deterministic output."""

    def __init__(self, *args, **kwargs):
        self.current_provider_name = "fake"
        self.current_model = "fake/model"
        self.switched_provider = False
        self.previous_provider = None
        self.researcher_calls = 0

    def generate(self, prompt, system_prompt=None, call_name=None, **kwargs):
        return LLMResponse(content="ok", model="fake/model", provider="fake")

    def extract_tags(self, prompt, system_prompt=None, call_name=None, **kwargs):
        if call_name and call_name.startswith("researcher"):
            self.researcher_calls += 1
            return {
                "name": f"candidate_{self.researcher_calls}",
                "motivation": "iterate on the previous attempt",
                "code": "print('hello from candidate')\n",
            }
        return {"analysis": "A stable result worth remembering for later runs."}


FAKE_EVALUATOR = (
    "import json, sys\n"
    "from pathlib import Path\n"
    "Path(sys.argv[2]).write_text(json.dumps({\n"
    "    'eval_score': 0.5,\n"
    "    'success': True,\n"
    "    'runtime': 0.01,\n"
    "    'stdout': 'candidate output',\n"
    "    'stderr': '',\n"
    "}))\n"
)
