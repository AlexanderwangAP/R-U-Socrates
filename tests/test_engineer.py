"""Engineer: evaluator resolution, argv contract, and process cleanup."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from services.worker.pipeline import Engineer

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_EVALUATOR = REPO_ROOT / "services" / "worker" / "evaluator.py"


@pytest.fixture
def work_dir(tmp_path):
    return tmp_path / "step"


# ── evaluator resolution ─────────────────────────────────────────────────────


def test_python_evaluator_resolves_to_python_interpreter(work_dir):
    cmd = Engineer()._resolve_cmd(str(BUNDLED_EVALUATOR), work_dir)
    assert cmd[0] == sys.executable
    assert cmd[1] == str(BUNDLED_EVALUATOR)


def test_evaluator_always_receives_code_and_results_argv(work_dir):
    cmd = Engineer()._resolve_cmd(str(BUNDLED_EVALUATOR), work_dir)
    assert cmd[-2:] == [str(work_dir / "code.py"), str(work_dir / "results.json")]


def test_shell_evaluator_honours_shebang(work_dir):
    work_dir.mkdir(parents=True, exist_ok=True)
    script = work_dir / "ev.sh"
    script.write_text("#!/usr/bin/env bash\n")
    cmd = Engineer()._resolve_cmd(str(script), work_dir)
    assert cmd[-2:] == [str(work_dir / "code.py"), str(work_dir / "results.json")]
    assert "bash" in cmd[:3]


def test_unknown_script_falls_back_to_a_shell(work_dir, tmp_path):
    script = tmp_path / "mystery.sh"
    script.write_text("echo fallback")
    script.chmod(0o755)
    cmd = Engineer()._resolve_cmd(str(script), work_dir)
    assert cmd[0] in ("bash", "sh")
    assert cmd[-2:] == [str(work_dir / "code.py"), str(work_dir / "results.json")]


def test_missing_interpreter_raises(work_dir, tmp_path, monkeypatch):
    script = tmp_path / "mystery.sh"
    script.write_text("echo fallback")

    import services.worker.pipeline as pipeline

    monkeypatch.setattr(pipeline.shutil, "which", lambda _name: None)
    with pytest.raises(FileNotFoundError):
        Engineer()._resolve_cmd(str(script), work_dir)


# ── end-to-end run with the bundled evaluator ────────────────────────────────


def test_bundled_evaluator_scores_a_candidate(work_dir):
    result = Engineer().run(
        code='print("hello from candidate")\n',
        work_dir=work_dir,
        eval_script=str(BUNDLED_EVALUATOR),
        timeout=60,
    )
    assert result["success"] is True
    assert result["eval_score"] > 0
    assert "hello from candidate" in result["stdout"]


def test_eval_code_is_written_and_executed(work_dir):
    evaluator = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "Path(sys.argv[2]).write_text(json.dumps({'eval_score': 0.42}))\n"
    )
    result = Engineer().run(
        code="x = 1\n",
        work_dir=work_dir,
        eval_code=evaluator,
        timeout=60,
    )
    assert result["success"] is True
    assert result["eval_score"] == pytest.approx(0.42)


def test_no_evaluator_is_skipped(work_dir):
    result = Engineer().run(code="x = 1\n", work_dir=work_dir)
    assert result["skipped"] is True
    assert result["eval_score"] == 0.0


# ── resource limits ──────────────────────────────────────────────────────────


def test_resource_limits_stop_a_runaway_candidate(work_dir):
    """A candidate that blows past the address-space ceiling must be stopped."""
    original = Engineer.MEMORY_LIMIT_MB
    Engineer.MEMORY_LIMIT_MB = 256
    try:
        result = Engineer().run(
            code="x = bytearray(2 * 1024 * 1024 * 1024)\n",  # 2 GB
            work_dir=work_dir,
            eval_script=str(BUNDLED_EVALUATOR),
            timeout=30,
        )
    finally:
        Engineer.MEMORY_LIMIT_MB = original

    assert result["success"] is False
    assert result["eval_score"] == 0.0


def test_resource_limits_do_not_break_a_normal_candidate(work_dir):
    original = Engineer.MEMORY_LIMIT_MB
    Engineer.MEMORY_LIMIT_MB = 1024
    try:
        result = Engineer().run(
            code="print('fine')\n",
            work_dir=work_dir,
            eval_script=str(BUNDLED_EVALUATOR),
            timeout=30,
        )
    finally:
        Engineer.MEMORY_LIMIT_MB = original

    assert result["success"] is True
    assert result["eval_score"] > 0


# ── process cleanup ──────────────────────────────────────────────────────────


def test_timeout_kills_children_and_leaves_no_orphans(work_dir):
    """A slow evaluator that spawns a child must not leave it behind."""
    work_dir.mkdir(parents=True, exist_ok=True)
    evaluator = work_dir / "slow.py"
    evaluator.write_text(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "time.sleep(120)\n"
    )

    result = Engineer().run(
        code="x = 1\n",
        work_dir=work_dir,
        eval_script=str(evaluator),
        timeout=2,
    )

    assert result["success"] is False
    assert "Timeout" in result["error"]

    # Give the kernel a moment, then confirm nothing from this test survives.
    time.sleep(0.5)
    out = subprocess.run(
        ["ps", "-eo", "args"], capture_output=True, text=True
    ).stdout
    assert "time.sleep(120)" not in out, "child process survived the timeout"
