"""End-to-end pipeline run with a stubbed LLM.

Proves the whole Researcher → Engineer → Analyzer loop works on a machine with
no embedding stack and no real model: the events stream in order, every
iteration is scored, and the candidate's output survives into the results.
"""

import asyncio

import services.worker.pipeline as pipeline_mod
from conftest import FAKE_EVALUATOR, FakeLLM
from services.worker.models import EventType, RunConfig


def _run(tmp_path, monkeypatch, max_iterations=2):
    monkeypatch.setattr(pipeline_mod, "LLMClient", FakeLLM)
    config = RunConfig(
        run_id="test-run",
        task_description="Optimise a trivial function",
        model="fake/model",
        max_iterations=max_iterations,
        eval_code=FAKE_EVALUATOR,
        data_dir=str(tmp_path),
    )

    events = []

    async def collect():
        async for event in pipeline_mod.Pipeline(config).run():
            events.append(event)

    asyncio.run(collect())
    return events


# ── the loop ─────────────────────────────────────────────────────────────────


def test_loop_runs_to_completion(tmp_path, monkeypatch):
    events = _run(tmp_path, monkeypatch, max_iterations=2)
    types = [e.type for e in events]

    assert types[0] == EventType.RUN_STARTED
    assert types[-1] == EventType.RUN_COMPLETE
    assert types.count(EventType.ITERATION_COMPLETE) == 2
    assert EventType.RESEARCHER_FAILED not in types
    assert EventType.ENGINEER_FAILED not in types


def test_every_iteration_is_scored_by_the_evaluator(tmp_path, monkeypatch):
    events = _run(tmp_path, monkeypatch, max_iterations=3)
    done = [e for e in events if e.type == EventType.ITERATION_COMPLETE]

    assert len(done) == 3
    assert all(e.eval_score == 0.5 for e in done)
    assert all(e.eval_success for e in done)


def test_candidate_output_survives_into_the_result(tmp_path, monkeypatch):
    """Regression: the evaluator's own stdout used to clobber the candidate's."""
    events = _run(tmp_path, monkeypatch, max_iterations=1)
    complete = next(e for e in events if e.type == EventType.ENGINEER_COMPLETE)
    assert complete.eval_stdout_preview.startswith("candidate output")


def test_run_complete_reports_the_best_score(tmp_path, monkeypatch):
    events = _run(tmp_path, monkeypatch, max_iterations=2)
    final = events[-1]

    assert final.best_score == 0.5
    assert final.stats["iterations"] == 2
    assert final.best_node is not None
    assert "hello from candidate" in final.best_node["code"]


def test_nodes_are_persisted_for_the_reasoning_tree(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, max_iterations=2)
    db_files = list(tmp_path.rglob("nodes.json"))
    assert db_files, "node database was never written"


def test_failed_attempts_do_not_shrink_the_run(tmp_path, monkeypatch):
    """10 requested iterations should still produce 10 completed ones."""

    class FlakyLLM(FakeLLM):
        """Fails both the tag pass and its raw-text fallback."""

        def _should_fail(self) -> bool:
            return self.researcher_calls <= 2

        def generate(self, prompt, system_prompt=None, call_name=None, **kwargs):
            if call_name and "researcher" in call_name and self._should_fail():
                raise RuntimeError("model unavailable")
            return super().generate(prompt, system_prompt, call_name, **kwargs)

        def extract_tags(self, prompt, system_prompt=None, call_name=None, **kwargs):
            if call_name and call_name.startswith("researcher"):
                self.researcher_calls += 1
                if self._should_fail():
                    raise ValueError("model returned no tags")
            return super().extract_tags(prompt, system_prompt, call_name, **kwargs)

    monkeypatch.setattr(pipeline_mod, "LLMClient", FlakyLLM)
    config = RunConfig(
        run_id="flaky-run",
        task_description="Optimise a trivial function",
        model="fake/model",
        max_iterations=4,
        eval_code=FAKE_EVALUATOR,
        data_dir=str(tmp_path),
    )

    events = []

    async def collect():
        async for event in pipeline_mod.Pipeline(config).run():
            events.append(event)

    asyncio.run(collect())

    final = events[-1]
    assert final.stats["iterations"] == 4
    assert final.stats["attempts"] > 4, "the failed attempts were not retried"
    assert [e.type for e in events].count(EventType.ITERATION_COMPLETE) == 4
