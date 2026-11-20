"""Pause and cancel control flow: the loop must actually stop."""

import asyncio

import pytest

import services.worker.pipeline as pipeline_mod
from conftest import FAKE_EVALUATOR, FakeLLM
from services.worker.models import EventType, RunConfig, RunCancelled
from services.api.store import RunEventStore


def _config(tmp_path, max_iterations=5):
    return RunConfig(
        run_id="control-run",
        task_description="Optimise a trivial function",
        model="fake/model",
        max_iterations=max_iterations,
        eval_code=FAKE_EVALUATOR,
        data_dir=str(tmp_path),
    )


def _collect(pipeline):
    events = []

    async def go():
        async for event in pipeline.run():
            events.append(event)

    return events, go


# ── store checkpoint ─────────────────────────────────────────────────────────


def test_checkpoint_raises_once_cancelled():
    store = RunEventStore.get("chk-cancel")

    async def go():
        await store.cancel()
        await store.checkpoint()

    with pytest.raises(RunCancelled):
        asyncio.run(go())


def test_checkpoint_blocks_while_paused_and_resumes():
    store = RunEventStore.get("chk-pause")

    async def go():
        await store.pause()
        waiter = asyncio.create_task(store.checkpoint())
        await asyncio.sleep(0.05)
        assert not waiter.done(), "checkpoint returned while paused"
        await store.resume()
        await asyncio.wait_for(waiter, timeout=2)

    asyncio.run(go())


def test_cancel_releases_a_paused_run():
    """Cancelling while paused must not leave the loop waiting forever."""
    store = RunEventStore.get("chk-both")

    async def go():
        await store.pause()
        waiter = asyncio.create_task(store.checkpoint())
        await asyncio.sleep(0.05)
        await store.cancel()
        with pytest.raises(RunCancelled):
            await asyncio.wait_for(waiter, timeout=2)

    asyncio.run(go())


# ── pipeline control flow ────────────────────────────────────────────────────


def test_cancel_stops_the_loop_before_max_iterations(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_mod, "LLMClient", FakeLLM)
    store = RunEventStore.get("pipe-cancel")

    async def go():
        pipeline = pipeline_mod.Pipeline(
            _config(tmp_path, max_iterations=10),
            control_check=store.checkpoint,
            should_cancel=lambda: store.cancelled,
        )
        started = asyncio.create_task(self_soon(store))
        events = []
        try:
            async for event in pipeline.run():
                events.append(event)
        except RunCancelled:
            pass
        await started
        return events

    async def self_soon(s):
        await asyncio.sleep(0.05)
        await s.cancel()

    events = asyncio.run(go())
    assert events, "pipeline produced no events before cancelling"
    assert EventType.RUN_COMPLETE not in [e.type for e in events]
    assert [e.type for e in events][0] == EventType.RUN_STARTED


def test_pause_stops_the_loop_from_starting_new_iterations(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_mod, "LLMClient", FakeLLM)
    store = RunEventStore.get("pipe-pause")

    async def go():
        pipeline = pipeline_mod.Pipeline(
            _config(tmp_path, max_iterations=3),
            control_check=store.checkpoint,
            should_cancel=lambda: store.cancelled,
        )
        await store.pause()
        running = asyncio.create_task(self_soon(store))
        events = []
        with pytest.raises(RunCancelled):
            async for event in pipeline.run():
                events.append(event)
        await running
        return events

    async def self_soon(s):
        # Cancelled after a moment: the loop must still be sitting in the pause
        # point rather than having run an iteration.
        await asyncio.sleep(0.05)
        await s.cancel()

    events = asyncio.run(go())
    types = [e.type for e in events]
    # RUN_STARTED is emitted before the loop; everything else must have waited.
    assert EventType.ITERATION_STARTED not in types
    assert EventType.RESEARCHER_STARTED not in types


def test_cancel_mid_evaluation_stops_before_the_analyzer(tmp_path, monkeypatch):
    """A cancelled evaluation must not be followed by a paid-for Analyzer call."""
    monkeypatch.setattr(pipeline_mod, "LLMClient", FakeLLM)
    state = {"cancel": False}

    slow_evaluator = "import time\ntime.sleep(30)\n"

    config = RunConfig(
        run_id="mid-cancel",
        task_description="Optimise a trivial function",
        model="fake/model",
        max_iterations=3,
        eval_code=slow_evaluator,
        eval_timeout=60,
        data_dir=str(tmp_path),
    )

    async def no_op():
        return None

    async def flip_soon():
        await asyncio.sleep(0.5)
        state["cancel"] = True

    async def go():
        pipeline = pipeline_mod.Pipeline(
            config,
            control_check=no_op,
            should_cancel=lambda: state["cancel"],
        )
        flipper = asyncio.create_task(flip_soon())
        events = []
        with pytest.raises(RunCancelled):
            async for event in pipeline.run():
                events.append(event)
        await flipper
        return events

    events = asyncio.run(go())
    types = [e.type for e in events]
    assert EventType.ANALYZER_STARTED not in types
    assert EventType.ANALYZER_COMPLETE not in types
    assert EventType.RUN_COMPLETE not in types
