"""RunEventStore: persistence, replay, and no duplicate delivery."""

import asyncio
import json

from services.api.store import RunEventStore, events_path


def test_events_are_written_to_disk_and_replayable(tmp_path):
    path = events_path("run-persist", tmp_path)

    async def go():
        store = RunEventStore.get("run-persist", persist_path=path)
        await store.publish({"type": "run_started", "run_id": "run-persist"})
        await store.publish({"type": "iteration_complete", "run_id": "run-persist"})

    asyncio.run(go())

    assert path.exists()
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]
    assert [e["type"] for e in lines] == ["run_started", "iteration_complete"]

    # The in-memory buffer is gone with the run; a reconnect reads the file.
    RunEventStore.remove("run-persist")
    store = RunEventStore.get("run-persist", persist_path=path)

    async def replay():
        return [e async for e in store.replay()]

    events = asyncio.run(replay())
    assert [e["type"] for e in events] == ["run_started", "iteration_complete"]


def test_replay_of_an_unknown_run_yields_nothing(tmp_path):
    store = RunEventStore.get("run-empty", persist_path=events_path("run-empty", tmp_path))

    async def replay():
        return [e async for e in store.replay()]

    assert asyncio.run(replay()) == []


def test_subscriber_sees_each_event_exactly_once(tmp_path):
    """Events published during the replay snapshot must not arrive twice."""
    store = RunEventStore.get("run-dup", persist_path=events_path("run-dup", tmp_path))

    async def go():
        await store.publish({"type": "a"})
        await store.publish({"type": "b"})

        seen = []

        async def collect():
            async for event in store.subscribe():
                if event.get("type") != "keepalive":
                    seen.append(event["type"])
                if len(seen) >= 4:
                    break

        consumer = asyncio.create_task(collect())
        # Published while the subscriber is mid-replay.
        await store.publish({"type": "c"})
        await store.publish({"type": "d"})
        await asyncio.wait_for(consumer, timeout=5)
        return seen

    seen = asyncio.run(go())
    assert seen == ["a", "b", "c", "d"]


def test_every_event_carries_a_monotonic_sequence(tmp_path):
    store = RunEventStore.get("run-seq", persist_path=events_path("run-seq", tmp_path))

    async def go():
        for i in range(5):
            await store.publish({"type": f"e{i}"})
        return list(store._events)

    events = asyncio.run(go())
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)
