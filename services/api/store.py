"""
In-memory SSE event store for active runs.

Each run gets a RunEventStore that:
1. Buffers all events (for replay to late-joining clients)
2. Appends every event to disk so it survives the process
3. Provides an async subscription queue for live streaming
4. Is cleaned up after the run completes

This is intentionally simple: in-process asyncio queue.
No Redis, no external broker. Works perfectly for single-user local deployment.
If multi-user is needed later, swap this for Redis pub/sub.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from ..worker.models import RunCancelled


def events_path(run_id: str, data_dir: Path) -> Path:
    """Where a run's event log lives on disk."""
    return Path(data_dir) / "runs" / run_id / "events.jsonl"


class RunEventStore:
    """Per-run event buffer + subscription queue, mirrored to disk."""

    _stores: Dict[str, "RunEventStore"] = {}

    def __init__(self, run_id: str):
        self.run_id = run_id
        self._events: List[Dict[str, Any]] = []
        self._subscribers: List[asyncio.Queue] = []
        self._closed = False
        self._paused = False
        self._cancelled = False
        self._seq = 0
        self._persist_path: Optional[Path] = None
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Initially not paused

    # ── registry ──────────────────────────────────────────────────────────

    @classmethod
    def get(cls, run_id: str, persist_path: Optional[Path] = None) -> "RunEventStore":
        store = cls._stores.get(run_id)
        if store is None:
            store = cls(run_id)
            cls._stores[run_id] = store
        if persist_path is not None:
            store.persist_to(persist_path)
        return store

    @classmethod
    def peek(cls, run_id: str) -> Optional["RunEventStore"]:
        """Return the live store for a run, or None.

        Unlike get(), this never creates one — a finished run should not get a
        fresh empty store just because a route touched it.
        """
        return cls._stores.get(run_id)

    @classmethod
    def remove(cls, run_id: str) -> None:
        cls._stores.pop(run_id, None)

    def persist_to(self, path: Path) -> None:
        """Mirror every published event to this file."""
        self._persist_path = Path(path)
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)

    # ── publish / subscribe ───────────────────────────────────────────────

    async def publish(self, event: Dict[str, Any]) -> None:
        """Publish an event to all subscribers, buffer it, and log it to disk."""
        self._seq += 1
        stamped = {**event, "seq": self._seq}
        self._events.append(stamped)
        self._append_to_disk(stamped)

        for q in list(self._subscribers):
            try:
                q.put_nowait(stamped)
            except asyncio.QueueFull:
                pass

    def _append_to_disk(self, event: Dict[str, Any]) -> None:
        if self._persist_path is None:
            return
        try:
            with open(self._persist_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError:
            # Losing the audit log is bad; losing the live run is worse.
            pass

    async def subscribe(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Subscribe to events. Replays history, then streams new ones.

        Each event carries a monotonic ``seq``; events seen in the replay are
        skipped when they turn up again on the queue, because the subscription
        is registered before the snapshot is taken and anything published in
        between lands in both.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=512)
        self._subscribers.append(q)
        try:
            last_seq = 0
            for event in list(self._events):
                last_seq = max(last_seq, event.get("seq", 0))
                yield event

            while not self._closed:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    yield {"type": "keepalive"}
                    continue

                if event.get("seq", 0) <= last_seq:
                    continue
                last_seq = event.get("seq", 0)

                yield event
                if event.get("type") in ("run_complete", "run_failed", "run_cancelled"):
                    break
        finally:
            if q in self._subscribers:
                self._subscribers.remove(q)

    async def replay(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Replay events for a finished run, from memory or from disk.

        The in-memory buffer is dropped when the run ends, so a reconnect has
        to come from the event log on disk.
        """
        if self._events:
            for event in self._events:
                yield event
            return

        if self._persist_path is None or not self._persist_path.exists():
            return
        with open(self._persist_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    # ── control flow ──────────────────────────────────────────────────────

    def is_paused(self) -> bool:
        """Check if the run is currently paused."""
        return self._paused

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    async def pause(self) -> None:
        """Pause the run."""
        self._paused = True
        self._pause_event.clear()
        await self.publish({
            "type": "run_paused",
            "run_id": self.run_id,
            "message": "Run has been paused"
        })

    async def resume(self) -> None:
        """Resume the run."""
        self._paused = False
        self._pause_event.set()
        await self.publish({
            "type": "run_resumed",
            "run_id": self.run_id,
            "message": "Run has been resumed"
        })

    async def cancel(self) -> None:
        """Cancel the run.

        Unlike close(), this is the signal the pipeline actually watches for.
        It also releases a paused run so it can notice the cancellation instead
        of waiting forever.
        """
        self._cancelled = True
        self._paused = False
        self._pause_event.set()
        await self.publish({
            "type": "run_cancelled",
            "run_id": self.run_id,
            "message": "Run cancelled by user",
        })

    async def checkpoint(self) -> None:
        """Pause point for the pipeline.

        Blocks while the run is paused and raises RunCancelled once the user
        cancels, so the loop can unwind instead of burning tokens.
        """
        while self._paused and not self._cancelled:
            await self._pause_event.wait()
        if self._cancelled:
            raise RunCancelled(f"run {self.run_id} was cancelled")

    async def close(self) -> None:
        """Mark the store as closed and notify all subscribers."""
        self._closed = True
        self._pause_event.set()  # Unblock any waiting tasks
        for q in list(self._subscribers):
            try:
                q.put_nowait({"type": "stream_closed"})
            except asyncio.QueueFull:
                pass
