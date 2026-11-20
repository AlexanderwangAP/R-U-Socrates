"""
Tasks router — CRUD + SSE stream endpoint.

Endpoints:
    POST   /api/tasks                      Create task and immediately start a run
    GET    /api/tasks                      List all tasks
    GET    /api/tasks/{task_id}            Get task detail
    DELETE /api/tasks/{task_id}            Delete a task and all its runs
    POST   /api/tasks/{task_id}/cancel     Cancel the active run for a task
    GET    /api/tasks/{task_id}/runs       List runs for a task
    GET    /api/tasks/{task_id}/stream     SSE: stream live research events for the latest run
"""

from __future__ import annotations

import json
import logging
import os
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..database import get_db, SessionLocal
from ..models import ExploredNode, Result, Run, Task
from ..schemas import TaskCreate, TaskResponse, RunResponse
from ..store import RunEventStore, events_path

# ---------------------------------------------------------------------------
# Worker package — imported as `services.worker`, same identity as the rest of
# the backend. Adding `services/` to sys.path and importing `worker` as a
# top-level module used to load a *second* copy, which made exception types
# from one copy unrecognisable to the other.
# ---------------------------------------------------------------------------
from ...worker import Pipeline, RunConfig, RunCancelled, EventType  # type: ignore[import]

# Location of the bundled default evaluator.
_WORKER_ROOT = Path(__file__).resolve().parent.parent.parent / "worker"

# Where run artefacts (nodes, cognition, event logs) live. Shared with the
# pipeline via RunConfig.data_dir so both sides agree on the path.
DATA_DIR = Path(os.getenv("RUS_DATA_DIR", "./data"))

router = APIRouter(prefix="/api/tasks", tags=["tasks"])
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Create task + start run
# ---------------------------------------------------------------------------

@router.post("", response_model=TaskResponse, status_code=201)
async def create_task(
    payload: TaskCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Create a research task and immediately queue a run."""
    task_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())

    task = Task(
        id=task_id,
        name=payload.name,
        description=payload.description,
        status="running",
        model=payload.model,
        max_iterations=payload.max_iterations,
    )
    run = Run(id=run_id, task_id=task_id, status="running")

    db.add(task)
    db.add(run)
    db.commit()
    db.refresh(task)

    # Launch pipeline in background (non-blocking)
    background_tasks.add_task(_run_pipeline, task_id, run_id, payload)

    return task.to_dict()


async def _run_pipeline(task_id: str, run_id: str, payload: TaskCreate):
    """Background task: run the research pipeline and persist events."""
    # Resolve evaluator.py relative to the worker package
    evaluator_path = str(_WORKER_ROOT / "evaluator.py")

    config = RunConfig(
        run_id=run_id,
        task_description=payload.description,
        models=payload.models,
        model=payload.model,
        max_iterations=payload.max_iterations,
        eval_script=evaluator_path if not payload.eval_code else None,
        eval_code=payload.eval_code,
        data_dir=str(DATA_DIR),
    )

    # The event log is written alongside the run's other artefacts so it
    # survives the process and can be replayed after the run ends.
    store = RunEventStore.get(run_id, persist_path=events_path(run_id, DATA_DIR))

    # Use a single long-lived DB session for the entire run (avoids per-persist
    # connection churn; safe because background task runs in a single thread).
    db = SessionLocal()
    try:
        pipeline = Pipeline(
            config,
            control_check=store.checkpoint,
            should_cancel=lambda: store.cancelled,
        )
        async for event in pipeline.run():
            event_dict = event.to_sse_dict()
            await store.publish(event_dict)

            # Persist completed iterations to DB
            if event.type == EventType.ITERATION_COMPLETE:
                _persist_iteration(db, run_id, event)
            elif event.type == EventType.RUN_COMPLETE:
                _persist_run_complete(db, run_id, task_id, event)
            elif event.type == EventType.RUN_FAILED:
                _persist_run_failed(db, run_id, task_id, str(event.message))

    except RunCancelled:
        logger.info(f"Run {run_id} cancelled by user")
        _persist_run_cancelled(db, run_id, task_id)
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        await store.publish({
            "type": "run_failed",
            "run_id": run_id,
            "message": error_msg,
        })
        _persist_run_failed(db, run_id, task_id, error_msg)
    finally:
        db.close()
        await store.close()
        # Remove store from global registry to free memory
        RunEventStore.remove(run_id)


def _persist_iteration(db: Session, run_id: str, event) -> None:
    """Persist iteration summary and explored node to DB."""
    run = db.get(Run, run_id)
    if run:
        run.best_score = max(run.best_score, event.best_score)
        run.total_nodes = event.total_nodes
        run.total_iterations = event.iteration

    # Persist the explored node for this iteration (powers L2 Reasoning Tree)
    node = ExploredNode(
        run_id=run_id,
        node_idx=event.iteration,
        name=event.node_name,
        motivation=event.node_motivation,
        code=event.node_code_preview,   # preview only — full code in worker fs
        analysis=event.analysis,
        score=event.eval_score,
        eval_success=event.eval_success,
    )
    db.add(node)
    db.commit()


def _persist_run_complete(db: Session, run_id: str, task_id: str, event) -> None:
    """Persist final result and update run/task status."""
    run = db.get(Run, run_id)
    task = db.get(Task, task_id)

    if run:
        run.status = "completed"
        run.completed_at = datetime.utcnow()
        run.best_score = event.best_score
        run.total_nodes = event.total_nodes
        run.total_iterations = event.stats.get("iterations", 0) if event.stats else 0

    if task:
        task.status = "completed"
        task.updated_at = datetime.utcnow()

    # Persist result
    if event.best_node:
        bn = event.best_node
        result = Result(
            id=str(uuid.uuid4()),
            run_id=run_id,
            best_score=event.best_score,
            best_node_name=bn.get("name", ""),
            best_node_motivation=bn.get("motivation", ""),
            best_node_code=bn.get("code", ""),
            best_node_analysis=bn.get("analysis", ""),
            stats_json=json.dumps(event.stats or {}),
        )
        db.add(result)

    db.commit()


def _persist_run_failed(db: Session, run_id: str, task_id: str, error: str) -> None:
    run = db.get(Run, run_id)
    task = db.get(Task, task_id)
    if run:
        run.status = "failed"
        run.error_message = error[:2000]
        run.completed_at = datetime.utcnow()
    if task:
        task.status = "failed"
        task.updated_at = datetime.utcnow()
    db.commit()


def _persist_run_cancelled(db: Session, run_id: str, task_id: str) -> None:
    """A cancelled run is not a failed one — record it as its own outcome."""
    run = db.get(Run, run_id)
    task = db.get(Task, task_id)
    if run:
        run.status = "cancelled"
        run.completed_at = datetime.utcnow()
    if task:
        task.status = "cancelled"
        task.updated_at = datetime.utcnow()
    db.commit()


# ---------------------------------------------------------------------------
# List / get tasks
# ---------------------------------------------------------------------------

@router.get("", response_model=list[TaskResponse])
def list_tasks(db: Session = Depends(get_db)):
    tasks = db.query(Task).order_by(Task.created_at.desc()).all()
    return [t.to_dict() for t in tasks]


@router.get("/{task_id}", response_model=TaskResponse)
def get_task(task_id: str, db: Session = Depends(get_db)):
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task.to_dict()


@router.delete("/{task_id}", status_code=204)
def delete_task(task_id: str, db: Session = Depends(get_db)):
    """Delete a task and all associated runs/results (cascade)."""
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    db.delete(task)
    db.commit()


@router.post("/{task_id}/cancel", status_code=200)
async def cancel_task(task_id: str, db: Session = Depends(get_db)):
    """
    Cancel the active run for a task.

    Sets the cancellation flag the pipeline polls, so the loop unwinds at its
    next checkpoint — or within a fraction of a second if it is mid-evaluation.
    The background task records the run as cancelled once it stops.
    """
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status not in ("running", "paused"):
        raise HTTPException(
            status_code=409,
            detail=f"Task is not running or paused (current status: {task.status})",
        )

    run = (
        db.query(Run)
        .filter(Run.task_id == task_id, Run.status.in_(("running", "paused")))
        .order_by(Run.started_at.desc())
        .first()
    )
    store = RunEventStore.peek(run.id) if run else None
    if store is not None:
        await store.cancel()

    task.status = "cancelled"
    task.updated_at = datetime.utcnow()
    db.commit()

    return {"status": "cancelled", "task_id": task_id}


@router.post("/{task_id}/pause", status_code=200)
async def pause_task(task_id: str, db: Session = Depends(get_db)):
    """Pause the active run for a task.

    Takes effect at the next checkpoint: an evaluation already in flight runs to
    completion, then the loop waits.
    """
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != "running":
        raise HTTPException(
            status_code=409,
            detail=f"Task is not running (current status: {task.status})",
        )

    run = (
        db.query(Run)
        .filter(Run.task_id == task_id, Run.status == "running")
        .order_by(Run.started_at.desc())
        .first()
    )
    store = RunEventStore.peek(run.id) if run else None
    if store is not None:
        await store.pause()

    task.status = "paused"
    task.updated_at = datetime.utcnow()
    db.commit()

    return {"status": "paused", "task_id": task_id}


@router.post("/{task_id}/resume", status_code=200)
async def resume_task(task_id: str, db: Session = Depends(get_db)):
    """Resume the paused run for a task."""
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != "paused":
        raise HTTPException(
            status_code=409,
            detail=f"Task is not paused (current status: {task.status})",
        )

    run = (
        db.query(Run)
        .filter(Run.task_id == task_id, Run.status == "paused")
        .order_by(Run.started_at.desc())
        .first()
    )
    store = RunEventStore.peek(run.id) if run else None
    if store is not None:
        await store.resume()

    task.status = "running"
    task.updated_at = datetime.utcnow()
    db.commit()

    return {"status": "running", "task_id": task_id}


@router.get("/{task_id}/runs", response_model=list[RunResponse])
def list_runs(task_id: str, db: Session = Depends(get_db)):
    runs = db.query(Run).filter(Run.task_id == task_id).order_by(Run.started_at.desc()).all()
    return [r.to_dict() for r in runs]


# ---------------------------------------------------------------------------
# SSE stream — the transparency core
# ---------------------------------------------------------------------------

@router.get("/{task_id}/stream")
async def stream_task(task_id: str, db: Session = Depends(get_db)):
    """
    Server-Sent Events endpoint.

    Returns real-time PipelineEvent objects as they are emitted by the pipeline.
    Frontend connects with EventSource('/api/tasks/{id}/stream').

    This is the core transparency mechanism: the user watches the research
    loop unfold — Researcher → Engineer → Analyzer — in real time.
    """
    # Find the latest run for this task
    run = (
        db.query(Run)
        .filter(Run.task_id == task_id)
        .order_by(Run.started_at.desc())
        .first()
    )
    if not run:
        raise HTTPException(status_code=404, detail="No runs found for this task")

    run_id = run.id
    log_path = events_path(run_id, DATA_DIR)
    live_store = RunEventStore.peek(run_id)

    async def event_generator() -> AsyncGenerator[str, None]:
        # A finished run has no live store any more. Read the event log without
        # registering anything — get() here would leave an empty store behind.
        if run.status in ("completed", "failed", "cancelled"):
            reader = live_store or RunEventStore(run_id)
            reader.persist_to(log_path)
            async for event_dict in reader.replay():
                yield f"data: {json.dumps(event_dict)}\n\n"
            return

        # Stream live events
        store = live_store or RunEventStore.get(run_id, persist_path=log_path)
        async for event_dict in store.subscribe():
            yield f"data: {json.dumps(event_dict)}\n\n"
            if event_dict.get("type") in ("run_complete", "run_failed", "run_cancelled"):
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
