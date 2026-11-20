"""
Research pipeline — the core of R U Socrates.

Forked and rewritten from ASI-Evolve pipeline/main.py with these changes:
1. Async generator interface: `run()` yields PipelineEvent at each step
2. No file-based config: receives RunConfig directly
3. No wandb / custom logger: uses stdlib logging
4. Windows-compatible subprocess (via engineer.py)
5. LiteLLM via LLMClient (llm.py)

The pipeline is a Python package imported directly by FastAPI —
no queue, no separate process, no Docker required.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import asyncio
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Any, Callable, Awaitable

from .models import (
    Node,
    CognitionItem,
    PipelineEvent,
    EventType,
    RunConfig,
    RunCancelled,
)
from .llm import LLMClient, LLMResponse, ProviderConfig
from .memory import NodeDatabase, CognitionStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates (inline, no Jinja2 dependency for now)
# ---------------------------------------------------------------------------

RESEARCHER_SYSTEM = """You are a research scientist tasked with evolving computational methods.
Given a task description, prior candidate solutions (nodes), and cognition items (lessons learned),
generate an improved candidate.

Respond with XML tags:
<name>short identifier for this candidate</name>
<motivation>why you believe this approach will improve over prior work</motivation>
<code>the complete Python implementation</code>
"""

RESEARCHER_PROMPT = """Task: {task_description}

Prior candidates (sampled by exploration-exploitation):
{context_nodes}

Relevant insights from previous experiments:
{cognition_items}

{base_code_section}

Generate the next improved candidate. Be specific about what you are changing and why."""


ANALYZER_SYSTEM = """You are a scientific analyst. Given experiment results, extract reusable lessons.

Respond with XML tags:
<analysis>
A 2-4 paragraph analysis explaining:
1. What this result reveals about the approach
2. Why it succeeded or failed relative to prior work  
3. What principle or insight this demonstrates for future exploration
4. A Socratic reflection: what question does this raise?
</analysis>
"""

ANALYZER_PROMPT = """Task: {task_description}

Code evaluated:
```python
{code}
```

Evaluation results:
{results}

Best prior candidate for comparison:
{best_node_info}

Analyze this experiment result and extract reusable insights."""


# ---------------------------------------------------------------------------
# Researcher
# ---------------------------------------------------------------------------

class Researcher:
    def __init__(self, llm: LLMClient, config: Optional[Dict] = None):
        self.llm = llm
        self.config = config or {}
        self.diff_based = self.config.get("diff_based_evolution", True)
        self.max_code_length = self.config.get("max_code_length", 10000)

    def run(
        self,
        task_description: str,
        context_nodes: List[Node],
        cognition_items: List[CognitionItem],
        base_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        logger.info(
            f"[Researcher] {len(context_nodes)} context nodes, "
            f"{'diff' if (self.diff_based and base_code) else 'full_rewrite'} mode"
        )

        context_text = self._format_nodes(context_nodes)
        cognition_text = self._format_cognition(cognition_items)

        if self.diff_based and base_code:
            base_section = f"Current best candidate (apply targeted improvements):\n```python\n{base_code[:3000]}\n```"
        else:
            base_section = "No prior candidate — generate a complete implementation from scratch."

        prompt = RESEARCHER_PROMPT.format(
            task_description=task_description,
            context_nodes=context_text,
            cognition_items=cognition_text,
            base_code_section=base_section,
        )

        try:
            result = self.llm.extract_tags(
                prompt,
                system_prompt=RESEARCHER_SYSTEM,
                call_name="researcher",
            )
        except ValueError:
            logger.warning("[Researcher] Tag extraction failed, attempting raw response fallback")
            response = self.llm.generate(prompt, system_prompt=RESEARCHER_SYSTEM, call_name="researcher_fallback")
            extracted_code = self._extract_code_block(response.content)
            result = {
                "name": f"candidate_{datetime.now().strftime('%H%M%S')}",
                # Preserve a meaningful motivation snippet from the raw LLM response
                # so the frontend can display something useful instead of a placeholder.
                "motivation": response.content[:500] if not extracted_code else response.content[:200],
                "code": extracted_code,
            }

        code = result.get("code", "")
        if len(code) > self.max_code_length:
            logger.warning(f"[Researcher] Code truncated: {len(code)} → {self.max_code_length} chars")
            code = code[:self.max_code_length]

        return {
            "name": result.get("name", "unnamed"),
            "motivation": result.get("motivation", ""),
            "code": code,
        }

    def _format_nodes(self, nodes: List[Node]) -> str:
        if not nodes:
            return "(none — this is the first iteration)"
        parts = []
        for i, n in enumerate(nodes, 1):
            parts.append(
                f"[{i}] {n.name} (score={n.score:.4f})\n"
                f"Motivation: {n.motivation[:200]}\n"
                f"Analysis: {(n.analysis or 'none')[:300]}"
            )
        return "\n\n".join(parts)

    def _format_cognition(self, items: List[CognitionItem]) -> str:
        if not items:
            return "(none)"
        return "\n".join(f"• {item.content[:300]}" for item in items)

    def _extract_code_block(self, text: str) -> str:
        match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"```\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()


# ---------------------------------------------------------------------------
# Engineer
# ---------------------------------------------------------------------------

class Engineer:
    """Execute and evaluate a candidate program."""

    def run(
        self,
        code: str,
        work_dir: Path,
        eval_script: Optional[str] = None,
        eval_code: Optional[str] = None,
        timeout: int = 300,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        work_dir.mkdir(parents=True, exist_ok=True)
        code_file = work_dir / "code.py"
        code_file.write_text(code, encoding="utf-8")

        if not eval_script and not eval_code:
            logger.info("[Engineer] No eval_script or eval_code provided, skipping evaluation")
            return {
                "eval_score": 0.0,
                "success": True,
                "runtime": 0.0,
                "stdout": "",
                "stderr": "",
                "skipped": True,
            }

        # If eval_code is provided, write it to a file in work_dir
        evaluator_path: Optional[Path] = None
        if eval_code:
            evaluator_path = work_dir / "evaluator.py"
            evaluator_path.write_text(eval_code, encoding="utf-8")
            eval_script = str(evaluator_path)

        # Resolve command cross-platform (ADR-002 / Phase 0 fix)
        cmd = self._resolve_cmd(eval_script, work_dir)
        cmd = self._with_resource_limits(cmd, timeout)
        logger.info(f"[Engineer] Running: {' '.join(cmd)}")

        start = time.time()
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=(platform.system() != "Windows"),
            )
        except OSError as exc:
            return {
                "eval_score": 0.0,
                "success": False,
                "runtime": 0.0,
                "error": f"Could not start evaluator: {exc}",
                "stdout": "",
                "stderr": "",
            }

        try:
            stdout, stderr = self._wait_for(proc, timeout, should_stop)
        except RunCancelled:
            logger.info("[Engineer] Cancelled while evaluating")
            self._kill_process_tree(proc)
            return {
                "eval_score": 0.0,
                "success": False,
                "runtime": time.time() - start,
                "error": "Run cancelled",
                "cancelled": True,
                "stdout": "",
                "stderr": "",
            }
        except subprocess.TimeoutExpired:
            logger.warning(f"[Engineer] Timeout after {timeout}s")
            self._kill_process_tree(proc)
            return {
                "eval_score": 0.0,
                "success": False,
                "runtime": timeout,
                "error": f"Timeout after {timeout}s",
                "stdout": "",
                "stderr": "",
            }
        except Exception:
            # Anything else would otherwise leave the evaluator running.
            self._kill_process_tree(proc)
            raise

        runtime = time.time() - start
        if proc.returncode != 0:
            return {
                "eval_score": 0.0,
                "success": False,
                "runtime": runtime,
                "stdout": stdout[:2000],
                "stderr": stderr[:2000],
                "error": f"Process exited with code {proc.returncode}",
            }

        # Parse results.json written by the evaluator
        results = self._parse_results(work_dir)
        results.setdefault("eval_score", 0.0)

        # results.json carries the *candidate's* stdout, stderr and runtime.
        # The evaluator's own console output is diagnostic only — letting it
        # overwrite those fields silently discarded everything the candidate
        # printed.
        results.setdefault("stdout", "")
        results.setdefault("stderr", "")
        results.setdefault("runtime", runtime)
        # The evaluator decides whether the *candidate* succeeded. Only
        # assume success when it says nothing about it — overwriting its
        # verdict here reported a crashed candidate as a clean run.
        results.setdefault("success", True)
        results["evaluator_stdout"] = stdout[:2000]
        results["evaluator_stderr"] = stderr[:2000]
        return results

    @staticmethod
    def _wait_for(
        proc: "subprocess.Popen",
        timeout: int,
        should_stop: Optional[Callable[[], bool]] = None,
        poll: float = 0.25,
    ) -> tuple:
        """Wait for the evaluator without getting stuck behind it.

        ``communicate(timeout=…)`` blocks for the whole timeout, so a 300s
        evaluation used to swallow every cancel request. Polling in short
        slices keeps the control loop responsive.

        Note: repeated communicate() is safe on POSIX; on Windows the pipes can
        wedge, which is why the poll interval is short and the whole call is
        bounded by the outer timeout.
        """
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(proc.args, timeout)
            try:
                return proc.communicate(timeout=min(poll, remaining))
            except subprocess.TimeoutExpired:
                if should_stop and should_stop():
                    raise RunCancelled("cancelled while evaluating")

    @staticmethod
    def _kill_process_tree(proc: "subprocess.Popen") -> None:
        """Kill the evaluator and everything it spawned.

        On POSIX the evaluator runs in its own session, so ``proc.kill()``
        only reaches the launcher — the candidate program survives as an
        orphan. Kill the whole process group instead.
        """
        try:
            if platform.system() != "Windows":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass

    def _resolve_cmd(self, eval_script: str, work_dir: Path) -> List[str]:
        """Build the command that runs an evaluator.

        Every evaluator — shell script or Python — gets the same argv:
            <code_file> <results_json>

        The interpreter is picked from the file itself (extension, then
        shebang), not from the host OS, so a .py evaluator runs under Python
        on every platform. Previously this unconditionally invoked ``bash``,
        which silently failed for Python evaluators on macOS/Linux.
        """
        eval_p = Path(eval_script)
        argv = [str(work_dir / "code.py"), str(work_dir / "results.json")]

        if eval_p.suffix.lower() == ".py":
            return [sys.executable, str(eval_p), *argv]

        shebang = self._read_shebang(eval_p)
        if shebang:
            exe = shutil.which(shebang[0]) or shebang[0]
            return [exe, *shebang[1:], str(eval_p), *argv]

        for shell in ("bash", "sh"):
            if shutil.which(shell):
                return [shell, str(eval_p), *argv]

        raise FileNotFoundError(
            f"No interpreter available for evaluator {eval_p}. "
            f"Install bash, or use a .py evaluator."
        )

    # Address-space ceiling for a candidate run, in MB. 0 disables the limit.
    MEMORY_LIMIT_MB = 4096
    # Largest file a candidate may write, in 512-byte blocks (32 MB).
    FILE_BLOCKS_LIMIT = 65536

    def _with_resource_limits(self, cmd: List[str], timeout: int) -> List[str]:
        """Cap CPU time, address space and file size for the evaluator.

        Damage limitation, not a sandbox: it stops a runaway candidate from
        eating the machine, but it cannot stop one from touching the file system
        or the network. Untrusted tasks still need a container.

        Limits go through a shell wrapper rather than ``preexec_fn``, which
        risks deadlock in a multithreaded process.
        """
        if platform.system() == "Windows":
            return cmd

        cpu = max(1, int(timeout))
        # bash's ulimit takes one resource per invocation, hence the sequence.
        calls = [
            f"ulimit -t {cpu}",
            f"ulimit -f {self.FILE_BLOCKS_LIMIT}",
        ]
        if self.MEMORY_LIMIT_MB:
            calls.append(f"ulimit -v {self.MEMORY_LIMIT_MB * 1024}")
        script = " 2>/dev/null; ".join(calls) + ' 2>/dev/null; exec "$@"'
        return ["bash", "-c", script, "evaluator", *cmd]

    @staticmethod
    def _read_shebang(path: Path) -> List[str]:
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                first = f.readline()
        except OSError:
            return []
        if not first.startswith("#!"):
            return []
        return first[2:].strip().split()

    def _parse_results(self, work_dir: Path) -> Dict[str, Any]:
        results_file = work_dir / "results.json"
        if results_file.exists():
            try:
                with open(results_file, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception as e:
                logger.warning(f"[Engineer] Failed to parse results.json: {e}")
        return {}


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class Analyzer:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(
        self,
        code: str,
        results: Dict[str, Any],
        task_description: str,
        best_node: Optional[Node] = None,
    ) -> str:
        results_str = json.dumps(
            {k: v for k, v in results.items() if k not in ("stdout", "stderr")},
            indent=2,
            ensure_ascii=False,
        )

        if best_node:
            best_info = (
                f"Name: {best_node.name}, Score: {best_node.score:.4f}\n"
                f"Motivation: {best_node.motivation[:300]}"
            )
        else:
            best_info = "(no prior best — first iteration)"

        prompt = ANALYZER_PROMPT.format(
            task_description=task_description,
            code=code[:2000],
            results=results_str,
            best_node_info=best_info,
        )

        try:
            result = self.llm.extract_tags(
                prompt,
                system_prompt=ANALYZER_SYSTEM,
                call_name="analyzer",
            )
            return result.get("analysis", "")
        except ValueError:
            logger.warning("[Analyzer] Tag extraction failed, using raw response as analysis")
            response = self.llm.generate(
                prompt, system_prompt=ANALYZER_SYSTEM, call_name="analyzer_fallback"
            )
            # Return full raw content (no truncation) so the frontend can render it
            return response.content


# ---------------------------------------------------------------------------
# Pipeline — the async generator that powers SSE
# ---------------------------------------------------------------------------

class Pipeline:
    """
    Orchestrate the research loop and yield PipelineEvent for every step.

    Usage (from FastAPI):

        async for event in Pipeline(config).run():
            yield f"data: {json.dumps(event.to_sse_dict())}\\n\\n"
    """

    def __init__(
        self,
        config: RunConfig,
        control_check: Optional[Callable[[], Awaitable[None]]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ):
        """
        control_check — awaited at every iteration boundary. Blocks while the
                        run is paused and raises RunCancelled to unwind.
        should_cancel — cheap synchronous probe polled while an evaluation is
                        running, so a cancel does not have to wait for it.
        """
        self.config = config
        self.control_check = control_check
        self.should_cancel = should_cancel

        # Data dirs
        data_root = Path(config.data_dir) / "runs" / config.run_id
        data_root.mkdir(parents=True, exist_ok=True)

        self.work_root = data_root / "steps"
        self.work_root.mkdir(parents=True, exist_ok=True)

        # Memory
        self.db = NodeDatabase(
            storage_dir=data_root / "nodes",
            embedding_model=config.embedding_model,
        )
        self.cognition = CognitionStore(
            storage_dir=data_root / "cognition",
            embedding_model=config.embedding_model,
        )

        # LLM - build provider configs
        if config.models:
            providers = [ProviderConfig(model=m) for m in config.models]
        else:
            providers = [ProviderConfig(model=config.model)]
        
        self.llm = LLMClient(
            providers=providers,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        
        # Track previous provider for fallback events
        self._previous_provider = None
        self._previous_model = None

        # Agents
        self.researcher = Researcher(
            self.llm,
            config={"diff_based_evolution": config.diff_based_evolution, "max_code_length": config.max_code_length},
        )
        self.engineer = Engineer()
        self.analyzer = Analyzer(self.llm)

    async def run(self) -> AsyncGenerator[PipelineEvent, None]:
        """
        Async generator: yields PipelineEvent for every meaningful step.

        Each event is serialisable and forwarded directly to the SSE stream.
        The frontend receives these in real time — this is how the research
        process is made visible to the user.
        """
        cfg = self.config
        run_id = cfg.run_id

        # Get initial provider info
        initial_provider = self.llm.current_provider_name
        initial_model = self.llm.current_model
        
        yield PipelineEvent(
            type=EventType.RUN_STARTED,
            run_id=run_id,
            message=f"Research run started. Task: {cfg.task_description[:120]}",
            current_provider=initial_provider,
            current_model=initial_model,
        )

        best_score = 0.0

        # A failed attempt must not silently shrink the run: 10 requested
        # iterations should still produce 10 completed ones. Extra attempts
        # give the Researcher room to recover.
        completed = 0
        attempt = 0
        max_attempts = cfg.max_iterations * 3

        while completed < cfg.max_iterations and attempt < max_attempts:
            attempt += 1
            iteration = attempt
            # Pause point — also raises RunCancelled if the user cancelled.
            if self.control_check:
                await self.control_check()
            yield PipelineEvent(
                type=EventType.ITERATION_STARTED,
                run_id=run_id,
                iteration=iteration,
                message=f"Starting iteration {iteration}/{cfg.max_iterations}",
                current_provider=self.llm.current_provider_name,
                current_model=self.llm.current_model,
            )

            # --- Memory: sample prior nodes ---
            context_nodes = self.db.sample(cfg.sample_n)
            yield PipelineEvent(
                type=EventType.MEMORY_SAMPLED,
                run_id=run_id,
                iteration=iteration,
                total_nodes=len(self.db),
                message=f"Sampled {len(context_nodes)} context nodes from {len(self.db)} total",
                current_provider=self.llm.current_provider_name,
                current_model=self.llm.current_model,
            )

            # --- Memory: retrieve cognition items ---
            cognition_items: List[CognitionItem] = []
            for node in context_nodes:
                query = node.analysis or node.motivation
                if query:
                    items = self.cognition.search(query, top_k=2)
                    cognition_items.extend(items)

            yield PipelineEvent(
                type=EventType.COGNITION_RETRIEVED,
                run_id=run_id,
                iteration=iteration,
                message=f"Retrieved {len(cognition_items)} cognition items",
                current_provider=self.llm.current_provider_name,
                current_model=self.llm.current_model,
            )

            # --- Researcher ---
            yield PipelineEvent(
                type=EventType.RESEARCHER_STARTED,
                run_id=run_id,
                iteration=iteration,
                agent_type="researcher",
                message="Researcher generating next candidate…",
                current_provider=self.llm.current_provider_name,
                current_model=self.llm.current_model,
            )

            base_code = context_nodes[0].code if (cfg.diff_based_evolution and context_nodes) else None

            try:
                researcher_result = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: self.researcher.run(
                        task_description=cfg.task_description,
                        context_nodes=context_nodes,
                        cognition_items=cognition_items,
                        base_code=base_code,
                    ),
                )
                # Check if provider switched
                if self.llm.switched_provider:
                    previous_provider = self.llm.previous_provider
                    previous_model = previous_provider.model if previous_provider else ""
                    previous_provider_name = previous_provider.model.split("/")[0] if (previous_provider and "/" in previous_provider.model) else (previous_provider.model if previous_provider else "")
                    yield PipelineEvent(
                        type=EventType.PROVIDER_SWITCHED,
                        run_id=run_id,
                        iteration=iteration,
                        message=f"Switched LLM provider to {self.llm.current_provider_name}",
                        current_provider=self.llm.current_provider_name,
                        current_model=self.llm.current_model,
                        previous_provider=previous_provider_name,
                        previous_model=previous_model,
                    )
            except Exception as exc:
                logger.error(f"[Pipeline] Researcher failed at iteration {iteration}: {exc}")
                yield PipelineEvent(
                    type=EventType.RESEARCHER_FAILED,
                    run_id=run_id,
                    iteration=iteration,
                    agent_type="researcher",
                    message=f"Researcher error: {type(exc).__name__}: {exc}",
                    current_provider=self.llm.current_provider_name,
                    current_model=self.llm.current_model,
                )
                continue

            node = Node(
                name=researcher_result["name"],
                motivation=researcher_result["motivation"],
                code=researcher_result["code"],
                parent=[n.id for n in context_nodes if n.id is not None],
            )

            yield PipelineEvent(
                type=EventType.RESEARCHER_COMPLETE,
                run_id=run_id,
                iteration=iteration,
                agent_type="researcher",
                node_name=node.name,
                node_motivation=node.motivation,
                node_code_preview=node.code[:300],
                message=f"Researcher generated: {node.name}",
                current_provider=self.llm.current_provider_name,
                current_model=self.llm.current_model,
            )

            # --- Engineer ---
            work_dir = self.work_root / f"iter_{iteration:03d}"

            yield PipelineEvent(
                type=EventType.ENGINEER_STARTED,
                run_id=run_id,
                iteration=iteration,
                agent_type="engineer",
                message="Engineer evaluating candidate…",
                current_provider=self.llm.current_provider_name,
                current_model=self.llm.current_model,
            )

            try:
                engineer_result = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: self.engineer.run(
                        code=node.code,
                        work_dir=work_dir,
                        eval_script=cfg.eval_script,
                        eval_code=cfg.eval_code,
                        timeout=cfg.eval_timeout,
                        should_stop=self.should_cancel,
                    ),
                )
            except Exception as exc:
                logger.error(f"[Pipeline] Engineer failed at iteration {iteration}: {exc}")
                engineer_result = {
                    "eval_score": 0.0,
                    "success": False,
                    "error": str(exc),
                    "runtime": 0.0,
                }

            # A cancelled evaluation must stop the run here. Falling through
            # would still spend an Analyzer call on a result nobody wants.
            if engineer_result.get("cancelled"):
                raise RunCancelled(f"run {run_id} cancelled during iteration {iteration}")

            node.score = engineer_result.get("eval_score", 0.0)
            node.results = {k: v for k, v in engineer_result.items() if k not in ("stdout", "stderr")}
            node.meta_info = {
                "success": engineer_result.get("success", False),
                "runtime": engineer_result.get("runtime", 0.0),
                "error": engineer_result.get("error"),
            }

            yield PipelineEvent(
                type=EventType.ENGINEER_COMPLETE if engineer_result.get("success", False) else EventType.ENGINEER_FAILED,
                run_id=run_id,
                iteration=iteration,
                agent_type="engineer",
                eval_score=node.score,
                eval_success=engineer_result.get("success", False),
                eval_runtime=engineer_result.get("runtime", 0.0),
                eval_stdout_preview=engineer_result.get("stdout", "")[:300],
                message=f"Engineer: score={node.score:.4f}, success={engineer_result.get('success', False)}",
                current_provider=self.llm.current_provider_name,
                current_model=self.llm.current_model,
            )

            # --- Analyzer ---
            yield PipelineEvent(
                type=EventType.ANALYZER_STARTED,
                run_id=run_id,
                iteration=iteration,
                agent_type="analyzer",
                message="Analyzer interpreting results…",
                current_provider=self.llm.current_provider_name,
                current_model=self.llm.current_model,
            )

            best_node_for_comparison = self.db.get_best()

            try:
                analysis = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: self.analyzer.run(
                        code=node.code,
                        results=engineer_result,
                        task_description=cfg.task_description,
                        best_node=best_node_for_comparison,
                    ),
                )
                # Check if provider switched
                if self.llm.switched_provider:
                    previous_provider = self.llm.previous_provider
                    previous_model = previous_provider.model if previous_provider else ""
                    previous_provider_name = previous_provider.model.split("/")[0] if (previous_provider and "/" in previous_provider.model) else (previous_provider.model if previous_provider else "")
                    yield PipelineEvent(
                        type=EventType.PROVIDER_SWITCHED,
                        run_id=run_id,
                        iteration=iteration,
                        message=f"Switched LLM provider to {self.llm.current_provider_name}",
                        current_provider=self.llm.current_provider_name,
                        current_model=self.llm.current_model,
                        previous_provider=previous_provider_name,
                        previous_model=previous_model,
                    )
            except Exception as exc:
                logger.error(f"[Pipeline] Analyzer failed at iteration {iteration}: {exc}")
                analysis = f"Analysis failed: {exc}"

            node.analysis = analysis

            yield PipelineEvent(
                type=EventType.ANALYZER_COMPLETE,
                run_id=run_id,
                iteration=iteration,
                agent_type="analyzer",
                analysis=analysis,
                message="Analyzer complete",
                current_provider=self.llm.current_provider_name,
                current_model=self.llm.current_model,
            )

            # --- Persist node ---
            node_id = self.db.add(node)

            # Distill analysis into cognition
            if analysis and len(analysis) > 50:
                self.cognition.add(
                    CognitionItem(content=analysis, source=f"run:{run_id}/iter:{iteration}")
                )

            # Update best score
            if node.score > best_score:
                best_score = node.score

            best_node = self.db.get_best()

            completed += 1
            yield PipelineEvent(
                type=EventType.ITERATION_COMPLETE,
                run_id=run_id,
                iteration=iteration,
                best_score=best_score,
                best_node_id=best_node.id if best_node else None,
                total_nodes=len(self.db),
                # Include node details so the API layer can persist ExploredNode
                node_name=node.name,
                node_motivation=node.motivation,
                node_code_preview=node.code[:300],
                analysis=analysis,
                eval_score=node.score,
                eval_success=engineer_result.get("success", False),
                message=f"Iteration {iteration} complete. Best score so far: {best_score:.4f}",
                current_provider=self.llm.current_provider_name,
                current_model=self.llm.current_model,
            )

        # --- Run complete ---
        if completed == 0:
            # Every attempt failed. Reporting this as a successful run would be
            # a lie the UI happily repeats.
            yield PipelineEvent(
                type=EventType.RUN_FAILED,
                run_id=run_id,
                best_score=best_score,
                total_nodes=len(self.db),
                stats={
                    "iterations": 0,
                    "attempts": attempt,
                    "total_nodes": len(self.db),
                    "best_score": best_score,
                    "model": cfg.model,
                },
                message=(
                    f"Run failed: no iteration completed after {attempt} attempts. "
                    f"Check the LLM configuration and the evaluator."
                ),
                current_provider=self.llm.current_provider_name,
                current_model=self.llm.current_model,
            )
            return

        best_node = self.db.get_best()
        yield PipelineEvent(
            type=EventType.RUN_COMPLETE,
            run_id=run_id,
            best_score=best_score,
            total_nodes=len(self.db),
            best_node=best_node.to_dict() if best_node else None,
            stats={
                "iterations": completed,
                "attempts": attempt,
                "total_nodes": len(self.db),
                "best_score": best_score,
                "model": cfg.model,
            },
            message=f"Research run complete. Best score: {best_score:.4f}",
            current_provider=self.llm.current_provider_name,
            current_model=self.llm.current_model,
        )
