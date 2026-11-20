# R U Socrates

> **The transparent research engine.** See how AI thinks — every step, every decision.

R U Socrates makes the AI research process visible. Instead of getting a black-box answer, you watch a loop of specialized agents — Researcher, Engineer, Analyzer — iterating in real time. Every hypothesis, every evaluation, every insight is streamed to your screen.

---

## What It Does

Most AI tools hide their reasoning. R U Socrates doesn't.

You give it a research task (optimize an algorithm, explore an architecture, improve a system). It runs a multi-agent loop and lets you observe:

- **Researcher** — generates a new candidate solution based on prior attempts and lessons learned
- **Engineer** — executes and evaluates the candidate, measuring performance
- **Analyzer** — interprets results and distills reusable insights for the next iteration

Every step is streamed live. You see the code, the scores, the analysis — as it happens.

---

## Quick Start

### Prerequisites

- Node.js 18+ and Python 3.10+
- An LLM API key (OpenAI, Anthropic, Qwen, Ollama, or any LiteLLM-compatible provider)

All commands below run from the repository root.

### 1. Install and build the frontend

```bash
npm install --legacy-peer-deps
npm run build:types          # shared TypeScript types
cd apps/web && npm run build # produces the static export in apps/web/out
```

### 2. Install the backend

```bash
pip install -r services/requirements.txt
```

### 3. Configure your LLM

Copy the example environment file and fill in your keys:

```bash
cp services/.env.example services/api/.env
```

```env
# For OpenAI
OPENAI_API_KEY=sk-...

# For Qwen (Alibaba DashScope)
DASHSCOPE_API_KEY=sk-...

# For Anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Start the API

```bash
uvicorn services.api.main:app --reload --port 8000
```

> The backend must be started as the `services.api.main` package from the
> repository root. Running `uvicorn main:app` from inside `services/api` fails
> with `ImportError: attempted relative import with no known parent package`.

### 5. Run the web app

```bash
cd apps/web && npm run dev     # http://localhost:3000
# or, for the built version:
cd apps/web && npm run preview
```

Click **New Task**, describe what you want to optimize, and watch the loop run.

---

## Desktop App

```bash
cd apps/web
npm install --legacy-peer-deps
npm run build                  # static export into apps/web/out
npm run build:electron         # packages the app for your platform
```

The Electron shell serves the static export from a loopback HTTP server and
spawns the Python backend. The packaged app needs **Python 3.10+ with the
backend requirements installed** on the target machine — it is a desktop shell
around a local deployment, not a self-contained installer.

---

## Architecture

```
apps/web/          Next.js 14 frontend — task list, live reasoning feed, results
services/api/      FastAPI — REST + SSE streaming
services/worker/   Research pipeline — Researcher / Engineer / Analyzer loop
packages/types/    Shared TypeScript types
tests/             pytest suite for the worker, API and memory layers
```

The pipeline is a pure Python async generator. FastAPI streams its events via Server-Sent Events (SSE). The frontend connects with `EventSource` and renders each event as it arrives — no polling, no refresh. Every event is also appended to `data/runs/<run_id>/events.jsonl`, so a finished run can be replayed after the process exits.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, React 18, TypeScript 5, Tailwind CSS, Shadcn/UI, Zustand |
| Backend | Python 3.10+, FastAPI, SQLAlchemy, SQLite |
| LLM Interface | LiteLLM (100+ models, one interface) |
| Vector Memory | FAISS + sentence-transformers (optional) |
| Code Execution | subprocess with CPU, memory and file-size limits |

The embedding stack is optional: without `faiss-cpu` and `sentence-transformers`
the app runs normally and simply skips semantic retrieval.

---

## Reasoning Visualization

R U Socrates provides three layers of transparency into the AI research process:

### L1 — Live Reasoning Feed
Real-time stream of every iteration showing:
- **Researcher** → generates candidate solutions with motivations
- **Engineer** → executes and benchmarks code
- **Analyzer** → interprets results and insights

### L2 — Reasoning Tree
Visual tree diagram showing:
- All explored nodes and their parent-child relationships
- Best path highlighted in green
- Currently exploring path in cyan
- Previously explored paths in gray

### L3 — Score Journey
Interactive chart showing:
- Score progression across iterations
- "New best" markers for peak performances
- Improvement tracking

---

## Security

**R U Socrates executes LLM-generated code on your machine.** That is the whole
point of the tool, and it is not safe by default.

What the code does protect against:

- CPU time, address-space and file-size caps on every evaluation
- The whole evaluator process group is killed on timeout or cancel
- Optional bearer token on the API (`RUS_API_TOKEN`)

What it does **not** protect against: a candidate reading, writing or deleting
files, or reaching the network. The limits are damage control, not a sandbox.

**Run untrusted tasks inside a container or a VM.** Keep the API on `127.0.0.1`.
If you expose it, set `RUS_API_TOKEN` and configure `RUS_ALLOWED_ORIGINS`.

---

## Tests

```bash
python -m pytest tests/ -q
```

The suite covers evaluator resolution and argv contracts, resource limits,
process cleanup, the full Researcher → Engineer → Analyzer loop (with a stubbed
LLM), pause/cancel control flow, the event store, and the API auth gate.

---

## Troubleshooting

**"API is offline" error in frontend**

Make sure the FastAPI backend is running from the repository root:
```bash
uvicorn services.api.main:app --reload --port 8000
```

**LLM API errors**

1. Verify your API key is set in `services/api/.env`
2. Check the model name is LiteLLM-compatible (e.g. `gpt-4o`, `deepseek-chat`)
3. For local models, ensure Ollama is running: `ollama serve`

**Frontend build fails**

```bash
cd apps/web
rm -rf node_modules package-lock.json
npm install --legacy-peer-deps
```

**Port already in use**

Change the port in `apps/web/next.config.mjs` or kill the existing process:
```bash
# Windows
netstat -ano | findstr :3000
taskkill /PID <pid> /F
```

---

## Credits

R U Socrates stands on two research projects from **GAIR-NLP**:

- **[ASI-Evolve](https://github.com/GAIR-NLP/ASI-Evolve)** —
  [*ASI-Evolve: AI Accelerates AI*](https://arxiv.org/abs/2603.29640).
  The `Researcher` / `Engineer` / `Analyzer` loop
  (`services/worker/pipeline.py`), the node and cognition stores
  (`services/worker/memory.py`) and the core data structures
  (`services/worker/models.py`) were forked and rewritten from this project,
  under Apache-2.0 as upstream released them.
- **[ASI-Arch](https://github.com/GAIR-NLP/ASI-Arch)** —
  [*AlphaGo Moment for Model Architecture Discovery*](https://arxiv.org/abs/2507.18074).
  Consulted as a secondary reference during the rework.

If you use this project, please cite and credit the upstream work.

This repository is itself a revision of
[zbbsdsb/R-U-Socrates](https://github.com/zbbsdsb/R-U-Socrates), used with the
original author's permission.

---

## Contributing

Contributions are welcome! Please read our [Developer Guide](docs/DEVELOPER_GUIDE.md) for setup instructions and code standards.

### Development Setup

```bash
# Clone the repository
git clone <this repo>
cd R-U-Socrates

# Install dependencies
npm install --legacy-peer-deps
npm run build:types
pip install -r services/requirements.txt

# Terminal 1 — API
uvicorn services.api.main:app --reload

# Terminal 2 — Frontend
cd apps/web && npm run dev
```

Please run `python -m pytest tests/ -q` before opening a PR.

---

## License

The `LICENSE` file in this repository is **Apache License 2.0**.

The core layer derives from ASI-Evolve and keeps Apache-2.0, as upstream
released it. Derived work in this repository is offered under the same terms.
