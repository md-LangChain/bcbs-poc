# BCBS RI — Agentic Prior Authorization POC

Three LangGraph agents that triage a prior-authorization request, check member
eligibility, evaluate clinical criteria, and stop for a human decision — plus a
chat UI for driving the whole thing.

## Quick start

```bash
cp .env.example .env     # then paste your two keys (see Environment below)
./dev.sh                 # starts the agents and the UI, Ctrl+C stops both
```

Open **http://localhost:3000** and click a case, or type `Review PA-1002`.

## The agents

| Agent | What it does | LLM |
| --- | --- | --- |
| **validation** (FA-2) | The one you talk to. Calls intake, evaluates clinical criteria, pauses for your decision. | yes |
| **intake** (FA-1) | Loads a case from the CSV, redacts PHI, checks for missing fields, flags high cost. | no |
| **eligibility** (FA-4) | Confirms coverage from a member id alone — no PHI crosses this boundary. | no |

Calls run one direction: **validation → intake → eligibility**. Validation
drives; intake is a tool it calls; eligibility is a bounded lookup intake makes.

All four graphs (including the `a2a-validation` variant) are registered in the
root [`langgraph.json`](langgraph.json), so one server serves them together.

## Human-in-the-loop

`evaluate_criteria` raises three interrupts, each rendered in the UI as an
approve / deny / request-more-info card:

- `intake_validation_failed` — missing or malformed case data
- `high_cost_hitl` — estimated cost above $10,000
- `borderline_hitl` — clinically inconclusive

The agent never decides. It produces a recommendation and stops.

## Environment

One `.env` at the repo root serves both the agents and the UI.
[`.env.example`](.env.example) documents every variable; only two must be filled in:

```bash
LANGSMITH_API_KEY=lsv2_...
OPENAI_API_KEY=sk-...          # or an lsv2_sk_ gateway key, see below
```

**Routing through the LangSmith LLM Gateway** — set `OPENAI_API_KEY` to the
gateway key and add:

```bash
OPENAI_BASE_URL=https://gateway.smith.langchain.com/v1
VALIDATION_MODEL=openai:openai/gpt-4.1-mini
```

The gateway requires `provider/model` form, which is why `openai` appears twice.

**Validation's system prompt** is pulled from LangSmith Context Hub at import.
Outside a local dev run the hub is required and an unavailable context raises at
startup; under `langgraph dev` the agent logs a warning and uses the built-in
`FALLBACK_SYSTEM_PROMPT` instead. Set `VALIDATION_PROMPT_REQUIRE_HUB` to
`true`/`false` to force either behaviour. Every validation run records which copy
executed as `prompt_source` and `prompt_sha` run metadata.

## Running the pieces separately

```bash
# agents only — the flags matter, see below
uv run langgraph dev --no-reload --n-jobs-per-worker 10

# UI only, against an already-running server
cd web && pnpm dev
```

`--n-jobs-per-worker` is not optional. The dev server defaults to **one**
concurrent job, and intake calls back into that same server to reach
eligibility — so a review ends up waiting on a worker it is itself holding, and
hangs. `--no-reload` keeps the file watcher off `web/node_modules`.

`./dev.sh` passes both for you.

## The UI

Next.js app in [`web/`](web/) — streaming chat, light/dark, and a sidebar of
previous chats backed by LangGraph threads. See [`web/README.md`](web/README.md).

## Data

[`data/BCBSRI_Synthetic_PriorAuth_Datasetv1.csv`](data/BCBSRI_Synthetic_PriorAuth_Datasetv1.csv)
— 25 synthetic cases. The `PlantedTestCondition` column is the test plan:

| Case | Exercises |
| --- | --- |
| PA-1001 | Clean baseline |
| PA-1002 | High cost — triggers the HITL gate |
| PA-1005 | Borderline criteria — human override |
| PA-1017, PA-1021 | Embedded PHI — tests redaction |
| PA-1024 | The only member eligibility denies |
| PA-1025 | Malformed record — safe stop |

No real PHI. Every identifier is synthetic.

## Evaluation

[`scripts/`](scripts/) builds LangSmith datasets and runs three experiments —
functional (which interrupt fired), PHI leakage, and prompt injection — wired
into [`.github/workflows/experiment-ci.yml`](.github/workflows/experiment-ci.yml)
so a PR touching an agent runs them as a gate.

Setting these up in a fresh workspace needs resources that do not exist yet
(an Application resource tag, the datasets, and a "PHI Detection" evaluator);
section 4 of [`.env.example`](.env.example) lists them.

## Known gaps

- **Eligibility is advisory only.** Intake returns an `eligible` flag and passes
  it along, but no code branches on it. A member with inactive coverage still
  gets a clinical recommendation. [`data/flow.md`](data/flow.md) step 5 says this
  should route to human review; that gate is not implemented.
- **`PriorAuth Golden Dataset`** is read by `scripts/runFunctionalExperiment.py`
  but no script creates it.
- **Local chat history is not durable.** `langgraph dev` keeps threads in memory;
  restarting the server can clear the sidebar.
