# Prior Authorization Review UI

Next.js (App Router) chat interface for the validation agent.

## Run

```bash
pnpm install
pnpm dev            # http://localhost:3000
```

The UI renders on its own. To actually talk to the agent, also run the
LangGraph server from the repo root:

```bash
uv run langgraph dev --no-reload --n-jobs-per-worker 10
```

`--n-jobs-per-worker` matters: the dev server defaults to **one** concurrent
job, and intake calls back into the same server to reach eligibility. With one
worker that run waits on a worker it is itself holding, and the request hangs.
`--no-reload` stops the watcher from churning on `web/node_modules`.

## Environment

`next.config.ts` loads the repo-root `.env`, so there is still one env file for
the whole POC. Relevant keys:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LANGSMITH_API_KEY` | — | Auth to the LangGraph server. Server-side only. |
| `LANGGRAPH_URL` | `http://127.0.0.1:2024` | Where the agent is running. |
| `VALIDATION_GRAPH` | `validation` | Graph id from the root `langgraph.json`. |

## Layout

```
app/api/chat/route.ts        creates the thread, streams a run as SSE
app/api/threads/route.ts     lists chats for the sidebar, creates one
app/api/threads/[id]/route   reopens or deletes one chat
components/Chat.tsx          conversation state, SSE parsing
components/Sidebar.tsx       chat history
components/InterruptCard     the approve / deny / more-info gate
lib/history.ts               thread state -> transcript
```

The browser never sees `LANGSMITH_API_KEY` — every call goes through the route
handlers in `app/api`.

## URL flags

- `?theme=light` / `?theme=dark` — force a theme, ignoring the saved preference

## Chat history

The sidebar is backed by LangGraph threads, not browser storage. Chats this UI
creates are tagged `app: bcbs-prior-auth-ui` in thread metadata and titled from
their opening message, so `threads.search` lists them without picking up
threads made by Studio, the eval scripts, or the other graphs.

Reopening a chat replays its stored messages *and* any interrupt it is still
paused on, so the decision buttons work on a chat from an earlier session — the
run genuinely resumes. Chats waiting on a decision show an amber dot.

Durability caveat: `langgraph dev` keeps threads in an in-memory store with
pickle snapshots under `.langgraph_api/`. History survives page reloads but is
not a database — restarting the server can clear it. A deployed LangGraph
server with Postgres persists properly.

## Interrupts

`evaluate_criteria` in the validation agent raises three interrupt types, each
rendered as an action card: `intake_validation_failed`, `high_cost_hitl`, and
`borderline_hitl`. Choosing an option resumes the run with
`command: { resume: "<decision>" }` on the same thread.
