"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Header from "./Header";
import InterruptCard from "./InterruptCard";
import Sidebar from "./Sidebar";
import ToolCall from "./ToolCall";
import { SendIcon } from "./Icons";
import type { ThreadSummary } from "@/lib/history";
import type { InterruptValue, Turn } from "@/lib/types";

const CASES = [
  { id: "PA-1001", blurb: "Clean baseline — MRI lumbar spine" },
  { id: "PA-1002", blurb: "High cost — pauses for approval" },
  { id: "PA-1005", blurb: "Borderline criteria — human override" },
  { id: "PA-1025", blurb: "Malformed record — safe stop" },
];

let seq = 0;
const nextId = () => `t${++seq}-${Date.now()}`;

export default function Chat() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [listLoading, setListLoading] = useState(true);
  const [opening, setOpening] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [draft, setDraft] = useState("");

  const scroller = useRef<HTMLElement>(null);
  const box = useRef<HTMLTextAreaElement>(null);
  // Read inside the streaming loop, which must not close over a stale value.
  const currentThread = useRef<string | null>(null);

  useEffect(() => {
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [turns]);

  const refreshThreads = useCallback(async () => {
    try {
      const res = await fetch("/api/threads");
      const data = await res.json();
      setThreads(data.threads ?? []);
    } catch {
      // The sidebar is not worth surfacing an error for; the chat pane will
      // report the same outage the moment anything is sent.
    } finally {
      setListLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshThreads();
  }, [refreshThreads]);

  const patchLast = useCallback((fn: (t: Turn) => Turn) => {
    setTurns((prev) => {
      const next = [...prev];
      next[next.length - 1] = fn(next[next.length - 1]);
      return next;
    });
  }, []);

  const run = useCallback(
    async (body: Record<string, unknown>) => {
      setBusy(true);
      setTurns((prev) => [
        ...prev,
        { id: nextId(), role: "assistant", text: "", tools: [], interrupts: [], pending: true },
      ]);

      let created = false;
      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ threadId: currentThread.current, ...body }),
        });

        const id = res.headers.get("X-Thread-Id");
        if (id && id !== currentThread.current) {
          created = true;
          currentThread.current = id;
          setThreadId(id);
        }

        if (!res.ok || !res.body) {
          const detail = await res.text();
          patchLast((t) => ({ ...t, pending: false, error: detail.slice(0, 500) || `HTTP ${res.status}` }));
          return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";
          for (const line of lines) {
            if (!line.startsWith("data:")) continue;
            try {
              applyFrame(JSON.parse(line.slice(5)), patchLast);
            } catch {
              // A truncated frame is not worth aborting the stream over.
            }
          }
        }
        patchLast((t) => ({ ...t, pending: false }));
      } catch (err) {
        patchLast((t) => ({
          ...t,
          pending: false,
          error: err instanceof Error ? err.message : String(err),
        }));
      } finally {
        setBusy(false);
        box.current?.focus();
        // Refresh so a new chat appears, and an existing one re-sorts to the
        // top. The second pass is because the server can still be marking the
        // run idle as the stream closes, which would leave a stale "paused"
        // dot on a chat that just finished.
        void refreshThreads();
        setTimeout(() => void refreshThreads(), 1200);
        if (created) setSidebarOpen(false);
      }
    },
    [patchLast, refreshThreads],
  );

  function send(text: string) {
    setTurns((prev) => [...prev, { id: nextId(), role: "user", text, tools: [], interrupts: [] }]);
    void run({ input: { messages: [{ role: "user", content: text }] } });
  }

  function submit() {
    const text = draft.trim();
    if (!text || busy) return;
    setDraft("");
    if (box.current) box.current.style.height = "auto";
    send(text);
  }

  function newChat() {
    currentThread.current = null;
    setThreadId(null);
    setTurns([]);
    setDraft("");
    setSidebarOpen(false);
    box.current?.focus();
  }

  async function openThread(id: string) {
    if (busy) return;
    setOpening(true);
    setSidebarOpen(false);
    try {
      const res = await fetch(`/api/threads/${id}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
      currentThread.current = id;
      setThreadId(id);
      setTurns(data.turns ?? []);
    } catch (err) {
      currentThread.current = id;
      setThreadId(id);
      setTurns([
        {
          id: nextId(),
          role: "assistant",
          text: "",
          tools: [],
          interrupts: [],
          error: err instanceof Error ? err.message : String(err),
        },
      ]);
    } finally {
      setOpening(false);
    }
  }

  async function deleteThread(id: string) {
    setThreads((prev) => prev.filter((t) => t.id !== id));
    if (id === currentThread.current) newChat();
    try {
      await fetch(`/api/threads/${id}`, { method: "DELETE" });
    } finally {
      void refreshThreads();
    }
  }

  return (
    <div className="app">
      <Sidebar
        threads={threads}
        activeId={threadId}
        open={sidebarOpen}
        loading={listLoading}
        onNew={newChat}
        onOpen={openThread}
        onDelete={deleteThread}
        onClose={() => setSidebarOpen(false)}
      />

      <div className="pane">
        <Header
          onNew={newChat}
          onToggleSidebar={() => setSidebarOpen((v) => !v)}
        />

        <main className="main" ref={scroller}>
          <div className="thread">
            {opening && <div className="loading">Loading chat…</div>}

            {!opening && turns.length === 0 && (
              <div className="empty">
                <h1>Prior authorization review</h1>
                <p>
                  Ask the validation agent to review a case. It runs intake,
                  checks member eligibility, evaluates clinical criteria, and
                  pauses for your decision before anything is finalized.
                </p>
                <div className="empty-label">Try a case</div>
                <div className="cases">
                  {CASES.map((c) => (
                    <button
                      key={c.id}
                      className="case"
                      disabled={busy}
                      onClick={() => send(`Review case ${c.id}`)}
                    >
                      <b>{c.id}</b>
                      <span>{c.blurb}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {!opening &&
              turns.map((turn, i) => {
                const continues = i > 0 && turns[i - 1].role === turn.role;
                return (
                  <div
                    className={`turn ${turn.role}${continues ? " continues" : ""}`}
                    key={turn.id}
                  >
                    <div className="avatar">
                      {continues ? "" : turn.role === "user" ? "You" : "AI"}
                    </div>
                    <div className="turn-body">
                      {!continues && (
                        <div className="turn-who">
                          {turn.role === "user" ? "You" : "Validation agent"}
                        </div>
                      )}

                      {turn.tools.map((t) => (
                        <ToolCall key={t.id} tool={t} />
                      ))}

                      {turn.interrupts.map((iv, k) => (
                        <InterruptCard
                          key={k}
                          value={iv}
                          disabled={busy}
                          onDecide={(decision) => void run({ command: { resume: decision } })}
                        />
                      ))}

                      {turn.text && <div className="turn-text">{turn.text}</div>}

                      {turn.pending && !turn.text && (
                        <div className="dots" aria-label="Thinking">
                          <span />
                          <span />
                          <span />
                        </div>
                      )}

                      {turn.error && (
                        <div className="error">
                          Could not reach the validation agent.
                          <code>{turn.error}</code>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
          </div>
        </main>

        <div className="composer">
          <div className="composer-wrap">
            <textarea
              ref={box}
              rows={1}
              value={draft}
              placeholder="Ask about a prior auth case, e.g. “Review PA-1002”"
              onChange={(e) => {
                setDraft(e.target.value);
                e.target.style.height = "auto";
                e.target.style.height = `${Math.min(e.target.scrollHeight, 176)}px`;
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  submit();
                }
              }}
            />
            <button className="send" onClick={submit} disabled={busy || !draft.trim()} aria-label="Send">
              <SendIcon />
            </button>
          </div>
          <div className="disclaimer">
            Recommendations are advisory. A human reviewer makes the final
            authorization decision.
          </div>
        </div>
      </div>
    </div>
  );
}

/* --------------------------------------------------------------------- */

type Patch = (fn: (t: Turn) => Turn) => void;

/** Fold one SSE frame from /api/chat into the streaming assistant turn. */
function applyFrame(frame: { event: string; data: unknown }, patch: Patch) {
  const { event, data } = frame;

  if (event === "error") {
    patch((t) => ({ ...t, pending: false, error: JSON.stringify(data) }));
    return;
  }

  // Token deltas. Only the agent's own model node is the visible reply: the
  // structured-output LLM inside evaluate_criteria also streams here (as
  // langgraph_node "tools"), and its raw JSON must not land in the transcript.
  if (event.startsWith("messages")) {
    const chunk = Array.isArray(data) ? data[0] : data;
    const meta = (Array.isArray(data) ? data[1] : {}) as { langgraph_node?: string };
    if (meta?.langgraph_node !== "model") return;
    const msg = chunk as { content?: unknown; type?: string; tool_call_id?: string };
    if (!msg || msg.type === "tool" || msg.tool_call_id) return;
    let piece = "";
    if (typeof msg.content === "string") piece = msg.content;
    else if (Array.isArray(msg.content))
      piece = msg.content
        .map((p) => (typeof p === "object" && p && "text" in p ? String(p.text) : ""))
        .join("");
    if (piece) patch((t) => ({ ...t, text: t.text + piece, pending: false }));
    return;
  }

  if (event === "updates" && data && typeof data === "object") {
    const record = data as Record<string, unknown>;

    const raised = record.__interrupt__;
    if (Array.isArray(raised)) {
      const values = raised.map(
        (i) => ((i as { value?: InterruptValue }).value ?? i) as InterruptValue,
      );
      patch((t) => ({ ...t, pending: false, interrupts: [...t.interrupts, ...values] }));
      return;
    }

    for (const update of Object.values(record)) {
      const messages = (update as { messages?: unknown[] })?.messages;
      if (!Array.isArray(messages)) continue;

      for (const raw of messages) {
        const m = raw as {
          type?: string;
          name?: string;
          content?: string;
          tool_call_id?: string;
          tool_calls?: { id: string; name: string; args: Record<string, unknown> }[];
        };

        if (m.type === "tool" || m.tool_call_id) {
          // The result completes the row the tool call already created —
          // merge into it by id rather than adding a second row.
          patch((t) => {
            const id = m.tool_call_id ?? nextId();
            const known = t.tools.some((x) => x.id === id);
            const result = m.content ?? "";
            const name = m.name ?? "tool";
            return {
              ...t,
              pending: false,
              tools: known
                ? t.tools.map((x) => (x.id === id ? { ...x, name, result } : x))
                : [...t.tools, { id, name, result }],
            };
          });
        } else if (m.tool_calls?.length) {
          patch((t) => {
            const known = new Set(t.tools.map((x) => x.id));
            const added = m.tool_calls!
              .filter((tc) => !known.has(tc.id))
              .map((tc) => ({ id: tc.id, name: tc.name, args: tc.args }));
            return added.length ? { ...t, pending: false, tools: [...t.tools, ...added] } : t;
          });
        }
      }
    }
  }
}
