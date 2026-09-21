import type { InterruptValue, ToolEvent, Turn } from "./types";

/** One message as the LangGraph thread-state endpoint returns it. */
type StoredMessage = {
  id?: string;
  type?: string;
  name?: string;
  content?: unknown;
  tool_call_id?: string;
  tool_calls?: { id: string; name: string; args: Record<string, unknown> }[];
};

export type ThreadSummary = {
  id: string;
  title: string;
  updatedAt: string;
  status?: string;
};

function textOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content))
    return content
      .map((p) =>
        typeof p === "object" && p && "text" in p ? String((p as { text: unknown }).text) : "",
      )
      .join("");
  return "";
}

/**
 * Rebuild the transcript from a thread's stored messages.
 *
 * Consecutive assistant and tool messages collapse into one assistant turn, so
 * a reopened chat looks the way it did while it streamed. Any interrupt the
 * thread is still paused on is attached to the final turn, which is what makes
 * the decision buttons live again on reopen.
 */
export function turnsFromState(
  messages: StoredMessage[],
  interrupts: { value?: InterruptValue }[] = [],
): Turn[] {
  const turns: Turn[] = [];
  let current: Turn | null = null;

  const startAssistant = (): Turn => {
    if (current) return current;
    current = {
      id: `a${turns.length}`,
      role: "assistant",
      text: "",
      tools: [],
      interrupts: [],
    };
    turns.push(current);
    return current;
  };

  for (const m of messages) {
    if (m.type === "human") {
      current = null;
      turns.push({
        id: m.id ?? `u${turns.length}`,
        role: "user",
        text: textOf(m.content),
        tools: [],
        interrupts: [],
      });
      continue;
    }

    if (m.type === "tool" || m.tool_call_id) {
      const turn = startAssistant();
      const id = m.tool_call_id ?? `t${turn.tools.length}`;
      const existing = turn.tools.find((x) => x.id === id);
      if (existing) {
        existing.result = textOf(m.content);
        existing.name = m.name ?? existing.name;
      } else {
        turn.tools.push({ id, name: m.name ?? "tool", result: textOf(m.content) });
      }
      continue;
    }

    if (m.type === "ai") {
      const turn = startAssistant();
      for (const tc of m.tool_calls ?? []) {
        if (!turn.tools.some((x) => x.id === tc.id)) {
          turn.tools.push({ id: tc.id, name: tc.name, args: tc.args } as ToolEvent);
        }
      }
      const text = textOf(m.content);
      if (text) turn.text += text;
    }
  }

  // A thread paused mid-run still owes the reviewer a decision.
  const pending = interrupts.map((i) => i.value).filter(Boolean) as InterruptValue[];
  if (pending.length) {
    const turn = turns.length && turns[turns.length - 1].role === "assistant"
      ? turns[turns.length - 1]
      : startAssistant();
    turn.interrupts.push(...pending);
  }

  return turns;
}

/** Fall back to the first user message when a thread has no stored title. */
export function deriveTitle(messages: StoredMessage[]): string {
  const first = messages.find((m) => m.type === "human");
  const text = textOf(first?.content).trim();
  return text ? text.slice(0, 80) : "New chat";
}
