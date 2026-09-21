import type { NextRequest } from "next/server";
import { getClient, unreachable } from "@/lib/langgraph";
import { deriveTitle, turnsFromState } from "@/lib/history";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type State = {
  values?: { messages?: unknown[] };
  interrupts?: unknown;
};

/** Reopen one chat: its transcript plus any interrupt it is still paused on. */
export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  try {
    const client = getClient();
    const [state, thread] = await Promise.all([
      client.threads.getState(id) as unknown as Promise<State>,
      client.threads.get(id),
    ]);

    const messages = (state.values?.messages ?? []) as Parameters<
      typeof turnsFromState
    >[0];

    // The server has returned interrupts as an array, and (older revisions) as
    // a map keyed by task id. Accept both.
    const raw = state.interrupts;
    const interrupts = Array.isArray(raw)
      ? raw
      : raw && typeof raw === "object"
        ? Object.values(raw).flat()
        : [];

    const meta = thread.metadata as Record<string, unknown> | undefined;

    return Response.json({
      thread_id: id,
      title:
        (typeof meta?.title === "string" && meta.title) || deriveTitle(messages),
      turns: turnsFromState(messages, interrupts as { value?: never }[]),
    });
  } catch (err) {
    return Response.json({ error: unreachable(err) }, { status: 502 });
  }
}

/** Delete a chat. */
export async function DELETE(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  try {
    await getClient().threads.delete(id);
    return Response.json({ ok: true });
  } catch (err) {
    return Response.json({ error: unreachable(err) }, { status: 502 });
  }
}
