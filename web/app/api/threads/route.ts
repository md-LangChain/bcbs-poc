import { APP_TAG, getClient, unreachable } from "@/lib/langgraph";
import { deriveTitle, type ThreadSummary } from "@/lib/history";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type ThreadRow = {
  thread_id: string;
  updated_at?: string;
  created_at?: string;
  status?: string;
  metadata?: Record<string, unknown>;
  values?: { messages?: { type?: string; content?: unknown }[] };
};

/** List this UI's chats, most recently updated first. */
export async function GET() {
  try {
    const rows = (await getClient().threads.search({
      metadata: { app: APP_TAG },
      limit: 100,
    })) as unknown as ThreadRow[];

    const threads: ThreadSummary[] = rows
      .map((t) => ({
        id: t.thread_id,
        title:
          (typeof t.metadata?.title === "string" && t.metadata.title) ||
          deriveTitle(t.values?.messages ?? []),
        updatedAt: t.updated_at ?? t.created_at ?? "",
        status: t.status,
      }))
      .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));

    return Response.json({ threads });
  } catch (err) {
    return Response.json({ error: unreachable(err), threads: [] }, { status: 502 });
  }
}

/** Create an empty chat. /api/chat does this implicitly on the first message. */
export async function POST(req: Request) {
  let title = "New chat";
  try {
    const body = await req.json();
    if (typeof body?.title === "string" && body.title.trim()) title = body.title.trim();
  } catch {
    // No body is fine — the thread just keeps the default title.
  }

  try {
    const thread = await getClient().threads.create({
      metadata: { app: APP_TAG, title },
    });
    return Response.json({ thread_id: thread.thread_id, title });
  } catch (err) {
    return Response.json({ error: unreachable(err) }, { status: 502 });
  }
}
