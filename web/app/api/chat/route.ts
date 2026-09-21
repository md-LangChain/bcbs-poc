import { APP_TAG, ASSISTANT_ID, describe, getClient, unreachable } from "@/lib/langgraph";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type Body = {
  threadId?: string | null;
  input?: { messages: { role: string; content: string }[] };
  command?: { resume: unknown };
};

/**
 * Streams one validation-agent run to the browser as SSE.
 *
 * Creates the thread on first call — tagged so the sidebar can find it, and
 * titled from the opening message — and returns its id in X-Thread-Id so the
 * client keeps the conversation (and any pending interrupt) on one thread.
 */
export async function POST(req: Request) {
  let body: Body;
  try {
    body = await req.json();
  } catch {
    return new Response("Invalid JSON body", { status: 400 });
  }

  const client = getClient();
  let threadId = body.threadId ?? null;
  let createdTitle: string | null = null;

  try {
    if (!threadId) {
      const opening = body.input?.messages?.find((m) => m.role === "user")?.content;
      createdTitle = opening?.trim().slice(0, 80) || "New chat";
      const thread = await client.threads.create({
        metadata: { app: APP_TAG, title: createdTitle },
      });
      threadId = thread.thread_id;
    }
  } catch (err) {
    return new Response(unreachable(err), { status: 502 });
  }

  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      const send = (event: string, data: unknown) =>
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify({ event, data })}\n\n`),
        );

      try {
        const run = client.runs.stream(threadId!, ASSISTANT_ID, {
          input: body.input,
          command: body.command,
          streamMode: ["messages-tuple", "updates"],
        });
        for await (const chunk of run) {
          send(chunk.event, chunk.data);
        }
      } catch (err) {
        send("error", describe(err));
      } finally {
        controller.close();
      }
    },
  });

  const headers: Record<string, string> = {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
    "X-Thread-Id": threadId,
  };
  if (createdTitle) headers["X-Thread-Title"] = encodeURIComponent(createdTitle);

  return new Response(stream, { headers });
}
