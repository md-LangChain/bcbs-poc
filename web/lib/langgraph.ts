import { Client } from "@langchain/langgraph-sdk";

/** URL of the LangGraph server. `uv run langgraph dev` serves 127.0.0.1:2024. */
export const LANGGRAPH_URL =
  process.env.LANGGRAPH_URL ?? "http://127.0.0.1:2024";

/** Graph id from the repo-root langgraph.json. */
export const ASSISTANT_ID = process.env.VALIDATION_GRAPH ?? "validation";

/**
 * Tags threads this UI created, so the sidebar lists only its own chats and
 * not threads made by Studio, the eval scripts, or the other graphs.
 */
export const APP_TAG = "bcbs-prior-auth-ui";

/**
 * Server-only client. The API key never reaches the browser — every call to
 * the LangGraph server goes through the route handlers in app/api.
 */
export function getClient(): Client {
  return new Client({
    apiUrl: LANGGRAPH_URL,
    apiKey: process.env.LANGSMITH_API_KEY,
  });
}

export function describe(err: unknown): string {
  if (err instanceof Error) return `${err.name}: ${err.message}`;
  return String(err);
}

export function unreachable(err: unknown): string {
  return (
    `Cannot reach the LangGraph server at ${LANGGRAPH_URL}. ` +
    `Is \`uv run langgraph dev\` running? (${describe(err)})`
  );
}
