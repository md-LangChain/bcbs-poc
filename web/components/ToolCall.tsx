"use client";

import { useState } from "react";
import type { ToolEvent } from "@/lib/types";

/** Pretty-print a JSON string, or pass it through when it is not JSON. */
function pretty(value: unknown): string {
  if (typeof value === "string") {
    try {
      return JSON.stringify(JSON.parse(value), null, 2);
    } catch {
      return value;
    }
  }
  return JSON.stringify(value, null, 2);
}

const TITLES: Record<string, string> = {
  run_intake: "Ran intake",
  evaluate_criteria: "Evaluated clinical criteria",
};

export default function ToolCall({ tool }: { tool: ToolEvent }) {
  const [open, setOpen] = useState(false);
  const body = tool.result ?? tool.args;
  const title =
    tool.result !== undefined
      ? (TITLES[tool.name] ?? `${tool.name} returned`)
      : `Calling ${tool.name}`;

  return (
    <div className="tool">
      <div
        className="tool-head"
        onClick={() => setOpen((v) => !v)}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            setOpen((v) => !v);
          }
        }}
      >
        <span className={`tool-caret${open ? " open" : ""}`}>▶</span>
        <b>{title}</b>
      </div>
      {open && body !== undefined && <pre className="code">{pretty(body)}</pre>}
    </div>
  );
}
