"use client";

import { useState } from "react";
import { DECISIONS, INTERRUPT_LABELS, type InterruptValue } from "@/lib/types";
import { PauseIcon } from "./Icons";

export default function InterruptCard({
  value,
  onDecide,
  disabled,
}: {
  value: InterruptValue;
  onDecide: (decision: string) => void;
  disabled?: boolean;
}) {
  const [chosen, setChosen] = useState<string | null>(null);
  const label =
    INTERRUPT_LABELS[value.type ?? ""] ?? value.type ?? "Human review required";

  return (
    <div className="interrupt">
      <div className="interrupt-kind">
        <PauseIcon />
        {label}
      </div>
      <p>{value.message ?? "The agent paused for a human decision."}</p>
      {value.payload && (
        <pre className="code">{JSON.stringify(value.payload, null, 2)}</pre>
      )}
      <div className="actions">
        {DECISIONS.map((d) => (
          <button
            key={d.value}
            className={d.tone === "neutral" ? "" : d.tone}
            disabled={disabled || chosen !== null}
            onClick={() => {
              setChosen(d.value);
              onDecide(d.value);
            }}
          >
            {chosen === d.value ? `${d.label} ✓` : d.label}
          </button>
        ))}
      </div>
    </div>
  );
}
