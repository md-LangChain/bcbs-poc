/** The three HITL gates raised by evaluate_criteria in the validation agent. */
export type InterruptType =
  | "intake_validation_failed"
  | "high_cost_hitl"
  | "borderline_hitl";

export type InterruptValue = {
  type?: InterruptType | string;
  message?: string;
  payload?: Record<string, unknown>;
};

export type ToolEvent = {
  id: string;
  name: string;
  /** Arguments when the model calls the tool; JSON result when it returns. */
  args?: Record<string, unknown>;
  result?: string;
};

export type Turn = {
  id: string;
  role: "user" | "assistant";
  text: string;
  tools: ToolEvent[];
  interrupts: InterruptValue[];
  error?: string;
  pending?: boolean;
};

/** Options offered on every interrupt, matching what the agent accepts. */
export const DECISIONS = [
  { value: "approve", label: "Approve", tone: "ok" as const },
  { value: "deny", label: "Deny", tone: "no" as const },
  {
    value: "request more information",
    label: "Request more info",
    tone: "neutral" as const,
  },
];

export const INTERRUPT_LABELS: Record<string, string> = {
  intake_validation_failed: "Intake validation failed",
  high_cost_hitl: "High cost — review required",
  borderline_hitl: "Borderline criteria — override required",
};
