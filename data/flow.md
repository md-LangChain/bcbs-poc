Build Scenario 3 as a LangGraph workflow for prior authorization processing.

The flow should be:

```text
START
  ↓
Receive prior authorization request
  ↓
FA-1: Intake / Triage
  ↓
Validate request
  ↓
Is request complete and usable?
  ├─ NO → Safe stop / flag for human review
  └─ YES
       ↓
FA-4: Check member eligibility through MCP tool
       ↓
Is member eligible / coverage active?
       ├─ NO → Route to human review
       └─ YES
            ↓
FA-2: Evaluate clinical criteria
            ↓
Does request meet clinical criteria?
            ├─ MEETS CRITERIA
            │      ↓
            │   Human review
            │
            └─ DOES NOT MEET CRITERIA
                   ↓
                Human review
                     ↓
              Human decision
              ├─ APPROVE
              ├─ DENY
              └─ REQUEST MORE INFORMATION
                     ↓
              Finalize outcome
                     ↓
                    END
```

## Step 1: Receive the prior authorization request

The workflow starts when a prior authorization request enters the system.

For the POC, this can simply be one row from the synthetic dataset.

The request then enters FA-1.

---

## Step 2: FA-1 performs intake and triage

The agent should read in the case and make sure all the data is there and not malformed. 

---

## Step 3: Validate the request

After intake, determine whether the request is valid enough to continue.

Ask questions such as:



### If validation fails

Do not continue blindly.

The workflow should stop safely and route the case for review or correction.

This is where the failure-recovery scenario belongs.

The important behavior is:

```text
Bad input
→ detect problem
→ do not hallucinate missing information
→ stop safely
→ explain why processing cannot continue
```

### If validation succeeds

Continue to eligibility.

---

## Step 4: FA-4 performs member eligibility lookup

FA-1 or the LangGraph workflow should invoke FA-4.

FA-4 should be treated as an MCP tool server, not as another autonomous reasoning agent.

For this scenario, its main responsibility is:

```text
Check whether the member has active eligibility / coverage
```

The workflow calls the eligibility tool and receives the result.

Conceptually:

```text
LangGraph
   ↓
FA-4 MCP eligibility tool
   ↓
eligibility response
   ↓
LangGraph
```

FA-4 does NOT generate prior authorization statistics.

FA-4 is just providing a bounded capability that the workflow needs.

---

## Step 5: Evaluate the eligibility result

After FA-4 returns the eligibility result, route based on the result.

### Eligible

Continue to FA-2.

```text
eligible
→ clinical criteria evaluation
```

### Not eligible / unclear eligibility

Do not automatically make a medical denial unless that is explicitly required by the business rules.

For the POC, route the case to human review.

```text
not eligible / eligibility unclear
→ human review
```

---

## Step 6: FA-2 evaluates clinical criteria

FA-2 determines whether the clinical facts satisfy the relevant medical-necessity criteria.

FA-2 receives the prior authorization request after intake and eligibility checks.

FA-2 should evaluate the case against the synthetic InterQual-style criteria.

The important output is one of:

```text
MEETS CRITERIA
```

or:

```text
DOES NOT MEET CRITERIA
```

Along with a rationale explaining why.

Example conceptually:

```text
Requested MRI
+
diagnosis
+
duration of symptoms
+
previous conservative treatment
+
clinical findings
        ↓
FA-2
        ↓
MEETS CRITERIA
because symptoms persisted for X weeks
and conservative therapy was attempted
```

FA-2 should not make the final authorization decision.

It produces a clinical recommendation.

---

## Step 7: Route the result to human review

Regardless of whether FA-2 returns:

```text
MEETS CRITERIA
```

or:

```text
DOES NOT MEET CRITERIA
```

the workflow should pause for a human reviewer.

The human should be able to see:

```text
Original prior authorization request

+

FA-1 intake / validation result

+

FA-4 eligibility result

+

FA-2 clinical criteria result

+

FA-2 rationale
```

This is the main human-in-the-loop governance point.

The LangGraph workflow should pause here instead of continuing automatically.

---

## Step 8: Human makes the authorization decision

The human reviewer chooses one of:

```text
APPROVE

DENY

REQUEST MORE INFORMATION
```

The agents are supporting the reviewer, not replacing the reviewer.

Conceptually:

```text
AI recommendation
      ↓
Human judgment
      ↓
Final decision
```

---

## Step 9: Finalize the case

Once the human makes the decision, the workflow resumes.

Translate the reviewer decision into the final case outcome.

Examples:

```text
Human = APPROVE
→ Prior authorization approved
```

```text
Human = DENY
→ Prior authorization denied
```

```text
Human = REQUEST MORE INFORMATION
→ Prior authorization remains pending
→ additional information requested
```

Then end the workflow.

---

# Core LangGraph Mental Model

The graph should conceptually look like:

```text
START
  ↓
FA-1 Intake
  ↓
Validation
  ↓
        ┌──────── invalid ───────→ Safe Stop
        │
        └──────── valid
                   ↓
             FA-4 Eligibility
                   ↓
        ┌──── eligibility issue ─→ Human Review
        │
        └──── eligible
                   ↓
             FA-2 Clinical
                   ↓
           Clinical Recommendation
                   ↓
              Human Review
                   ↓
          Approve / Deny / More Info
                   ↓
                Finalize
                   ↓
                  END
```

## Key architectural rule

Do not build this as several autonomous agents having an open-ended conversation with each other.

Build it primarily as an explicit LangGraph workflow:

```text
workflow logic
+
small specialized agent/tool steps
+
conditional routing
+
human interrupt
```

The responsibilities are:

```text
FA-1
= intake + triage

FA-4
= MCP tools, especially eligibility lookup

FA-2
= clinical criteria evaluation

LangGraph
= orchestration and routing

Human reviewer
= final authorization authority
```

## Build order

Implement the flow in this order:

```text
1. Receive one prior auth request

2. Run FA-1 intake

3. Validate it

4. If invalid, stop safely

5. If valid, call FA-4 eligibility

6. Route based on eligibility

7. Run FA-2 clinical criteria evaluation

8. Produce MEETS / DOES NOT MEET + rationale

9. Pause for human review

10. Human chooses approve / deny / request more information

11. Finalize the case

12. End
```

Keep the POC focused on demonstrating that orchestration clearly.
