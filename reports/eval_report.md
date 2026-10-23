# ReturnGuard evaluation

Images: 60 (30 defective). Active backend for end-to-end run: heuristic. LLM provider: auto.

## Vision (defective vs intact)

| Backend | Precision | Recall | F1 | Inconclusive | Mean IoU | IoU>=0.3 | Type acc | ms/img |
|---|---|---|---|---|---|---|---|---|
| heuristic | 1.00 | 1.00 | 1.00 | 0.00 | 0.35 | 0.50 | 0.93 | 56 |

## End-to-end decisions

| Scenario | Cases | Correct decision | Escalated |
|---|---|---|---|
| honest defect claim -> approve | 30 | 1.00 | 0.00 |
| honest change of mind -> approve | 30 | 1.00 | 0.00 |
| defect claim on intact photo -> escalate | 30 | 1.00 | 1.00 |
| unmentioned damage -> escalate | 30 | 1.00 | 1.00 |
| outside window -> reject | 60 | 1.00 | 0.00 |

Mean agent latency: 73 ms per case. All cited clauses exist in the policy: True.
