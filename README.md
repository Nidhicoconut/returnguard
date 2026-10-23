# ReturnGuard

An agentic vision copilot for retail return verification. A customer uploads a photo of an item and says what is wrong. The system **looks** at the photo (computer vision), **checks** the order and the store policy (LLM agent with tools and policy RAG), **decides** (approve, reject, or send to a human), opens a ticket, and shows all of its evidence in a web UI where staff can correct it.

```
 Browser (static/index.html)
    │  photo + order + reason                      staff review / corrections
    ▼                                                      ▲
 FastAPI (app/main.py) ── SQLite (cases, orders, audit events)
    │
    ▼
 LangGraph agent (app/agent.py)
    lookup_order ──► analyze_image ──► retrieve_policy ──► decide ──► finalize
        │ (no order: skip vision)           │                 │           │
        ▼                                   ▼                 ▼           ▼
   orders table                   vision.py (CLIP zero-shot   llm.py +    ticket +
                                  or classical OpenCV)        decision.py audit log
                                                              policy.py (TF-IDF RAG)
```

**Design rule: the LLM proposes, the rules dispose.** `app/decision.py` encodes the policy as code. The LLM writes the explanation and proposes a decision, but it can never approve what the rules would not, it can only cite clauses that really exist, and if it disagrees with the rules on a soft case the case goes to a human. If no LLM is configured (or the call fails) the app still works on rules alone.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # optional: add an LLM key or endpoint
uvicorn app.main:app --reload   # open http://localhost:8000
```

Try the photos in `samples/` (`crack.png`, `scratch.png`, `stain.png`, `intact_1.png`, `intact_2.png`) against the seeded orders:

| Try this | Expected |
|---|---|
| ORD-1001, "arrived with a crack", `crack.png` | Approved, full refund or replacement (R2) |
| ORD-1002, "changed my mind", `intact_1.png` | Approved, store credit (R3) |
| ORD-1007, "the strap is torn", `intact_2.png` | Needs human review: claim and photo disagree (R6) |
| ORD-1004 (delivered 45 days ago), any photo | Rejected: outside window (R1) |
| ORD-1006 (₹14,999), "cracked", `crack.png` | Needs human review: above the approval limit (R7) |
| ORD-0000 (unknown order) | Needs human review (R7) |

To reproduce the evaluation (the 60-image synthetic set is regenerated, not shipped):

```bash
python scripts/make_synthetic_data.py --out data/eval --n 30
python scripts/evaluate.py --data data/eval --backends heuristic   # add ,clip if torch+transformers are installed
pytest -q
```

Seed dates are relative to the day the database was created. Run `python -c "from app import db; db.init_db(reset=True)"` to refresh them.

### Choosing the vision backend and the LLM (all optional, set in `.env`)

| Setting | Options |
|---|---|
| `VISION_BACKEND` | `auto` (CLIP if torch+transformers are installed, else classical), `clip`, `heuristic` |
| `LLM_PROVIDER` | `auto`, `anthropic` (set `ANTHROPIC_API_KEY`), `openai` (any OpenAI-compatible endpoint), `rules` (offline) |

No-cost options for the LLM: Ollama locally (`OPENAI_BASE_URL=http://localhost:11434/v1`), or the Gemini or Groq free tiers through their OpenAI-compatible endpoints. API keys stay on the server and are never sent to the browser.

## What is in the box

| Path | What it does |
|---|---|
| `app/vision.py` | Two backends with one output format. **Classical:** Otsu thresholding, morphology, large-median "expected appearance", local colour/brightness anomaly, connected components, defect typing by chroma and thickness. **CLIP:** zero-shot prompts on the whole photo plus 2x2 and 3x3 tiles for coarse localisation. |
| `app/agent.py` | LangGraph workflow with a conditional edge, a trace entry and an audit event per step. |
| `app/decision.py` | Policy-as-code verdict, merge with the LLM proposal, citation validation. |
| `app/llm.py` | Anthropic or OpenAI-compatible client, strict-JSON prompt, safe failure. |
| `app/policy.py`, `policy/return_policy.md` | Policy split into clauses R1 to R7, TF-IDF retrieval. |
| `app/tools.py`, `app/mcp_server.py` | Four tools (order lookup, vision, policy retrieval, ticket). The MCP server exposes the same functions: `python -m app.mcp_server`. |
| `app/main.py`, `app/db.py` | REST API, upload validation, SQLite, audit trail. |
| `static/index.html` | Upload, photo with marked damage regions, decision label, cited policy, agent steps, staff approve/reject, vision feedback, history. |
| `scripts/make_synthetic_data.py` | Synthetic product photos with known defect boxes (no consent or licensing issues). |
| `scripts/evaluate.py` | Vision and end-to-end metrics, writes `reports/eval_report.md`. |
| `tests/` | 11 tests: policy RAG, decision engine, API scenarios, review override, upload validation. |

API: `GET /api/health`, `GET /api/orders`, `POST /api/cases` (multipart: `order_id`, `reason`, `image`), `GET /api/cases`, `GET /api/cases/{id}`, `POST /api/cases/{id}/review`.

## Mapping to the rubric

| Rubric row | Where it is covered |
|---|---|
| CV-1 | Defect localisation and typing with regions (`vision.py`); tile-grid regions for CLIP. Add a trained U-Net or YOLO later if you have labelled data. |
| CV-2 | Zero-shot learning with CLIP versus the classical baseline, compared in `scripts/evaluate.py` (run with `--backends heuristic,clip`). That comparison is the "informative comparison". |
| Module 1 | `llm.py` prompt design (strict JSON, grounded in supplied facts, cite-only-provided-clauses). Write up why prompting plus rules was chosen over fine-tuning; LoRA is optional. |
| Module 2 | LangGraph agent with tools and a conditional route; the same tools served over MCP. A2A is not used because no independent agents need to delegate. |
| Module 3 | Image plus text in one workflow, evidence inspectable in the UI. Speech is not built (the rubric lists it as optional). |
| Module 4 | Runnable app with config and logs. Keep your own examples of AI-assisted requirements, code, debugging, tests and docs for the report. |
| Module 5 | `evaluate.py`: precision/recall/F1, IoU, decision accuracy per scenario, escalation rate, citation validity, latency. Add human evaluation of explanation quality on real photos. |
| Frontend / Backend | Upload, overlays, answers, corrections, history and case status; inference endpoints, validation, SQLite, error handling, keys server-side. |
| Dataset | Synthetic set included. For the report, collect 50 to 100 consented real photos, keep one split per physical item, and drop them into `data/eval/` with a `labels.json` in the same format. |

Suggested split by role: agent and LLM lead (`agent.py`, `decision.py`, `llm.py`, `policy.py`), CV lead (`vision.py`, CV-2 comparison), backend and cloud lead (`main.py`, `db.py`, deployment), frontend and evaluation lead (`static/`, `evaluate.py`, user testing).

## Honest notes (read before you present)

- **The included metrics are optimistic.** `reports/eval_report.md` is on synthetic photos with plain surfaces from the same generator. Expect real photos to be much harder, especially for the classical backend (printed logos, textures and busy backgrounds look like damage to it). Report numbers from your own consented photos.
- **Not verified in the build environment:** the CLIP backend (the model download was blocked, so its thresholds are uncalibrated: sweep them on your pilot data), the Anthropic call (no key), and the UI in a real browser. Verified: the full agent and API with the classical backend, the OpenAI-compatible LLM path against a mock server (agreement, soft disagreement, hard rule, citation stripping, offline fallback), the MCP tool list, and a live uvicorn run.
- The classical baseline reports `inconclusive` when it cannot find the product, and low-confidence or inconclusive photos always go to a human.
- Defect-claim detection from the customer's text is a keyword match (`decision.py`), chosen so the guardrail is deterministic. Replace it with an LLM classifier if you need more language coverage.

## Suggested plan (matches the milestone rows)

1. Week 1: agree the MVP, collect the pilot photos, confirm which LLM and hardware you will use.
2. Weeks 2 to 3: run `evaluate.py` on real photos, calibrate CLIP thresholds, show evidence in the UI.
3. Week 4: connect your LLM, check citations and corrections work, extend the policy and orders.
4. Weeks 5 to 6: held-out test plus a few user trials, cut errors and latency, record the demo video.
