# ReturnGuard

An agentic vision copilot for clothing return verification. A customer signs in, picks an order, describes the problem and uploads photos. The agent **checks the photos** (quality and integrity), **looks for damage** (trained hole and stain detectors, plus an independent vision-language check), **reads the policy** (RAG) and reaches one of three outcomes: **auto-approve**, **ask for another photo**, or **send to a reviewer**. The one automatic decline is a return outside the 10-day window (the customer can appeal; turn it off with `AUTO_REJECT_OUTSIDE_WINDOW=0` to match the plan's "never auto-reject"). Every other rejection is made by a reviewer, with a reason code. An approved return is then **collected and checked at the door** by a pickup partner (policy R9): they compare the item with the order and the customer's photos, the customer confirms the handover with a 4-digit pickup code, and only then is the refund issued. A failed doorstep check goes back to a reviewer. Clothing only for now.

```
 /login ── one sign-in ───► /account (customer)                    /admin (reviewer)
                           new return, live agent steps,          queue, evidence pack,
                           outcome, extra photos, appeal          approve / reject / request info
                                   │                                      ▲
                                   ▼                                      │
 FastAPI (app/main.py) ── SQLite: users, sessions, orders, cases, media, tool_calls, reviews, events
                                   │
                                   ▼
 LangGraph agent (app/agent.py), max 8 tool calls / 120 s per case
   get_order ► parse_claim ► check_photos ► verify_product ─(wrong item / no usable photo)─► retrieve_policy ► decide ► ticket
      (T0)       (LLM)       (T1 + T7)        (T3)        ► detect_defects ─(weak)► zoom_in ─┤
                                              (T4, YOLO)              (T4 tiles)
                                                   └─(needs 2nd look)► verify_with_vlm (T9, Gemini, blind) ┘
```

**Design rules.** The rules in `app/decision.py` decide; the LLM only proposes and writes the wording. The LLM can make an outcome more cautious (it can stop an automatic approval) but never less. A defect is only auto-approved when the detector **and** the independent vision check both see it, and at least one photo shows the ordered item (product check T3: Gemini compares each photo with the order's type and colour, and a CLIP match against the catalog photos can confirm it). A photo of a different item, or only close-ups, leads to a request for a photo of the whole item. Photo integrity flags, high-value orders (above ₹5,000) and any disagreement go to a reviewer; late returns are declined without checking the photos.

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
pip install ultralytics                            # the trained clothing models (CV/runs/deploy)
copy .env.example .env                             # then add your keys (see below)
uvicorn app.main:app --reload                      # open http://localhost:8000
```

Everyone signs in on the same page; reviewers are sent to `/admin` and customers to `/account`. Local accounts (the login page offers them under "Test accounts"; hide that with `SHOW_DEMO_ACCOUNTS=0` for a public deployment). Pick an account from the "Test account" dropdown to fill in the form; the ↻ button next to it resets the demo: it deletes every return and its photos, and case numbers start again at RG-000001:

| Role | Username | Password | Orders |
|---|---|---|---|
| Customer | `asha` | `demo1234` | ORD-3001 crop sweater (white), ORD-3002 ribbed knit tee (army green) |
| Customer | `ravi` | `demo1234` | ORD-3003 cable knit sweater set, ORD-3004 striped pyjama set |
| Customer | `neha` | `demo1234` | ORD-3005 quilted tights (navy), ORD-3006 thermal knit set, 2-pack |
| Reviewer | `reviewer` | `review1234` | Sees every case |
| Pickup partner | `courier` | `pickup1234` | Collects approved returns at `/pickup` and checks them at the door |

The demo orders are real Amazon Fashion products. Their listing photos and the matching damaged review photos are in `demo_amazon/` (local only, gitignored: Amazon Reviews 2023 is for research use only). `demo_amazon/README.md` lists which photo to upload for each order.

Keys in `.env` (all optional; without them the app still runs on rules and keyword parsing):

| Setting | What it does |
|---|---|
| `LLM_PROVIDER`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` | Text LLM for claim parsing and report wording (Groq, Gemini, Ollama or any OpenAI-compatible endpoint; or `ANTHROPIC_API_KEY`). |
| `GEMINI_API_KEY`, `VLM_MODEL` | The vision-language check (T9). Free key at https://aistudio.google.com/apikey. Without it, defect claims go to a reviewer instead of being auto-approved. |
| `CLOTHES_SECOND_OPINION=1` | Grounding DINO second opinion inside the detector (needs `CV/.pydeps`, slow on CPU). |
| `RETURN_WINDOW_DAYS`, `AUTO_APPROVE_MAX_PRICE` | Business rules (defaults 10 days and ₹5,000, from the project plan). |

To start again with an empty database: stop the server and delete `data/returnguard.db`.

## How a case flows

1. **New return** (customer): order, description (up to 1,000 characters), 1 to 4 photos. Each photo gets a quality pre-check as it is added ("✓ img 1 OK", "⚠ img 2 blurry - retake").
2. **Verifying**: the agent runs in the background; the page shows its steps as they finish.
3. **Pre-analysis** (`app/preanalysis.py`): resolution, sharpness and exposure (limits checked on 244 real garment photos), exact and near-duplicate photos across cases (SHA-256 and a 256-bit difference hash), camera date before delivery, editing software. EXIF is read, then stripped from the stored photo. Faces are blurred before a crop is sent to Gemini.
4. **Detection**: the trained detector and locator from `CV/` (see `CV/README.md`). If the result is weak, or the customer claims damage the first pass did not see, the agent retries on four zoomed tiles.
5. **Vision check (T9)**: Gemini describes the cropped region without seeing the claim; the comparison with the claim is done in code.
6. **Decision**: auto-approve (refund for a confirmed defect, store credit for an undamaged change of mind), ask for a specific photo (at most twice), or escalate with a recommendation for the reviewer.
7. **Review** (reviewer): queue sorted by priority, evidence pack with photos and boxes, parsed claim, findings, order, policy and every tool call; approve, reject or request info, each with a mandatory reason code. A customer whose return was declined can appeal.

## What is in the box

| Path | What it does |
|---|---|
| `app/main.py` | Pages, sign-in, customer and reviewer API, upload validation, private photo serving. |
| `app/auth.py`, `app/security.py` | Session cookies, role checks, PBKDF2 password hashing. |
| `app/db.py` | SQLite schema (users, sessions, orders, cases, media, tool_calls, reviews, events) and demo seed data. |
| `app/agent.py` | LangGraph agent: conditional routing, zoom-in retry, guardrails, live trace, tool-call log. |
| `app/tools.py`, `app/mcp_server.py` | The tools (T0, T1+T7, T4, T8, T9, T10, claim parsing); the MCP server exposes them: `python -m app.mcp_server`. |
| `app/preanalysis.py` | Quality gate, integrity checks, EXIF, face blurring. |
| `app/vision.py` | Trained clothing models (with tiled zoom), with CLIP and an OpenCV heuristic as fallbacks. |
| `app/vlm.py` | Gemini blind region check and the claim comparison. |
| `app/llm.py` | Claim parsing (schema-validated, keyword fallback) and the report proposal. |
| `app/decision.py`, `app/policy.py`, `policy/return_policy.md` | Clothing policy R1 to R8, TF-IDF retrieval, rules engine, reviewer reason codes. |
| `static/login.html`, `customer.html`, `review.html` | The pages, following the team's Figma wireframes (S1 to S5). |
| `weights/face_detection_yunet_2023mar.onnx` | OpenCV YuNet face detector (MIT licence) used for blurring. |
| `tests/` | 26 tests: rules, quality gate, claim parsing, sign-in and access control, end-to-end flows, reviewer and appeal. |

## Mapping to the rubric

| Rubric row | Where it is covered |
|---|---|
| CV-1 | Trained YOLO11 hole and stain detector and locator (`CV/`), boxes shown in the evidence pack, zoomed retry. |
| CV-2 | Zero-shot learning with CLIP versus the classical baseline, compared in `scripts/evaluate.py` (run with `--backends heuristic,clip`). That comparison is the "informative comparison". |
| Module 1 | `llm.py` prompt design (strict JSON, grounded in supplied facts, cite-only-provided-clauses). Write up why prompting plus rules was chosen over fine-tuning; LoRA is optional. |
| Module 2 | LangGraph agent that picks its next tool from the evidence (skips, zoom-in retry, VLM only when needed), guardrails and a tool-call log; the same tools served over MCP. A2A is not used. |
| Module 3 | Image plus text: detector, blind VLM check of crops, LLM claim parsing; evidence pack in the UI. Speech is not built (optional). |
| Module 4 | Runnable app with config and logs. Keep your own examples of AI-assisted requirements, code, debugging, tests and docs for the report. |
| Module 5 | `evaluate.py`: precision/recall/F1, IoU, decision accuracy per scenario, escalation rate, citation validity, latency. Add human evaluation of explanation quality on real photos. |
| Frontend / Backend | Upload, overlays, answers, corrections, history and case status; inference endpoints, validation, SQLite, error handling, keys server-side. |
| Dataset | Synthetic set included. For the report, collect 50 to 100 consented real photos, keep one split per physical item, and drop them into `data/eval/` with a `labels.json` in the same format. |

Suggested split by role: agent and LLM lead (`agent.py`, `decision.py`, `llm.py`, `policy.py`), CV lead (`vision.py`, CV-2 comparison), backend and cloud lead (`main.py`, `db.py`, deployment), frontend and evaluation lead (`static/`, `evaluate.py`, user testing).

## Honest notes (read before you present)

- **The included metrics are optimistic.** `reports/eval_report.md` is on synthetic photos with plain surfaces from the same generator. Expect real photos to be much harder, especially for the classical backend (printed logos, textures and busy backgrounds look like damage to it). Report numbers from your own consented photos.
- **Not verified in the build environment:** the CLIP backend (the model download was blocked, so its thresholds are uncalibrated: sweep them on your pilot data), the Anthropic call (no key), and the UI in a real browser. Verified: the full agent and API with the classical backend, the OpenAI-compatible LLM path against a mock server (agreement, soft disagreement, hard rule, citation stripping, offline fallback), the MCP tool list, and a live uvicorn run.
- The classical baseline reports `inconclusive` when it cannot find the product, and low-confidence or inconclusive photos always go to a human.
- The customer's claim is parsed by the LLM into a schema-validated structure, with a keyword parser as the fallback. Whether the customer claims damage decides between the defect rules and the change-of-mind rules.

## Suggested plan (matches the milestone rows)

1. Week 1: agree the MVP, collect the pilot photos, confirm which LLM and hardware you will use.
2. Weeks 2 to 3: run `evaluate.py` on real photos, calibrate CLIP thresholds, show evidence in the UI.
3. Week 4: connect your LLM, check citations and corrections work, extend the policy and orders.
4. Weeks 5 to 6: held-out test plus a few user trials, cut errors and latency, record the demo video.
