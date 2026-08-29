# Payment Failure Doctor

**A hybrid ML + LLM system that diagnoses why payments fail and estimates how much of that failed revenue is actually recoverable.**

Built for the Razorpay AI Builder internship track ("build what you believe should exist").

---

## The problem

When a payment fails, a merchant sees a gateway error code (`BAD_REQUEST_ERROR`, `issuer_general_decline`, `GATEWAY_ERROR`...) and nothing else. They don't know if that transaction was lost for good, worth retrying immediately, or a genuine fraud block that shouldn't be retried at all. In aggregate, this is real leaked revenue that nobody is triaging — every failed transaction just gets treated the same, when in reality some categories recover at 80%+ on a simple retry and others recover essentially never.

## What this does

Given a batch of failed transactions, the system:

1. **Classifies** each one into a root-cause bucket (network timeout, checkout drop-off, invalid details, bank decline, insufficient funds, or a risk/fraud block) using a trained classifier — not a hand-written if/else chain.
2. **Scores** each one with a calibrated retry-success probability using a second trained model.
3. **Aggregates** this into merchant-facing numbers: total failed value, estimated recoverable value, and which transactions should never be auto-retried.
4. **Explains** individual transactions and answers free-form merchant questions in plain language, via an LLM call that is strictly downstream of the model output.

## Why the architecture is split this way

This is the part worth reading before the code. It would be easy to point an LLM at raw transaction logs and ask it to "explain what happened" — that's a demo, not a system, and it would hallucinate probabilities that sound plausible and are not real. Instead:

- **The model owns every number.** Root-cause bucket and retry probability come from two scikit-learn models (`backend/train.py`), trained on labelled data with a held-out test set. These are real, evaluated models — see metrics below, not just a demo screenshot.
- **The LLM owns every sentence.** It only runs after classification is complete, receives the model's output as fixed ground truth in the prompt, and is explicitly instructed not to re-derive or question it. Its job is translation and explanation — the one thing LLMs are actually reliable at — not judgment.

This separation is also why the system is auditable: you can inspect exactly why a transaction was bucketed a certain way (feature importances, confidence score) independent of whatever the LLM says about it.

## Model evaluation (actual results, not projected)

Trained on 6,000 synthetic-but-realistic labelled transactions (80/20 train/test split), generated with injected label noise and feature interactions so the classifier has to genuinely learn patterns rather than look up a code in a dictionary.

**Root-cause classifier (RandomForest, 300 trees):**
```
test accuracy: 0.929

              precision    recall  f1-score   support
     decline       0.92      0.88      0.90       138
     dropoff       0.94      0.94      0.94       270
       funds       0.93      0.92      0.93       144
     invalid       0.92      0.96      0.94       260
        risk       0.91      0.82      0.86       135
     timeout       0.93      0.97      0.95       253
```

**Retry-success probability model (GradientBoosting):**
```
test accuracy: 0.680
brier score (lower = better calibrated): 0.198
log loss: 0.580
```

Top predictive features for the bucket classifier were, as expected, the error reason and error code fields — but the model also picks up meaningful signal from amount, prior retry count, and new-customer status, which is what lets it separate cases a pure lookup table would get wrong (e.g. a large, first-time, "general decline" transaction is disproportionately a risk block, not a routine decline — the model learns this interaction).

Re-run `python train.py` any time to regenerate data and retrain — it prints these metrics fresh.

## Architecture

```
payment-failure-doctor/
├── backend/
│   ├── train.py          # generates data, trains + evaluates + saves both models
│   ├── main.py            # FastAPI app: /classify /batch /diagnose /ask
│   ├── llm.py              # thin LLM layer, strictly downstream of model output
│   ├── models/             # saved .joblib models (created by train.py)
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js               # auto-detects backend; falls back to client-side
│                             # heuristic + direct LLM call if it's not running
└── data/
    ├── failures_train.csv   # full training set (created by train.py)
    └── failures_sample.csv  # 60-row demo sample, used when no CSV is uploaded
```

## Running it

**1. Backend**
```bash
cd backend
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
python train.py                 # trains models, prints metrics, ~10 seconds
cp .env.example .env             # then paste in an ANTHROPIC_API_KEY
uvicorn main:app --reload --port 8000
```

**2. Frontend**

Just open `frontend/index.html` in a browser (or serve it with any static server — e.g. `python -m http.server 5500` from inside `frontend/`). It talks to `http://localhost:8000` by default.

The frontend is a single merged client that **auto-detects the backend**:
- Backend running → it calls `/batch`, `/diagnose`, `/ask` and everything you see comes from the real trained models. The status pill at the top reads "Connected — RandomForest / GradientBoosting backend".
- Backend not running → it falls back to a small client-side heuristic (same taxonomy, hand-written scoring instead of a trained model) and calls the Anthropic API directly from the browser for diagnosis/Q&A. The status pill reads "No backend detected — running client-side heuristic fallback" and every confidence label in the UI switches from "Model confidence" to "Heuristic confidence" so it's never ambiguous which mode produced a given number.

This means the page is inspectable in two seconds with zero setup, but the thing actually being evaluated — the trained classifier and calibrated probability model — is one `uvicorn` command away.

Without an API key set (either in `backend/.env` or hardcoded in the fallback path), classification still works fully in both modes — only the "Get diagnosis" and "Ask the doctor" buttons need it.

## API reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check |
| `/classify` | POST | Classify a single transaction |
| `/batch` | POST | Classify a CSV upload (or bundled sample if none provided) |
| `/diagnose` | POST | LLM explanation for one already-classified transaction |
| `/ask` | POST | LLM answer to a free-form question, grounded in batch summary |

## What's synthetic vs. real here

The transaction data is synthetic (there's no production payment data to train on outside a real gateway), generated with a deliberately probabilistic, noisy process so the classification problem is genuine rather than trivial. The **models are real** — actually trained, actually evaluated on held-out data, with the metrics above coming straight from the training run. In a production setting, `train.py`'s `generate_dataset()` would be replaced by a query against real historical failure + retry-outcome logs; nothing else in the pipeline would need to change.

## Next steps if this went further

- Swap synthetic data for real gateway webhook logs
- Add a feedback loop: log actual retry outcomes and periodically retrain the retry-probability model on them (true online learning, not a static snapshot)
- Auto-trigger retries for the `autoRetry` category directly via the Payment Links API instead of just recommending it
- Multilingual diagnosis output for merchants who don't operate in English
