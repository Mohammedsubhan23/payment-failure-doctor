"""
LLM layer. Deliberately thin: every function here receives numbers that
have already been produced by the trained models in train.py / main.py,
and its only job is to turn them into a clear sentence or answer a
question grounded in them. Nothing here is allowed to invent a
classification or a statistic.
"""

import os
import requests

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-6"


def _call(prompt: str) -> str:
    if not ANTHROPIC_API_KEY:
        return ("[No ANTHROPIC_API_KEY set in the environment — add one to backend/.env "
                "to enable live diagnosis. See .env.example.]")
    resp = requests.post(
        API_URL,
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": MODEL,
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()


def diagnose_transaction(t: dict) -> str:
    prompt = f"""You are writing a short, merchant-facing diagnosis inside a payments \
dashboard called "Payment Failure Doctor". Write 2-3 tight sentences in plain, \
non-technical language, then one clear recommended action on its own final line \
starting with "Recommended:". Do not restate the raw fields as a list.

Transaction: ₹{t['amount']} via {t['method']}, bank: {t['bank']}, gateway error: \
{t['error_code']} ({t['reason']}).
Model-classified root cause: "{t['bucket']}" with {t['confidence']*100:.0f}% confidence.
Model-estimated retry-success probability: {t['retry_probability']*100:.0f}%.
Category: {t['category']}.

Treat the classification and probability as ground truth from the model — do not \
question or re-derive them. Just explain what it means for this merchant and what \
to do next."""
    return _call(prompt)


def answer_question(question: str, summary: dict, bucket_breakdown: list) -> str:
    breakdown_lines = "\n".join(
        f"{b['bucket']}: ₹{b['amount']:,.0f}" for b in bucket_breakdown
    )
    prompt = f"""You are "the doctor" inside a payments dashboard, answering a \
merchant's question using ONLY the computed data below. Never invent numbers not \
present here. Answer in 3-5 sentences, plain language, no bullet lists, no headers.

Total failed value: ₹{summary['total_failed_value']:,.0f} across \
{summary['total_transactions']} transactions.
Estimated recoverable value: ₹{summary['estimated_recoverable_value']:,.0f} \
({summary['recoverable_pct']}%).
Risk-flagged (no auto-retry): {summary['risk_flagged_count']}.

Breakdown by root cause:
{breakdown_lines}

Merchant's question: "{question}\""""
    return _call(prompt)
