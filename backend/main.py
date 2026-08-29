"""
Payment Failure Doctor — API layer.

Serves the two trained models (backend/train.py) over REST, and exposes
two LLM-backed endpoints that are strictly downstream of the model output
— the LLM is never asked to classify anything, only to explain a result
that has already been computed.

Run:
    uvicorn main:app --reload --port 8000
"""

import os
import io
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llm import diagnose_transaction, answer_question

MODELS_DIR = Path(__file__).parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"

app = FastAPI(title="Payment Failure Doctor API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

bucket_model = None
retry_model = None


@app.on_event("startup")
def load_models():
    global bucket_model, retry_model
    bp = MODELS_DIR / "bucket_model.joblib"
    rp = MODELS_DIR / "retry_model.joblib"
    if not bp.exists() or not rp.exists():
        raise RuntimeError(
            "Trained models not found. Run `python train.py` inside backend/ first."
        )
    bucket_model = joblib.load(bp)
    retry_model = joblib.load(rp)


BUCKET_LABELS = {
    "timeout": "Network / Gateway Timeout",
    "dropoff": "Checkout Drop-off",
    "invalid": "Invalid Card / Details",
    "decline": "Bank / Issuer Decline",
    "funds": "Insufficient Funds",
    "risk": "Risk / Fraud Block",
}
BUCKET_CATEGORY = {
    "timeout": "autoRetry", "dropoff": "autoRetry", "invalid": "userAction",
    "decline": "altMethod", "funds": "delayedRetry", "risk": "manualReview",
}

FEATURES = ["amount", "method", "bank", "hour", "day_of_week",
            "prior_retries", "is_new_customer", "error_code", "reason"]


class Transaction(BaseModel):
    id: str
    amount: float
    method: str
    bank: str
    hour: int
    day_of_week: int = 0
    prior_retries: int = 0
    is_new_customer: int = 0
    error_code: str
    reason: str


def classify_df(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES]
    bucket_pred = bucket_model.predict(X)
    bucket_proba = bucket_model.predict_proba(X)
    confidence = bucket_proba.max(axis=1)
    retry_proba = retry_model.predict_proba(X)[:, 1]

    out = df.copy()
    out["bucket_key"] = bucket_pred
    out["bucket_label"] = [BUCKET_LABELS[b] for b in bucket_pred]
    out["category"] = [BUCKET_CATEGORY[b] for b in bucket_pred]
    out["confidence"] = confidence
    out["retry_probability"] = retry_proba
    out["recoverable_value"] = out.apply(
        lambda r: 0.0 if r["category"] == "manualReview" else r["amount"] * r["retry_probability"],
        axis=1,
    )
    return out


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": bucket_model is not None}


@app.post("/classify")
def classify_one(txn: Transaction):
    df = pd.DataFrame([txn.dict()])
    result = classify_df(df).iloc[0]
    return {
        "id": txn.id,
        "bucket": result["bucket_label"],
        "bucket_key": result["bucket_key"],
        "category": result["category"],
        "confidence": round(float(result["confidence"]), 4),
        "retry_probability": round(float(result["retry_probability"]), 4),
    }


@app.post("/batch")
async def classify_batch(file: Optional[UploadFile] = File(None)):
    """
    Classify a CSV of failed transactions. If no file is uploaded, falls
    back to the bundled demo sample so the app is usable with zero setup.
    Expected columns: id, amount, method, bank, hour, day_of_week,
    prior_retries, is_new_customer, error_code, reason
    """
    if file is not None:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    else:
        sample_path = DATA_DIR / "failures_sample.csv"
        if not sample_path.exists():
            raise HTTPException(404, "No sample data found. Run train.py first.")
        df = pd.read_csv(sample_path)
        df["id"] = [f"pay_{i:08d}" for i in range(len(df))]

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Missing required columns: {missing}")

    result = classify_df(df)

    total_value = float(df["amount"].sum())
    recoverable_value = float(result["recoverable_value"].sum())
    risk_count = int((result["category"] == "manualReview").sum())

    bucket_breakdown = (
        result.groupby("bucket_label")["amount"].sum().sort_values(ascending=False)
    )

    return {
        "summary": {
            "total_transactions": len(df),
            "total_failed_value": total_value,
            "estimated_recoverable_value": round(recoverable_value, 2),
            "recoverable_pct": round(100 * recoverable_value / total_value, 1) if total_value else 0,
            "risk_flagged_count": risk_count,
        },
        "bucket_breakdown": [
            {"bucket": k, "amount": float(v)} for k, v in bucket_breakdown.items()
        ],
        "transactions": result[[
            "id", "amount", "method", "bank", "error_code", "reason",
            "bucket_label", "bucket_key", "category", "confidence", "retry_probability",
        ]].to_dict(orient="records"),
    }


class DiagnoseRequest(BaseModel):
    id: str
    amount: float
    method: str
    bank: str
    error_code: str
    reason: str
    bucket: str
    category: str
    confidence: float
    retry_probability: float


@app.post("/diagnose")
def diagnose(req: DiagnoseRequest):
    text = diagnose_transaction(req.dict())
    return {"id": req.id, "diagnosis": text}


class AskRequest(BaseModel):
    question: str
    summary: dict
    bucket_breakdown: list


@app.post("/ask")
def ask(req: AskRequest):
    text = answer_question(req.question, req.summary, req.bucket_breakdown)
    return {"answer": text}
