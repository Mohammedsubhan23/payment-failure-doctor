"""
Payment Failure Doctor — training pipeline.

Generates a synthetic-but-realistic labelled dataset of failed payment
transactions, then trains two real models on it:

  1. bucket_model   — multi-class classifier: which root-cause bucket does
                       this failure belong to?
  2. retry_model    — probability model: if we retried this transaction,
                       how likely is it to succeed?

Both are genuinely trained (train/test split, held-out evaluation) rather
than hand-coded rules, and both are saved to backend/models/ so the API
can load and serve them. Run this file once before starting the API:

    python train.py

It prints accuracy / classification report / feature importances so you
have real evidence of model quality to cite, not just a demo screenshot.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, brier_score_loss, log_loss
)

RNG = np.random.default_rng(42)
MODELS_DIR = Path(__file__).parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Root-cause taxonomy — this mirrors real payment gateway failure taxonomies
# ---------------------------------------------------------------------------
BUCKETS = ["timeout", "dropoff", "invalid", "decline", "funds", "risk"]

REASON_TO_BUCKET = {
    "issuer_unavailable": "timeout",
    "internal_error": "timeout",
    "otp_timeout": "dropoff",
    "user_cancelled": "dropoff",
    "invalid_cvv": "invalid",
    "invalid_expiry": "invalid",
    "issuer_general_decline": "decline",
    "insufficient_funds": "funds",
    "issuer_risk_block": "risk",
}
REASONS = list(REASON_TO_BUCKET.keys())
METHODS = ["UPI", "Card", "Netbanking", "Wallet"]
BANKS = ["HDFC Bank", "ICICI Bank", "SBI", "Axis Bank", "Kotak Mahindra", "Yes Bank", "IDFC First"]

# base retry-success rate per bucket (ground truth the model has to recover)
BUCKET_RETRY_BASE = {
    "timeout": 0.82, "dropoff": 0.66, "invalid": 0.52,
    "decline": 0.33, "funds": 0.20, "risk": 0.03,
}


def generate_dataset(n=6000, label_noise=0.08):
    rows = []
    for _ in range(n):
        reason = RNG.choice(REASONS)
        true_bucket = REASON_TO_BUCKET[reason]

        amount = int(np.clip(RNG.lognormal(mean=7.5, sigma=1.0), 100, 60000))
        method = RNG.choice(METHODS, p=[0.42, 0.33, 0.15, 0.10])
        bank = RNG.choice(BANKS)
        hour = int(RNG.integers(0, 24))
        day_of_week = int(RNG.integers(0, 7))
        prior_retries = int(RNG.choice([0, 1, 2, 3], p=[0.55, 0.28, 0.12, 0.05]))
        is_new_customer = int(RNG.random() < 0.35)
        error_code = "GATEWAY_ERROR" if true_bucket in ("timeout", "risk") else \
                     "SERVER_ERROR" if true_bucket == "timeout" and RNG.random() < 0.3 else \
                     "BAD_REQUEST_ERROR"

        # inject realistic label noise: large, late-night, new-customer
        # transactions on "decline" reasons are sometimes actually risk
        # blocks in practice — the model has to learn this interaction,
        # it isn't a clean lookup from reason alone.
        bucket = true_bucket
        if true_bucket == "decline" and amount > 25000 and is_new_customer and RNG.random() < 0.5:
            bucket = "risk"
        if RNG.random() < label_noise:
            bucket = RNG.choice(BUCKETS)

        # retry outcome: probabilistic function of bucket + context, then sampled
        p_retry = BUCKET_RETRY_BASE[bucket]
        p_retry -= 0.15 if amount > 20000 else 0
        p_retry += 0.06 if method == "UPI" else 0
        p_retry -= 0.05 if hour < 6 else 0
        p_retry -= 0.04 * prior_retries
        p_retry = float(np.clip(p_retry + RNG.normal(0, 0.05), 0.01, 0.98))
        retry_success = int(RNG.random() < p_retry)

        rows.append(dict(
            amount=amount, method=method, bank=bank, hour=hour,
            day_of_week=day_of_week, prior_retries=prior_retries,
            is_new_customer=is_new_customer, error_code=error_code,
            reason=reason, bucket=bucket, retry_success=retry_success,
        ))
    return pd.DataFrame(rows)


FEATURES = ["amount", "method", "bank", "hour", "day_of_week",
            "prior_retries", "is_new_customer", "error_code", "reason"]
CATEGORICAL = ["method", "bank", "error_code", "reason"]
NUMERIC = ["amount", "hour", "day_of_week", "prior_retries", "is_new_customer"]


def build_preprocessor():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ], remainder="passthrough")


def main():
    print("Generating synthetic training data...")
    df = generate_dataset(n=6000)
    df.to_csv(DATA_DIR / "failures_train.csv", index=False)
    df.sample(60, random_state=1).to_csv(DATA_DIR / "failures_sample.csv", index=False)
    print(f"  {len(df)} rows -> {DATA_DIR/'failures_train.csv'}")
    print(f"  60-row demo sample -> {DATA_DIR/'failures_sample.csv'}")

    X = df[FEATURES]
    y_bucket = df["bucket"]
    y_retry = df["retry_success"]

    X_train, X_test, yb_train, yb_test, yr_train, yr_test = train_test_split(
        X, y_bucket, y_retry, test_size=0.2, random_state=42, stratify=y_bucket
    )

    # --- model 1: root-cause bucket classifier ---
    print("\nTraining bucket classifier (RandomForest)...")
    bucket_pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, class_weight="balanced")),
    ])
    bucket_pipe.fit(X_train, yb_train)
    yb_pred = bucket_pipe.predict(X_test)
    acc = accuracy_score(yb_test, yb_pred)
    print(f"  test accuracy: {acc:.3f}")
    print(classification_report(yb_test, yb_pred))

    # --- model 2: retry-success probability model ---
    print("Training retry-success probability model (GradientBoosting)...")
    retry_pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", GradientBoostingClassifier(n_estimators=250, max_depth=3, random_state=42)),
    ])
    retry_pipe.fit(X_train, yr_train)
    yr_proba = retry_pipe.predict_proba(X_test)[:, 1]
    yr_pred = retry_pipe.predict(X_test)
    print(f"  test accuracy: {accuracy_score(yr_test, yr_pred):.3f}")
    print(f"  brier score (lower=better calibrated): {brier_score_loss(yr_test, yr_proba):.3f}")
    print(f"  log loss: {log_loss(yr_test, yr_proba):.3f}")

    joblib.dump(bucket_pipe, MODELS_DIR / "bucket_model.joblib")
    joblib.dump(retry_pipe, MODELS_DIR / "retry_model.joblib")
    print(f"\nSaved models to {MODELS_DIR}/")

    # feature importance from the bucket model (evidence for the writeup)
    ohe = bucket_pipe.named_steps["prep"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL))
    all_names = cat_names + [c for c in FEATURES if c not in CATEGORICAL]
    importances = bucket_pipe.named_steps["clf"].feature_importances_
    top = sorted(zip(all_names, importances), key=lambda x: -x[1])[:8]
    print("\nTop features for bucket classification:")
    for name, imp in top:
        print(f"  {name:30s} {imp:.3f}")


if __name__ == "__main__":
    main()
