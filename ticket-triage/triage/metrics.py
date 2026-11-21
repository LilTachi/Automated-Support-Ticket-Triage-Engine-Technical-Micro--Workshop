"""Evaluation helpers shared by training script and notebook."""
import time

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def evaluate_head(y_true_idx, probs, labels) -> dict:
    pred = probs.argmax(1)
    rep = classification_report(y_true_idx, pred, labels=list(range(len(labels))),
                                target_names=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true_idx, pred)),
        "macro_f1": float(f1_score(y_true_idx, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_idx, pred, average="weighted", zero_division=0)),
        "per_class": {l: {k: float(rep[l][k]) for k in ("precision", "recall", "f1-score", "support")} for l in labels},
        "confusion_matrix": confusion_matrix(y_true_idx, pred, labels=list(range(len(labels)))).tolist(),
    }


def evaluate_model(model, texts, cat_idx, urg_idx) -> dict:
    pc, pu = model.predict_proba(texts)
    res = {"category": evaluate_head(cat_idx, pc, model.categories), "urgency": evaluate_head(urg_idx, pu, model.urgencies)}
    res["mean_macro_f1"] = (res["category"]["macro_f1"] + res["urgency"]["macro_f1"]) / 2
    return res


def latency_ms(model, texts, n: int = 200) -> dict:
    """Single-request latency (batch size 1), the shape of a real API call."""
    texts = [texts[i % len(texts)] for i in range(n)]
    model.predict_proba(texts[:5])                       # warm-up
    times = []
    for t in texts:
        s = time.perf_counter()
        model.predict_proba([t])
        times.append((time.perf_counter() - s) * 1000)
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95))}
