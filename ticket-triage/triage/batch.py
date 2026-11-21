"""Score many tickets at once, no server needed.

    python -m triage.batch --input sample_tickets.csv --output predictions.csv
    python -m triage.batch --input tickets.csv --column message     # custom text column
"""
import argparse

import pandas as pd

from .inference import Predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="sample_tickets.csv", help="CSV file containing the ticket text")
    ap.add_argument("--column", default="text", help="name of the text column (default: text)")
    ap.add_argument("--output", default="predictions.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.input)
    if a.column not in df.columns:
        raise SystemExit(f"Column '{a.column}' not found. Available: {list(df.columns)}")
    texts = df[a.column].fillna("").astype(str).tolist()
    p = Predictor()
    res = p.predict([t if t.strip() else "." for t in texts])
    out = df.copy()
    out["category"] = [r["category"]["label"] for r in res]
    out["category_conf"] = [r["category"]["confidence"] for r in res]
    out["urgency"] = [r["urgency"]["label"] for r in res]
    out["urgency_conf"] = [r["urgency"]["confidence"] for r in res]
    out["needs_human_review"] = [r["needs_human_review"] for r in res]
    out["review_reasons"] = [",".join(r["review_reasons"]) for r in res]
    out.to_csv(a.output, index=False)

    show = out[[a.column, "category", "category_conf", "urgency", "urgency_conf", "needs_human_review"]].copy()
    show[a.column] = show[a.column].str.slice(0, 55)
    print(f"model: {p.name}   tickets: {len(out)}   flagged for review: {int(out.needs_human_review.sum())}\n")
    print(show.to_string(index=False))
    print(f"\nsaved -> {a.output}")


if __name__ == "__main__":
    main()
