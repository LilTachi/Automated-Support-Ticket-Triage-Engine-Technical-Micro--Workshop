"""Training entry point.

    python -m triage.train                       # synthetic data, full pipeline
    python -m triage.train --data path/to.csv    # your own CSV (text, category, urgency)
Outputs go to artifacts/: baseline.joblib, bilstm.pt, vocab.json, metadata.json
"""
import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .data import clean_dataframe, label_sets, load_or_create, split
from .metrics import evaluate_model, latency_ms
from .models import BaselineModel, BiLSTMModel, TicketNet, pad_batch
from .preprocess import Vocab


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def class_weights(idx: np.ndarray, n: int) -> torch.Tensor:
    counts = np.bincount(idx, minlength=n).astype(float)
    w = counts.sum() / (n * np.maximum(counts, 1))        # 'balanced' inverse-frequency
    return torch.tensor(w, dtype=torch.float)


def train_bilstm(train, val, cat_idx_tr, urg_idx_tr, cat_idx_va, urg_idx_va, cats, urgs, log=print):
    vocab = Vocab.build(train["clean"])
    model = BiLSTMModel(vocab, cats, urgs)
    net = model.net
    opt = torch.optim.AdamW(net.parameters(), lr=C.LR, weight_decay=1e-2)
    ce_cat = nn.CrossEntropyLoss(weight=class_weights(cat_idx_tr, len(cats)))
    ce_urg = nn.CrossEntropyLoss(weight=class_weights(urg_idx_tr, len(urgs)))
    seqs = [vocab.encode(t, C.MAX_SEQ_LEN) for t in train["clean"]]
    ycat, yurg = torch.tensor(cat_idx_tr), torch.tensor(urg_idx_tr)

    best, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(1, C.MAX_EPOCHS + 1):
        net.train()
        order = np.random.permutation(len(seqs))
        total = 0.0
        for i in range(0, len(order), C.BATCH):
            b = order[i:i + C.BATCH]
            lc, lu = net(pad_batch([seqs[j] for j in b]))
            loss = ce_cat(lc, ycat[b]) + ce_urg(lu, yurg[b])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); total += loss.item() * len(b)
        score = evaluate_model(model, list(val["clean"]), cat_idx_va, urg_idx_va)["mean_macro_f1"]
        history.append({"epoch": epoch, "train_loss": total / len(order), "val_mean_macro_f1": score})
        log(f"epoch {epoch:2d}  loss {total/len(order):.4f}  val mean macro-F1 {score:.4f}")
        if score > best + 1e-4:
            best, bad = score, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= C.PATIENCE:
                log("early stopping"); break
    net.load_state_dict(best_state)
    return model, vocab, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="CSV with text,category,urgency (or the Kaggle ticket CSV)")
    ap.add_argument("--n", type=int, default=6000, help="synthetic dataset size")
    ap.add_argument("--artifacts", default=None, help="output folder (default: artifacts/). Use a new name to keep earlier models")
    args = ap.parse_args()
    if args.artifacts:
        C.ARTIFACTS_DIR = Path(args.artifacts)

    seed_everything(C.SEED)
    C.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_or_create(args.data, args.n)
    df = clean_dataframe(raw)
    print(f"raw rows {len(raw)} -> clean rows {len(df)}")
    train, val, test = split(df)
    cats, urgs = label_sets(df)
    print("categories:", cats, "| urgencies:", urgs)
    enc = lambda d: (d["category"].map(cats.index).to_numpy(), d["urgency"].map(urgs.index).to_numpy())
    (ctr, utr), (cva, uva), (cte, ute) = enc(train), enc(val), enc(test)

    # ---- baseline
    print("\n== Baseline: TF-IDF + Logistic Regression")
    base = BaselineModel(cats, urgs).fit(train["clean"], train["category"], train["urgency"])
    # ---- neural
    print("\n== BiLSTM multi-task")
    lstm, vocab, history = train_bilstm(train, val, ctr, utr, cva, uva, cats, urgs)

    results = {}
    for m in (base, lstm):
        results[m.name] = {
            "val": evaluate_model(m, list(val["clean"]), cva, uva),
            "test": evaluate_model(m, list(test["clean"]), cte, ute),
            "latency": latency_ms(m, list(test["clean"])),
        }
        r = results[m.name]
        print(f"{m.name:9s} val {r['val']['mean_macro_f1']:.4f}  test {r['test']['mean_macro_f1']:.4f}  "
              f"p50 {r['latency']['p50_ms']:.2f}ms p95 {r['latency']['p95_ms']:.2f}ms")

    # ---- sanity check: is there any signal in the text at all?
    # Train the same baseline on SHUFFLED labels. If the real model is not clearly
    # better than a model trained on noise, the labels are unrelated to the text.
    rng = np.random.default_rng(C.SEED)
    noise = BaselineModel(cats, urgs).fit(train["clean"], rng.permutation(train["category"].to_numpy()),
                                          rng.permutation(train["urgency"].to_numpy()))
    noise_res = evaluate_model(noise, list(val["clean"]), cva, uva)
    for head in ("category", "urgency"):
        real, chance = results["baseline"]["val"][head]["macro_f1"], noise_res[head]["macro_f1"]
        print(f"signal check [{head}]: macro-F1 real {real:.3f} vs shuffled-labels {chance:.3f}")
        if real < chance + 0.05:
            print(f"WARNING: the {head} labels look unrelated to the text (no better than shuffled labels). "
                  f"No model can fix that: fix or re-derive the labels before modelling.")

    # ---- selection on VALIDATION only (test is touched once, for reporting)
    gain = results["bilstm"]["val"]["mean_macro_f1"] - results["baseline"]["val"]["mean_macro_f1"]
    selected = "bilstm" if gain > C.SELECTION_MARGIN else "baseline"
    print(f"\nBiLSTM val gain over baseline: {gain:+.4f} (margin {C.SELECTION_MARGIN}) -> selected: {selected}")

    joblib.dump(base, C.ARTIFACTS_DIR / "baseline.joblib")
    torch.save({"state_dict": lstm.net.state_dict(), "vocab_size": len(vocab), "n_cat": len(cats), "n_urg": len(urgs)}, C.ARTIFACTS_DIR / "bilstm.pt")
    (C.ARTIFACTS_DIR / "vocab.json").write_text(json.dumps(vocab.itos))
    (C.ARTIFACTS_DIR / "metadata.json").write_text(json.dumps({
        "selected_model": selected, "selection_margin": C.SELECTION_MARGIN, "val_gain": gain,
        "categories": cats, "urgencies": urgs, "review_threshold": C.REVIEW_THRESHOLD,
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "results": results, "bilstm_history": history,
    }, indent=2))
    # keep the held-out split so the notebook evaluates exactly the same rows
    for name, d in (("train", train), ("val", val), ("test", test)):
        d.to_csv(C.ARTIFACTS_DIR / f"split_{name}.csv", index=False)
    print(f"artifacts saved to {C.ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
