"""Inference: load saved artifacts and expose one Predictor used by the API and CLI."""
import json
import os
from pathlib import Path

import joblib
import torch

from . import config as C
from .models import BiLSTMModel, TicketNet
from .preprocess import Vocab, clean_text, tokenize


class Predictor:
    def __init__(self, artifacts_dir: Path = C.ARTIFACTS_DIR, model_name: str | None = None):
        meta_path = Path(artifacts_dir) / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No trained artifacts in {artifacts_dir}. Run: python -m triage.train")
        self.meta = json.loads(meta_path.read_text())
        self.name = model_name or os.getenv("TRIAGE_MODEL") or self.meta["selected_model"]
        self.categories, self.urgencies = self.meta["categories"], self.meta["urgencies"]
        self.vocab = Vocab(json.loads((Path(artifacts_dir) / "vocab.json").read_text()))
        if self.name == "baseline":
            self.model = joblib.load(Path(artifacts_dir) / "baseline.joblib")
        elif self.name == "bilstm":
            vocab = self.vocab
            ckpt = torch.load(Path(artifacts_dir) / "bilstm.pt", map_location="cpu", weights_only=True)
            net = TicketNet(ckpt["vocab_size"], ckpt["n_cat"], ckpt["n_urg"])
            net.load_state_dict(ckpt["state_dict"])
            self.model = BiLSTMModel(vocab, self.categories, self.urgencies, net)
        else:
            raise ValueError(f"Unknown model '{self.name}' (use 'baseline' or 'bilstm')")

    @staticmethod
    def _head(probs_row, labels):
        i = int(probs_row.argmax())
        return {"label": labels[i], "confidence": round(float(probs_row[i]), 4),
                "probabilities": {l: round(float(p), 4) for l, p in zip(labels, probs_row)}}

    def coverage(self, clean: str) -> float:
        toks = tokenize(clean)
        return sum(t in self.vocab.stoi for t in toks) / len(toks) if toks else 0.0

    def predict(self, texts: list[str]) -> list[dict]:
        cleaned = [clean_text(t) for t in texts]
        pc, pu = self.model.predict_proba(cleaned)
        out = []
        for c_row, u_row, clean in zip(pc, pu, cleaned):
            cat, urg = self._head(c_row, self.categories), self._head(u_row, self.urgencies)
            reasons = []
            if min(cat["confidence"], urg["confidence"]) < C.REVIEW_THRESHOLD:
                reasons.append("low_confidence")
            if self.coverage(clean) < C.MIN_VOCAB_COVERAGE:   # softmax is confidently wrong off-topic
                reasons.append("unfamiliar_text")
            out.append({"category": cat, "urgency": urg,
                        "needs_human_review": bool(reasons), "review_reasons": reasons})
        return out


if __name__ == "__main__":      # quick CLI:  python -m triage.inference "my text"
    import sys
    p = Predictor()
    print(json.dumps(p.predict([" ".join(sys.argv[1:]) or "The API is down, URGENT!"])[0], indent=2))
