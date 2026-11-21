"""Model definitions: sklearn baseline + multi-task BiLSTM (PyTorch).
Label lists are passed in (derived from the data), not hard-coded."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config as C


# ----------------------------------------------------------------- baseline
def make_baseline_pipeline() -> Pipeline:
    """TF-IDF (word 1-2 grams) + Logistic Regression. class_weight='balanced'
    is our imbalance handling: rare classes get proportionally larger loss."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                  token_pattern=r"<[a-z]+>|[a-z']+|[!?]")),
        ("clf", LogisticRegression(C=5.0, max_iter=2000, class_weight="balanced")),
    ])


class BaselineModel:
    """Two independent pipelines (one per target) behind a common interface."""
    name = "baseline"

    def __init__(self, categories=None, urgencies=None):
        self.categories = list(categories or C.CATEGORIES)
        self.urgencies = list(urgencies or C.URGENCIES)
        self.cat = make_baseline_pipeline()
        self.urg = make_baseline_pipeline()

    def fit(self, texts, cat_y, urg_y):
        self.cat.fit(texts, cat_y)
        self.urg.fit(texts, urg_y)
        return self

    def predict_proba(self, texts):
        """Returns (cat_probs, urg_probs), columns ordered as self.categories / self.urgencies."""
        def ordered(pipe, labels):
            p = pipe.predict_proba(texts)
            idx = [list(pipe.classes_).index(l) for l in labels]
            return p[:, idx]
        return ordered(self.cat, self.categories), ordered(self.urg, self.urgencies)


# ------------------------------------------------------------------- BiLSTM
class TicketNet(nn.Module):
    """Embedding -> BiLSTM -> [max-pool || mean-pool] -> two classification heads.

    Multi-task: one shared encoder, one head for category, one for urgency.
    Sharing the encoder is cheaper at inference (one forward pass) and lets the
    tasks regularise each other.
    """

    def __init__(self, vocab_size, n_cat, n_urg, emb=C.EMB_DIM, hidden=C.HIDDEN, dropout=C.DROPOUT):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.head_cat = nn.Linear(4 * hidden, n_cat)
        self.head_urg = nn.Linear(4 * hidden, n_urg)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)                                   # (B, T, 1)
        h, _ = self.lstm(self.drop(self.emb(x)))                        # (B, T, 2H)
        mx = h.masked_fill(~mask, -1e9).max(dim=1).values
        mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        z = self.drop(torch.cat([mx, mean], dim=-1))
        return self.head_cat(z), self.head_urg(z)


def pad_batch(seqs: list[list[int]]) -> torch.Tensor:
    L = max(len(s) for s in seqs)
    out = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s)
    return out


class BiLSTMModel:
    name = "bilstm"

    def __init__(self, vocab, categories=None, urgencies=None, net: TicketNet | None = None):
        self.vocab = vocab
        self.categories = list(categories or C.CATEGORIES)
        self.urgencies = list(urgencies or C.URGENCIES)
        self.net = net or TicketNet(len(vocab), len(self.categories), len(self.urgencies))

    @torch.no_grad()
    def predict_proba(self, texts, batch_size: int = 256):
        """Batch-independent inference. Tickets are grouped by length so no padding is
        ever added: with padding, the backward LSTM would read pad tokens first and a
        ticket's answer would depend on what else was in the same request."""
        self.net.eval()
        seqs = [self.vocab.encode(t, C.MAX_SEQ_LEN) for t in texts]
        cat = np.zeros((len(seqs), len(self.categories)), dtype=np.float32)
        urg = np.zeros((len(seqs), len(self.urgencies)), dtype=np.float32)
        by_len = {}
        for i, sq in enumerate(seqs):
            by_len.setdefault(len(sq), []).append(i)
        for idx in by_len.values():
            for j in range(0, len(idx), batch_size):
                chunk = idx[j:j + batch_size]
                lc, lu = self.net(torch.tensor([seqs[i] for i in chunk], dtype=torch.long))
                cat[chunk] = torch.softmax(lc, -1).numpy()
                urg[chunk] = torch.softmax(lu, -1).numpy()
        return cat, urg
