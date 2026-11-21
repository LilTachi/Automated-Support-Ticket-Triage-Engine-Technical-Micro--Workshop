"""Text cleaning, tokenisation and vocabulary. Shared by training AND serving
so that there is exactly one definition of 'cleaned text' (avoids train/serve skew)."""
import re
from collections import Counter

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_NUM = re.compile(r"\b\d[\d,.\-/]*\b")
_ALLCAPS = re.compile(r"\b[A-Z]{4,}\b")
_PLACEHOLDER = re.compile(r"\{[a-z_]+\}")
_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"<[a-z]+>|[a-z']+|[!?]")

PAD, UNK = "<pad>", "<unk>"


def clean_text(text: str) -> str:
    """Normalise raw ticket text.

    Shouting (ALL-CAPS words such as URGENT) is a strong urgency signal that
    lowercasing would destroy, so we keep it as an explicit <caps> token.
    """
    if not isinstance(text, str):
        return ""
    has_caps = bool(_ALLCAPS.search(text))
    t = _PLACEHOLDER.sub(" <field> ", text)          # unfilled template slots, e.g. {product_purchased}
    t = _URL.sub(" <url> ", t)
    t = _EMAIL.sub(" <email> ", t)
    t = _NUM.sub(" <num> ", t)
    t = t.lower()
    t = _WS.sub(" ", t).strip()
    return f"{t} <caps>" if has_caps and t else t


def tokenize(clean: str) -> list[str]:
    return _TOKEN.findall(clean)


class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    @classmethod
    def build(cls, cleaned_texts, min_freq: int = 2, max_size: int = 20000) -> "Vocab":
        counts = Counter(tok for t in cleaned_texts for tok in tokenize(t))
        words = [w for w, c in counts.most_common(max_size) if c >= min_freq]
        return cls([PAD, UNK] + words)

    def encode(self, clean: str, max_len: int) -> list[int]:
        ids = [self.stoi.get(tok, 1) for tok in tokenize(clean)][:max_len]
        return ids or [1]

    def __len__(self):
        return len(self.itos)
