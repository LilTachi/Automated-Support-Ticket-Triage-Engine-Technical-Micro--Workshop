"""Data: synthetic generator, optional Kaggle loader, cleaning and splitting.

The synthetic generator is deliberately *imperfect* so metrics are not trivially
100%: ~15% of tickets describe an issue one severity level away from their
label, ~8% carry a distractor sentence from another category, urgency cue words
are sometimes missing or misleading, there are typos, and ~3% of category labels
are flipped (annotator error). Class imbalance is built in.
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CATEGORIES, DATA_DIR, SEED, URGENCIES
from .preprocess import clean_text

# ---- phrase banks: ISSUES[category][severity 0..2] --------------------------
ISSUES = {
    "Billing": [
        ["I have a question about the line items on my last invoice",
         "Could you explain the tax on my latest statement",
         "I would like a copy of my receipt from last month",
         "How do I update the billing address on my invoice"],
        ["I was charged for a plan I did not choose",
         "My refund from two weeks ago still has not arrived",
         "The discount code was not applied at checkout",
         "There is an unexpected fee on my subscription"],
        ["I was charged twice and my account is now overdrawn",
         "A fraudulent charge from your company appeared on my card",
         "You billed me ten times the agreed amount and my payment bounced",
         "Our payroll run failed because of your billing error"],
    ],
    "Technical": [
        ["The dark mode toggle looks slightly misaligned",
         "I have a question about the export feature",
         "Is there a keyboard shortcut for the search bar",
         "The font in the mobile app looks small on my tablet"],
        ["The app crashes whenever I open the reports page",
         "Notifications are delayed by several hours",
         "File uploads keep failing for large documents",
         "The dashboard loads very slowly since the last update"],
        ["The whole platform is down for our entire team",
         "We are losing data, every save returns an error",
         "The API returns 500 errors for all production requests",
         "Our checkout is completely broken and customers cannot pay"],
    ],
    "Account": [
        ["How do I change my display name",
         "I want to update the phone number on my profile",
         "Can I merge two of my workspaces",
         "Where do I find the language setting for my account"],
        ["I cannot reset my password, the email never arrives",
         "My two factor codes are not being accepted",
         "I was removed from my team workspace by mistake",
         "My profile shows the wrong subscription tier"],
        ["My account was hacked and the email address was changed",
         "I am locked out of my admin account and our deadline is today",
         "Someone else is logging into my account right now",
         "All of our staff accounts were suspended suddenly"],
    ],
    "General": [
        ["Do you offer student discounts",
         "I am curious whether you have a partner program",
         "Where can I find your documentation",
         "What are your support hours"],
        ["I have been waiting a week for a reply on my partnership request",
         "Please follow up on my earlier ticket, nobody has answered",
         "I need a formal quote for our company by Friday",
         "Your sales team promised a callback that never happened"],
        ["I need to speak to a manager immediately about a legal matter",
         "This is a formal complaint and I am escalating to my lawyer",
         "Our contract renewal expires today and nobody responds",
         "I am a journalist and we publish our article in two hours"],
    ],
}
CUES = [
    ["no rush", "whenever you get a chance", "just curious", "when you have time"],
    ["please help when possible", "this is getting frustrating", "I would appreciate a quick reply", "I need this sorted this week"],
    ["URGENT", "ASAP", "this is critical", "I need help right now", "this cannot wait"],
]
GREET = ["Hi team,", "Hello,", "Hi,", "Good morning,", "Dear support,", ""]
CLOSE = ["Thanks", "Thank you", "Regards", "Cheers", ""]
NAMES = ["Ana", "Ben", "Carla", "Dev", "Elena", "Farid", "Grace", "Hugo", "Ines", "Jon"]
CAT_P = [0.27, 0.40, 0.21, 0.12]                       # imbalanced categories
URG_P = {"Billing": [.40, .40, .20], "Technical": [.35, .40, .25],
         "Account": [.45, .40, .15], "General": [.65, .28, .07]}


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 5 or rng.random() > 0.03:
        return word
    i = rng.randrange(1, len(word) - 1)
    return word[:i] + word[i + 1:] if rng.random() < 0.5 else word[:i] + word[i + 1] + word[i] + word[i + 2:]


def _one(rng: random.Random, i: int) -> dict:
    cat = rng.choices(CATEGORIES, CAT_P)[0]
    urg_i = rng.choices(range(3), URG_P[cat])[0]
    # ambiguity: sometimes the described issue is one severity step off
    sev = urg_i if rng.random() < 0.85 else min(2, max(0, urg_i + rng.choice([-1, 1])))
    parts = [rng.choice(ISSUES[cat][sev])]
    if rng.random() < 0.08:                              # distractor from another category
        other = rng.choice([c for c in CATEGORIES if c != cat])
        parts.append(rng.choice(ISSUES[other][rng.randrange(3)]))
    if rng.random() < 0.5:
        parts.append(f"My order number is {rng.randint(10000, 99999)}")
    r = rng.random()                                     # urgency cue: right / adjacent / absent
    if r < 0.65:
        cue = rng.choice(CUES[urg_i])
    elif r < 0.75:
        cue = rng.choice(CUES[min(2, max(0, urg_i + rng.choice([-1, 1])))])
    else:
        cue = ""
    if cue:
        parts.insert(rng.choice([0, len(parts)]), cue)
    body = ". ".join(p[0].upper() + p[1:] if not p.isupper() else p for p in parts) + rng.choice([".", "!", "..."])
    if urg_i == 2 and rng.random() < 0.3:
        body += "!!"
    text = " ".join(x for x in [rng.choice(GREET), body, rng.choice(CLOSE), rng.choice(NAMES) if rng.random() < .5 else ""] if x)
    text = " ".join(_typo(w, rng) for w in text.split())
    if rng.random() < 0.1:
        text = text.lower()
    label_cat = cat if rng.random() > 0.03 else rng.choice([c for c in CATEGORIES if c != cat])  # label noise
    return {"ticket_id": f"T{100000 + i}", "text": text, "category": label_cat, "urgency": URGENCIES[urg_i]}


def generate_synthetic(n: int = 6000, seed: int = SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    df = pd.DataFrame([_one(rng, i) for i in range(n)])
    # inject realistic mess for the cleaning step: duplicates + empty/null texts
    dups = df.sample(int(n * 0.015), random_state=seed).copy()
    bad = df.sample(int(n * 0.005), random_state=seed + 1).copy()
    bad["text"] = np.where(np.arange(len(bad)) % 2 == 0, "", None)
    return pd.concat([df, dups, bad], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def load_kaggle(path: str | Path) -> pd.DataFrame:
    """Adapter for the Kaggle 'Customer Support Ticket Dataset'
    (columns: Ticket Subject, Ticket Description, Ticket Type, Ticket Priority).
    Ticket Type -> category, Priority (Critical folded into High) -> urgency.
    Caution: check label learnability first; that dataset is largely synthetic."""
    raw = pd.read_csv(path)
    missing = [c for c in ("Ticket Subject", "Ticket Description", "Ticket Type", "Ticket Priority") if c not in raw.columns]
    if missing:
        raise ValueError(f"Kaggle CSV is missing columns: {missing}. Found: {list(raw.columns)}")
    df = pd.DataFrame({
        "ticket_id": raw.get("Ticket ID", pd.Series(range(len(raw)))).astype(str),
        "text": (raw["Ticket Subject"].fillna("") + ". " + raw["Ticket Description"].fillna("")).str.strip(),
        "category": raw["Ticket Type"],
        "urgency": raw["Ticket Priority"].replace({"Critical": "High"}),
    })
    return df


def load_or_create(path: str | Path | None = None, n: int = 6000) -> pd.DataFrame:
    if path:
        p = Path(path)
        return load_kaggle(p) if "Ticket Subject" in pd.read_csv(p, nrows=1).columns else pd.read_csv(p)
    p = DATA_DIR / "tickets.csv"
    if p.exists():
        return pd.read_csv(p)
    DATA_DIR.mkdir(exist_ok=True)
    df = generate_synthetic(n)
    df.to_csv(p, index=False)
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop empty texts, normalise, de-duplicate on the cleaned text, drop unknown labels."""
    out = df.copy()
    out["clean"] = out["text"].map(clean_text)
    out = out[out["clean"].str.len() > 0]
    out = out.drop_duplicates(subset=["clean", "category", "urgency"])
    out = out[out["category"].notna() & out["urgency"].notna()]
    return out.reset_index(drop=True)


def split(df: pd.DataFrame, seed: int = SEED):
    """70/15/15 stratified on the joint (category, urgency) label so that rare
    combinations appear in every split."""
    strat = df["category"] + "|" + df["urgency"]
    train, rest = train_test_split(df, test_size=0.30, stratify=strat, random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=(rest["category"] + "|" + rest["urgency"]), random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def label_sets(df: pd.DataFrame):
    """Derive label lists from the data so any dataset works. Keeps the
    configured order when the data uses (a subset of) the default labels;
    otherwise orders categories by frequency and urgencies Low<Medium<High."""
    def order(found, default, by_freq):
        found_set = set(found)
        if found_set <= set(default):
            return [l for l in default if l in found_set]
        return list(by_freq)
    cats = order(df["category"].unique(), CATEGORIES, df["category"].value_counts().index)
    canon = ["Low", "Medium", "High"]
    urgs = order(df["urgency"].unique(), URGENCIES, [u for u in canon if u in set(df["urgency"])] + sorted(set(df["urgency"]) - set(canon)))
    return cats, urgs
