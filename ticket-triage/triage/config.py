"""Central configuration. Everything tunable lives here or in env vars."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = Path(os.getenv("TRIAGE_ARTIFACTS", ROOT / "artifacts"))

SEED = 42
CATEGORIES = ["Billing", "Technical", "Account", "General"]
URGENCIES = ["Low", "Medium", "High"]

# Tickets whose weakest head is below this confidence are flagged for a human.
REVIEW_THRESHOLD = float(os.getenv("TRIAGE_REVIEW_THRESHOLD", "0.60"))

# Share of tokens seen during training. In-domain tickets score >= 0.90 (99th pct
# on held-out data); off-topic text scores far lower. Below this -> human review.
MIN_VOCAB_COVERAGE = float(os.getenv("TRIAGE_MIN_COVERAGE", "0.80"))

MAX_TEXT_CHARS = 5000
MAX_SEQ_LEN = 80

# BiLSTM hyper-parameters
EMB_DIM = 128
HIDDEN = 128
DROPOUT = 0.3
LR = 2e-3
BATCH = 64
MAX_EPOCHS = 20
PATIENCE = 3

# A neural model must beat the baseline by this much (mean macro-F1 on
# validation) to be selected; otherwise the simpler model wins.
SELECTION_MARGIN = 0.01
