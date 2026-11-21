
# Support Ticket Triage Engine

An end-to-end NLP system that reads a raw customer-support ticket and predicts

- **Category**: `Billing`, `Technical`, `Account` or `General`
- **Urgency**: `Low`, `Medium` or `High`

along with a **confidence score** for each prediction and a **human-review flag** for tickets the model should not decide alone.
It includes the data pipeline, a PyTorch model compared against a classical baseline, and a FastAPI service.

```
raw text -> clean_text -> model (TF-IDF + LogReg  |  multi-task BiLSTM) -> labels + confidences + review flag
```

Example: `"I was charged twice and my payment bounced, URGENT!"` -> **Billing** (0.99), **High** (1.00), no review needed.

---

## Deliverables checklist

| Requirement | Where to find it |
|---|---|
| Exploratory notebook: EDA, baseline comparison, confusion matrices, F1 / precision / recall | [`notebooks/01_eda_and_evaluation.ipynb`](notebooks/01_eda_and_evaluation.ipynb) (already executed; outputs are visible on GitHub) |
| Production Python script: **training** | [`triage/train.py`](triage/train.py) (`python -m triage.train`) |
| Production Python script: **inference** | [`triage/inference.py`](triage/inference.py) (`Predictor` class and CLI) |
| Production Python script: **/predict API** | [`triage/api.py`](triage/api.py) (FastAPI, `POST /predict`) |
| `requirements.txt` | [`requirements.txt`](requirements.txt) |
| Clear instructions to run locally | [Quick start](#quick-start) below |
| Data pipeline: cleaning, EDA, class imbalance, train/val/test split | [`triage/data.py`](triage/data.py), [`triage/preprocess.py`](triage/preprocess.py), notebook sections 2-5 |
| Neural model (PyTorch) + simpler baseline + justified final choice | [`triage/models.py`](triage/models.py), [Results](#results-and-model-choice) |

The training slide deck and the micro-teaching video (Part B) are submitted alongside this repository.

---

## Quick start

### Prerequisites
- **Python 3.10 or newer** (developed and tested on 3.12). Check with `python --version`.
- **About 1 GB of free disk space** (PyTorch is the large download).
- No GPU needed. Everything runs on a normal CPU.

### 1. Get the code
```bash
git clone <your-repo-url>
cd ticket-triage
```
All commands below must be run **from inside this `ticket-triage` folder** (the one that contains `triage/` and `requirements.txt`).

### 2. Create a virtual environment and install
**Windows (PowerShell):**
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope Process Bypass` and try again.

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Verify the installation
```bash
python -m pytest -q
```
Expected: `12 passed`.

### 4. Start the API
```bash
uvicorn triage.api:app --port 8000
```
Wait for `Application startup complete.`, then leave this terminal running. Pre-trained model files are included in `artifacts/`, so **no training is required** to try the API.

### 5. Try it in the browser
Open **http://localhost:8000/**. This is a small built-in page for the API (it works offline, nothing to install):

- **One ticket** tab: paste a message and click **Analyze ticket** (or press Ctrl+Enter). You get the category and urgency, how sure the model is, a bar showing the probability of every class, and a verdict: *Safe to route automatically* or *Send to a person*. Four example buttons are provided; **Off-topic** shows the review flag in action.
- **Many tickets** tab: paste one ticket per line (up to 64), click **Analyze all tickets**, read the results table, and use **Download CSV** to save it.
- **Show raw API response** under each result displays the exact JSON the endpoint returned.

The page simply calls the JSON endpoints described below. It does not change them.

### 6. Or call the API directly
Interactive Swagger docs are at **http://localhost:8000/docs**: click `POST /predict`, **Try it out**, paste `{"text": "I was charged twice and my payment bounced, URGENT!"}` and click **Execute**. From a second terminal:

| System | Command |
|---|---|
| macOS / Linux | `curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"text": "My password reset email never arrives"}'` |
| Windows PowerShell | `Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType "application/json" -Body '{"text": "My password reset email never arrives"}'` |

> Opening `/predict` in a browser address bar shows **"Method Not Allowed"**. That is expected: the endpoint only accepts **POST**. Use the page at `/`, `/docs`, or one of the commands above.

---

## Using the API

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Web page for trying the API (see Quick start, step 5) |
| `/predict` | POST | One ticket. Body: `{"text": "..."}` (1 to 5000 characters) |
| `/predict/batch` | POST | Up to 64 tickets. Body: `{"texts": ["...", "..."]}`. Returns a list in the same order |
| `/health` | GET | Liveness check and active model |
| `/metadata` | GET | Selected model and its test metrics |
| `/docs` | GET | Interactive Swagger UI |

### Response

```json
{
  "category": {"label": "Billing", "confidence": 0.99,
               "probabilities": {"Billing": 0.99, "Technical": 0.005, "Account": 0.003, "General": 0.002}},
  "urgency":  {"label": "High", "confidence": 1.0,
               "probabilities": {"Low": 0.0, "Medium": 0.0, "High": 1.0}},
  "needs_human_review": false,
  "review_reasons": [],
  "model": "bilstm",
  "latency_ms": 2.4
}
```

| Field | Meaning |
|---|---|
| `label` | The predicted class for each head |
| `confidence` | Probability of the predicted class (0 to 1) |
| `probabilities` | Full distribution over all classes; use it to spot close calls |
| `needs_human_review` | `true` means do not auto-route this ticket |
| `review_reasons` | `low_confidence` (lower of the two confidences is below 0.60) and/or `unfamiliar_text` (fewer than 80% of the words were seen in training) |
| `latency_ms` | Model time for this request (average per ticket in batch mode) |

Invalid input (empty, whitespace-only or over 5000 characters) returns HTTP `422`.

---

## Score many tickets at once (no server needed)

```bash
python -m triage.batch                                   # uses sample_tickets.csv
python -m triage.batch --input mytickets.csv --column message --output results.csv
```
Prints a table and saves a CSV with the predictions, confidences and review flags.

## Retrain the model

```bash
python -m triage.train
```
Takes about a minute on a laptop CPU. It generates the synthetic dataset (`data/tickets.csv`) if missing, then cleans, splits, trains both models, evaluates, selects the final model and writes everything to `artifacts/`. The run is seeded and reproducible.
Use `--artifacts my_folder` to keep earlier models, and `--data file.csv` to train on your own CSV with columns `text, category, urgency`.

---

## Results and model choice

Held-out **test set** (894 tickets), synthetic data:

| Model | Category macro-F1 | Urgency macro-F1 | Mean macro-F1 | p50 / p95 latency* |
|---|---|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 0.928 | 0.904 | 0.916 | 0.7 / 0.8 ms |
| **Multi-task BiLSTM (selected)** | **0.970** | 0.902 | **0.936** | 0.7 / 1.3 ms |

\*Model only, batch size 1, one CPU core. The HTTP round trip adds a few milliseconds.

**Why the BiLSTM.** The selection rule is in code (`triage/train.py`): the neural model ships only if it beats the baseline on **validation** mean macro-F1 by at least 0.01; otherwise the simpler model is used. The BiLSTM cleared the bar (+0.022). The gain is almost entirely in Category (0.928 to 0.970); Urgency is effectively a tie, and both models answer in under a millisecond, so latency does not decide it. Repeating BiLSTM training with three different random seeds on the same split gave test Category macro-F1 of 0.966 to 0.969 and mean macro-F1 of 0.934 to 0.947, so the gain is not a lucky seed. It is not mainly about typos either: the BiLSTM is ahead both on tickets with only known words (accuracy 0.979 vs 0.949) and on tickets with unseen words (0.956 vs 0.881). The tickets it fixes are mostly ones that mention two topics, which is consistent with it using word order (the main complaint comes first in this synthetic data) where the bag-of-words baseline cannot. That explanation is consistent with the evidence but untested, and it may not carry over to real tickets. The test set is used once, for reporting only.

Full per-class precision, recall and F1, confusion matrices, training curves, error analysis and the review-threshold trade-off are in the notebook.

---

## How it works

1. **Cleaning** (`triage/preprocess.py`): masks emails, URLs and numbers, lowercases, and keeps ALL-CAPS shouting as a `<caps>` token because it is a strong urgency signal. The same function is used in training and serving, which prevents train/serve skew.
2. **Data** (`triage/data.py`): a synthetic generator with deliberate realism (typos, ambiguous severity, distractor sentences, duplicates, empty rows, about 3% label noise) so scores are not trivially perfect. Rows are cleaned and de-duplicated, then split **70/15/15, stratified on the joint (category, urgency) label**.
3. **Class imbalance**: stratified splits, class-weighted loss, and **macro-F1** as the headline metric. In the notebook, weighting did not measurably help on this mildly imbalanced data; it stays on as insurance and should be re-tested on real data.
4. **Models** (`triage/models.py`):
   - Baseline: TF-IDF (1-2 grams) + Logistic Regression, one per target.
   - Neural: Embedding -> BiLSTM -> max+mean pooling -> two heads (category, urgency). One shared encoder means a single forward pass answers both questions.
5. **Serving** (`triage/inference.py`, `triage/api.py`): the model loads once at startup; input is validated; every response carries confidences and a review flag. Inference is batch-independent: tickets are grouped by length so no padding is added, which means a ticket gets the same answer alone or inside a batch.
6. **Human-in-the-loop**: softmax confidence alone is not reliable on off-topic text (a neural net can be confidently wrong), so a vocabulary-coverage check also routes unfamiliar text to a person.

## Project structure

```
ticket-triage/
├── README.md
├── requirements.txt
├── sample_tickets.csv                      example input for the batch tool
├── triage/
│   ├── config.py                           labels, hyper-parameters, thresholds (env-overridable)
│   ├── preprocess.py                       cleaning, tokenizer, vocabulary
│   ├── data.py                             synthetic generator, loaders, cleaning, splitting
│   ├── models.py                           baseline + BiLSTM
│   ├── metrics.py                          evaluation and latency helpers
│   ├── train.py                            training, evaluation, model selection, export
│   ├── inference.py                        Predictor (used by API, batch tool, CLI)
│   ├── api.py                              FastAPI app (JSON API + serves the web page)
│   ├── static/index.html                   self-contained web page, no CDN or build step
│   └── batch.py                            score a CSV from the command line
├── notebooks/01_eda_and_evaluation.ipynb   EDA, baseline comparison, metrics
├── tests/test_core.py                      preprocessing, API and web-page tests
├── artifacts/                              trained models and metadata (created by train.py)
└── data/tickets.csv                        generated synthetic dataset
```

## Configuration

| Environment variable | Default | Effect |
|---|---|---|
| `TRIAGE_MODEL` | selected model | Force `baseline` or `bilstm` in the API |
| `TRIAGE_REVIEW_THRESHOLD` | `0.60` | Confidence below this sets `needs_human_review` |
| `TRIAGE_MIN_COVERAGE` | `0.80` | Minimum share of known words before text is flagged as unfamiliar |
| `TRIAGE_ARTIFACTS` | `artifacts/` | Folder to load models from |

Example (Windows PowerShell): `$env:TRIAGE_MODEL="baseline"; uvicorn triage.api:app --port 8000`
Example (macOS / Linux): `TRIAGE_MODEL=baseline uvicorn triage.api:app --port 8000`

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `No module named triage...` | You are not in the project root, or have an old copy. Run `cd` into the folder that contains `triage/` and check that `triage/batch.py` exists |
| `Method Not Allowed` on `/predict` | Browsers send GET. Use the page at `/`, `/docs`, or a POST command (see Quick start) |
| The page at `/` shows "Cannot reach the API" | The uvicorn server stopped or runs on another port. Restart it and reload the page |
| `/docs` page is blank | Swagger loads its assets from a CDN; offline, firewall or ad-blocker can block it. Use the page at `/` (it has no external dependencies) or the curl / PowerShell commands |
| `No trained artifacts ... Run: python -m triage.train` | The `artifacts/` folder is missing or `TRIAGE_ARTIFACTS` points elsewhere. Run `python -m triage.train` |
| Pickle / version warning when loading the model | Different scikit-learn or torch version than the one used for training. Run `python -m triage.train` to regenerate |
| `Address already in use` | Port 8000 is taken. Use `--port 8001` |
| `pip install` is slow or fails on torch | PyTorch is a large download. Retry on a stable connection, or install the CPU build from pytorch.org first, then rerun `pip install -r requirements.txt` |

## Limitations

- **Synthetic data.** Scores demonstrate that the pipeline works, not how it would perform on real customer tickets, which are longer, multi-topic and inconsistently labelled. One split and one seed: a point or two between models is within noise.
- **Urgency leans on cue words** such as "URGENT" and "no rush". A ticket like "charged twice, no rush" is labelled Low. Real deployments need a severity policy agreed with the support team.
- **Word-level vocabulary.** Unseen words and typos become unknown tokens. A pretrained sub-word model such as DistilBERT is the natural next step on real data; it is not implemented here.
- **No drift monitoring yet.** The intended first signals are the share of `unfamiliar_text` flags and the confidence distribution over time.
- **Optional public data.** The Kaggle "Customer Support Ticket Dataset" can be loaded with `python -m triage.train --data customer_support_tickets.csv --artifacts artifacts_kaggle`. The training script prints a signal check against shuffled labels; on that dataset it warns that the labels look unrelated to the text, which is why this project uses the synthetic dataset.
