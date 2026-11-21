"""FastAPI service.   uvicorn triage.api:app --port 8000"""
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from . import __version__
from . import config as C
from .inference import Predictor

log = logging.getLogger("triage.api")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["predictor"] = Predictor()          # load once at start-up, not per request
    log.info("loaded model: %s", state["predictor"].name)
    yield
    state.clear()


app = FastAPI(title="Support Ticket Triage API", version=__version__, lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(..., max_length=C.MAX_TEXT_CHARS, examples=["I was charged twice and my payment bounced, URGENT!"])

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty or whitespace")
        return v


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64)


class HeadOut(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


class PredictResponse(BaseModel):
    category: HeadOut
    urgency: HeadOut
    needs_human_review: bool
    review_reasons: list[str] = []
    model: str
    latency_ms: float


_UI = Path(__file__).parent / "static" / "index.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui():
    """Small self-contained web page (no CDN) for trying /predict and /predict/batch."""
    if not _UI.exists():
        raise HTTPException(404, "UI file missing; the JSON API at /docs still works.")
    return HTMLResponse(_UI.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {"status": "ok", "model": state["predictor"].name, "version": __version__}


@app.get("/metadata")
def metadata():
    m = state["predictor"].meta
    return {"selected_model": state["predictor"].name, "test_metrics": {k: {"mean_macro_f1": v["test"]["mean_macro_f1"], "latency": v["latency"]} for k, v in m["results"].items()}}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    res = state["predictor"].predict([req.text])[0]
    return {**res, "model": state["predictor"].name, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


@app.post("/predict/batch", response_model=list[PredictResponse])
def predict_batch(req: BatchRequest):
    if any(not t.strip() or len(t) > C.MAX_TEXT_CHARS for t in req.texts):
        raise HTTPException(422, "every text must be non-empty and <= %d chars" % C.MAX_TEXT_CHARS)
    t0 = time.perf_counter()
    res = state["predictor"].predict(req.texts)
    ms = round((time.perf_counter() - t0) * 1000 / len(res), 2)
    return [{**r, "model": state["predictor"].name, "latency_ms": ms} for r in res]
