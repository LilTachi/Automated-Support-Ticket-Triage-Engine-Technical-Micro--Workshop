import pytest
from fastapi.testclient import TestClient

from triage.api import app
from triage.preprocess import clean_text, tokenize


def test_clean_text_masks_and_keeps_shouting():
    out = clean_text("URGENT!! mail me at a@b.com, order 12345 http://x.io")
    assert "<email>" in out and "<num>" in out and "<url>" in out and out.endswith("<caps>")


def test_clean_text_handles_non_strings():
    assert clean_text(None) == "" and clean_text(float("nan")) == ""
    assert tokenize("hello world!") == ["hello", "world", "!"]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_shape_and_probabilities(client):
    r = client.post("/predict", json={"text": "The whole platform is down for our team, URGENT!"})
    assert r.status_code == 200
    body = r.json()
    assert body["category"]["label"] == "Technical" and body["urgency"]["label"] == "High"
    assert abs(sum(body["category"]["probabilities"].values()) - 1) < 1e-2
    assert 0 <= body["urgency"]["confidence"] <= 1


def test_predict_billing_low(client):
    body = client.post("/predict", json={"text": "Hi, could you send me a copy of my receipt? no rush"}).json()
    assert body["category"]["label"] == "Billing" and body["urgency"]["label"] == "Low"


@pytest.mark.parametrize("bad", ["", "   ", "x" * 5001])
def test_predict_rejects_bad_input(client, bad):
    assert client.post("/predict", json={"text": bad}).status_code == 422


def test_gibberish_does_not_crash(client):
    r = client.post("/predict", json={"text": "asdkjh qwe 🙂🙂 zzzz"})
    assert r.status_code == 200


def test_batch(client):
    r = client.post("/predict/batch", json={"texts": ["my password reset email never arrives", "do you have student discounts"]})
    assert r.status_code == 200 and len(r.json()) == 2


def test_offtopic_text_is_flagged_for_review(client):
    body = client.post("/predict", json={"text": "what is the best pizza in town"}).json()
    assert body["needs_human_review"] and "unfamiliar_text" in body["review_reasons"]
    clean = client.post("/predict", json={"text": "The whole platform is down for our team, URGENT!"}).json()
    assert not clean["needs_human_review"]


def test_web_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "Support ticket triage" in r.text and "/predict/batch" in r.text
