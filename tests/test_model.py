import time
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from src.utils.preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
TEST_CSV = ROOT / "data" / "processed" / "test.csv"


@pytest.fixture(scope="module")
def model():
    vec = joblib.load(MODELS / "tfidf_vectorizer.joblib")
    clf = joblib.load(MODELS / "sentiment_model.joblib")
    return vec, clf


def predict(model, texts):
    vec, clf = model
    return clf.predict(vec.transform([clean_text(t) for t in texts]))


@pytest.mark.skipif(not TEST_CSV.exists(), reason="test.csv absent : lancer clean_data")
def test_performance_test_set(model):
    df = pd.read_csv(TEST_CSV).dropna()
    pred = predict(model, df["text"].tolist())
    assert accuracy_score(df["label"], pred) >= 0.80
    assert (f1_score(df["label"], pred, average=None) > 0.75).all()


EDGE_CASES = {
    "tres_court": "ok",
    "emojis_seuls": "😍😍😍",
    "tres_long": "great video " * 500,
    "emojis_et_texte": "I love it 😍🔥",
    "langues_melangees": "This video est vraiment top, merci!",
    "url_seule": "https://youtube.com",
    "majuscules": "THIS IS THE WORST",
}


@pytest.mark.parametrize("text", EDGE_CASES.values(), ids=EDGE_CASES.keys())
def test_cas_limites(model, text):
    assert predict(model, [text])[0] in (-1, 0, 1)


def test_sentiments_evidents(model):
    pred = predict(model, ["I absolutely love this, amazing work!",
                           "This is terrible, worst video ever"])
    assert pred[0] == 1 and pred[1] == -1


def test_temps_inference(model):
    vec, clf = model
    batch = [clean_text(f"This is sample comment number {i}") for i in range(50)]
    clf.predict_proba(vec.transform(batch))          # échauffement
    t0 = time.perf_counter()
    for _ in range(20):
        clf.predict_proba(vec.transform(batch))
    ms = (time.perf_counter() - t0) / 20 * 1000
    print(f"\nInférence batch de 50 : {ms:.2f} ms")
    assert ms < 100