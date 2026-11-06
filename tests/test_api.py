import time

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # "with" déclenche le lifespan, donc le chargement du modèle
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


@pytest.mark.parametrize("size", [1, 10, 50, 100, 500])
def test_tailles_de_batch(client, size):
    comments = [f"I really like this video {i}" for i in range(size)]
    t0 = time.perf_counter()
    r = client.post("/predict_batch", json={"comments": comments})
    ms = (time.perf_counter() - t0) * 1000
    print(f"\nBatch de {size:>3} : {ms:.1f} ms")
    assert r.status_code == 200
    stats = r.json()["stats"]
    assert len(r.json()["predictions"]) == size
    assert stats["positive"] + stats["neutral"] + stats["negative"] == size


@pytest.mark.parametrize(
    "payload",
    [{"comments": []}, {"comments": ["", "  "]}, {"texte": ["abc"]},
     {"comments": "pas une liste"}, {"comments": ["x"] * 501}],
    ids=["batch_vide", "tous_vides", "champ_absent", "mauvais_type", "trop_grand"],
)
def test_donnees_invalides(client, payload):
    assert client.post("/predict_batch", json=payload).status_code == 422


def test_structure_reponse(client):
    p = client.post("/predict_batch", json={"comments": ["Great!", "Awful"]}).json()["predictions"][0]
    assert set(p) == {"text", "sentiment", "label", "confidence"}
    assert 0 <= p["confidence"] <= 1