import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.utils.preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
NAMES = ["Négatif", "Neutre", "Positif"]

vec = joblib.load(MODELS / "tfidf_vectorizer.joblib")
clf = joblib.load(MODELS / "sentiment_model.joblib")
test = pd.read_csv(ROOT / "data" / "processed" / "test.csv").dropna()

X = vec.transform(test["text"].apply(clean_text))
pred = clf.predict(X)
acc = accuracy_score(test["label"], pred)
f1 = f1_score(test["label"], pred, average=None)

batch = vec.transform(test["text"].iloc[:50].apply(clean_text))
t0 = time.perf_counter()
for _ in range(20):
    clf.predict_proba(batch)
ms = (time.perf_counter() - t0) / 20 * 1000

params = json.loads((MODELS / "metrics.json").read_text(encoding="utf-8"))["best_params"]
print(f"Modèle : {type(clf).__name__} | meilleurs paramètres : {params}\n")
print(classification_report(test["label"], pred, target_names=NAMES, digits=4))
print("=== Critères du TP ===")
print(f"Accuracy >= 0.80        : {acc:.4f}  {'OK' if acc >= 0.80 else 'KO'}")
print(f"F1 par classe > 0.75    : {'OK' if (f1 > 0.75).all() else 'KO'}  {f1.round(4)}")
print(f"Inférence < 100 ms (50) : {ms:.2f} ms  {'OK' if ms < 100 else 'KO'}")