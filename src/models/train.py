import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIG = ROOT / "reports" / "figures"
RS = 42
NAMES = ["Négatif", "Neutre", "Positif"]


def load_data():
    train = pd.read_csv(PROC / "train.csv").dropna()
    test = pd.read_csv(PROC / "test.csv").dropna()
    return train["text"], train["label"], test["text"], test["label"]


def make_pipeline(clf):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9,
                            sublinear_tf=True, max_features=50000)
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def compare_models(X, y):
    """Comparaison par validation croisée sur le train (le test reste intact)."""
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "LinearSVC": LinearSVC(class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                               n_jobs=-1, random_state=RS),
    }
    print("=== Comparaison des modèles (CV 3 folds, F1 macro) ===")
    for name, clf in candidates.items():
        t0 = time.perf_counter()
        scores = cross_val_score(make_pipeline(clf), X, y, cv=3,
                                 scoring="f1_macro", n_jobs=-1)
        print(f"{name:<20} F1 = {scores.mean():.4f} ± {scores.std():.4f}"
              f"  ({time.perf_counter() - t0:.0f}s)")


def tune_logreg(X, y):
    grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2],
        "clf__C": [0.5, 1, 5, 10],
    }
    search = GridSearchCV(
        make_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced")),
        grid, cv=3, scoring="f1_macro", n_jobs=-1, verbose=1,
    )
    search.fit(X, y)
    print(f"\nMeilleurs paramètres : {search.best_params_}")
    print(f"Meilleur F1 (CV)     : {search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_


def evaluate(pipe, X_test, y_test):
    pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1_per_class = f1_score(y_test, pred, average=None)
    print("\n=== Évaluation sur le test set ===")
    print(classification_report(y_test, pred, target_names=NAMES, digits=4))

    # Matrice de confusion
    FIG.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=NAMES,
                                            cmap="Blues", normalize="true")
    plt.title("Matrice de confusion (normalisée)")
    plt.tight_layout()
    plt.savefig(FIG / "confusion_matrix.png", dpi=150)
    plt.close()

    # Temps d'inférence : batch de 50 commentaires, moyenne sur 20 essais
    batch = X_test.iloc[:50].tolist()
    t0 = time.perf_counter()
    for _ in range(20):
        pipe.predict_proba(batch)
    infer_ms = (time.perf_counter() - t0) / 20 * 1000

    metrics = {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_score(y_test, pred, average="macro"), 4),
        "f1_per_class": {n: round(float(v), 4) for n, v in zip(NAMES, f1_per_class)},
        "inference_ms_batch50": round(infer_ms, 2),
    }
    print("=== Critères du TP ===")
    print(f"Accuracy >= 0.80        : {acc:.4f}  {'OK' if acc >= 0.80 else 'KO'}")
    print(f"F1 par classe > 0.75    : {'OK' if (f1_per_class > 0.75).all() else 'KO'}")
    print(f"Inférence < 100 ms (50) : {infer_ms:.2f} ms  {'OK' if infer_ms < 100 else 'KO'}")
    return metrics


def save(pipe, metrics, params):
    MODELS.mkdir(exist_ok=True)
    joblib.dump(pipe.named_steps["tfidf"], MODELS / "tfidf_vectorizer.joblib")
    joblib.dump(pipe.named_steps["clf"], MODELS / "sentiment_model.joblib")
    metrics["best_params"] = {k: str(v) for k, v in params.items()}
    (MODELS / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nModèle, vectoriseur et métriques sauvegardés dans {MODELS}")


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    compare_models(X_train, y_train)
    best_pipe, best_params = tune_logreg(X_train, y_train)
    metrics = evaluate(best_pipe, X_test, y_test)
    save(best_pipe, metrics, best_params)