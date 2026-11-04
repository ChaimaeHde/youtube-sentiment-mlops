from pathlib import Path

import joblib
import numpy as np

from src.utils.preprocessing import clean_text

LABEL_NAMES = {-1: "negative", 0: "neutral", 1: "positive"}


class SentimentPredictor:
    def __init__(self, model_dir: Path):
        self.vectorizer = joblib.load(model_dir / "tfidf_vectorizer.joblib")
        self.model = joblib.load(model_dir / "sentiment_model.joblib")
        self.classes = self.model.classes_          # array([-1, 0, 1])

    def predict(self, comments: list[str]) -> list[dict]:
        cleaned = [clean_text(c) for c in comments]
        proba = self.model.predict_proba(self.vectorizer.transform(cleaned))
        idx = proba.argmax(axis=1)

        results = []
        for text, clean, i, p in zip(comments, cleaned, idx, proba):
            if not clean:                            # ex. commentaire uniquement en emojis
                label, conf = 0, 0.0
            else:
                label, conf = int(self.classes[i]), float(p[i])
            results.append({"text": text, "label": label,
                            "sentiment": LABEL_NAMES[label],
                            "confidence": round(conf, 4)})
        return results

    @staticmethod
    def compute_stats(results: list[dict]) -> dict:
        total = len(results)
        counts = {s: sum(r["sentiment"] == s for r in results)
                  for s in ("positive", "neutral", "negative")}
        pct = {f"{s}_pct": round(100 * n / total, 2) for s, n in counts.items()}
        avg = float(np.mean([r["confidence"] for r in results]))
        return {"total": total, **counts, **pct, "avg_confidence": round(avg, 4)}