from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "reddit_clean.csv"
FIG_DIR = ROOT / "reports" / "figures"
NAMES = {-1: "Négatif", 0: "Neutre", 1: "Positif"}


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    df["sentiment"] = df["label"].map(NAMES)
    df["n_words"] = df["text"].str.split().str.len()

    # 1. Distribution des classes
    counts = df["sentiment"].value_counts()
    print("Distribution :\n", counts)
    print(f"Ratio max/min : {counts.max() / counts.min():.2f}")
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="sentiment", order=list(NAMES.values()))
    plt.title("Distribution des classes")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_distribution.png", dpi=150)
    plt.close()

    # 2. Longueur des textes
    print("\nLongueur (mots) par classe :")
    print(df.groupby("sentiment")["n_words"].describe().round(1))
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df[df["n_words"] < 150], x="n_words", hue="sentiment", bins=50)
    plt.title("Longueur des commentaires (< 150 mots)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "text_length.png", dpi=150)
    plt.close()

    print(f"\nFigures sauvegardées dans {FIG_DIR}")


if __name__ == "__main__":
    main()