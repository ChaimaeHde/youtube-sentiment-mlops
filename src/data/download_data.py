import sys
from pathlib import Path

import pandas as pd
import requests

URL = ("https://raw.githubusercontent.com/Himanshu-1703/"
       "reddit-sentiment-analysis/refs/heads/main/data/reddit.csv")
ROOT = Path(__file__).resolve().parents[2]          # racine du projet
RAW_PATH = ROOT / "data" / "raw" / "reddit.csv"


def download(url: str = URL, dest: Path = RAW_PATH) -> Path:
    """Télécharge le CSV et le sauvegarde dans data/raw/."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()                          # erreur si code HTTP != 200
    dest.write_bytes(resp.content)
    print(f"Fichier sauvegardé : {dest}")
    return dest


def show_stats(path: Path) -> None:
    """Affiche les statistiques demandées par le TP."""
    df = pd.read_csv(path)
    print(f"\nColonnes : {list(df.columns)}")
    print(f"Nombre de commentaires : {len(df)}")
    print(f"Valeurs manquantes :\n{df.isna().sum()}")

    counts = df["category"].value_counts().sort_index()
    names = {-1: "Négatif", 0: "Neutre", 1: "Positif"}
    print("\nDistribution des labels :")
    for label, n in counts.items():
        print(f"  {label:>2} ({names[label]:<7}) : {n:>6}  ({n / len(df):.1%})")


if __name__ == "__main__":
    try:
        path = download()
    except requests.RequestException as e:
        print(f"Erreur de téléchargement : {e}")
        sys.exit(1)
    show_stats(path)