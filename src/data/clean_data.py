from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "reddit.csv"
PROC_DIR = ROOT / "data" / "processed"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    df = df.rename(columns={"clean_comment": "text", "category": "label"})
    n0 = len(df)

    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([-1, 0, 1])]

    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset="text")

    print(f"Lignes : {n0} -> {len(df)} après nettoyage")
    return df[["text", "label"]].reset_index(drop=True)


def split_and_save(df: pd.DataFrame) -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    train, test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label"]
    )
    df.to_csv(PROC_DIR / "reddit_clean.csv", index=False)
    train.to_csv(PROC_DIR / "train.csv", index=False)
    test.to_csv(PROC_DIR / "test.csv", index=False)

    print(f"Train : {len(train)} | Test : {len(test)}")
    print("Distribution train (%) :")
    print((train["label"].value_counts(normalize=True).sort_index() * 100).round(1))


if __name__ == "__main__":
    data = load_and_clean()
    min_class = data["label"].value_counts().min()
    assert len(data) >= 1000 and min_class >= 300, "Dataset trop petit (spec : 1000+, 300/classe)"
    split_and_save(data)