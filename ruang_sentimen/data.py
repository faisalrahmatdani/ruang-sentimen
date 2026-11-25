"""Loading, cleaning, de-duplicating and splitting the two datasets."""
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import EMOTION, EMOTION3, SENTIMENT, Task

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_REPEAT = re.compile(r"(.)\1{3,}")  # "baguuuuus" -> "baguus"
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Light cleaning only: BERT sees slang/emoji-free but otherwise raw text."""
    text = str(text)
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = _REPEAT.sub(r"\1\1", text)
    return _WS.sub(" ", text).strip()


def load_task_frame(task: Task, keep_genre: bool = False) -> pd.DataFrame:
    """Return DataFrame[text, label(int)] with duplicates removed (plus genre for the Play Store set if asked)."""
    raw = pd.read_csv(task.csv)
    if task is EMOTION:
        df = pd.DataFrame({"text": raw["Customer Review"], "name": raw["Emotion"]})
    elif task is EMOTION3:
        merged = raw["Emotion"].replace({"Fear": "Anger", "Sadness": "Anger"})
        df = pd.DataFrame({"text": raw["Customer Review"], "name": merged})
    elif task is SENTIMENT:
        df = pd.DataFrame({"text": raw["content"], "name": raw["label"].str.strip().str.lower(),
                           "genre": raw["genre"]})
    else:
        raise ValueError(task)

    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0]
    df = df[df["name"].isin(task.labels)]

    # Drop texts that appear with conflicting labels, then exact duplicates.
    conflict = df.groupby("text")["name"].nunique()
    df = df[~df["text"].isin(conflict[conflict > 1].index)]
    df = df.drop_duplicates("text")

    label2id = {n: i for i, n in enumerate(task.labels)}
    df["label"] = df["name"].map(label2id)
    cols = ["text", "label"] + (["genre"] if keep_genre else [])
    return df[cols].reset_index(drop=True)


def split_frame(df: pd.DataFrame, seed: int = 42, val: float = 0.1, test: float = 0.1):
    """Stratified train/val/test split."""
    train_df, rest = train_test_split(
        df, test_size=val + test, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        rest, test_size=test / (val + test), stratify=rest["label"], random_state=seed
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def genre_folds(genres, k: int = 5, seed: int = 42) -> list[list[str]]:
    """Deterministically partition genres into k disjoint groups (for cross-genre tests)."""
    names = sorted(set(genres))
    order = np.random.RandomState(seed).permutation(len(names))
    return [sorted(names[i] for i in order[f::k]) for f in range(k)]


def split_by_genre(df: pd.DataFrame, fold: int, k: int = 5, seed: int = 42, val: float = 0.1):
    """Test = every review of the held-out genres; val = stratified slice of the remaining genres."""
    held = genre_folds(df["genre"], k, seed)[fold]
    test_df = df[df["genre"].isin(held)]
    rest = df[~df["genre"].isin(held)]
    train_df, val_df = train_test_split(rest, test_size=val, stratify=rest["label"], random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True), held
