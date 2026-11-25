"""Task definitions: which dataset, which IndoBERT checkpoint, which labels."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS = Path.home() / "Downloads"

PRDECT_CSV = DOWNLOADS / "PRDECT-ID Dataset.csv"
PLAYSTORE_CSV = DOWNLOADS / "PRIVATE Crawl playstore Dataset.csv"


@dataclass(frozen=True)
class Task:
    name: str
    csv: Path
    default_model: str
    labels: tuple  # index == class id


# Emotion: PRDECT-ID e-commerce reviews, 5 emotions (Sentiment column is
# fully determined by Emotion, so it is derived at inference time).
EMOTION = Task(
    name="emotion",
    csv=PRDECT_CSV,
    default_model="indobenchmark/indobert-base-p1",
    labels=("Anger", "Fear", "Happy", "Love", "Sadness"),
)

# Emotion, 3 classes. Fear and Sadness are folded into "Anger" (the negative
# emotion): their labels in PRDECT-ID are not separable (5-class accuracy ~75%).
EMOTION3 = Task(
    name="emotion3",
    csv=PRDECT_CSV,
    default_model="indobenchmark/indobert-large-p1",
    labels=("Anger", "Happy", "Love"),
)

# Sentiment: Play Store reviews, binary. Label comes from the star score
# (1-2 negatif, 4-5 positif); the crawl has no 3-star reviews. An optional
# "netral" band is derived from the predicted probability (see grading.py).
SENTIMENT = Task(
    name="sentiment",
    csv=PLAYSTORE_CSV,
    default_model="indobenchmark/indobert-base-p1",
    labels=("negatif", "positif"),
)

TASKS = {"emotion": EMOTION, "emotion3": EMOTION3, "sentiment": SENTIMENT}

EMOTION_TO_POLARITY = {
    "Anger": "negatif", "Fear": "negatif", "Sadness": "negatif",
    "Happy": "positif", "Love": "positif",
}
