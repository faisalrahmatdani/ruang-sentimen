"""Turn model probabilities into the output shape the user asked for.

Sentiment model is binary (negatif, positif). "netral" is not a learned class:
it is the band where the model cannot decide, and its width is a parameter.
"""
from .config import EMOTION_TO_POLARITY


def grade_sentiment(probs, labels, mode="binary", neutral_band=(0.35, 0.65)):
    """mode: 'binary' | 'three' (adds netral when P(positif) is inside neutral_band)."""
    pos = probs[labels.index("positif")]
    if mode == "binary":
        return "positif" if pos >= 0.5 else "negatif"
    if mode == "three":
        lo, hi = neutral_band
        return "negatif" if pos <= lo else "positif" if pos >= hi else "netral"
    raise ValueError(f"unknown mode {mode!r}")


def grade_emotion(probs, labels, mode="emotion"):
    """mode: 'emotion' (Anger/Happy/Love) | 'polarity' (positif/negatif)."""
    if mode == "emotion":
        return labels[max(range(len(probs)), key=probs.__getitem__)]
    if mode == "polarity":
        pos = sum(p for p, l in zip(probs, labels) if EMOTION_TO_POLARITY[l] == "positif")
        return "positif" if pos >= 0.5 else "negatif"
    raise ValueError(f"unknown mode {mode!r}")
