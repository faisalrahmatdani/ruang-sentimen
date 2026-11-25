"""Inference wrapper.

    from ruang_sentimen.predict import SentimentAnalyzer
    a = SentimentAnalyzer()
    a.sentiment(["aplikasinya bagus"], mode="three")
    a.emotion(["barang rusak, kecewa banget"], mode="emotion")

CLI:
    python -m ruang_sentimen.predict --kind sentiment --mode three "aplikasinya lumayan"
"""
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import ROOT
from .data import clean_text
from .grading import grade_emotion, grade_sentiment
from .train import pick_device


def model_file(path, name):
    """Local file inside a model dir, or fetched from the Hub when path is a repo id."""
    local = Path(str(path)) / name
    if local.exists():
        return local
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(str(path), name))
    except Exception:
        return None


class _Model:
    def __init__(self, path, max_len=128):
        self.device = pick_device()
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()
        cfg = self.model.config
        self.labels = [cfg.id2label[i] for i in range(cfg.num_labels)]
        self.max_len = max_len
        self.temperature = 1.0
        calib = model_file(path, "calibration.json")
        if calib is not None and calib.exists():
            self.temperature = float(json.loads(calib.read_text())["temperature"])

    @torch.no_grad()
    def probs(self, texts, batch_size=16, on_progress=None):
        """Class probabilities per text, in input order. Batches are length-sorted
        so padding stays small; on_progress(done, total) is called after each batch."""
        texts = [clean_text(t) for t in texts]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [None] * len(texts)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            enc = self.tok([texts[i] for i in idx], truncation=True, max_length=self.max_len,
                           padding=True, return_tensors="pt").to(self.device)
            for i, p in zip(idx, torch.softmax(self.model(**enc).logits / self.temperature, -1).cpu().tolist()):
                out[i] = p
            if on_progress:
                on_progress(min(start + batch_size, len(order)), len(order))
        return out


class SentimentAnalyzer:
    def __init__(self, sentiment_dir=None, emotion_dir=None):
        # Paths may also be Hugging Face Hub ids; env vars let a deployment
        # (e.g. a Space) point at hub repos without code changes.
        self._dirs = {
            "sentiment": sentiment_dir or os.environ.get("MODEL_SENTIMENT") or ROOT / "models/sentiment",
            "emotion": emotion_dir or os.environ.get("MODEL_EMOTION") or ROOT / "models/emotion3",
        }
        self._models = {}

    def path(self, kind):
        return self._dirs[kind]

    def _get(self, kind):
        if kind not in self._models:
            self._models[kind] = _Model(self._dirs[kind])
        return self._models[kind]

    def probs(self, kind, texts, on_progress=None):
        """(labels, probs) for kind in {'sentiment', 'emotion'}."""
        m = self._get(kind)
        return m.labels, m.probs(texts, on_progress=on_progress)

    def sentiment(self, texts, mode="three", **grade_kw):
        m = self._get("sentiment")
        return [
            {"text": t, "label": grade_sentiment(p, m.labels, mode, **grade_kw),
             "probs": dict(zip(m.labels, [round(x, 4) for x in p]))}
            for t, p in zip(texts, m.probs(texts))
        ]

    def emotion(self, texts, mode="emotion"):
        m = self._get("emotion")
        return [
            {"text": t, "label": grade_emotion(p, m.labels, mode),
             "probs": dict(zip(m.labels, [round(x, 4) for x in p]))}
            for t, p in zip(texts, m.probs(texts))
        ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["sentiment", "emotion"], required=True)
    ap.add_argument("--mode", help="sentiment: binary|three; emotion: emotion|grouped|polarity")
    ap.add_argument("texts", nargs="+")
    args = ap.parse_args()
    a = SentimentAnalyzer()
    if args.kind == "sentiment":
        res = a.sentiment(args.texts, mode=args.mode or "three")
    else:
        res = a.emotion(args.texts, mode=args.mode or "emotion")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
