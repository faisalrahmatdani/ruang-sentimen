"""Fit a temperature on the validation split so probabilities are not saturated.

    python -m ruang_sentimen.calibrate --task sentiment
Writes <model dir>/calibration.json; predictions (argmax) are unchanged.
"""
import argparse
import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import ROOT, TASKS
from .data import load_task_frame, split_frame
from .train import pick_device


@torch.no_grad()
def val_logits(model_dir, df, device, batch=16):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    out = []
    texts = df["text"].tolist()
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], truncation=True, max_length=128, padding=True,
                  return_tensors="pt").to(device)
        out.append(model(**enc).logits.float().cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS, required=True)
    args = ap.parse_args()
    task = TASKS[args.task]
    model_dir = ROOT / f"models/{task.name}"
    _, val_df, _ = split_frame(load_task_frame(task))
    logits = val_logits(model_dir, val_df, pick_device())
    y = torch.tensor(val_df["label"].tolist())

    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    nll = lambda t: torch.nn.functional.cross_entropy(logits / t, y)

    def closure():
        opt.zero_grad()
        loss = nll(log_t.exp())
        loss.backward()
        return loss

    before = nll(torch.tensor(1.0)).item()
    opt.step(closure)
    t = log_t.exp().item()
    print(f"{task.name}: T={t:.3f} val NLL {before:.4f} -> {nll(torch.tensor(t)).item():.4f}")
    (model_dir / "calibration.json").write_text(json.dumps({"temperature": t}))


if __name__ == "__main__":
    main()
