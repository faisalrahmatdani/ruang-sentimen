"""Fine-tune an IndoBERT checkpoint for emotion or sentiment.

    python -m ruang_sentimen.train --task emotion
    python -m ruang_sentimen.train --task sentiment
"""
import argparse
import json
import random

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .config import ROOT, TASKS
from .data import load_task_frame, split_by_genre, split_frame


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(df, tok, max_len, batch_size, shuffle):
    texts, labels = df["text"].tolist(), df["label"].tolist()

    def collate(idx):
        enc = tok([texts[i] for i in idx], truncation=True, max_length=max_len,
                  padding=True, pad_to_multiple_of=16, return_tensors="pt")
        enc["labels"] = torch.tensor([labels[i] for i in idx])
        return enc

    return DataLoader(list(range(len(df))), batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, gold, loss_sum, n = [], [], 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss_sum += out.loss.item() * len(batch["labels"])
        n += len(batch["labels"])
        preds += out.logits.argmax(-1).tolist()
        gold += batch["labels"].tolist()
    return gold, preds, loss_sum / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--model", help="HF checkpoint (default: per-task choice in config.py)")
    p.add_argument("--out", help="output dir (default: models/<task>)")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--patience", type=int, default=2, help="early-stop on val macro-F1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, help="seed for the data split (default: same as --seed)")
    p.add_argument("--genre-fold", type=int, help="sentiment only: hold out genre group K of 5 as the test set")
    p.add_argument("--limit", type=int, help="subsample N rows (smoke test)")
    args = p.parse_args()

    task = TASKS[args.task]
    model_name = args.model or task.default_model
    out_dir = ROOT / (args.out or f"models/{task.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = pick_device()

    df = load_task_frame(task, keep_genre=args.genre_fold is not None)
    if args.limit:
        df = df.groupby("label", group_keys=False).apply(
            lambda g: g.sample(min(len(g), max(2, args.limit // len(task.labels))), random_state=args.seed)
        )
    if args.genre_fold is not None:
        train_df, val_df, test_df, held = split_by_genre(df, args.genre_fold, seed=42)
        print("held-out genres:", held)
    else:
        train_df, val_df, test_df = split_frame(df, seed=args.split_seed if args.split_seed is not None else args.seed)
    print(f"device={device} model={model_name} train/val/test={len(train_df)}/{len(val_df)}/{len(test_df)}")
    print("train label dist:", train_df["label"].value_counts().sort_index().to_dict())

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(task.labels),
        id2label={i: n for i, n in enumerate(task.labels)},
        label2id={n: i for i, n in enumerate(task.labels)},
    ).to(device)

    train_dl = make_loader(train_df, tok, args.max_len, args.batch_size, True)
    val_dl = make_loader(val_df, tok, args.max_len, args.batch_size * 2, False)
    test_dl = make_loader(test_df, tok, args.max_len, args.batch_size * 2, False)

    steps = len(train_dl) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    # Inverse-frequency class weights (sqrt-damped) to soften imbalance.
    counts = np.bincount(train_df["label"], minlength=len(task.labels)).astype(float)
    weights = torch.tensor((counts.sum() / (len(counts) * np.maximum(counts, 1))) ** 0.5,
                           dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    best_f1, bad, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_dl, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            loss = loss_fn(model(**batch).logits, labels)
            if not torch.isfinite(loss):  # MPS can fail silently under memory pressure
                raise RuntimeError(f"non-finite loss at epoch {epoch} step {step}; aborting")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            running += loss.item()
            if device.type == "mps" and step % 25 == 0:
                torch.mps.empty_cache()
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_dl)} loss {running / step:.4f}", flush=True)

        gold, pred, vloss = evaluate(model, val_dl, device)
        f1 = f1_score(gold, pred, average="macro")
        history.append({"epoch": epoch, "train_loss": running / len(train_dl), "val_loss": vloss, "val_macro_f1": f1})
        print(f"epoch {epoch}: val_loss={vloss:.4f} val_macro_f1={f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1, bad = f1, 0
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break

    # Final test evaluation with the best checkpoint.
    model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(device)
    gold, pred, _ = evaluate(model, test_dl, device)
    report = classification_report(gold, pred, target_names=task.labels, output_dict=True, zero_division=0)
    print(classification_report(gold, pred, target_names=task.labels, zero_division=0))
    metrics = {
        "task": task.name, "base_model": model_name, "max_len": args.max_len,
        "labels": list(task.labels), "best_val_macro_f1": best_f1, "history": history,
        "test_report": report, "test_confusion_matrix": confusion_matrix(gold, pred).tolist(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
