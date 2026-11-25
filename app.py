"""Ruang Sentimen web app. Run: uvicorn app:app --port 7860"""
import json
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ruang_sentimen.predict import SentimentAnalyzer, model_file
from ruang_sentimen.report import (
    MAX_ROWS, build_frame, label_one, make_chart, read_upload, summarize, to_pdf, to_xlsx,
)
from ruang_sentimen.template import TEMPLATE_PATH, make_template

ROOT = Path(__file__).parent
MAX_TEXT_CHARS = 5000
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
KINDS = ("sentiment", "emotion")

app = FastAPI(title="Ruang Sentimen")
analyzer = SentimentAnalyzer()
model_lock = threading.Lock()          # one inference at a time (CPU, shared model)
worker = ThreadPoolExecutor(max_workers=1)
jobs: dict = {}
if not Path(TEMPLATE_PATH).exists():
    make_template()


def _check(kind, margin):
    if kind not in KINDS:
        raise HTTPException(400, "Jenis analisis tidak dikenal.")
    if not 0.0 <= margin <= 0.5:
        raise HTTPException(400, "Lebar zona netral harus antara 0 dan 0.5.")


class TextRequest(BaseModel):
    text: str
    kind: str = "sentiment"
    netral: bool = False
    margin: float = 0.15


@app.post("/api/analyze-text")
def analyze_text(req: TextRequest):
    _check(req.kind, req.margin)
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Tulis ulasan terlebih dahulu.")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(400, f"Ulasan terlalu panjang (maks. {MAX_TEXT_CHARS} karakter).")
    with model_lock:
        labels, probs = analyzer.probs(req.kind, [text])
    p = probs[0]
    return {"label": label_one(req.kind, labels, p, req.netral, req.margin),
            "confidence": round(max(p) * 100, 1),
            "probs": {l: round(x * 100, 1) for l, x in zip(labels, p)}}


def _run_job(job_id, texts, kind, netral, margin):
    job = jobs[job_id]
    try:
        with model_lock:
            labels, probs = analyzer.probs(kind, texts, on_progress=lambda d, t: job.update(done=d))
        df = build_frame(kind, texts, labels, probs, netral, margin)
        summary = summarize(df, kind, labels)
        fig = make_chart(summary)
        pdf = to_pdf(df, summary, fig, str(job["dir"] / "laporan_ruang_sentimen.pdf"))
        xlsx = to_xlsx(df, summary, str(job["dir"] / "hasil_ruang_sentimen.xlsx"))
        import matplotlib.pyplot as plt
        plt.close(fig)
        job.update(status="done", summary=summary, rows=df.to_dict("records"), pdf=pdf, xlsx=xlsx)
    except Exception as e:  # surfaced to the UI
        job.update(status="error", error=f"Analisis gagal: {e.__class__.__name__}")


@app.post("/api/analyze-batch")
async def analyze_batch(file: UploadFile = File(...), kind: str = Form("sentiment"),
                        netral: bool = Form(False), margin: float = Form(0.15)):
    _check(kind, margin)
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "File terlalu besar (maks. 5 MB).")
    job_id = uuid.uuid4().hex[:12]
    workdir = Path(tempfile.mkdtemp(prefix="rs_"))
    src = workdir / ("upload" + (".csv" if (file.filename or "").lower().endswith(".csv") else ".xlsx"))
    src.write_bytes(data)
    try:
        texts, note = read_upload(str(src))
    except ValueError as e:
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(400, str(e))
    jobs[job_id] = {"status": "running", "done": 0, "total": len(texts), "note": note, "dir": workdir}
    for old in list(jobs)[:-20]:                       # keep only the 20 newest jobs
        shutil.rmtree(jobs.pop(old)["dir"], ignore_errors=True)
    worker.submit(_run_job, job_id, texts, kind, netral, margin)
    return {"job_id": job_id, "total": len(texts), "note": note}


def _job(job_id):
    if job_id not in jobs:
        raise HTTPException(404, "Hasil tidak ditemukan atau sudah kedaluwarsa.")
    return jobs[job_id]


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    j = _job(job_id)
    out = {k: j[k] for k in ("status", "done", "total", "note")}
    if j["status"] == "done":
        out.update(summary=j["summary"], rows=j["rows"])
    if j["status"] == "error":
        out["error"] = j["error"]
    return out


@app.get("/api/jobs/{job_id}/pdf")
def job_pdf(job_id: str):
    j = _job(job_id)
    if j["status"] != "done":
        raise HTTPException(409, "Analisis belum selesai.")
    return FileResponse(j["pdf"], filename="laporan_ruang_sentimen.pdf", media_type="application/pdf")


@app.get("/api/jobs/{job_id}/xlsx")
def job_xlsx(job_id: str):
    j = _job(job_id)
    if j["status"] != "done":
        raise HTTPException(409, "Analisis belum selesai.")
    return FileResponse(j["xlsx"], filename="hasil_ruang_sentimen.xlsx")


@app.get("/api/template")
def template():
    return FileResponse(TEMPLATE_PATH, filename="template_ulasan.xlsx")


@app.get("/api/model-info")
def model_info():
    out = {"max_rows": MAX_ROWS}
    for kind in KINDS:
        f = model_file(analyzer.path(kind), "metrics.json")
        if f is None:
            continue
        m = json.loads(Path(f).read_text())
        r = m["test_report"]
        out[kind] = {
            "base_model": m["base_model"],
            "labels": m["labels"],
            "test_size": sum(map(sum, m["test_confusion_matrix"])),
            "accuracy": round(r["accuracy"] * 100, 1),
            "macro_f1": round(r["macro avg"]["f1-score"], 3),
            "per_class": {l: {"precision": round(r[l]["precision"], 2), "recall": round(r[l]["recall"], 2),
                              "f1": round(r[l]["f1-score"], 2), "support": int(r[l]["support"])}
                          for l in m["labels"]},
        }
    return out


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
