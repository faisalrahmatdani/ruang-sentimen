"""Batch upload parsing, result tables, charts, and Excel/PDF export."""
import io
import os
from datetime import datetime
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

from .grading import grade_sentiment  # noqa: E402

MAX_ROWS = int(os.environ.get("MAX_ROWS", "500"))
PDF_MAX_ROWS = 300
_FORMULA_PREFIXES = ("=", "+", "-", "@")

TEXT_COLUMNS = ("ulasan", "review", "teks", "text", "content", "komentar", "customer review")

KIND_TITLES = {"sentiment": "Sentimen Positif / Negatif", "emotion": "Deteksi Emosi"}
COLORS = {
    "positif": "#3D6B4A", "negatif": "#B4452F", "netral": "#8D917E",
    "Anger": "#B4452F", "Happy": "#3F6A9A", "Love": "#3D6B4A",
}


def read_upload(path: str, max_rows: int = MAX_ROWS):
    """Return (texts, note). Raises ValueError with a user-facing message."""
    try:
        if str(path).lower().endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path, sheet_name=0)
    except Exception as e:
        raise ValueError(f"File tidak bisa dibaca. Pastikan formatnya .xlsx ({e.__class__.__name__}).") from e

    col = next((c for c in df.columns if str(c).strip().lower() in TEXT_COLUMNS), None)
    if col is None and len(df.columns) == 1:
        col = df.columns[0]
    if col is None:
        raise ValueError("Kolom 'ulasan' tidak ditemukan. Gunakan template yang disediakan.")

    texts = [str(t).strip() for t in df[col].dropna() if str(t).strip()]
    if not texts:
        raise ValueError("Tidak ada ulasan yang terbaca di kolom 'ulasan'.")
    note = ""
    if len(texts) > max_rows:
        note = f"File berisi {len(texts)} ulasan; hanya {max_rows} baris pertama yang diproses."
        texts = texts[:max_rows]
    return texts, note


def label_one(kind, labels, probs, netral=False, margin=0.15):
    if kind == "sentiment":
        return grade_sentiment(probs, labels, "three" if netral else "binary",
                               neutral_band=(0.5 - margin, 0.5 + margin))
    return labels[max(range(len(probs)), key=probs.__getitem__)]


def build_frame(kind, texts, labels, probs, netral=False, margin=0.15) -> pd.DataFrame:
    rows = []
    for i, (t, p) in enumerate(zip(texts, probs), start=1):
        row = {"No": i, "Ulasan": t, "Hasil": label_one(kind, labels, p, netral, margin),
               "Keyakinan (%)": round(max(p) * 100, 1)}
        for lab, x in zip(labels, p):
            row[f"P({lab}) %"] = round(x * 100, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, kind: str, labels) -> dict:
    total = len(df)
    order = list(labels) + (["netral"] if "netral" in set(df["Hasil"]) else [])
    counts = {l: int((df["Hasil"] == l).sum()) for l in order}
    avg_prob = {l: round(float(df[f"P({l}) %"].mean()), 1) for l in labels}
    return {
        "kind": kind,
        "total": total,
        "counts": counts,
        "percent": {l: round(c / total * 100, 1) for l, c in counts.items()},
        "avg_prob": avg_prob,
        "avg_confidence": round(float(df["Keyakinan (%)"].mean()), 1),
        "dominant": max(counts, key=counts.get),
    }


def make_chart(summary: dict):
    """Donut of label distribution + bars of average probability per class."""
    counts = {k: v for k, v in summary["counts"].items() if v > 0}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
    a1.pie(list(counts.values()), labels=[f"{k}\n{v}" for k, v in counts.items()],
           colors=[COLORS.get(k, "#999") for k in counts], startangle=90,
           wedgeprops={"width": 0.42, "edgecolor": "white"}, textprops={"fontsize": 9})
    a1.set_title("Distribusi hasil", fontsize=11)
    names = list(summary["avg_prob"])
    vals = [summary["avg_prob"][n] for n in names]
    bars = a2.barh(names, vals, color=[COLORS.get(n, "#999") for n in names])
    a2.set_xlim(0, 100)
    a2.invert_yaxis()
    a2.set_xlabel("Rata-rata probabilitas (%)")
    a2.set_title("Rata-rata skor per kelas", fontsize=11)
    for b, v in zip(bars, vals):
        a2.text(v + 1, b.get_y() + b.get_height() / 2, f"{v}%", va="center", fontsize=9)
    for ax in (a2,):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def summary_markdown(summary: dict) -> str:
    lines = [f"**Jenis analisis:** {KIND_TITLES[summary['kind']]}  ",
             f"**Jumlah ulasan:** {summary['total']}  ",
             f"**Hasil dominan:** {summary['dominant']} ({summary['percent'][summary['dominant']]}%)  ",
             f"**Rata-rata keyakinan model:** {summary['avg_confidence']}%", "",
             "| Kelas | Jumlah | Persentase | Rata-rata probabilitas |", "|---|---|---|---|"]
    for k, c in summary["counts"].items():
        avg = summary["avg_prob"].get(k)
        lines.append(f"| {k} | {c} | {summary['percent'][k]}% | {f'{avg}%' if avg is not None else '-'} |")
    return "\n".join(lines)


def to_xlsx(df: pd.DataFrame, summary: dict, path: str) -> str:
    rows = [{"Kelas": k, "Jumlah": c, "Persentase (%)": summary["percent"][k],
             "Rata-rata probabilitas (%)": summary["avg_prob"].get(k)} for k, c in summary["counts"].items()]
    rows.append({"Kelas": "Rata-rata keyakinan model", "Jumlah": None, "Persentase (%)": None,
                 "Rata-rata probabilitas (%)": summary["avg_confidence"]})
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="Hasil", index=False)
        pd.DataFrame(rows).to_excel(xw, sheet_name="Ringkasan", index=False)
        ws = xw.sheets["Hasil"]
        ws.column_dimensions["B"].width = 90
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                # Ulasan berawalan = + - @ bisa dibaca Excel sebagai formula; simpan sebagai teks.
                if isinstance(cell.value, str) and cell.value.lstrip().startswith(_FORMULA_PREFIXES):
                    cell.data_type = "s"
                    cell.quotePrefix = True
    return path


def _pdf_safe(text: str) -> str:
    """Built-in PDF fonts are Latin-1 only: drop emoji / other scripts."""
    return escape(text.encode("latin-1", "ignore").decode("latin-1"))


def to_pdf(df: pd.DataFrame, summary: dict, fig, path: str) -> str:
    styles = getSampleStyleSheet()
    cell = ParagraphStyle("cell", parent=styles["BodyText"], fontSize=8, leading=10)
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm, title="Laporan Ruang Sentimen")
    story = [Paragraph("Laporan Analisis - Ruang Sentimen", styles["Title"]),
             Paragraph(f"{KIND_TITLES[summary['kind']]} &nbsp;|&nbsp; {datetime.now():%d %B %Y %H:%M}",
                       styles["Normal"]), Spacer(1, 10)]

    head = ["Kelas", "Jumlah", "Persentase", "Rata-rata prob."]
    body = [[k, c, f"{summary['percent'][k]}%",
             f"{summary['avg_prob'][k]}%" if k in summary["avg_prob"] else "-"]
            for k, c in summary["counts"].items()]
    t = Table([head] + body, hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B4A3A")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D9D8CC")),
                           ("FONTSIZE", (0, 0), (-1, -1), 9)]))
    story += [Paragraph(f"Total ulasan: <b>{summary['total']}</b> &nbsp; Hasil dominan: "
                        f"<b>{_pdf_safe(summary['dominant'])}</b> &nbsp; Rata-rata keyakinan model: "
                        f"<b>{summary['avg_confidence']}%</b>", styles["Normal"]), Spacer(1, 8), t, Spacer(1, 12)]

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    buf.seek(0)
    story += [Image(buf, width=17 * cm, height=17 * cm * 3.8 / 9), Spacer(1, 12),
              Paragraph("Daftar hasil", styles["Heading2"])]

    shown = df.head(PDF_MAX_ROWS)
    rows = [["No", "Ulasan", "Hasil", "Yakin"]] + [
        [str(r["No"]), Paragraph(_pdf_safe(r["Ulasan"][:400]), cell), r["Hasil"], f"{r['Keyakinan (%)']}%"]
        for _, r in shown.iterrows()]
    tbl = Table(rows, colWidths=[1.1 * cm, 12 * cm, 2.4 * cm, 1.8 * cm], repeatRows=1)
    tbl.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B4A3A")),
                             ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                             ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D9D8CC")),
                             ("VALIGN", (0, 0), (-1, -1), "TOP"), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
    story.append(tbl)
    if len(df) > PDF_MAX_ROWS:
        story.append(Paragraph(f"Menampilkan {PDF_MAX_ROWS} dari {len(df)} baris. Unduh Excel untuk data lengkap.",
                               styles["Italic"]))
    doc.build(story)
    return path
