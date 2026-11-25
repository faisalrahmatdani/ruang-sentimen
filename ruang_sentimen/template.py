"""Excel template users download, fill in and upload for batch analysis."""
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

from .config import ROOT

TEMPLATE_PATH = ROOT / "assets" / "template_ulasan.xlsx"

SAMPLES = [
    "Barang bagus, pengiriman cepat, penjual ramah. Terima kasih!",
    "Aplikasinya sering error dan iklannya mengganggu banget.",
    "Kecewa, barang yang datang tidak sesuai foto dan rusak.",
    "Fiturnya lengkap, sangat membantu pekerjaan saya sehari-hari.",
    "Suka banget sama produknya, pasti beli lagi!",
]

INSTRUCTIONS = [
    "Petunjuk pengisian",
    "1. Isi satu ulasan per baris di kolom 'ulasan' pada sheet 'Ulasan'.",
    "2. Jangan mengubah nama kolom (baris pertama) dan jangan menambah kolom di sebelah kiri.",
    "3. Hapus baris contoh sebelum mengunggah, atau timpa dengan ulasan Anda.",
    "4. Baris kosong akan dilewati.",
    "5. Simpan sebagai .xlsx lalu unggah di tab 'Unggah Excel'.",
]


def make_template(path: Path = TEMPLATE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "Ulasan"
    ws["A1"] = "ulasan"
    ws["A1"].font = Font(bold=True, color="FFFFFF")
    ws["A1"].fill = PatternFill("solid", fgColor="2B4A3A")
    for i, text in enumerate(SAMPLES, start=2):
        ws.cell(row=i, column=1, value=text).alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 100

    info = wb.create_sheet("Petunjuk")
    for i, line in enumerate(INSTRUCTIONS, start=1):
        info.cell(row=i, column=1, value=line)
    info["A1"].font = Font(bold=True, size=13)
    info.column_dimensions["A"].width = 95
    wb.save(path)
    return path


if __name__ == "__main__":
    print(make_template())
