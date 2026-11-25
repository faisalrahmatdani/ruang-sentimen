# Ruang Sentimen — sisi Machine Learning

Dua model IndoBERT:

| Model | Dataset | Output latih | Checkpoint default |
|---|---|---|---|
| `emotion3` | PRDECT-ID (5.400 ulasan e-commerce) | Anger / Happy / Love (Fear & Sadness digabung ke Anger) | `indobenchmark/indobert-large-p1` |
| `sentiment` | Crawl Play Store (16.111 ulasan) | negatif / positif | `indobenchmark/indobert-base-p1` |


## Pakai
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m ruang_sentimen.train --task emotion3
.venv/bin/python -m ruang_sentimen.train --task sentiment
.venv/bin/python -m ruang_sentimen.predict --kind sentiment --mode five "aplikasinya lumayan"
```
Hasil latih: `models/<task>/` (model, tokenizer, `metrics.json` berisi F1 val/test + confusion matrix).
Device otomatis: CUDA → MPS (Apple Silicon) → CPU.

## Output yang bisa diatur user
- Sentimen: `binary` (positif/negatif) atau `three` (+netral).
- Emosi: `emotion` (Anger/Happy/Love) atau `polarity` (positif/negatif).
- Ambang netral: `neutral_band=(0.35, 0.65)` — lihat `grading.py`.

## Catatan penting
- **Dataset Play Store tidak punya kelas netral** (tidak ada bintang 3; label = turunan skor). "Netral" dan gradasi di sini dihitung dari distribusi probabilitas model, bukan dipelajari dari data. Untuk validasi, sebaiknya label manual ±200 ulasan netral lalu tuning ambang.
- Label Play Store diturunkan dari bintang, jadi ada noise (ulasan berbintang 4 yang isinya mengeluh).
- Kolom `Sentiment` PRDECT-ID sepenuhnya ditentukan oleh `Emotion`, jadi polaritas diturunkan dari emosi.
- Teks duplikat / berlabel konflik dibuang sebelum split stratified 80/10/10. Kolom `text_light`/`text_full` tidak dipakai (BERT lebih baik dengan teks asli).

## Hasil (data test)
- sentiment (biner): accuracy 95,7%, macro-F1 0,956
- emotion3 (Anger/Happy/Love): accuracy 92,5%, macro-F1 0,886
- (5 kelas asli hanya 75,4% karena label Anger/Fear/Sadness tidak konsisten; modelnya di `models_old/`)

## Web app
`uvicorn app:app --port 7860` (FastAPI + frontend di `static/`): analisis teks tunggal, batch Excel dengan progress (template di `assets/template_ulasan.xlsx`),
ringkasan + grafik + rata-rata, unduh PDF/Excel. Probabilitas dikalibrasi (`python -m ruang_sentimen.calibrate --task <task>`).
Panduan deploy gratis: lihat [DEPLOY.md](DEPLOY.md).

## Baseline pembanding (tidak dideploy)
Folder `baseline/` berisi model SVM dan Naive Bayes serta laporan perbandingan dengan IndoBERT (`baseline/report/`). Terpisah dari app; lihat `baseline/README.md`.
