# Deploy gratis lewat GitHub Codespaces

Hugging Face Spaces (Docker/Gradio di CPU basic) sekarang berbayar, jadi app dijalankan di GitHub Codespaces.
Kuota gratis akun pribadi: 120 core-jam/bulan (≈ 60 jam di mesin 2-core 8 GB) dan 15 GB storage.
Link aktif selama codespace menyala; codespace berhenti sendiri bila lama tidak dipakai.

Model sudah ada di Hugging Face Hub (repo public, gratis, tanpa token):
`FaisalRd/ruang-sentimen-sentiment` dan `FaisalRd/ruang-sentimen-emotion`.
Konfigurasi ada di `.devcontainer/` (instal dependensi, unduh model, jalankan app di port 7860).

## 1. Push kode ke GitHub (sekali saja)
Buat repo **private** kosong di github.com/new (tanpa README), lalu:
```bash
git add . && git commit -m "Ruang Sentimen"
git remote add origin https://github.com/<user-github>/ruang-sentimen.git
git push -u origin main
```
`.gitignore` sudah mengecualikan `models/`, `models_old/`, `.venv/`, `baseline/`, `validasi/`, `skripsi/`, `logs/`, dan CSV.

## 2. Jalankan
1. Di halaman repo: **Code → Codespaces → Create codespace on main**.
2. Pembuatan pertama ±5–10 menit (instal torch + unduh model ±1,8 GB). Setelah itu app jalan otomatis.
3. Buka tab **Ports** → baris 7860 → klik kanan → **Port Visibility → Public**. Salin link
   `https://<nama>-7860.app.github.dev` dan bagikan ke penguji (tanpa login GitHub).

Jika app belum jalan: `bash .devcontainer/start.sh`, lihat log di `/tmp/ruang-sentimen.log`.

## Tips hari sidang
- Nyalakan codespace 10 menit sebelumnya (github.com/codespaces → **Start**), lalu coba satu analisis
  agar model termuat ke memori. Port kembali **Private** setiap restart — set ulang ke Public.
- Perpanjang batas idle: github.com → Settings → Codespaces → **Default idle timeout** = 240 menit.
- **Stop** codespace setelah selesai agar kuota jam tidak habis.
- Cadangan: jalankan lokal di laptop.

## Jalankan lokal
```bash
.venv/bin/pip install -r requirements.txt torch
.venv/bin/python -m uvicorn app:app --port 7860   # http://127.0.0.1:7860
```
