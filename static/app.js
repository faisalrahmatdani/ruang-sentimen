const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const COLORS = {positif:"#3d6b4a", negatif:"#b4452f", netral:"#8d917e", Anger:"#b4452f", Happy:"#3f6a9a", Love:"#3d6b4a"};
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const cap = s => s.charAt(0).toUpperCase() + s.slice(1);
const fmt = n => String(n).replace(".", ",");

const state = {kind: "sentiment", file: null, jobId: null, rows: [], shown: 50, timer: null};
const KIND_TXT = {sentiment: "Sentimen", emotion: "Emosi"};
const CALLOUT = {
  sentiment: "<b>Hanya positif dan negatif.</b> Model dilatih pada ulasan Play Store berlabel dari bintang (1–2 negatif, 4–5 positif). Kategori netral bukan kelas yang dipelajari: ia hanya zona ragu saat peluang positif mendekati 50%. Lihat cakupan dan pengujian di <u>Tentang model</u>.",
  emotion: "<b>Tiga emosi: Anger, Happy, Love.</b> Anger mencakup emosi negatif secara umum (marah, takut, sedih), karena ketiganya tidak bisa dibedakan secara andal dari label data latih PRDECT-ID. Lihat pengujian di <u>Tentang model</u>.",
};

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = "Terjadi kesalahan.";
    try { msg = (await r.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return r;
}

/* ---------- navigation ---------- */
$$(".nav").forEach(b => b.onclick = () => {
  $$(".nav").forEach(x => x.classList.toggle("active", x === b));
  $("#view-analyze").hidden = b.dataset.view !== "analyze";
  $("#view-about").hidden = b.dataset.view !== "about";
  $("#crumb").textContent = b.dataset.view === "analyze" ? "Analisis sentimen" : "Tentang model";
  if (b.dataset.view === "about") loadAbout();
});
$$(".tab").forEach(b => b.onclick = () => {
  $$(".tab").forEach(x => x.classList.toggle("active", x === b));
  $("#panel-text").hidden = b.dataset.tab !== "text";
  $("#panel-batch").hidden = b.dataset.tab !== "batch";
});
$$("#kind button").forEach(b => b.onclick = () => {
  state.kind = b.dataset.kind;
  $$("#kind button").forEach(x => x.classList.toggle("on", x === b));
  $("#netral-row").hidden = state.kind !== "sentiment";
  $("#callout-text").innerHTML = CALLOUT[state.kind];
  resetTextResult();
});
$("#callout-text").innerHTML = CALLOUT.sentiment;
let rerunTimer;
function settingsChanged() {
  clearTimeout(rerunTimer);
  if (!$("#res-body").hidden && text.value.trim()) rerunTimer = setTimeout(run, 250);   // live update
  if (!$("#batch-out").hidden) $("#batch-status").textContent = "Pengaturan berubah. Klik “Analisis Excel” untuk menerapkannya pada file.";
}
$("#netral").onchange = () => { $("#margin-wrap").style.opacity = $("#netral").checked ? 1 : .4; settingsChanged(); };
$("#netral").onchange();
$("#margin").oninput = e => { $("#margin-val").textContent = e.target.value; if ($("#netral").checked) settingsChanged(); };
const opts = () => ({netral: state.kind === "sentiment" && $("#netral").checked, margin: $("#margin").value / 100});

/* ---------- text analysis ---------- */
function zoneNote(r) {
  if (state.kind !== "sentiment" || !("positif" in r.probs)) return "";
  const m = Math.round($("#margin").value), pos = r.probs.positif;
  if (!$("#netral").checked) return `<p class="note">Kategori netral nonaktif: ulasan dengan peluang positif ≥ 50% dianggap positif.</p>`;
  const inside = pos > 50 - m && pos < 50 + m;
  return `<p class="note"><b>Zona netral</b>: peluang positif ${50 - m}%–${50 + m}%. Peluang positif ulasan ini ${fmt(pos)}%, jadi ${inside ? "<b>berada di dalam zona → netral</b>" : "berada di luar zona"}. Model jarang ragu, sehingga netral hanya muncul pada ulasan yang benar-benar ambigu.</p>`;
}
const text = $("#text");
text.oninput = () => $("#count").textContent = `${text.value.length.toLocaleString("id-ID")} / 5.000`;
$$(".chip").forEach(c => c.onclick = () => { text.value = c.dataset.s; text.oninput(); });
$("#clear").onclick = () => { text.value = ""; text.oninput(); resetTextResult(); $("#text-status").textContent = ""; };
function resetTextResult() {
  $("#res-empty").hidden = false; $("#res-body").hidden = true;
  $("#res-state").textContent = "Menunggu input";
  $("#res-foot").innerHTML = `<i></i>Diproses oleh IndoBERT · ${state.kind === "sentiment" ? "2 kelas" : "3 kelas"}`;
}
resetTextResult();
$("#run-text").onclick = run;
text.onkeydown = e => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run(); };
async function run() {
  const st = $("#text-status"), btn = $("#run-text");
  if (!text.value.trim()) { st.textContent = "Tulis ulasan terlebih dahulu."; return; }
  btn.disabled = true; st.textContent = "Menganalisis…";
  try {
    const r = await (await api("/api/analyze-text", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text: text.value, kind: state.kind, ...opts()})})).json();
    const rows = Object.entries(r.probs).sort((a, b) => b[1] - a[1]).map(([l, p]) =>
      `<div class="bar-row"><span style="color:${COLORS[l]}">${esc(cap(l))}</span><div class="track"><i style="width:${p}%;background:${COLORS[l]}"></i></div><b>${fmt(p)}%</b></div>`).join("");
    $("#res-body").innerHTML = `<div class="eyebrow" style="color:#7a8274">${KIND_TXT[state.kind]} terdeteksi</div>
      <div class="big" style="color:${COLORS[r.label]}">${esc(cap(r.label))}</div>
      <div class="field-label">Skor keluaran model per kelas</div><div class="bars">${rows}</div>
      ${zoneNote(r)}
      <p class="note">Keyakinan ${fmt(r.confidence)}%. Skor sudah dikalibrasi pada data validasi, tetapi tetap perkiraan statistik, bukan jaminan ketepatan.</p>`;
    $("#res-empty").hidden = true; $("#res-body").hidden = false;
    $("#res-state").textContent = "Analisis selesai"; st.textContent = "Selesai. Ulasan diproses di server dan tidak disimpan.";
  } catch (e) { st.textContent = e.message; }
  btn.disabled = false;
}

/* ---------- batch upload ---------- */
const drop = $("#drop"), fileIn = $("#file");
drop.onclick = () => fileIn.click();
drop.onkeydown = e => { if (e.key === "Enter") fileIn.click(); };
["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => setFile(e.dataTransfer.files[0]));
fileIn.onchange = () => setFile(fileIn.files[0]);
function setFile(f) {
  if (!f) return;
  state.file = f;
  $("#drop-title").textContent = f.name;
  $("#drop-sub").textContent = `${(f.size / 1024).toFixed(0)} KB · klik untuk mengganti`;
  $("#run-batch").disabled = false; $("#batch-status").textContent = "";
}
$("#run-batch").onclick = async () => {
  const st = $("#batch-status"), btn = $("#run-batch");
  st.textContent = ""; $("#batch-out").hidden = true; btn.disabled = true;
  const fd = new FormData();
  fd.append("file", state.file); fd.append("kind", state.kind);
  const o = opts(); fd.append("netral", o.netral); fd.append("margin", o.margin);
  try {
    const r = await (await api("/api/analyze-batch", {method: "POST", body: fd})).json();
    state.jobId = r.job_id;
    $("#progress").hidden = false; $("#bar").style.width = "2%";
    $("#prog-text").textContent = `Memuat model… (${r.total} ulasan)`;
    poll(r.note);
  } catch (e) { st.textContent = e.message; btn.disabled = false; }
};
function poll(note) {
  clearTimeout(state.timer);
  state.timer = setTimeout(async () => {
    try {
      const j = await (await api(`/api/jobs/${state.jobId}`)).json();
      if (j.status === "running") {
        $("#bar").style.width = Math.max(2, j.done / j.total * 100) + "%";
        $("#prog-text").textContent = `Menganalisis ${j.done} / ${j.total}`;
        return poll(note);
      }
      $("#progress").hidden = true; $("#run-batch").disabled = false;
      if (j.status === "error") { $("#batch-status").textContent = j.error; return; }
      $("#batch-status").textContent = note ? "⚠ " + note : "";
      showBatch(j);
    } catch (e) { $("#progress").hidden = true; $("#run-batch").disabled = false; $("#batch-status").textContent = e.message; }
  }, 700);
}

function showBatch(j) {
  const s = j.summary; state.rows = j.rows; state.shown = 50;
  $("#batch-out").hidden = false;
  $("#st-total").textContent = s.total.toLocaleString("id-ID");
  $("#st-dom").textContent = cap(s.dominant); $("#st-dom").style.color = COLORS[s.dominant];
  $("#st-dom-sub").textContent = `${fmt(s.percent[s.dominant])}% dari ulasan`;
  $("#st-conf").textContent = fmt(s.avg_confidence) + "%";
  $("#dl-pdf").href = `/api/jobs/${state.jobId}/pdf`; $("#dl-xlsx").href = `/api/jobs/${state.jobId}/xlsx`;

  // donut
  const names = Object.keys(s.counts), R = 52, C = 2 * Math.PI * R; let off = 0;
  const arcs = names.filter(n => s.counts[n] > 0).map(n => {
    const len = s.counts[n] / s.total * C;
    const el = `<circle r="${R}" cx="70" cy="70" fill="none" stroke="${COLORS[n]}" stroke-width="20" stroke-dasharray="${len} ${C - len}" stroke-dashoffset="${-off}" transform="rotate(-90 70 70)"/>`;
    off += len; return el;
  }).join("");
  $("#donut").innerHTML = `<div class="donut-wrap"><svg width="140" height="140" viewBox="0 0 140 140">${arcs}
    <text x="70" y="68" text-anchor="middle" font-family="Instrument Serif" font-size="26" fill="#2b4a3a">${s.total}</text>
    <text x="70" y="84" text-anchor="middle" font-size="8" letter-spacing="1.5" fill="#7a8274">ULASAN</text></svg>
    <div class="legend">${names.map(n => `<span><i style="background:${COLORS[n]}"></i>${esc(cap(n))} · ${s.counts[n]}</span>`).join("")}</div></div>`;
  // average probability bars
  $("#avgbars").innerHTML = `<div class="bars">${Object.entries(s.avg_prob).map(([l, p]) =>
    `<div class="bar-row"><span style="color:${COLORS[l]}">${esc(cap(l))}</span><div class="track"><i style="width:${p}%;background:${COLORS[l]}"></i></div><b>${fmt(p)}%</b></div>`).join("")}</div>`;
  $("#sumtable").innerHTML = `<tr><th>Kelas</th><th>Jumlah</th><th>Persentase</th><th>Rata-rata prob.</th></tr>` +
    names.map(n => `<tr><td>${esc(cap(n))}</td><td>${s.counts[n]}</td><td>${fmt(s.percent[n])}%</td><td>${n in s.avg_prob ? fmt(s.avg_prob[n]) + "%" : "–"}</td></tr>`).join("") +
    `<tr><td colspan="3"><b>Rata-rata keyakinan model</b></td><td><b>${fmt(s.avg_confidence)}%</b></td></tr>`;

  $("#filter").innerHTML = `<option value="all">Semua hasil</option>` + names.map(n => `<option value="${esc(n)}">${esc(cap(n))}</option>`).join("");
  renderTable();
  $("#batch-out").scrollIntoView({behavior: "smooth", block: "start"});
}
$("#filter").onchange = () => { state.shown = 50; renderTable(); };
$("#more").onclick = () => { state.shown += 100; renderTable(); };
function renderTable() {
  const f = $("#filter").value, all = state.rows.filter(r => f === "all" || r.Hasil === f), part = all.slice(0, state.shown);
  const probCols = Object.keys(state.rows[0] || {}).filter(k => k.startsWith("P("));
  $("#tbl").innerHTML = `<tr><th>No.</th><th>Ulasan</th><th>Hasil</th><th>Keyakinan</th></tr>` + part.map(r =>
    `<tr><td>${r.No}</td><td>${esc(r.Ulasan)}</td><td><span class="tag t-${esc(r.Hasil)}">${esc(cap(r.Hasil))}</span></td><td>${fmt(r["Keyakinan (%)"])}%</td></tr>`).join("");
  $("#row-count").textContent = `${state.rows.length} baris`;
  $("#tbl-note").textContent = `Menampilkan ${part.length} dari ${all.length} baris sesuai filter. Unduhan Excel memuat seluruh baris dan skor per kelas.`;
  $("#more").hidden = part.length >= all.length;
}

/* ---------- about ---------- */
let aboutLoaded = false;
async function loadAbout() {
  if (aboutLoaded) return;
  const info = await (await api("/api/model-info")).json();
  const NAMES = {sentiment: ["Sentimen positif / negatif", "Ulasan Play Store (16.111 ulasan, label dari bintang)"],
                 emotion: ["Deteksi emosi", "PRDECT-ID, ulasan e-commerce (5.400 ulasan)"]};
  $("#max-rows").textContent = info.max_rows;
  $("#about-body").innerHTML = ["sentiment", "emotion"].filter(k => info[k]).map(k => {
    const m = info[k];
    return `<div class="card about-card"><div class="eyebrow">${NAMES[k][1]}</div>
      <h2>${NAMES[k][0]}<span class="badge">${m.labels.length} kelas</span></h2>
      <p class="muted">Basis: ${esc(m.base_model)} · fine-tuning untuk klasifikasi.</p>
      <div class="kv"><div>Accuracy (data uji)<b>${fmt(m.accuracy)}%</b></div><div>Macro-F1<b>${fmt(m.macro_f1)}</b></div><div>Jumlah data uji<b>${m.test_size.toLocaleString("id-ID")}</b></div></div>
      <table class="mini"><tr><th>Kelas</th><th>Precision</th><th>Recall</th><th>F1</th><th>Data uji</th></tr>${Object.entries(m.per_class).map(([l, c]) =>
        `<tr><td>${esc(cap(l))}</td><td>${fmt(c.precision)}</td><td>${fmt(c.recall)}</td><td>${fmt(c.f1)}</td><td>${c.support}</td></tr>`).join("")}</table></div>`;
  }).join("") + `<div class="callout"><span>↳</span><p>Batas yang perlu diketahui: Fear dan Sadness digabung ke Anger karena label aslinya tidak konsisten (akurasi 5 kelas hanya sekitar 75%). Kategori netral pada sentimen bukan kelas hasil latihan.</p></div>`;
  aboutLoaded = true;
}
fetch("/api/model-info").then(r => r.json()).then(i => $("#max-rows").textContent = i.max_rows).catch(() => {});
