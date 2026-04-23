from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io, base64

app = Flask(__name__)

CHANNELS = ['R', 'G', 'B']

# ─── Helpers ─────────────────────────────────────────────────────────────────
def img_to_b64(img: Image.Image) -> str: 
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def b64_to_arr(b64: str) -> np.ndarray:
    """Decode base64 → RGB numpy array (H, W, 3)"""
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB'))

def histogram_rgb(arr: np.ndarray) -> dict:
    """Return histogram for each R, G, B channel"""
    return {ch: np.bincount(arr[..., i].flatten(), minlength=256).tolist()
            for i, ch in enumerate(CHANNELS)}

def equalize_channel(ch_arr: np.ndarray):
    """Equalize a single channel. Returns (result_arr, table, cdf_min, N)"""
    hist = np.bincount(ch_arr.flatten(), minlength=256)
    cdf  = hist.cumsum()
    cdf_min = int(cdf[cdf > 0].min())
    N = int(ch_arr.size)
    lut = np.round((cdf - cdf_min) / (N - cdf_min) * 255).clip(0, 255).astype(np.uint8)
    table = [{'r': int(i), 'freq': int(hist[i]), 'cdf': int(cdf[i]), 'lut': int(lut[i])}
             for i in range(256) if hist[i] > 0]
    return lut[ch_arr], table, cdf_min, N

def equalize_rgb(arr: np.ndarray):
    out = np.empty_like(arr)
    tables, metas = {}, {}
    for i, ch in enumerate(CHANNELS):
        out[..., i], tables[ch], cdf_min, N = equalize_channel(arr[..., i])
        metas[ch] = {'cdf_min': cdf_min, 'N': N}
    return out, tables, metas

def match_channel(src_ch: np.ndarray, ref_ch: np.ndarray):
    """Match histogram of one channel. Returns (result_arr, table)"""
    sh = np.bincount(src_ch.flatten(), minlength=256)
    rh = np.bincount(ref_ch.flatten(), minlength=256)
    sc = sh.cumsum() / src_ch.size
    rc = rh.cumsum() / ref_ch.size
    lut = np.searchsorted(rc, sc).clip(0, 255).astype(np.uint8)
    table = [{'r': int(i), 'freq': int(sh[i]),
              'cdf_s': round(float(sc[i]), 4),
              'cdf_r': round(float(rc[lut[i]]), 4),
              'z': int(lut[i])}
             for i in range(256) if sh[i] > 0]
    return lut[src_ch], table

def match_rgb(src: np.ndarray, ref: np.ndarray):
    out = np.empty_like(src)
    tables = {}
    for i, ch in enumerate(CHANNELS):
        out[..., i], tables[ch] = match_channel(src[..., i], ref[..., i])
    return out, tables

# ─── API ─────────────────────────────────────────────────────────────────────
@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """Upload & analyze RGB image — returns histogram per channel + table"""
    try:
        arr = b64_to_arr(request.json['image'])
        img = Image.fromarray(arr)
        hists = histogram_rgb(arr)
        tables = {ch: [{'r': i, 'freq': hists[ch][i]}
                        for i in range(256) if hists[ch][i] > 0]
                  for ch in CHANNELS}
        return jsonify({'image': img_to_b64(img), 'histograms': hists,
                        'tables': tables, 'width': img.width, 'height': img.height})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
@app.route('/api/equalize', methods=['POST'])
def api_equalize():
    try:
        arr = b64_to_arr(request.json['image'])
        eq, tables, metas = equalize_rgb(arr)
        out = Image.fromarray(eq)
        return jsonify({'image': img_to_b64(out), 'histograms': histogram_rgb(eq),
                        'tables': tables, 'metas': metas})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/match', methods=['POST'])
def api_match():
    try:
        src = b64_to_arr(request.json['source'])
        ref = b64_to_arr(request.json['reference'])
        matched, tables = match_rgb(src, ref)
        out = Image.fromarray(matched)
        return jsonify({'image': img_to_b64(out), 'histograms': histogram_rgb(matched),
                        'tables': tables})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ─── HTML ─────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Histogram Lab — RGB</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#f0f2f5;--surface:#fff;--surface2:#f8f9fb;--border:#e2e5ea;
  --text:#1a1d23;--muted:#6b7280;
  --blue:#2563eb;--blue-bg:#eff6ff;--blue-lt:#bfdbfe;
  --green:#16a34a;--green-bg:#f0fdf4;--green-lt:#bbf7d0;
  --purple:#7c3aed;--purp-bg:#f5f3ff;--purp-lt:#ddd6fe;
  --orange:#ea580c;--org-bg:#fff7ed;--org-lt:#fed7aa;
  --red:#dc2626;--red-bg:#fef2f2;
  --radius:10px;--mono:'JetBrains Mono',monospace;--sans:'Inter',sans-serif;
  --shadow:0 1px 3px rgba(0,0,0,.08),0 1px 2px rgba(0,0,0,.06);
}
body{font-family:var(--sans);background:var(--bg);color:var(--text);min-height:100vh}

/* Header */
.header{background:var(--surface);border-bottom:1px solid var(--border);padding:0 2rem;
  display:flex;align-items:center;gap:1rem;height:60px;position:sticky;top:0;z-index:100;box-shadow:var(--shadow)}
.header-logo{font-weight:700;font-size:1.1rem;color:var(--blue)}
.header-sub{font-size:.8rem;color:var(--muted)}
.header-badge{margin-left:auto;font-size:.7rem;font-weight:600;
  background:var(--blue-bg);color:var(--blue);border:1px solid var(--blue-lt);padding:.2rem .6rem;border-radius:99px}

.page{max-width:1280px;margin:0 auto;padding:2rem}
.steps{display:flex;flex-direction:column;gap:1.5rem}

/* Step Card */
.step-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow);overflow:hidden}
.step-header{display:flex;align-items:center;gap:.75rem;padding:1rem 1.25rem;
  border-bottom:1px solid var(--border);background:var(--surface2)}
.step-num{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-weight:700;font-size:.8rem;flex-shrink:0}
.step-num.blue  {background:var(--blue);  color:#fff}
.step-num.green {background:var(--green); color:#fff}
.step-num.purple{background:var(--purple);color:#fff}
.step-num.orange{background:var(--orange);color:#fff}
.step-title{font-weight:600;font-size:.95rem}
.step-desc{font-size:.8rem;color:var(--muted);margin-top:.1rem}
.step-body{padding:1.25rem}

/* Upload */
.upload-row{display:flex;gap:1rem;align-items:flex-start;flex-wrap:wrap}
.upload-zone{flex:0 0 220px;border:2px dashed var(--border);border-radius:8px;
  padding:1.25rem 1rem;text-align:center;cursor:pointer;transition:.2s;position:relative;background:var(--surface2)}
.upload-zone:hover{border-color:var(--blue);background:var(--blue-bg)}
.upload-zone.filled{border-style:solid;border-color:var(--green);background:var(--green-bg)}
.upload-zone input{position:absolute;inset:0;opacity:0;cursor:pointer}
.upload-icon{font-size:1.6rem;display:block;margin-bottom:.4rem}
.upload-text{font-size:.78rem;color:var(--muted);line-height:1.5}
.upload-name{font-size:.75rem;font-weight:500;color:var(--green);margin-top:.4rem;word-break:break-all}
.preview-wrap{flex:0 0 200px;display:none}
.preview-wrap.show{display:block}
.preview-label{font-size:.7rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:.4rem}
.preview-img{width:100%;aspect-ratio:4/3;object-fit:contain;background:#f1f3f5;border:1px solid var(--border);border-radius:6px}

/* Buttons */
.action-row{display:flex;align-items:center;gap:1rem;flex-wrap:wrap;
  margin-top:1rem;padding-top:1rem;border-top:1px solid var(--border)}
.btn{display:inline-flex;align-items:center;gap:.4rem;padding:.5rem 1.1rem;
  font-family:var(--sans);font-size:.82rem;font-weight:600;
  border:none;border-radius:6px;cursor:pointer;transition:.15s;white-space:nowrap}
.btn:disabled{opacity:.4;cursor:not-allowed}
.btn-blue  {background:var(--blue);  color:#fff} .btn-blue:not(:disabled):hover  {background:#1d4ed8}
.btn-green {background:var(--green); color:#fff} .btn-green:not(:disabled):hover {background:#15803d}
.btn-purple{background:var(--purple);color:#fff} .btn-purple:not(:disabled):hover{background:#6d28d9}
.btn-orange{background:var(--orange);color:#fff} .btn-orange:not(:disabled):hover{background:#c2410c}

/* Status */
.status-pill{font-size:.75rem;padding:.25rem .65rem;border-radius:99px;display:none;align-items:center;gap:.35rem}
.status-pill.show{display:inline-flex}
.status-pill.ok     {background:var(--green-bg);color:var(--green);border:1px solid var(--green-lt)}
.status-pill.loading{background:var(--blue-bg); color:var(--blue); border:1px solid var(--blue-lt)}
.status-pill.err    {background:var(--red-bg);  color:var(--red);  border:1px solid #fecaca}
.spinner{width:10px;height:10px;border:2px solid transparent;border-top-color:currentColor;
  border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* Results */
.result-area{margin-top:1.25rem;padding-top:1.25rem;border-top:1px solid var(--border);display:none}
.result-area.show{display:block}
.result-cols{display:grid;grid-template-columns:1fr 1fr;gap:1.25rem}
.result-cols.triple{grid-template-columns:repeat(3,1fr)}
.result-box{border:1px solid var(--border);border-radius:8px;overflow:hidden}
.result-box-header{padding:.5rem .85rem;font-size:.72rem;font-weight:600;text-transform:uppercase;
  letter-spacing:.06em;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:.4rem}
.result-box-header.blue  {background:var(--blue-bg); color:var(--blue)}
.result-box-header.green {background:var(--green-bg);color:var(--green)}
.result-box-header.purple{background:var(--purp-bg); color:var(--purple)}
.result-box-header.orange{background:var(--org-bg);  color:var(--orange)}
.result-img{width:100%;aspect-ratio:4/3;object-fit:contain;background:#f8f9fb;display:block}
.chart-note{font-size:.68rem;color:var(--muted);padding:.35rem .85rem;
  border-top:1px solid var(--border);background:var(--surface2)}

/* RGB Histogram tabs */
.hist-tabs{display:flex;border-bottom:1px solid var(--border);background:var(--surface2)}
.hist-tab{padding:.4rem .9rem;font-size:.75rem;font-weight:600;cursor:pointer;
  border-bottom:2px solid transparent;transition:.15s;color:var(--muted)}
.hist-tab:hover{color:var(--text)}
.hist-tab.active-r{color:#dc2626;border-bottom-color:#dc2626}
.hist-tab.active-g{color:#16a34a;border-bottom-color:#16a34a}
.hist-tab.active-b{color:#2563eb;border-bottom-color:#2563eb}
.hist-pane{display:none;padding:.5rem;height:130px;position:relative}
.hist-pane.show{display:block}

/* Info box */
.info-box{background:var(--blue-bg);border:1px solid var(--blue-lt);
  border-radius:8px;padding:.85rem 1rem;font-size:.8rem;color:#1e40af;line-height:1.65}
.info-box.green {background:var(--green-bg);border-color:var(--green-lt);color:#166534}
.info-box.purple{background:var(--purp-bg); border-color:var(--purp-lt); color:#4c1d95}
.info-box.orange{background:var(--org-bg);  border-color:var(--org-lt);  color:#7c2d12}
.info-box strong{font-weight:600}
.info-box ul{margin:.4rem 0 0 1.1rem}
.info-box li{margin-bottom:.15rem}
.formula{margin-top:.55rem;font-family:var(--mono);font-size:.75em;
  background:rgba(0,0,0,.07);padding:.4rem .7rem;border-radius:5px;display:inline-block}

/* Locked */
.step-card.locked .step-body{opacity:.4;pointer-events:none;user-select:none}
.step-card.locked .step-header{opacity:.65}

/* Divider */
.divider-label{display:flex;align-items:center;gap:.6rem;font-size:.72rem;font-weight:600;
  text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:.9rem}
.divider-label::before,.divider-label::after{content:'';flex:1;height:1px;background:var(--border)}

/* Calc Panel */
.calc-toggle{display:inline-flex;align-items:center;gap:.4rem;margin-top:1rem;padding:.4rem .9rem;
  font-size:.78rem;font-weight:600;color:var(--muted);background:var(--surface2);
  border:1px solid var(--border);border-radius:6px;cursor:pointer;transition:.15s;user-select:none}
.calc-toggle:hover{color:var(--text);border-color:#c0c4cc}
.calc-toggle .arrow{transition:transform .2s;font-style:normal}
.calc-toggle.open .arrow{transform:rotate(90deg)}
.calc-panel{display:none;margin-top:.75rem;border:1px solid var(--border);border-radius:8px;overflow:hidden}
.calc-panel.show{display:block}
.calc-toolbar{display:flex;align-items:center;gap:1rem;flex-wrap:wrap;
  padding:.65rem 1rem;background:var(--surface2);border-bottom:1px solid var(--border);font-size:.78rem}
.calc-toolbar label{font-weight:600;color:var(--muted);white-space:nowrap}
.calc-toolbar input[type=range]{width:160px;accent-color:var(--blue)}
.range-val{font-family:var(--mono);font-size:.78rem;font-weight:600;color:var(--blue);min-width:90px}
.total-rows{margin-left:auto;color:var(--muted);font-size:.73rem}

/* Calc channel tabs */
.calc-ch-tabs{display:flex;background:var(--surface2);border-bottom:1px solid var(--border)}
.calc-ch-tab{padding:.35rem .85rem;font-size:.73rem;font-weight:700;cursor:pointer;
  border-bottom:2px solid transparent;transition:.15s;color:var(--muted)}
.calc-ch-tab.r{color:#dc2626} .calc-ch-tab.r.active{border-bottom-color:#dc2626;background:#fff5f5}
.calc-ch-tab.g{color:#16a34a} .calc-ch-tab.g.active{border-bottom-color:#16a34a;background:#f0fdf4}
.calc-ch-tab.b{color:#2563eb} .calc-ch-tab.b.active{border-bottom-color:#2563eb;background:#eff6ff}

.calc-scroll{overflow-x:auto;max-height:300px;overflow-y:auto}
table.calc-table{width:100%;border-collapse:collapse;font-size:.78rem;font-family:var(--mono)}
table.calc-table thead th{position:sticky;top:0;background:#f1f3f5;padding:.5rem .85rem;
  text-align:left;font-size:.68rem;text-transform:uppercase;letter-spacing:.07em;
  color:var(--muted);border-bottom:2px solid var(--border);white-space:nowrap}
table.calc-table tbody tr:nth-child(even){background:#fafafa}
table.calc-table tbody tr:hover{background:var(--blue-bg)}
table.calc-table td{padding:.4rem .85rem;border-bottom:1px solid #f0f0f0;white-space:nowrap}
table.calc-table td.num{text-align:right;color:#374151}
table.calc-table td.hi-r{font-weight:700;color:#dc2626}
table.calc-table td.hi-g{font-weight:700;color:#16a34a}
table.calc-table td.hi-b{font-weight:700;color:#2563eb}
table.calc-table td.hi-o{font-weight:700;color:var(--orange)}
.calc-note{padding:.5rem 1rem;font-size:.7rem;color:var(--muted);
  background:var(--surface2);border-top:1px solid var(--border)}

@media(max-width:700px){
  .result-cols,.result-cols.triple{grid-template-columns:1fr}
  .upload-row{flex-direction:column}
  .upload-zone,.preview-wrap{flex:1 1 100%}
}
</style>
</head>
<body>

<div class="header">
  <div>
    <div class="header-logo">🎨 Histogram Lab</div>
    <div class="header-sub">Pengolahan Citra Digital — RGB</div>
  </div>
  <div class="header-badge">Python · Flask · NumPy · PIL</div>
</div>

<div class="page"><div class="steps">

<!-- ══ STEP 1 — UPLOAD & ANALISIS RGB ══ -->
<div class="step-card" id="step1">
  <div class="step-header">
    <div class="step-num blue">1</div>
    <div>
      <div class="step-title">Upload Gambar &amp; Analisis Histogram RGB</div>
      <div class="step-desc">Unggah gambar berwarna. Histogram akan ditampilkan per channel R, G, dan B secara terpisah.</div>
    </div>
  </div>
  <div class="step-body">

    <div class="info-box" style="margin-bottom:1.1rem">
      <strong>Histogram RGB</strong> — Setiap gambar berwarna punya 3 channel: <span style="color:#dc2626;font-weight:600">Merah (R)</span>, <span style="color:#16a34a;font-weight:600">Hijau (G)</span>, <span style="color:#2563eb;font-weight:600">Biru (B)</span>. Histogram dihitung secara terpisah untuk tiap channel — masing-masing berisi 256 nilai intensitas (0–255). Klik tab R / G / B untuk beralih antar histogram.
    </div>

    <div class="upload-row">
      <div class="upload-zone" id="mainZone">
        <input type="file" id="mainFile" accept="image/*">
        <span class="upload-icon">🖼️</span>
        <div class="upload-text">Klik atau drag &amp; drop<br><span style="color:#9ca3af;font-size:.72rem">JPG · PNG · BMP · TIFF</span></div>
        <div class="upload-name" id="mainFileName"></div>
      </div>
      <div class="preview-wrap" id="mainPreviewWrap">
        <div class="preview-label">Preview</div>
        <img class="preview-img" id="mainThumb">
      </div>
    </div>

    <div class="action-row">
      <button class="btn btn-blue" id="btnAnalyze" disabled>🔍 Analisis Histogram RGB</button>
      <div class="status-pill" id="statusAnalyze"></div>
    </div>

    <div class="result-area" id="resAnalyze">
      <div class="divider-label">Hasil Analisis</div>
      <div class="result-cols">
        <div class="result-box">
          <div class="result-box-header blue">🖼️ Gambar Asli</div>
          <img class="result-img" id="imgOrig">
        </div>
        <div class="result-box">
          <div class="result-box-header blue">📊 Histogram per Channel</div>
          <div class="hist-tabs" id="tabsAnalyze">
            <div class="hist-tab active-r" onclick="switchHistTab('analyze','R',this)">R — Merah</div>
            <div class="hist-tab" onclick="switchHistTab('analyze','G',this)">G — Hijau</div>
            <div class="hist-tab" onclick="switchHistTab('analyze','B',this)">B — Biru</div>
          </div>
          <div class="hist-pane show" id="pane-analyze-R"><canvas id="chart-analyze-R"></canvas></div>
          <div class="hist-pane"      id="pane-analyze-G"><canvas id="chart-analyze-G"></canvas></div>
          <div class="hist-pane"      id="pane-analyze-B"><canvas id="chart-analyze-B"></canvas></div>
          <div class="chart-note">X = intensitas (0–255) · Y = jumlah piksel · Hover untuk detail</div>
        </div>
      </div>

      <div class="calc-toggle" onclick="toggleCalc('calcAnalyze',this)"><i class="arrow">▶</i> Lihat Tabel Perhitungan Histogram</div>
      <div class="calc-panel" id="calcAnalyze">
        <div class="calc-toolbar">
          <label>Tampilkan:</label>
          <input type="range" id="sliderAnalyze" min="1" max="256" value="20" oninput="updateTable('analyze')">
          <span class="range-val" id="rv-analyze">20 baris</span>
          <span class="total-rows" id="tr-analyze"></span>
        </div>
        <div class="calc-ch-tabs">
          <div class="calc-ch-tab r active" onclick="switchCalcTab('analyze','R',this)">R — Merah</div>
          <div class="calc-ch-tab g" onclick="switchCalcTab('analyze','G',this)">G — Hijau</div>
          <div class="calc-ch-tab b" onclick="switchCalcTab('analyze','B',this)">B — Biru</div>
        </div>
        <div class="calc-scroll">
          <table class="calc-table">
            <thead><tr><th>Intensitas (r)</th><th>Frekuensi</th></tr></thead>
            <tbody id="tbody-analyze"></tbody>
          </table>
        </div>
        <div class="calc-note">Frekuensi = jumlah piksel dengan nilai intensitas tersebut pada channel yang dipilih.</div>
      </div>
    </div>
  </div>
</div>

<!-- ══ STEP 2 — EQUALIZATION ══ -->
<div class="step-card locked" id="step2">
  <div class="step-header">
    <div class="step-num green">2</div>
    <div>
      <div class="step-title">Histogram Equalization (RGB)</div>
      <div class="step-desc">Equalization dilakukan secara independen pada tiap channel R, G, B menggunakan CDF masing-masing.</div>
    </div>
  </div>
  <div class="step-body">

    <div class="info-box green" style="margin-bottom:1.1rem">
      <strong>Cara Kerja:</strong> Tiap channel (R, G, B) diequalize secara terpisah dengan formula yang sama:
      <div class="formula">s = round( (CDF(r) − CDF_min) / (N − CDF_min) × 255 )</div>
      Hasilnya digabung kembali menjadi gambar RGB.
    </div>

    <div class="action-row" style="margin-top:0;padding-top:0;border-top:none">
      <button class="btn btn-green" id="btnEqual" disabled>📈 Jalankan Equalization</button>
      <div class="status-pill" id="statusEqual"></div>
    </div>

    <div class="result-area" id="resEqual">
      <div class="divider-label">Perbandingan Sebelum vs Sesudah</div>
      <div class="result-cols">
        <div class="result-box">
          <div class="result-box-header blue">📷 Sebelum</div>
          <img class="result-img" id="imgBeforeEq">
          <div class="hist-tabs"><div class="hist-tab active-r" onclick="switchHistTab('eq-before','R',this)">R</div><div class="hist-tab" onclick="switchHistTab('eq-before','G',this)">G</div><div class="hist-tab" onclick="switchHistTab('eq-before','B',this)">B</div></div>
          <div class="hist-pane show" id="pane-eq-before-R"><canvas id="chart-eq-before-R"></canvas></div>
          <div class="hist-pane"      id="pane-eq-before-G"><canvas id="chart-eq-before-G"></canvas></div>
          <div class="hist-pane"      id="pane-eq-before-B"><canvas id="chart-eq-before-B"></canvas></div>
          <div class="chart-note">Histogram menumpuk → kontras rendah</div>
        </div>
        <div class="result-box">
          <div class="result-box-header green">✅ Sesudah</div>
          <img class="result-img" id="imgAfterEq">
          <div class="hist-tabs"><div class="hist-tab active-r" onclick="switchHistTab('eq-after','R',this)">R</div><div class="hist-tab" onclick="switchHistTab('eq-after','G',this)">G</div><div class="hist-tab" onclick="switchHistTab('eq-after','B',this)">B</div></div>
          <div class="hist-pane show" id="pane-eq-after-R"><canvas id="chart-eq-after-R"></canvas></div>
          <div class="hist-pane"      id="pane-eq-after-G"><canvas id="chart-eq-after-G"></canvas></div>
          <div class="hist-pane"      id="pane-eq-after-B"><canvas id="chart-eq-after-B"></canvas></div>
          <div class="chart-note">Histogram tersebar merata → kontras meningkat</div>
        </div>
      </div>

      <div class="calc-toggle" onclick="toggleCalc('calcEq',this)"><i class="arrow">▶</i> Lihat Tabel Perhitungan Equalization per Channel</div>
      <div class="calc-panel" id="calcEq">
        <div class="calc-toolbar">
          <label>Tampilkan:</label>
          <input type="range" id="sliderEq" min="1" max="256" value="20" oninput="updateTable('eq')">
          <span class="range-val" id="rv-eq">20 baris</span>
          <span class="total-rows" id="tr-eq"></span>
        </div>
        <div class="calc-ch-tabs">
          <div class="calc-ch-tab r active" onclick="switchCalcTab('eq','R',this)">R — Merah</div>
          <div class="calc-ch-tab g" onclick="switchCalcTab('eq','G',this)">G — Hijau</div>
          <div class="calc-ch-tab b" onclick="switchCalcTab('eq','B',this)">B — Biru</div>
        </div>
        <div class="calc-scroll">
          <table class="calc-table">
            <thead><tr><th>Intensitas (r)</th><th>Frekuensi</th><th>CDF(r)</th><th>CDF−CDF_min</th><th>÷(N−CDF_min)</th><th>×255 = s (LUT)</th></tr></thead>
            <tbody id="tbody-eq"></tbody>
          </table>
        </div>
        <div class="calc-note" id="note-eq"></div>
      </div>
    </div>
  </div>
</div>

<!-- ══ STEP 3 — REFERENSI ══ -->
<div class="step-card locked" id="step3">
  <div class="step-header">
    <div class="step-num purple">3</div>
    <div>
      <div class="step-title">Upload Gambar Referensi</div>
      <div class="step-desc">Unggah gambar yang distribusi warnanya ingin diterapkan ke gambar sumber melalui Histogram Matching.</div>
    </div>
  </div>
  <div class="step-body">
    <div class="info-box purple" style="margin-bottom:1.1rem">
      <strong>Gambar Referensi</strong> — Histogram tiap channel RGB dari gambar ini akan menjadi "target" yang dikejar oleh gambar sumber.
    </div>
    <div class="upload-row">
      <div class="upload-zone" id="refZone">
        <input type="file" id="refFile" accept="image/*">
        <span class="upload-icon">📎</span>
        <div class="upload-text">Klik atau drag &amp; drop<br><span style="color:#9ca3af;font-size:.72rem">Gambar Referensi</span></div>
        <div class="upload-name" id="refFileName"></div>
      </div>
      <div class="preview-wrap" id="refPreviewWrap">
        <div class="preview-label">Preview Referensi</div>
        <img class="preview-img" id="refThumb">
      </div>
      <div id="refHistBox" style="flex:1;min-width:220px;display:none">
        <div class="result-box">
          <div class="result-box-header purple">📊 Histogram Referensi</div>
          <div class="hist-tabs"><div class="hist-tab active-r" onclick="switchHistTab('ref','R',this)">R</div><div class="hist-tab" onclick="switchHistTab('ref','G',this)">G</div><div class="hist-tab" onclick="switchHistTab('ref','B',this)">B</div></div>
          <div class="hist-pane show" id="pane-ref-R"><canvas id="chart-ref-R"></canvas></div>
          <div class="hist-pane"      id="pane-ref-G"><canvas id="chart-ref-G"></canvas></div>
          <div class="hist-pane"      id="pane-ref-B"><canvas id="chart-ref-B"></canvas></div>
          <div class="chart-note">Target distribusi untuk matching</div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ══ STEP 4 — MATCHING ══ -->
<div class="step-card locked" id="step4">
  <div class="step-header">
    <div class="step-num orange">4</div>
    <div>
      <div class="step-title">Histogram Specification / Matching (RGB)</div>
      <div class="step-desc">Sesuaikan distribusi warna gambar sumber agar mendekati warna gambar referensi, per channel R, G, B.</div>
    </div>
  </div>
  <div class="step-body">

    <div class="info-box orange" style="margin-bottom:1.1rem">
      <strong>Cara Kerja:</strong> Untuk tiap channel, cari intensitas <em>z</em> di referensi yang CDF-nya paling mendekati CDF sumber:
      <div class="formula">z = argmin |CDF_ref(z) − CDF_src(r)|</div>
      Ketiga channel hasil digabung menjadi gambar berwarna.
    </div>

    <div class="action-row" style="margin-top:0;padding-top:0;border-top:none">
      <button class="btn btn-orange" id="btnMatch" disabled>🔗 Jalankan Histogram Matching</button>
      <div class="status-pill" id="statusMatch"></div>
    </div>

    <div class="result-area" id="resMatch">
      <div class="divider-label">Perbandingan: Sumber · Referensi · Hasil</div>
      <div class="result-cols triple">
        <div class="result-box">
          <div class="result-box-header blue">📷 Sumber</div>
          <img class="result-img" id="imgMatchSrc">
          <div class="hist-tabs"><div class="hist-tab active-r" onclick="switchHistTab('ms-src','R',this)">R</div><div class="hist-tab" onclick="switchHistTab('ms-src','G',this)">G</div><div class="hist-tab" onclick="switchHistTab('ms-src','B',this)">B</div></div>
          <div class="hist-pane show" id="pane-ms-src-R"><canvas id="chart-ms-src-R"></canvas></div>
          <div class="hist-pane"      id="pane-ms-src-G"><canvas id="chart-ms-src-G"></canvas></div>
          <div class="hist-pane"      id="pane-ms-src-B"><canvas id="chart-ms-src-B"></canvas></div>
          <div class="chart-note">Distribusi awal</div>
        </div>
        <div class="result-box">
          <div class="result-box-header purple">📎 Referensi</div>
          <img class="result-img" id="imgMatchRef">
          <div class="hist-tabs"><div class="hist-tab active-r" onclick="switchHistTab('ms-ref','R',this)">R</div><div class="hist-tab" onclick="switchHistTab('ms-ref','G',this)">G</div><div class="hist-tab" onclick="switchHistTab('ms-ref','B',this)">B</div></div>
          <div class="hist-pane show" id="pane-ms-ref-R"><canvas id="chart-ms-ref-R"></canvas></div>
          <div class="hist-pane"      id="pane-ms-ref-G"><canvas id="chart-ms-ref-G"></canvas></div>
          <div class="hist-pane"      id="pane-ms-ref-B"><canvas id="chart-ms-ref-B"></canvas></div>
          <div class="chart-note">Target distribusi</div>
        </div>
        <div class="result-box">
          <div class="result-box-header orange">✅ Hasil Matching</div>
          <img class="result-img" id="imgMatchOut">
          <div class="hist-tabs"><div class="hist-tab active-r" onclick="switchHistTab('ms-out','R',this)">R</div><div class="hist-tab" onclick="switchHistTab('ms-out','G',this)">G</div><div class="hist-tab" onclick="switchHistTab('ms-out','B',this)">B</div></div>
          <div class="hist-pane show" id="pane-ms-out-R"><canvas id="chart-ms-out-R"></canvas></div>
          <div class="hist-pane"      id="pane-ms-out-G"><canvas id="chart-ms-out-G"></canvas></div>
          <div class="hist-pane"      id="pane-ms-out-B"><canvas id="chart-ms-out-B"></canvas></div>
          <div class="chart-note">Mendekati distribusi referensi</div>
        </div>
      </div>

      <div class="calc-toggle" onclick="toggleCalc('calcMatch',this)"><i class="arrow">▶</i> Lihat Tabel Perhitungan Matching per Channel</div>
      <div class="calc-panel" id="calcMatch">
        <div class="calc-toolbar">
          <label>Tampilkan:</label>
          <input type="range" id="sliderMatch" min="1" max="256" value="20" oninput="updateTable('match')">
          <span class="range-val" id="rv-match">20 baris</span>
          <span class="total-rows" id="tr-match"></span>
        </div>
        <div class="calc-ch-tabs">
          <div class="calc-ch-tab r active" onclick="switchCalcTab('match','R',this)">R — Merah</div>
          <div class="calc-ch-tab g" onclick="switchCalcTab('match','G',this)">G — Hijau</div>
          <div class="calc-ch-tab b" onclick="switchCalcTab('match','B',this)">B — Biru</div>
        </div>
        <div class="calc-scroll">
          <table class="calc-table">
            <thead><tr><th>Intensitas (r)</th><th>Frekuensi</th><th>CDF_src(r)</th><th>CDF_ref(z)</th><th>→ z (baru)</th></tr></thead>
            <tbody id="tbody-match"></tbody>
          </table>
        </div>
        <div class="calc-note">Setiap piksel dengan intensitas r diganti menjadi z — nilai di referensi yang CDF-nya paling dekat.</div>
      </div>
    </div>
  </div>
</div>

</div></div>

<script>
// ── State ──────────────────────────────────────
let srcB64=null, refB64=null;
const charts={};

// Store table data per step per channel
const tableData = { analyze:{}, eq:{}, match:{} };
const calcCh    = { analyze:'R', eq:'R', match:'R' };
const eqMetas   = {};

// Channel colors
const CH_COLOR = { R:'#dc2626', G:'#16a34a', B:'#2563eb' };

// ── Chart ──────────────────────────────────────
function drawChart(id, data, color) {
  const canvas=document.getElementById(id);
  if(!canvas) return;
  if(charts[id]) charts[id].destroy();
  charts[id]=new Chart(canvas.getContext('2d'),{
    type:'line',
    data:{ labels:Array.from({length:256},(_,i)=>i),
      datasets:[{data,fill:true,borderColor:color,backgroundColor:color+'33',
        borderWidth:2,pointRadius:0,tension:0.3}]},
    options:{responsive:true,maintainAspectRatio:false,animation:{duration:300},
      plugins:{legend:{display:false},tooltip:{callbacks:{
        title:i=>'Intensitas: '+i[0].label,
        label:i=>'Jumlah piksel: '+i.raw.toLocaleString('id')}}},
      scales:{x:{display:false},
        y:{display:true,grid:{color:'#f0f0f0'},ticks:{color:'#aaa',font:{size:9},maxTicksLimit:3}}}}
  });
}

function drawRGB(prefix, hists) {
  ['R','G','B'].forEach(ch => drawChart(`chart-${prefix}-${ch}`, hists[ch], CH_COLOR[ch]));
}

// ── Histogram tabs ─────────────────────────────
function switchHistTab(prefix, ch, el) {
  const tabs=el.parentElement.querySelectorAll('.hist-tab');
  tabs.forEach(t=>t.className='hist-tab');
  el.className=`hist-tab active-${ch.toLowerCase()}`;
  const box=el.closest('.result-box');
  box.querySelectorAll('.hist-pane').forEach(p=>p.classList.remove('show'));
  box.querySelector(`#pane-${prefix}-${ch}`).classList.add('show');
}

// ── Calc panel ─────────────────────────────────
function toggleCalc(id, btn) {
  document.getElementById(id).classList.toggle('show');
  btn.classList.toggle('open');
}

function switchCalcTab(step, ch, el) {
  const tabs=el.parentElement.querySelectorAll('.calc-ch-tab');
  tabs.forEach(t=>t.classList.remove('active'));
  el.classList.add('active');
  calcCh[step]=ch;
  updateTable(step);
}

function sliderLabel(v, total) {
  return v>=total ? `Semua (${total})` : `${v} dari ${total}`;
}

function updateTable(step) {
  const ch  = calcCh[step];
  const rows= tableData[step][ch] || [];
  const v   = +document.getElementById(`slider${cap(step)}`).value;
  document.getElementById(`rv-${step}`).textContent = sliderLabel(v, rows.length);
  document.getElementById(`tr-${step}`).textContent = `Total: ${rows.length} intensitas (channel ${ch})`;

  const slice = rows.slice(0, v);
  const hiClass= ch==='R'?'hi-r':ch==='G'?'hi-g':'hi-b';

  let html='';
  if(step==='analyze') {
    html=slice.map(d=>`<tr><td class="num ${hiClass}">${d.r}</td><td class="num">${d.freq.toLocaleString('id')}</td></tr>`).join('');
  } else if(step==='eq') {
    const {cdf_min,N}=eqMetas[ch]||{cdf_min:0,N:1};
    const denom=N-cdf_min;
    html=slice.map(d=>`<tr>
      <td class="num ${hiClass}">${d.r}</td>
      <td class="num">${d.freq.toLocaleString('id')}</td>
      <td class="num">${d.cdf.toLocaleString('id')}</td>
      <td class="num">${(d.cdf-cdf_min).toLocaleString('id')}</td>
      <td class="num">${((d.cdf-cdf_min)/denom).toFixed(4)}</td>
      <td class="num hi-o">${d.lut}</td>
    </tr>`).join('');
  } else if(step==='match') {
    html=slice.map(d=>`<tr>
      <td class="num ${hiClass}">${d.r}</td>
      <td class="num">${d.freq.toLocaleString('id')}</td>
      <td class="num">${d.cdf_s.toFixed(4)}</td>
      <td class="num">${d.cdf_r.toFixed(4)}</td>
      <td class="num hi-o">${d.z}</td>
    </tr>`).join('');
  }
  document.getElementById(`tbody-${step}`).innerHTML=html;
}

function cap(s){return s.charAt(0).toUpperCase()+s.slice(1)}

function initSlider(step, data) {
  // data = {R:[...], G:[...], B:[...]}
  tableData[step]=data;
  const maxLen=Math.max(...Object.values(data).map(d=>d.length));
  const slider=document.getElementById(`slider${cap(step)}`);
  slider.max=maxLen;
  slider.value=Math.min(20,maxLen);
  updateTable(step);
}

// ── Status / unlock ────────────────────────────
function setStatus(id,msg,type){
  const el=document.getElementById(id);
  el.className='status-pill show '+type;
  el.innerHTML=type==='loading'?`<div class="spinner"></div>${msg}`:msg;
}
function unlock(id){document.getElementById(id).classList.remove('locked')}

// ── Upload ─────────────────────────────────────
function setupUpload(inputId,zoneId,thumbId,nameId,wrapId,onLoad){
  const input=document.getElementById(inputId),zone=document.getElementById(zoneId);
  function load(file){
    const reader=new FileReader();
    reader.onload=e=>{
      const b64=e.target.result.split(',')[1];
      document.getElementById(thumbId).src=e.target.result;
      document.getElementById(wrapId).classList.add('show');
      zone.classList.add('filled');
      document.getElementById(nameId).textContent='✓ '+file.name;
      onLoad(b64);
    };
    reader.readAsDataURL(file);
  }
  input.addEventListener('change',()=>input.files[0]&&load(input.files[0]));
  zone.addEventListener('dragover',e=>{e.preventDefault();zone.style.borderColor='var(--blue)'});
  zone.addEventListener('dragleave',()=>zone.style.borderColor='');
  zone.addEventListener('drop',e=>{e.preventDefault();zone.style.borderColor='';if(e.dataTransfer.files[0])load(e.dataTransfer.files[0])});
}

async function api(url,body){
  const res=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  return res.json();
}

// ── Step 1: Analyze ────────────────────────────
setupUpload('mainFile','mainZone','mainThumb','mainFileName','mainPreviewWrap',b64=>{
  srcB64=b64;
  document.getElementById('btnAnalyze').disabled=false;
});

document.getElementById('btnAnalyze').addEventListener('click',async()=>{
  setStatus('statusAnalyze','Menganalisis...','loading');
  const data=await api('/api/analyze',{image:srcB64});
  if(data.error){setStatus('statusAnalyze','✗ '+data.error,'err');return;}

  srcB64=data.image;
  document.getElementById('imgOrig').src='data:image/png;base64,'+data.image;
  drawRGB('analyze', data.histograms);
  document.getElementById('resAnalyze').classList.add('show');
  setStatus('statusAnalyze',`✓ Selesai · ${data.width}×${data.height} px`,'ok');

  initSlider('analyze', data.tables);
  unlock('step2'); document.getElementById('btnEqual').disabled=false;
  unlock('step3');
});

// ── Step 2: Equalize ───────────────────────────
document.getElementById('btnEqual').addEventListener('click',async()=>{
  setStatus('statusEqual','Menghitung CDF & equalisasi...','loading');
  const data=await api('/api/equalize',{image:srcB64});
  if(data.error){setStatus('statusEqual','✗ '+data.error,'err');return;}

  document.getElementById('imgBeforeEq').src='data:image/png;base64,'+srcB64;
  // reuse analyze histograms for "before"
  ['R','G','B'].forEach(ch=>{
    const orig=charts[`chart-analyze-${ch}`];
    if(orig) drawChart(`chart-eq-before-${ch}`,orig.data.datasets[0].data.slice(),CH_COLOR[ch]);
  });

  document.getElementById('imgAfterEq').src='data:image/png;base64,'+data.image;
  drawRGB('eq-after', data.histograms);
  document.getElementById('resEqual').classList.add('show');
  setStatus('statusEqual','✓ Equalization selesai','ok');

  // store metas per channel
  ['R','G','B'].forEach(ch=>eqMetas[ch]=data.metas[ch]);
  const note=document.getElementById('note-eq');
  note.innerHTML=['R','G','B'].map(ch=>{
    const m=data.metas[ch];
    return `<span style="color:${CH_COLOR[ch]};font-weight:600">${ch}</span>: N=${m.N.toLocaleString('id')} · CDF_min=${m.cdf_min} · penyebut=${(m.N-m.cdf_min).toLocaleString('id')}`;
  }).join(' &nbsp;|&nbsp; ');

  initSlider('eq', data.tables);
});

// ── Step 3: Upload ref ─────────────────────────
setupUpload('refFile','refZone','refThumb','refFileName','refPreviewWrap',async b64=>{
  const data=await api('/api/analyze',{image:b64});
  if(!data.error){
    refB64=data.image;
    document.getElementById('refHistBox').style.display='block';
    drawRGB('ref', data.histograms);
  } else { refB64=b64; }
  if(srcB64){ unlock('step4'); document.getElementById('btnMatch').disabled=false; }
});

// ── Step 4: Match ──────────────────────────────
document.getElementById('btnMatch').addEventListener('click',async()=>{
  setStatus('statusMatch','Mencocokkan histogram...','loading');
  const data=await api('/api/match',{source:srcB64,reference:refB64});
  if(data.error){setStatus('statusMatch','✗ '+data.error,'err');return;}

  document.getElementById('imgMatchSrc').src='data:image/png;base64,'+srcB64;
  document.getElementById('imgMatchRef').src='data:image/png;base64,'+refB64;
  document.getElementById('imgMatchOut').src='data:image/png;base64,'+data.image;

  ['R','G','B'].forEach(ch=>{
    const orig=charts[`chart-analyze-${ch}`];
    if(orig) drawChart(`chart-ms-src-${ch}`,orig.data.datasets[0].data.slice(),CH_COLOR[ch]);
    const ref=charts[`chart-ref-${ch}`];
    if(ref) drawChart(`chart-ms-ref-${ch}`,ref.data.datasets[0].data.slice(),CH_COLOR[ch]);
  });
  drawRGB('ms-out', data.histograms);

  document.getElementById('resMatch').classList.add('show');
  setStatus('statusMatch','✓ Matching selesai','ok');
  initSlider('match', data.tables);
});
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return HTML

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)