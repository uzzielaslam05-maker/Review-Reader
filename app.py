"""
IMDb Sentiment Classifier — PythonAnywhere-ready version.

Why this file looks different from the local version:
PythonAnywhere's free tier caps you at 512MB of disk space, and full
TensorFlow alone is ~1.9GB installed — it simply won't fit. This version
swaps it for two lightweight pieces that add up to a few dozen MB instead:

  1. The model is exported as .tflite (imdb_sentiment.tflite) instead of
     .keras, and run with `ai-edge-litert` (~48MB) instead of `tensorflow`
     (~1.9GB). Same architecture, same trained weights, same predictions —
     just a lighter-weight execution format.
  2. The tokenizer is exported as plain JSON (tokenizer.json) instead of a
     pickled Keras Tokenizer object, because unpickling a Keras Tokenizer
     still requires TensorFlow to be installed just to read the file. This
     version reimplements tokenizing and padding in plain Python, verified
     to produce byte-identical output to the original Keras pipeline.

Setup on PythonAnywhere:
    pip install --user flask numpy ai-edge-litert

Files needed in the same folder:
    app.py, imdb_sentiment.tflite, tokenizer.json

Then point your PythonAnywhere Web app's WSGI config at this file's `app`
object (see the deployment steps for the exact line to add).
"""

import re
import string
import json

import numpy as np
from flask import Flask, request, jsonify, render_template_string
from ai_edge_litert.interpreter import Interpreter

MAX_LEN = 120

print("Loading tokenizer...")
with open("tokenizer.json") as f:
    _tok = json.load(f)
WORD_INDEX = _tok["word_index"]
NUM_WORDS = _tok["num_words"]
OOV_INDEX = WORD_INDEX[_tok["oov_token"]]

print("Loading TFLite model...")
interpreter = Interpreter(model_path="imdb_sentiment.tflite")
interpreter.allocate_tensors()
INPUT_DETAILS = interpreter.get_input_details()
OUTPUT_DETAILS = interpreter.get_output_details()
print("Ready.")

app = Flask(__name__)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def texts_to_sequence(text: str) -> list:
    cleaned = clean_text(text)
    words = cleaned.split(" ") if cleaned else []
    seq = []
    for w in words:
        idx = WORD_INDEX.get(w)
        seq.append(idx if idx is not None and idx < NUM_WORDS else OOV_INDEX)
    return seq


def pad_pre(seq: list, maxlen: int = MAX_LEN) -> list:
    seq = seq[:maxlen]                      # truncating="post": cut the end
    if len(seq) < maxlen:
        seq = [0] * (maxlen - len(seq)) + seq  # padding="pre": pad the front
    return seq


def predict(text: str):
    seq = texts_to_sequence(text)
    padded = np.array([pad_pre(seq)], dtype=np.float32)
    interpreter.set_tensor(INPUT_DETAILS[0]["index"], padded)
    interpreter.invoke()
    prob = float(interpreter.get_tensor(OUTPUT_DETAILS[0]["index"])[0][0])
    label = "Positive" if prob >= 0.5 else "Negative"
    confidence = prob if prob >= 0.5 else 1 - prob
    return label, prob, confidence


PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sentiment — Screening Room</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --ink: #0b0c0f;
    --panel: #15171c;
    --panel-2: #191c22;
    --line: rgba(255,255,255,0.08);
    --line-soft: rgba(255,255,255,0.05);
    --text: #f1efe7;
    --muted: #8b909a;
    --muted-2: #5c616b;
    --gold: #e3b341;
    --gold-soft: rgba(227,179,65,0.14);
    --teal: #4fa8a0;
    --neg: #d0685c;
    --neg-soft: rgba(208,104,92,0.14);
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0;
    background: var(--ink);
    color: var(--text);
    font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 32px 20px;
    position: relative;
    overflow-x: hidden;
  }
  .glow {
    position: fixed;
    inset: -20%;
    background:
      radial-gradient(600px circle at 20% 20%, rgba(227,179,65,0.07), transparent 60%),
      radial-gradient(500px circle at 85% 75%, rgba(79,168,160,0.06), transparent 60%);
    filter: blur(10px);
    animation: drift 18s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
  }
  @keyframes drift {
    0%   { transform: translate(0px, 0px) scale(1); }
    100% { transform: translate(30px, -20px) scale(1.05); }
  }
  .grain {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 1;
    opacity: 0.035;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  }
  .wrap { position: relative; z-index: 2; width: 100%; max-width: 620px; }
  .top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 18px;
    padding: 0 4px;
  }
  .eyebrow {
    font-size: 11px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--gold);
    font-family: "JetBrains Mono", monospace;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--gold);
    box-shadow: 0 0 8px var(--gold);
    animation: pulse 2s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }
  .modelbadge {
    font-family: "JetBrains Mono", monospace;
    font-size: 10.5px;
    color: var(--muted-2);
    border: 1px solid var(--line);
    padding: 4px 9px;
    border-radius: 100px;
  }
  .card {
    background: linear-gradient(180deg, var(--panel-2), var(--panel));
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 36px;
    box-shadow: 0 30px 60px -20px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.03);
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.5s ease;
  }
  .card.positive { box-shadow: 0 30px 70px -20px rgba(227,179,65,0.15), inset 0 1px 0 rgba(255,255,255,0.03); }
  .card.negative { box-shadow: 0 30px 70px -20px rgba(208,104,92,0.15), inset 0 1px 0 rgba(255,255,255,0.03); }
  h1 {
    font-family: "Fraunces", serif;
    font-size: 28px;
    font-weight: 600;
    margin: 0 0 6px;
    letter-spacing: -0.01em;
  }
  .sub { font-size: 13px; color: var(--muted); margin-bottom: 26px; }
  .field { position: relative; }
  textarea {
    width: 100%;
    min-height: 132px;
    background: #0c0e11;
    border: 1px solid var(--line);
    border-radius: 12px;
    color: var(--text);
    padding: 16px;
    font-size: 15px;
    line-height: 1.55;
    font-family: inherit;
    resize: vertical;
    outline: none;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
  }
  textarea::placeholder { color: var(--muted-2); }
  textarea:focus { border-color: var(--gold); box-shadow: 0 0 0 4px var(--gold-soft); }
  .charcount {
    position: absolute;
    bottom: 10px; right: 12px;
    font-size: 10.5px;
    font-family: "JetBrains Mono", monospace;
    color: var(--muted-2);
    pointer-events: none;
  }
  .chips { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 12px; }
  .chip {
    font-size: 12px;
    color: var(--muted);
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--line);
    padding: 6px 12px;
    border-radius: 100px;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  .chip:hover { border-color: var(--gold); color: var(--gold); background: var(--gold-soft); }
  button.predict {
    margin-top: 18px;
    width: 100%;
    padding: 15px;
    border: none;
    border-radius: 12px;
    background: linear-gradient(135deg, #f0c25b, var(--gold));
    color: #14161a;
    font-weight: 700;
    font-size: 15px;
    cursor: pointer;
    position: relative;
    overflow: hidden;
    transition: transform 0.15s ease, filter 0.2s ease;
  }
  button.predict::after {
    content: "";
    position: absolute; inset: 0;
    background: linear-gradient(120deg, transparent 20%, rgba(255,255,255,0.35) 50%, transparent 80%);
    transform: translateX(-120%);
  }
  button.predict:hover:not(:disabled)::after { transform: translateX(120%); transition: transform 0.7s ease; }
  button.predict:hover:not(:disabled) { filter: brightness(1.05); }
  button.predict:active:not(:disabled) { transform: scale(0.985); }
  button.predict:disabled { opacity: 0.55; cursor: default; }
  .spinner {
    display: none;
    width: 15px; height: 15px;
    border: 2px solid rgba(20,22,26,0.25);
    border-top-color: #14161a;
    border-radius: 50%;
    margin-right: 8px;
    animation: spin 0.7s linear infinite;
  }
  button.loading .spinner { display: inline-block; }
  button.loading .btn-label { font-size: 0; }
  button.loading .btn-label::after { content: "Analyzing…"; font-size: 15px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .btn-inner { display: flex; align-items: center; justify-content: center; }
  .error { color: var(--neg); font-size: 12.5px; margin-top: 10px; display: none; font-family: "JetBrains Mono", monospace; }
  .result {
    margin-top: 24px;
    padding: 22px;
    border-radius: 14px;
    max-height: 0;
    opacity: 0;
    overflow: hidden;
    transition: max-height 0.5s cubic-bezier(.2,.8,.2,1), opacity 0.4s ease, padding 0.4s ease, margin 0.4s ease;
  }
  .result.show { max-height: 220px; opacity: 1; }
  .result.positive { background: var(--gold-soft); border: 1px solid rgba(227,179,65,0.35); }
  .result.negative { background: var(--neg-soft); border: 1px solid rgba(208,104,92,0.35); }
  .result-top { display: flex; align-items: center; gap: 12px; }
  .icon {
    width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 17px;
    flex-shrink: 0;
    transform: scale(0);
    animation: pop 0.45s cubic-bezier(.34,1.56,.64,1) forwards;
    animation-delay: 0.1s;
  }
  @keyframes pop { to { transform: scale(1); } }
  .positive .icon { background: rgba(227,179,65,0.25); }
  .negative .icon { background: rgba(208,104,92,0.25); }
  .label { font-family: "Fraunces", serif; font-size: 22px; font-weight: 600; }
  .positive .label { color: var(--gold); }
  .negative .label { color: var(--neg); }
  .bar-track { height: 8px; background: rgba(255,255,255,0.06); border-radius: 4px; overflow: hidden; margin-top: 16px; }
  .bar-fill { height: 100%; width: 0%; border-radius: 4px; transition: width 0.9s cubic-bezier(.16,1,.3,1); }
  .positive .bar-fill { background: linear-gradient(90deg, #c99a2e, var(--gold)); }
  .negative .bar-fill { background: linear-gradient(90deg, #a84a3f, var(--neg)); }
  .meta-row { display: flex; justify-content: space-between; margin-top: 9px; font-size: 11px; font-family: "JetBrains Mono", monospace; color: var(--muted); }
  .history { margin-top: 26px; border-top: 1px solid var(--line-soft); padding-top: 16px; }
  .history-title { font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted-2); font-family: "JetBrains Mono", monospace; margin-bottom: 10px; }
  .history-item { display: flex; align-items: center; gap: 10px; padding: 8px 0; font-size: 12.5px; color: var(--muted); border-bottom: 1px solid var(--line-soft); cursor: pointer; }
  .history-item:last-child { border-bottom: none; }
  .history-item:hover .htext { color: var(--text); }
  .history-item .hdot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
  .history-item.positive .hdot { background: var(--gold); }
  .history-item.negative .hdot { background: var(--neg); }
  .htext { flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; transition: color 0.2s ease; }
  .hpct { font-family: "JetBrains Mono", monospace; font-size: 11px; color: var(--muted-2); }
  @media (max-width: 480px) { .card { padding: 24px; } h1 { font-size: 23px; } }
</style>
</head>
<body>
<div class="glow"></div>
<div class="grain"></div>
<div class="wrap">
  <div class="top">
    <div class="eyebrow"><span class="dot"></span> Live Inference</div>
    <div class="modelbadge">SimpleRNN · TFLite</div>
  </div>
  <div class="card" id="card">
    <h1>Sentiment, read in real time</h1>
    <div class="sub">Type a review below — the model reads it the moment you ask it to.</div>
    <div class="field">
      <textarea id="reviewInput" placeholder="e.g. The pacing dragged in the middle, but that ending redeemed the whole film..." maxlength="600"></textarea>
      <div class="charcount" id="charcount">0 / 600</div>
    </div>
    <div class="chips">
      <div class="chip" data-text="An absolute triumph — beautifully shot, brilliantly acted, and endlessly rewatchable.">✦ triumphant</div>
      <div class="chip" data-text="Painfully slow, poorly written, and I regret every minute I spent on it.">✦ scathing</div>
      <div class="chip" data-text="Great acting, but the plot fell apart in the third act.">✦ mixed</div>
      <div class="chip" data-text="A solid Sunday-afternoon watch — nothing groundbreaking, but I enjoyed it.">✦ mild praise</div>
    </div>
    <button class="predict" id="predictBtn">
      <div class="btn-inner"><div class="spinner"></div><span class="btn-label">Predict sentiment</span></div>
    </button>
    <div class="error" id="errorMsg"></div>
    <div class="result" id="result">
      <div class="result-top">
        <div class="icon" id="resultIcon"></div>
        <div class="label" id="resultLabel"></div>
      </div>
      <div class="bar-track"><div class="bar-fill" id="resultBar"></div></div>
      <div class="meta-row"><span id="resultConfidence"></span><span id="resultScore"></span></div>
    </div>
    <div class="history" id="historyWrap" style="display:none;">
      <div class="history-title">Recent</div>
      <div id="historyList"></div>
    </div>
  </div>
</div>
<script>
const btn = document.getElementById('predictBtn');
const input = document.getElementById('reviewInput');
const charcount = document.getElementById('charcount');
const card = document.getElementById('card');
const result = document.getElementById('result');
const resultIcon = document.getElementById('resultIcon');
const label = document.getElementById('resultLabel');
const bar = document.getElementById('resultBar');
const confidence = document.getElementById('resultConfidence');
const scoreEl = document.getElementById('resultScore');
const errorMsg = document.getElementById('errorMsg');
const historyWrap = document.getElementById('historyWrap');
const historyList = document.getElementById('historyList');
let history = [];
input.addEventListener('input', () => { charcount.textContent = `${input.value.length} / 600`; });
document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    input.value = chip.dataset.text;
    charcount.textContent = `${input.value.length} / 600`;
    input.focus();
  });
});
function animateCount(el, target, suffix, duration) {
  const start = performance.now();
  function frame(now) {
    const p = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = (target * eased).toFixed(1) + suffix;
    if (p < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}
function renderHistory() {
  if (!history.length) { historyWrap.style.display = 'none'; return; }
  historyWrap.style.display = 'block';
  historyList.innerHTML = history.slice(0, 5).map(h => `
    <div class="history-item ${h.label.toLowerCase()}" data-text="${h.text.replace(/"/g,'&quot;')}">
      <div class="hdot"></div>
      <div class="htext">${h.text}</div>
      <div class="hpct">${(h.confidence*100).toFixed(0)}%</div>
    </div>
  `).join('');
  historyList.querySelectorAll('.history-item').forEach(item => {
    item.addEventListener('click', () => {
      input.value = item.dataset.text;
      charcount.textContent = `${input.value.length} / 600`;
    });
  });
}
async function runPredict() {
  const text = input.value.trim();
  errorMsg.style.display = 'none';
  if (!text) { input.focus(); return; }
  btn.disabled = true;
  btn.classList.add('loading');
  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    if (!res.ok) throw new Error('Server error');
    const data = await res.json();
    result.className = 'result show ' + data.label.toLowerCase();
    card.className = 'card ' + data.label.toLowerCase();
    resultIcon.textContent = data.label === 'Positive' ? '✓' : '✕';
    label.textContent = data.label;
    scoreEl.textContent = `raw score ${data.prob.toFixed(4)}`;
    bar.style.width = '0%';
    void bar.offsetWidth;
    bar.style.width = (data.confidence * 100).toFixed(1) + '%';
    animateCount(confidence, data.confidence * 100, '% confidence', 700);
    history.unshift({ text, label: data.label, confidence: data.confidence });
    renderHistory();
  } catch (e) {
    errorMsg.textContent = 'Something went wrong — is the server still running?';
    errorMsg.style.display = 'block';
  } finally {
    btn.disabled = false;
    btn.classList.remove('loading');
  }
}
btn.addEventListener('click', runPredict);
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) runPredict();
});
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(PAGE)


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "empty text"}), 400
    label, prob, confidence = predict(text)
    return jsonify({"label": label, "prob": prob, "confidence": confidence})


if __name__ == "__main__":
    # Local testing only. On PythonAnywhere, the WSGI config imports `app`
    # directly and this block never runs.
    app.run(debug=False, host="0.0.0.0", port=5000)
