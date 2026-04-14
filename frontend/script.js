// ============================================================
// FoodAI Vision — Premium JS v2
// ============================================================

(function () {
  'use strict';

  // ── Inject SVG gradient for ring ──────────────────────────
  const svgNs = 'http://www.w3.org/2000/svg';
  const defs = document.createElementNS(svgNs, 'svg');
  defs.setAttribute('width', '0');
  defs.setAttribute('height', '0');
  defs.style.cssText = 'position:absolute;pointer-events:none;';
  defs.innerHTML = `<defs>
    <linearGradient id="ringGradient" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#7c3aed"/>
      <stop offset="100%" stop-color="#06b6d4"/>
    </linearGradient>
  </defs>`;
  document.body.prepend(defs);

  // ── Particles ─────────────────────────────────────────────
  const particleBox = document.getElementById('bgParticles');
  if (particleBox) {
    const colors = ['#7c3aed','#06b6d4','#a78bfa','#22d3ee','#10b981'];
    for (let i = 0; i < 35; i++) {
      const p = document.createElement('div');
      p.className = 'particle';
      const sz = Math.random() * 3.5 + 1;
      const c = colors[Math.floor(Math.random() * colors.length)];
      p.style.cssText = `width:${sz}px;height:${sz}px;left:${Math.random()*100}vw;background:${c};color:${c};animation-duration:${Math.random()*22+14}s;animation-delay:${Math.random()*-25}s;box-shadow:0 0 ${sz*3}px currentColor;`;
      particleBox.appendChild(p);
    }
  }

  // ── DOM refs ──────────────────────────────────────────────
  const $ = id => document.getElementById(id);

  const dropZone         = $('dropZone');
  const dropZoneInner    = $('dropZoneInner');
  const imageInput       = $('imageInput');
  const browseBtn        = $('browseBtn');
  const predictBtn       = $('predictButton');
  const btnContent       = $('btnContent');
  const btnLoader        = $('btnLoader');
  const previewContainer = $('previewContainer');
  const previewImg       = $('previewImg');
  const previewName      = $('previewName');
  const previewMeta      = $('previewMeta');
  const removeBtn        = $('removeBtn');
  const scanOverlay      = $('scanOverlay');

  const resultSection    = $('resultSection');
  const resultIcon       = $('resultIcon');
  const resultLabel      = $('resultLabel');
  const resultTimestamp   = $('resultTimestamp');
  const inferenceChip    = $('inferenceChip');
  const inferenceMs      = $('inferenceMs');
  const predictionTag    = $('predictionTag');
  const ringFill         = $('ringFill');
  const ringPct          = $('ringPct');
  const probabilityBars  = $('probabilityBars');
  const interpIcon       = $('interpIcon');
  const interpTitle      = $('interpTitle');
  const interpDesc       = $('interpDesc');
  const reanalyzeBtn     = $('reanalyzeBtn');
  const downloadBtn      = $('downloadBtn');
  const resultImg        = $('resultImg');
  const imgInfoPills     = $('imgInfoPills');
  const imgMetaRow       = $('imgMetaRow');
  const metaFilenameVal  = $('metaFilenameVal');
  const metaSizeVal      = $('metaSizeVal');
  const metaDimsVal      = $('metaDimsVal');
  const metaFormatVal    = $('metaFormatVal');

  const navDetect        = $('navDetect');
  const navHistory       = $('navHistory');
  const detectView       = $('detectView');
  const historySection   = $('historySection');
  const historyList      = $('historyList');
  const historyEmpty     = $('historyEmpty');
  const clearHistoryBtn  = $('clearHistoryBtn');
  const histBadge        = $('histBadge');

  const modelStatus      = $('modelStatus');
  const statusDot        = $('statusDot');
  const statusText       = $('statusText');
  const sbarDevice       = $('sbarDevice');
  const sbarPredictions  = $('sbarPredictions');
  const sbarUptime       = $('sbarUptime');
  const toastContainer   = $('toastContainer');

  // ── State ─────────────────────────────────────────────────
  let currentFile = null;
  let lastResult  = null;
  let history = JSON.parse(localStorage.getItem('foodai_history') || '[]');

  // ── Toast ─────────────────────────────────────────────────
  function toast(msg, type = 'info') {
    const el = document.createElement('div');
    el.className = `toast toast--${type}`;
    el.textContent = msg;
    toastContainer.appendChild(el);
    el.addEventListener('animationend', e => {
      if (e.animationName === 'toastOut') el.remove();
    });
    setTimeout(() => el.remove(), 3600);
  }

  // ── Health / Stats ────────────────────────────────────────
  async function fetchHealth() {
    try {
      const r = await fetch('/health');
      const d = await r.json();
      modelStatus.classList.remove('hidden');
      if (d.model && d.model.loaded) {
        statusDot.className = 'status-dot ok';
        statusText.textContent = 'Model Ready';
      } else {
        statusDot.className = 'status-dot warn';
        statusText.textContent = 'Model Unavailable';
      }
      if (d.model) sbarDevice.textContent = d.model.device || '—';
      if (d.server) {
        sbarPredictions.textContent = d.server.total_predictions ?? '—';
        sbarUptime.textContent = d.server.uptime || '—';
      }
    } catch {
      modelStatus.classList.remove('hidden');
      statusDot.className = 'status-dot err';
      statusText.textContent = 'Offline';
    }
  }
  fetchHealth();
  setInterval(fetchHealth, 15000);

  // ── Drag & Drop ───────────────────────────────────────────
  ['dragenter','dragover','dragleave','drop'].forEach(ev =>
    dropZone.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); })
  );
  dropZone.addEventListener('dragenter', () => dropZone.classList.add('drag-over'));
  dropZone.addEventListener('dragover',  () => dropZone.classList.add('drag-over'));
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) selectFile(file);
    else toast('Please drop a valid image file.', 'error');
  });

  // clicks
  dropZone.addEventListener('click', e => {
    if (e.target.closest('#removeBtn')) return;
    if (previewContainer.classList.contains('hidden')) imageInput.click();
  });
  dropZone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); imageInput.click(); }});
  browseBtn.addEventListener('click', e => { e.stopPropagation(); imageInput.click(); });
  imageInput.addEventListener('change', () => { if (imageInput.files.length) selectFile(imageInput.files[0]); });
  removeBtn.addEventListener('click', e => { e.stopPropagation(); clearFile(); });

  function selectFile(file) {
    if (file.size > 10 * 1024 * 1024) { toast('File too large (max 10 MB).', 'error'); return; }
    currentFile = file;
    const reader = new FileReader();
    reader.onload = ev => {
      previewImg.src = ev.target.result;
      previewName.textContent = file.name;

      // get image dimensions
      const img = new Image();
      img.onload = () => {
        previewMeta.innerHTML = '';
        const pills = [
          `${img.naturalWidth}×${img.naturalHeight}`,
          formatBytes(file.size),
          file.type.split('/')[1].toUpperCase(),
        ];
        pills.forEach(t => {
          const s = document.createElement('span');
          s.className = 'meta-pill';
          s.textContent = t;
          previewMeta.appendChild(s);
        });
      };
      img.src = ev.target.result;

      dropZoneInner.classList.add('hidden');
      previewContainer.classList.remove('hidden');
      predictBtn.disabled = false;
      resultSection.classList.add('hidden');
    };
    reader.readAsDataURL(file);
  }

  function clearFile() {
    currentFile = null;
    lastResult = null;
    imageInput.value = '';
    previewImg.src = '';
    previewContainer.classList.add('hidden');
    dropZoneInner.classList.remove('hidden');
    predictBtn.disabled = true;
    resultSection.classList.add('hidden');
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }

  // ── Prediction ────────────────────────────────────────────
  predictBtn.addEventListener('click', runPrediction);

  async function runPrediction() {
    if (!currentFile) return;
    setLoading(true);

    const form = new FormData();
    form.append('file', currentFile);

    try {
      const r = await fetch('/predict', { method: 'POST', body: form });
      const d = await r.json();
      if (!r.ok) { showError(d.detail || JSON.stringify(d)); return; }
      lastResult = d;
      displayResult(d);
      toast('Analysis complete!', 'success');
      fetchHealth();               // refresh stats bar
    } catch (err) {
      showError(`Request failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  function setLoading(on) {
    predictBtn.disabled = on;
    btnContent.classList.toggle('hidden', on);
    btnLoader.classList.toggle('hidden', !on);
    scanOverlay.classList.toggle('hidden', !on);
  }

  // ── Display Result ────────────────────────────────────────
  const CLASS_META = {
    spoilage_detection: {
      label: '⚠️ Spoilage Detected',
      icon: '⚠️',
      tagText: 'Spoilage Detected',
      tagCls: 'spoilage_detection',
      interpIcon: '⚠️',
      interpTitle: 'Spoilage Indicators Found',
      getDesc: c => `The model detected visual signs of food spoilage with ${pct(c)}% confidence. Degradation patterns, discoloration, or mold-like textures were identified in the image. It is recommended to discard this food item.`,
    },
    ai_detection: {
      label: '🤖 AI-Generated Image',
      icon: '🤖',
      tagText: 'AI-Generated',
      tagCls: 'ai_detection',
      interpIcon: '🤖',
      interpTitle: 'AI-Generated Image Detected',
      getDesc: c => `This food image is likely AI-generated with ${pct(c)}% confidence. Pixel-level artifacts typical of GAN or diffusion model outputs were identified.`,
    },
  };

  function pct(v) { return (v * 100).toFixed(1); }

  function displayResult(data) {
    const { predicted_class, confidence, probabilities, metadata } = data;
    const cfg = CLASS_META[predicted_class] || {
      label: predicted_class, icon: '🔍', tagText: predicted_class, tagCls: '',
      interpIcon: 'ℹ️', interpTitle: 'Analysis Complete',
      getDesc: c => `Predicted: ${predicted_class} (${pct(c)}% confidence).`,
    };

    // Header
    resultIcon.textContent = cfg.icon;
    resultLabel.textContent = cfg.label;
    resultTimestamp.textContent = new Date().toLocaleTimeString([], { hour:'2-digit', minute:'2-digit', second:'2-digit' });

    // Inference chip
    if (metadata && metadata.inference_ms != null) {
      inferenceChip.hidden = false;
      inferenceMs.textContent = metadata.inference_ms;
    } else {
      inferenceChip.hidden = true;
    }

    // Prediction tag
    predictionTag.textContent = cfg.tagText;
    predictionTag.className = `prediction-tag ${cfg.tagCls}`;

    // Confidence ring
    const c = 2 * Math.PI * 50;
    const off = c - c * confidence;
    ringFill.style.strokeDashoffset = c;
    requestAnimationFrame(() => {
      requestAnimationFrame(() => { ringFill.style.strokeDashoffset = off; });
    });
    animateCount(ringPct, 0, Math.round(confidence * 100), 1200, v => v + '%');

    // Probabilities
    probabilityBars.innerHTML = '';
    Object.entries(probabilities).forEach(([cls, prob], i) => {
      const pv = (prob * 100).toFixed(1);
      const name = cls.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
      const el = document.createElement('div');
      el.className = 'prob-item';
      el.style.animationDelay = `${i * 0.12}s`;
      el.innerHTML = `
        <div class="prob-header"><span class="prob-name">${name}</span><span class="prob-value">${pv}%</span></div>
        <div class="prob-bar-bg"><div class="prob-bar-fill" data-w="${pv}"></div></div>`;
      probabilityBars.appendChild(el);
    });
    setTimeout(() => {
      document.querySelectorAll('.prob-bar-fill').forEach(b => { b.style.width = b.dataset.w + '%'; });
    }, 80);

    // Interpretation
    interpIcon.textContent = cfg.interpIcon;
    interpTitle.textContent = cfg.interpTitle;
    interpDesc.textContent = cfg.getDesc(confidence);

    // Result image
    if (previewImg.src) resultImg.src = previewImg.src;
    imgInfoPills.innerHTML = '';
    if (metadata) {
      const pills = [];
      if (metadata.inference_ms != null) pills.push(`${metadata.inference_ms} ms`);
      if (metadata.device) pills.push(metadata.device.toUpperCase());
      if (metadata.prediction_id) pills.push(`#${metadata.prediction_id}`);
      pills.forEach(t => {
        const s = document.createElement('span');
        s.className = 'iip'; s.textContent = t;
        imgInfoPills.appendChild(s);
      });

      imgMetaRow.hidden = false;
      metaFilenameVal.textContent = metadata.filename || '—';
      metaSizeVal.textContent = metadata.file_size_kb ? metadata.file_size_kb + ' KB' : '—';
      metaFormatVal.textContent = currentFile ? currentFile.type.split('/')[1].toUpperCase() : '—';
      // dimensions from img element
      const img = new Image();
      img.onload = () => { metaDimsVal.textContent = `${img.naturalWidth} × ${img.naturalHeight}`; };
      img.src = previewImg.src;
    } else {
      imgMetaRow.hidden = true;
    }

    // Show
    resultSection.classList.remove('hidden');
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // History
    addToHistory(data, currentFile);
  }

  function showError(msg) {
    resultSection.classList.remove('hidden');
    resultIcon.textContent = '❌';
    resultLabel.textContent = 'Detection Failed';
    predictionTag.textContent = 'Error';
    predictionTag.className = 'prediction-tag';
    ringPct.textContent = '—';
    ringFill.style.strokeDashoffset = 314;
    probabilityBars.innerHTML = '';
    interpIcon.textContent = '❌';
    interpTitle.textContent = 'Error';
    interpDesc.textContent = msg;
    inferenceChip.hidden = true;
    imgInfoPills.innerHTML = '';
    imgMetaRow.hidden = true;
    resultImg.src = previewImg.src || '';
    toast(msg, 'error');
  }

  function animateCount(el, from, to, dur, fmt) {
    const t0 = performance.now();
    (function tick(now) {
      const p = Math.min((now - t0) / dur, 1);
      const e = 1 - Math.pow(1 - p, 3);
      el.textContent = fmt(Math.round(from + (to - from) * e));
      if (p < 1) requestAnimationFrame(tick);
    })(t0);
  }

  // ── Download JSON ─────────────────────────────────────────
  downloadBtn.addEventListener('click', () => {
    if (!lastResult) return;
    const blob = new Blob([JSON.stringify(lastResult, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `foodai_result_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast('JSON downloaded!', 'success');
  });

  // ── Re-analyze ────────────────────────────────────────────
  reanalyzeBtn.addEventListener('click', () => {
    clearFile();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  // ── Navigation ────────────────────────────────────────────
  function showView(view) {
    const isDetect = view === 'detect';
    detectView.classList.toggle('hidden', !isDetect);
    historySection.classList.toggle('hidden', isDetect);
    navDetect.classList.toggle('chip--active', isDetect);
    navDetect.setAttribute('aria-pressed', isDetect);
    navHistory.classList.toggle('chip--active', !isDetect);
    navHistory.setAttribute('aria-pressed', !isDetect);
    if (!isDetect) renderHistory();
  }

  navDetect.addEventListener('click', () => showView('detect'));
  navDetect.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') showView('detect'); });
  navHistory.addEventListener('click', () => showView('history'));
  navHistory.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') showView('history'); });

  // ── History ───────────────────────────────────────────────
  function updateBadge() {
    if (history.length > 0) {
      histBadge.textContent = history.length;
      histBadge.classList.remove('hidden');
    } else {
      histBadge.classList.add('hidden');
    }
  }
  updateBadge();

  function addToHistory(data, file) {
    const reader = new FileReader();
    reader.onload = ev => {
      // downscale thumbnail
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        const MAX = 80;
        const ratio = Math.min(MAX / img.width, MAX / img.height, 1);
        canvas.width = img.width * ratio;
        canvas.height = img.height * ratio;
        canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
        const thumb = canvas.toDataURL('image/jpeg', 0.6);

        history.unshift({
          id: Date.now(),
          filename: file.name,
          thumb,
          predicted_class: data.predicted_class,
          confidence: data.confidence,
          inference_ms: data.metadata ? data.metadata.inference_ms : null,
          time: new Date().toLocaleString(),
        });
        if (history.length > 30) history.pop();
        localStorage.setItem('foodai_history', JSON.stringify(history));
        updateBadge();
      };
      img.src = ev.target.result;
    };
    reader.readAsDataURL(file);
  }

  function renderHistory() {
    historyList.innerHTML = '';
    if (!history.length) {
      const empty = historyEmpty.cloneNode(true);
      empty.id = '';
      historyList.appendChild(empty);
      return;
    }
    history.forEach((e, i) => {
      const div = document.createElement('div');
      div.className = 'history-item';
      div.setAttribute('role', 'listitem');
      div.style.animationDelay = `${i * 0.04}s`;
      const cls = e.predicted_class || '';
      const displayName = cls.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
      div.innerHTML = `
        <img class="history-thumb" src="${e.thumb}" alt="thumbnail" loading="lazy"/>
        <div class="history-info">
          <div class="history-filename">${e.filename}</div>
          <div class="history-meta">${e.time}${e.inference_ms != null ? ' · ' + e.inference_ms + ' ms' : ''}</div>
        </div>
        <div class="history-result">
          <span class="history-class ${cls}">${displayName}</span>
          <div class="history-conf">${pct(e.confidence)}%</div>
        </div>`;
      historyList.appendChild(div);
    });
  }

  clearHistoryBtn.addEventListener('click', () => {
    history = [];
    localStorage.removeItem('foodai_history');
    renderHistory();
    updateBadge();
    toast('History cleared.', 'info');
  });

  // ── Keyboard Shortcuts ────────────────────────────────────
  document.addEventListener('keydown', e => {
    // Ignore if typing in an input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    if (e.key === 'v' || e.key === 'V') {
      e.preventDefault();
      imageInput.click();
    }
    if (e.key === 'Enter' && !predictBtn.disabled) {
      e.preventDefault();
      runPrediction();
    }
    if (e.key === 'Escape') {
      clearFile();
      showView('detect');
    }
  });

})();
