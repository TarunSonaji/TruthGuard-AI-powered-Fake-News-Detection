/* =========================================================
   TruthGuard — main.js
   Handles: AJAX prediction, character counter, dark mode,
   toast notifications, copy/download result, clear button.
   ========================================================= */

document.addEventListener('DOMContentLoaded', () => {
  initThemeToggle();
  initCharCounter();
  initAnalyzeForm();
  initClearButton();
  initResultActions();
});

/* ---------------------------------------------------------
   Toast helper — reusable for JS-triggered notifications
--------------------------------------------------------- */
function showToast(message, category = 'info') {
  const container = document.getElementById('jsToastContainer');
  if (!container) return;

  const toastEl = document.createElement('div');
  toastEl.className = `toast align-items-center text-bg-${category} border-0 tg-toast`;
  toastEl.setAttribute('role', 'alert');
  toastEl.innerHTML = `
    <div class="d-flex">
      <div class="toast-body">${message}</div>
      <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
    </div>`;
  container.appendChild(toastEl);

  const toast = new bootstrap.Toast(toastEl, { delay: 4000 });
  toast.show();
  toastEl.addEventListener('hidden.bs.toast', () => toastEl.remove());
}

/* ---------------------------------------------------------
   Dark mode toggle (persisted via localStorage)
--------------------------------------------------------- */
function initThemeToggle() {
  const root = document.documentElement;
  const btn = document.getElementById('themeToggle');
  const saved = localStorage.getItem('tg-theme') || 'light';
  root.setAttribute('data-theme', saved);
  updateThemeIcon(saved);

  if (!btn) return;
  btn.addEventListener('click', () => {
    const current = root.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    localStorage.setItem('tg-theme', next);
    updateThemeIcon(next);
  });
}

function updateThemeIcon(theme) {
  const btn = document.getElementById('themeToggle');
  if (!btn) return;
  btn.innerHTML = theme === 'dark'
    ? '<i class="bi bi-sun-fill"></i>'
    : '<i class="bi bi-moon-stars-fill"></i>';
}

/* ---------------------------------------------------------
   Character counter for the article textarea
--------------------------------------------------------- */
function initCharCounter() {
  const textarea = document.getElementById('articleText');
  const counter = document.getElementById('charCounter');
  if (!textarea || !counter) return;

  const maxLen = parseInt(textarea.getAttribute('maxlength'), 10) || 20000;
  textarea.addEventListener('input', () => {
    counter.textContent = `${textarea.value.length} / ${maxLen}`;
  });
}

/* ---------------------------------------------------------
   Analyze form submission (AJAX -> /predict)
--------------------------------------------------------- */
function initAnalyzeForm() {
  const form = document.getElementById('analyzeForm');
  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    if (window.TG_AUTHENTICATED === false) {
      showToast('Please log in to analyze articles.', 'warning');
      return;
    }

    const textarea = document.getElementById('articleText');
    const article = textarea.value.trim();

    if (article.split(/\s+/).filter(Boolean).length < 4) {
      showToast('Please enter a longer piece of text.', 'warning');
      return;
    }

    const analyzeBtn = document.getElementById('analyzeBtn');
    const progressWrap = document.getElementById('progressWrap');
    const progressBar = document.getElementById('progressBar');
    const resultCard = document.getElementById('resultCard');

    analyzeBtn.disabled = true;
    resultCard.classList.add('d-none');
    progressWrap.classList.remove('d-none');

    // Fake progress animation while the real request is in flight
    let progress = 0;
    const progressTimer = setInterval(() => {
      progress = Math.min(progress + Math.random() * 18, 90);
      progressBar.style.width = `${progress}%`;
    }, 220);

    try {
      const formData = new FormData();
      formData.append('article', article);

      const response = await fetch('/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();

      clearInterval(progressTimer);
      progressBar.style.width = '100%';

      setTimeout(() => {
        progressWrap.classList.add('d-none');
        progressBar.style.width = '0%';
        analyzeBtn.disabled = false;

        if (!data.success) {
          showToast(data.error || 'Something went wrong.', 'danger');
          return;
        }
        renderResult(data);
        showToast('Analysis complete!', 'success');
      }, 300);

    } catch (err) {
      clearInterval(progressTimer);
      progressWrap.classList.add('d-none');
      analyzeBtn.disabled = false;
      showToast('Network error. Please try again.', 'danger');
    }
  });
}

function renderResult(data) {
  const resultCard = document.getElementById('resultCard');
  const resultIcon = document.getElementById('resultIcon');
  const resultLabel = document.getElementById('resultLabel');
  const resultTimestamp = document.getElementById('resultTimestamp');
  const confidenceValue = document.getElementById('confidenceValue');
  const confidenceMeter = document.getElementById('confidenceMeter');
  const suspiciousWrap = document.getElementById('suspiciousWrap');
  const suspiciousWords = document.getElementById('suspiciousWords');

  const isFake = data.prediction === 'Fake';

  resultCard.classList.remove('d-none', 'tg-result-fake', 'tg-result-real');
  resultCard.classList.add(isFake ? 'tg-result-fake' : 'tg-result-real');

  resultIcon.innerHTML = isFake
    ? '<i class="bi bi-x-octagon-fill text-danger"></i>'
    : '<i class="bi bi-check-circle-fill text-success"></i>';
  resultLabel.textContent = isFake ? 'Likely Fake News' : 'Likely Real News';
  resultLabel.style.color = isFake ? 'var(--tg-danger)' : 'var(--tg-success)';
  resultTimestamp.textContent = `Analyzed at ${data.timestamp}`;

  confidenceValue.textContent = data.confidence;
  const deg = (data.confidence / 100) * 360;
  const color = isFake ? '#ef4444' : '#10b981';
  confidenceMeter.style.background = `conic-gradient(${color} ${deg}deg, rgba(99,102,241,0.15) ${deg}deg)`;
  confidenceMeter.textContent = `${Math.round(data.confidence)}%`;
  confidenceMeter.style.color = color;
  confidenceMeter.style.fontSize = '0.85rem';

  suspiciousWords.innerHTML = '';
  if (data.suspicious_words && data.suspicious_words.length > 0) {
    suspiciousWrap.classList.remove('d-none');
    data.suspicious_words.forEach((w) => {
      const chip = document.createElement('span');
      chip.className = 'tg-word-chip';
      chip.textContent = w;
      suspiciousWords.appendChild(chip);
    });
  } else {
    suspiciousWrap.classList.add('d-none');
  }

  // Stash the latest result for copy/download actions
  window.TG_LAST_RESULT = data;

  resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

/* ---------------------------------------------------------
   Clear button
--------------------------------------------------------- */
function initClearButton() {
  const clearBtn = document.getElementById('clearBtn');
  if (!clearBtn) return;

  clearBtn.addEventListener('click', () => {
    const textarea = document.getElementById('articleText');
    const counter = document.getElementById('charCounter');
    const resultCard = document.getElementById('resultCard');

    textarea.value = '';
    if (counter) counter.textContent = `0 / ${textarea.getAttribute('maxlength') || 20000}`;
    resultCard.classList.add('d-none');
    textarea.focus();
  });
}

/* ---------------------------------------------------------
   Copy result / Download report buttons
--------------------------------------------------------- */
function initResultActions() {
  const copyBtn = document.getElementById('copyResultBtn');
  const downloadBtn = document.getElementById('downloadResultBtn');

  if (copyBtn) {
    copyBtn.addEventListener('click', () => {
      const result = window.TG_LAST_RESULT;
      if (!result) return;
      const text =
        `TruthGuard Prediction Report\n` +
        `-----------------------------\n` +
        `Result: ${result.prediction}\n` +
        `Confidence: ${result.confidence}%\n` +
        `Suspicious words: ${(result.suspicious_words || []).join(', ') || 'None'}\n` +
        `Analyzed at: ${result.timestamp}`;

      navigator.clipboard.writeText(text).then(() => {
        showToast('Result copied to clipboard!', 'success');
      }).catch(() => {
        showToast('Could not copy automatically — please select manually.', 'warning');
      });
    });
  }

  if (downloadBtn) {
    downloadBtn.addEventListener('click', () => {
      const result = window.TG_LAST_RESULT;
      if (!result) return;

      const text =
        `TRUTHGUARD - FAKE NEWS DETECTION REPORT\n` +
        `========================================\n\n` +
        `Result:              ${result.prediction}\n` +
        `Confidence:          ${result.confidence}%\n` +
        `Suspicious words:    ${(result.suspicious_words || []).join(', ') || 'None detected'}\n` +
        `Analyzed at:         ${result.timestamp}\n\n` +
        `Generated by TruthGuard - AI Powered Fake News Detection System\n`;

      const blob = new Blob([text], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `truthguard_report_${Date.now()}.txt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      showToast('Report downloaded!', 'success');
    });
  }
}
