// ---------------------------------------------------------------- setup --
const SAMPLES = {
  real: "Officials confirmed on Tuesday that the new infrastructure funding bill will move forward after a review by the Congressional Budget Office, according to a statement released this week. The measure passed committee with a vote of 24 to 11, and is expected to take effect over the next 12 months. Analysts at several independent research firms said the impact would likely be gradual, citing prior similar measures.",
  fake: "BREAKING: Secret documents EXPOSE the real agenda behind the vaccine rollout. A whistleblower who wishes to remain anonymous says this changes EVERYTHING. Wake up, people — they have been LYING to us for years and this PROVES it beyond any doubt! Share this before it gets DELETED!"
};

let lastResult = null;
let lastText = "";

// ---------------------------------------------------------- mode tabs --
document.querySelectorAll(".mode-tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".mode-tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".mode-panel").forEach(p => p.style.display = "none");
    tab.classList.add("active");
    document.getElementById(`mode-${tab.dataset.mode}`).style.display = "block";
  });
});

// ---------------------------------------------------------- text mode --
const textarea = document.getElementById("newsText");
const charCount = document.getElementById("charCount");
const analyzeBtn = document.getElementById("analyzeBtn");
const placeholder = document.getElementById("resultPlaceholder");
const resultContent = document.getElementById("resultContent");
const resultError = document.getElementById("resultError");

if (textarea) {
  textarea.addEventListener("input", () => {
    charCount.textContent = `${textarea.value.length} characters`;
  });
}

document.querySelectorAll(".sample-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    textarea.value = SAMPLES[btn.dataset.sample];
    charCount.textContent = `${textarea.value.length} characters`;
    textarea.focus();
  });
});

if (analyzeBtn) {
  analyzeBtn.addEventListener("click", async () => {
    const text = textarea.value.trim();
    resultError.style.display = "none";
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Analyzing...";

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Something went wrong.");

      lastResult = data;
      lastText = text;
      renderResult(data);
    } catch (err) {
      placeholder.style.display = "none";
      resultContent.style.display = "none";
      resultError.style.display = "block";
      resultError.textContent = err.message;
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = "Analyze article";
    }
  });
}

function renderResult(data) {
  placeholder.style.display = "none";
  resultError.style.display = "none";
  resultContent.style.display = "block";

  const isReal = data.prediction === "REAL";
  document.getElementById("verdictLabel").textContent = isReal ? "LIKELY REAL" : "LIKELY FAKE";
  document.getElementById("verdictLabel").className = "verdict-label " + (isReal ? "c-real" : "c-fake");
  document.getElementById("verdictConfidence").textContent = `${data.confidence}% confidence`;

  document.getElementById("realBar").style.width = data.real_probability + "%";
  document.getElementById("fakeBar").style.width = data.fake_probability + "%";
  document.getElementById("realPct").textContent = data.real_probability + "%";
  document.getElementById("fakePct").textContent = data.fake_probability + "%";

  document.getElementById("modelUsed").textContent = data.model_used;
  document.getElementById("wordCount").textContent = data.word_count;
  document.getElementById("charCountResult").textContent = data.char_count;

  const indicatorBox = document.getElementById("indicatorWords");
  if (indicatorBox) {
    let html = "";
    if (data.fake_indicator_words && data.fake_indicator_words.length) {
      html += `<h5>Words pushing toward FAKE</h5><div class="word-chip-row">${
        data.fake_indicator_words.map(w => `<span class="word-chip chip-fake">${w}</span>`).join("")
      }</div>`;
    }
    if (data.real_indicator_words && data.real_indicator_words.length) {
      html += `<h5>Words pushing toward REAL</h5><div class="word-chip-row">${
        data.real_indicator_words.map(w => `<span class="word-chip chip-real">${w}</span>`).join("")
      }</div>`;
    }
    indicatorBox.innerHTML = html;
  }

  const factcheckBox = document.getElementById("factcheckBox");
  if (factcheckBox) {
    if (data.factchecks && data.factchecks.length) {
      let fcHtml = `<h5>Related professional fact-checks</h5>`;
      data.factchecks.forEach(fc => {
        fcHtml += `
          <div class="factcheck-item">
            <span class="factcheck-rating">${fc.rating}</span>
            <p class="factcheck-text">"${fc.text}"</p>
            <a href="${fc.url}" target="_blank" rel="noopener" class="factcheck-source">${fc.publisher} →</a>
          </div>`;
      });
      factcheckBox.innerHTML = fcHtml;
      factcheckBox.style.display = "block";
    } else {
      factcheckBox.style.display = "none";
    }
  }

  document.getElementById("resultNote").textContent = isReal
    ? "This reads consistent with the sourced, measured style typical of real reporting in the training data — but always verify important claims independently."
    : "This reads consistent with the sensational, unsourced style typical of fabricated articles in the training data — treat with caution and verify before sharing.";
}

// ---------------------------------------------------------- copy result --
const copyResultBtn = document.getElementById("copyResultBtn");
if (copyResultBtn) {
  copyResultBtn.addEventListener("click", () => {
    if (!lastResult) return;
    const summary = `VeriFact result: ${lastResult.prediction} (${lastResult.confidence}% confidence) — analyzed via ${lastResult.model_used}.`;
    navigator.clipboard.writeText(summary).then(() => {
      copyResultBtn.textContent = "✓ Copied!";
      copyResultBtn.classList.add("copied");
      setTimeout(() => {
        copyResultBtn.textContent = "📋 Copy result as text";
        copyResultBtn.classList.remove("copied");
      }, 1800);
    });
  });
}

// ---------------------------------------------------------- PDF export --
const exportPdfBtn = document.getElementById("exportPdfBtn");
if (exportPdfBtn) {
  exportPdfBtn.addEventListener("click", async () => {
    if (!lastResult) return;
    exportPdfBtn.disabled = true;
    exportPdfBtn.textContent = "Generating...";
    try {
      const res = await fetch("/api/export-pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: lastText, result: lastResult })
      });
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "verifact-report.pdf";
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (err) {
      alert("Couldn't generate the PDF report.");
    } finally {
      exportPdfBtn.disabled = false;
      exportPdfBtn.textContent = "Download PDF report";
    }
  });
}

// ---------------------------------------------------------- voice input --
const voiceBtn = document.getElementById("voiceBtn");
if (voiceBtn) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    voiceBtn.title = "Voice input isn't supported in this browser";
    voiceBtn.style.opacity = "0.4";
  } else {
    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.lang = "en-US";
    let listening = false;

    voiceBtn.addEventListener("click", () => {
      if (listening) {
        recognition.stop();
        return;
      }
      recognition.start();
      listening = true;
      voiceBtn.classList.add("recording");
    });

    recognition.onresult = (event) => {
      let transcript = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
      }
      textarea.value += (textarea.value ? " " : "") + transcript;
      charCount.textContent = `${textarea.value.length} characters`;
    };
    recognition.onend = () => {
      listening = false;
      voiceBtn.classList.remove("recording");
    };
    recognition.onerror = () => {
      listening = false;
      voiceBtn.classList.remove("recording");
    };
  }
}

// ---------------------------------------------------------- URL mode --
const analyzeUrlBtn = document.getElementById("analyzeUrlBtn");
if (analyzeUrlBtn) {
  analyzeUrlBtn.addEventListener("click", async () => {
    const url = document.getElementById("newsUrl").value.trim();
    const panel = document.getElementById("urlResultPanel");
    analyzeUrlBtn.disabled = true;
    analyzeUrlBtn.textContent = "Fetching...";

    try {
      const res = await fetch("/api/predict-url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Something went wrong.");

      const isReal = data.prediction === "REAL";
      panel.innerHTML = `
        <div class="verdict-block">
          <span class="verdict-label ${isReal ? 'c-real' : 'c-fake'}">${isReal ? 'LIKELY REAL' : 'LIKELY FAKE'}</span>
          <span class="verdict-confidence">${data.confidence}% confidence</span>
        </div>
        <div class="prob-bars">
          <div class="prob-row"><span class="prob-tag c-real">REAL</span><div class="prob-track"><div class="prob-fill fill-real" style="width:${data.real_probability}%"></div></div><span class="prob-pct">${data.real_probability}%</span></div>
          <div class="prob-row"><span class="prob-tag c-fake">FAKE</span><div class="prob-track"><div class="prob-fill fill-fake" style="width:${data.fake_probability}%"></div></div><span class="prob-pct">${data.fake_probability}%</span></div>
        </div>
        <div class="result-note">Extracted preview: "${data.extracted_preview}..."</div>
      `;
    } catch (err) {
      panel.innerHTML = `<div class="result-error">${err.message}</div>`;
    } finally {
      analyzeUrlBtn.disabled = false;
      analyzeUrlBtn.textContent = "Fetch & analyze";
    }
  });
}

// ---------------------------------------------------------- batch mode --
const analyzeBatchBtn = document.getElementById("analyzeBatchBtn");
if (analyzeBatchBtn) {
  analyzeBatchBtn.addEventListener("click", async () => {
    const fileInput = document.getElementById("batchFile");
    const resultsBox = document.getElementById("batchResults");
    if (!fileInput.files.length) {
      resultsBox.innerHTML = `<div class="result-error">Please choose a CSV file first.</div>`;
      return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    analyzeBatchBtn.disabled = true;
    analyzeBatchBtn.textContent = "Processing...";

    try {
      const res = await fetch("/api/predict-batch", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Something went wrong.");

      let html = `<div class="batch-row"><span>Text</span><span>Verdict</span><span>Confidence</span></div>`;
      data.results.forEach(r => {
        const cls = r.prediction === "REAL" ? "c-real" : "c-fake";
        html += `<div class="batch-row"><span>${r.text_preview}</span><span class="${cls}"><strong>${r.prediction}</strong></span><span>${r.confidence}%</span></div>`;
      });
      resultsBox.innerHTML = `<p style="color:var(--text-muted); margin-bottom:10px;">${data.count} articles analyzed</p>` + html;
    } catch (err) {
      resultsBox.innerHTML = `<div class="result-error">${err.message}</div>`;
    } finally {
      analyzeBatchBtn.disabled = false;
      analyzeBatchBtn.textContent = "Analyze batch";
    }
  });
}
