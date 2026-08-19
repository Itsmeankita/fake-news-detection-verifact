const compareBtn = document.getElementById("compareBtn");
const compareError = document.getElementById("compareError");
const compareResults = document.getElementById("compareResults");

function renderCompareResult(panel, result) {
  const isReal = result.prediction === "REAL";
  panel.innerHTML = `
    <div class="verdict-block">
      <span class="verdict-label ${isReal ? 'c-real' : 'c-fake'}">${isReal ? 'LIKELY REAL' : 'LIKELY FAKE'}</span>
      <span class="verdict-confidence">${result.confidence}% confidence</span>
    </div>
    <div class="prob-bars">
      <div class="prob-row">
        <span class="prob-tag c-real">REAL</span>
        <div class="prob-track"><div class="prob-fill fill-real" style="width:${result.real_probability}%"></div></div>
        <span class="prob-pct">${result.real_probability}%</span>
      </div>
      <div class="prob-row">
        <span class="prob-tag c-fake">FAKE</span>
        <div class="prob-track"><div class="prob-fill fill-fake" style="width:${result.fake_probability}%"></div></div>
        <span class="prob-pct">${result.fake_probability}%</span>
      </div>
    </div>
  `;
}

if (compareBtn) {
  compareBtn.addEventListener("click", async () => {
    const text_a = document.getElementById("textA").value.trim();
    const text_b = document.getElementById("textB").value.trim();
    compareError.style.display = "none";
    compareBtn.disabled = true;
    compareBtn.textContent = "Comparing...";

    try {
      const res = await fetch("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text_a, text_b })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Something went wrong.");

      compareResults.style.display = "grid";
      renderCompareResult(document.getElementById("resultPanelA"), data.result_a);
      renderCompareResult(document.getElementById("resultPanelB"), data.result_b);
    } catch (err) {
      compareError.style.display = "block";
      compareError.textContent = err.message;
    } finally {
      compareBtn.disabled = false;
      compareBtn.textContent = "Compare articles";
    }
  });
}
