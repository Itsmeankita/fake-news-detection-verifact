const textarea = document.getElementById("text");
const btn = document.getElementById("checkBtn");
const result = document.getElementById("result");

// If the user right-clicked selected text, pre-fill it.
chrome.storage.local.get("selectedText", (data) => {
  if (data.selectedText) {
    textarea.value = data.selectedText;
    chrome.storage.local.remove("selectedText");
  }
});

btn.addEventListener("click", async () => {
  const text = textarea.value.trim();
  result.style.display = "none";
  if (text.length < 20) {
    result.style.display = "block";
    result.innerHTML = `<div class="err">Please enter at least 20 characters.</div>`;
    return;
  }

  btn.disabled = true;
  btn.textContent = "Analyzing...";

  try {
    const res = await fetch("http://127.0.0.1:5000/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Something went wrong.");

    const isReal = data.prediction === "REAL";
    result.style.display = "block";
    result.innerHTML = `
      <span class="verdict ${isReal ? 'real' : 'fake'}">${isReal ? 'LIKELY REAL' : 'LIKELY FAKE'}</span>
      <span class="conf">${data.confidence}% confidence</span>
    `;
  } catch (err) {
    result.style.display = "block";
    result.innerHTML = `<div class="err">${err.message || "Couldn't reach VeriFact. Is the Flask app running on port 5000?"}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyze";
  }
});
