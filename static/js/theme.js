(function () {
  const body = document.body;
  const btn = document.getElementById("themeToggle");
  const saved = localStorage_safe_get("verifact-theme") || "dark";
  body.setAttribute("data-theme", saved);

  if (btn) {
    btn.addEventListener("click", () => {
      const current = body.getAttribute("data-theme");
      const next = current === "dark" ? "light" : "dark";
      body.setAttribute("data-theme", next);
      localStorage_safe_set("verifact-theme", next);
    });
  }

  // Wrapped so this still works gracefully in contexts without localStorage.
  function localStorage_safe_get(key) {
    try { return localStorage.getItem(key); } catch (e) { return null; }
  }
  function localStorage_safe_set(key, value) {
    try { localStorage.setItem(key, value); } catch (e) { /* ignore */ }
  }
})();
