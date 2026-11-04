const $ = (id) => document.getElementById(id);
const LABELS_FR = { positive: "Positif", neutral: "Neutre", negative: "Négatif" };
let allResults = [];
let currentFilter = "all";

/* ---------- Thème ---------- */
function applyTheme(theme) {
  document.body.classList.toggle("dark", theme === "dark");
  $("themeToggle").textContent = theme === "dark" ? "☀️" : "🌙";
}
async function initTheme() {
  const { theme } = await chrome.storage.local.get("theme");
  applyTheme(theme || "light");
}
$("themeToggle").addEventListener("click", async () => {
  const theme = document.body.classList.contains("dark") ? "light" : "dark";
  applyTheme(theme);
  await chrome.storage.local.set({ theme });
});

/* ---------- États ---------- */
function showStatus(msg, type = "info") {
  $("status").textContent = msg;
  $("status").className = `status ${type}`;
}
const hideStatus = () => ($("status").className = "status hidden");

/* ---------- Extraction ---------- */
async function getComments(tabId, autoScroll) {
  const msg = { action: "getComments", autoScroll, max: CONFIG.MAX_COMMENTS };
  try {
    return await chrome.tabs.sendMessage(tabId, msg);
  } catch {
    // Content script absent (page ouverte avant l'installation de l'extension)
    await chrome.scripting.executeScript({ target: { tabId }, files: ["content.js"] });
    return await chrome.tabs.sendMessage(tabId, msg);
  }
}

/* ---------- Appel API par batch ---------- */
async function callApi(comments) {
  const results = [];
  let apiMs = 0;
  for (let i = 0; i < comments.length; i += CONFIG.BATCH_SIZE) {
    const batch = comments.slice(i, i + CONFIG.BATCH_SIZE);
    const res = await fetch(`${CONFIG.API_URL}/predict_batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comments: batch.map((c) => c.text) }),
      signal: AbortSignal.timeout(CONFIG.TIMEOUT_MS),
    });
    if (!res.ok) throw new Error(`Erreur API (${res.status})`);
    const data = await res.json();
    apiMs += data.processing_time_ms;
    data.predictions.forEach((p, j) => results.push({ ...p, isReply: batch[j].isReply }));
  }
  return { results, apiMs };
}

/* ---------- Statistiques ---------- */
function computeStats(results) {
  const total = results.length;
  const count = (s) => results.filter((r) => r.sentiment === s).length;
  const pct = (n) => (total ? Math.round((100 * n) / total) : 0);
  const pos = count("positive"), neu = count("neutral"), neg = count("negative");
  return { total, pos, neu, neg, posPct: pct(pos), neuPct: pct(neu), negPct: pct(neg) };
}

function renderStats(st, totalMs, apiMs) {
  $("totalCount").textContent = st.total;
  $("posPct").textContent = `${st.posPct}% (${st.pos})`;
  $("neuPct").textContent = `${st.neuPct}% (${st.neu})`;
  $("negPct").textContent = `${st.negPct}% (${st.neg})`;
  const a = st.posPct, b = st.posPct + st.neuPct;
  $("donut").style.background =
    `conic-gradient(var(--positive) 0 ${a}%, var(--neutral) ${a}% ${b}%, var(--negative) ${b}% 100%)`;
  $("meta").textContent =
    `${st.total} commentaires · API ${apiMs.toFixed(1)} ms · total ${totalMs} ms`;
}

/* ---------- Liste (textContent : protège contre l'injection HTML) ---------- */
function renderList() {
  const list = $("commentList");
  list.replaceChildren();
  const items = currentFilter === "all"
    ? allResults
    : allResults.filter((r) => r.sentiment === currentFilter);

  if (!items.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "Aucun commentaire pour ce filtre.";
    list.append(li);
    return;
  }
  for (const r of items) {
    const li = document.createElement("li");
    li.className = `comment ${r.sentiment}`;
    const badge = document.createElement("span");
    badge.className = `badge ${r.sentiment}`;
    badge.textContent = `${LABELS_FR[r.sentiment]} · ${Math.round(r.confidence * 100)}%`;
    li.append(badge);
    if (r.isReply) {
      const tag = document.createElement("span");
      tag.className = "reply-tag";
      tag.textContent = "↳ réponse";
      li.append(tag);
    }
    const p = document.createElement("p");
    p.textContent = r.text;
    li.append(p);
    list.append(li);
  }
}

/* ---------- Filtres ---------- */
document.querySelectorAll(".filters button").forEach((btn) =>
  btn.addEventListener("click", () => {
    document.querySelectorAll(".filters button").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    currentFilter = btn.dataset.filter;
    renderList();
  })
);

/* ---------- Copier / Exporter ---------- */
function flash(id, text) {
  const btn = $(id), old = btn.textContent;
  btn.textContent = text;
  setTimeout(() => (btn.textContent = old), 1500);
}

$("copyBtn").addEventListener("click", async () => {
  const st = computeStats(allResults);
  const lines = [
    `Analyse de ${st.total} commentaires YouTube`,
    `Positif : ${st.posPct}% | Neutre : ${st.neuPct}% | Négatif : ${st.negPct}%`,
    "",
    ...allResults.map((r) => `[${LABELS_FR[r.sentiment]} ${Math.round(r.confidence * 100)}%] ${r.text}`),
  ];
  await navigator.clipboard.writeText(lines.join("\n"));
  flash("copyBtn", "Copié !");
});

$("exportBtn").addEventListener("click", () => {
  const esc = (s) => `"${String(s).replace(/"/g, '""')}"`;
  const rows = allResults.map((r) => [esc(r.text), r.sentiment, r.confidence, r.isReply].join(","));
  const csv = ["text,sentiment,confidence,is_reply", ...rows].join("\n");
  const url = URL.createObjectURL(new Blob(["\ufeff" + csv], { type: "text/csv;charset=utf-8" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = "sentiment_youtube.csv";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

/* ---------- Analyse ---------- */
$("analyzeBtn").addEventListener("click", async () => {
  const btn = $("analyzeBtn");
  btn.disabled = true;
  $("results").classList.add("hidden");
  const t0 = performance.now();
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.url?.startsWith("https://www.youtube.com/watch")) {
      throw new Error("Ouvrez une vidéo YouTube (youtube.com/watch…).");
    }
    showStatus("Extraction des commentaires…");
    const resp = await getComments(tab.id, $("autoScroll").checked);
    if (resp?.error) throw new Error(resp.error);
    if (!resp?.comments?.length) {
      throw new Error("Aucun commentaire trouvé. Faites défiler jusqu'aux commentaires puis réessayez.");
    }

    showStatus(`Analyse de ${resp.comments.length} commentaires…`);
    const { results, apiMs } = await callApi(resp.comments);
    allResults = results;
    renderStats(computeStats(results), Math.round(performance.now() - t0), apiMs);
    renderList();
    hideStatus();
    $("results").classList.remove("hidden");
  } catch (e) {
    let msg = e.message;
    if (e.name === "TimeoutError" || e.name === "AbortError") msg = "L'API ne répond pas (délai dépassé).";
    else if (msg === "Failed to fetch") msg = "Impossible de joindre l'API. Est-elle lancée ?";
    showStatus(msg, "error");
  } finally {
    btn.disabled = false;
  }
});

initTheme();