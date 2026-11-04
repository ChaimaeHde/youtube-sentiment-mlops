// Garde-fou : évite une double déclaration si le script est injecté deux fois
if (!window.__ytSentimentLoaded) {
  window.__ytSentimentLoaded = true;

  const SELECTORS = [
    "ytd-comment-thread-renderer #content-text",
    "ytd-comment-view-model #content-text",
    "ytd-comment-renderer #content-text",
  ].join(",");

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  function extractComments(max) {
    const seen = new Set();
    const out = [];
    for (const node of document.querySelectorAll(SELECTORS)) {
      const text = node.innerText.trim();
      if (!text || seen.has(text)) continue;
      seen.add(text);
      out.push({ text, isReply: !!node.closest("ytd-comment-replies-renderer") });
      if (out.length >= max) break;
    }
    return out;
  }

  // YouTube charge les commentaires au scroll : on fait défiler la page
  async function autoScroll(max, rounds = 8) {
    for (let i = 0; i < rounds; i++) {
      if (extractComments(max).length >= max) break;
      window.scrollBy(0, window.innerHeight * 2);
      await sleep(900);
    }
  }

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg.action !== "getComments") return;
    (async () => {
      if (!location.pathname.startsWith("/watch")) {
        sendResponse({ error: "Ouvrez une vidéo YouTube." });
        return;
      }
      if (msg.autoScroll) await autoScroll(msg.max);
      sendResponse({ comments: extractComments(msg.max) });
    })();
    return true; // réponse asynchrone
  });
}