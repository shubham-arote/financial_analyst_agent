"use strict";

// ---- block-type colors (shared by canvas + legend) ----
const COLORS = {
  text: "#4f9cf9", title: "#ef4444", list: "#22c55e", table: "#a855f7",
  figure: "#f59e0b", caption: "#14b8a6", formula: "#ec4899",
  header: "#94a3b8", footer: "#94a3b8", page_number: "#94a3b8",
};
const colorFor = (t) => COLORS[t] || "#9ca3af";

// ---- DOM ----
const $ = (id) => document.getElementById(id);
const img = $("pageImg"), canvas = $("overlay"), ctx = canvas.getContext("2d");
const stats = $("stats"), placeholder = $("placeholder"), stageWrap = $("stageWrap");
const spinner = $("spinner"), spinnerText = $("spinnerText");

const tip = document.createElement("div");           // floating block tooltip
tip.className = "tooltip"; tip.style.display = "none";
document.body.appendChild(tip);

// ---- state ----
let docId = null;
let pageW = 1, pageH = 1, scale = 1;
let blocks = new Map();             // id -> {id, type, bbox, recognized, order, content}
let parseWS = null, askWS = null;
let pipelineInfo = {};
let hoverId = null;
const highlightIds = new Set();
const sourceRows = new Map();        // "page:block_id" -> source row element (page→answer link)
let multipage = false, currentPage = 1, totalPages = 1;
let zoomW = 0, fitMode = "width";    // page sizing: "width" (default, readable) | "page" | "custom"
let parsedPages = 0, pendingPage = null;   // background-parse progress

// ---- init ----
buildLegend();
loadStatus();
$("sampleBtn").onclick = () => start(fetch("/load-sample", { method: "POST" }));
$("fileInput").onchange = (e) => {
  if (!e.target.files.length) return;
  const fd = new FormData();
  fd.append("file", e.target.files[0]);
  start(fetch("/upload", { method: "POST", body: fd }));
};
$("askForm").onsubmit = onAsk;
$("prevPage").onclick = () => loadPdfPage(currentPage - 1);
$("nextPage").onclick = () => loadPdfPage(currentPage + 1);
$("jumpBtn").onclick = () => loadPdfPage(parseInt($("jumpPage").value) || 1);
$("jumpPage").onkeydown = (e) => { if (e.key === "Enter") loadPdfPage(parseInt($("jumpPage").value) || 1); };
window.addEventListener("resize", layoutPage);
$("fitBtn").onclick = () => { fitMode = (fitMode === "page" ? "width" : "page"); layoutPage(); };
$("zoomOut").onclick = () => zoomBy(0.8);
$("zoomIn").onclick = () => zoomBy(1.25);
canvas.style.pointerEvents = "auto";
canvas.addEventListener("mousemove", onCanvasHover);
canvas.addEventListener("mouseleave", () => { hoverId = null; tip.style.display = "none"; redraw(); });
canvas.addEventListener("click", onCanvasClick);
$("heroSample").onclick = () => start(fetch("/load-sample", { method: "POST" }));
$("heroUpload").onclick = () => $("fileInput").click();
$("detailsToggle").onclick = () => {
  const d = $("details"); d.hidden = !d.hidden;
  $("detailsToggle").textContent = d.hidden ? "Detection details ▾" : "Detection details ▴";
};

async function loadStatus() {
  try {
    const s = await (await fetch("/api/status")).json();
    $("badge").textContent = s.cloud ? "✓ AI answers" : "Offline (extractive)";
    $("badge").title = s.cloud ? s.provider : "no API key — answers are extractive";
  } catch { /* ignore */ }
}

function buildLegend() {
  $("legend").innerHTML = Object.keys(COLORS).slice(0, 7).map((t) =>
    `<span class="chip"><span class="swatch" style="background:${COLORS[t]}"></span>${t}</span>`
  ).join("");
}

// ---- start a run (image/sample vs multi-page PDF) ----
async function start(req) {
  showSpinner("Reading the document… (large or scanned PDFs can take a few minutes)");
  let doc;
  try { doc = await (await req).json(); } catch { showError("upload failed — is the server running?"); return; }
  if (!doc || doc.error) { showError((doc && doc.error) || "upload failed"); return; }

  docId = doc.doc_id;
  blocks = new Map(); highlightIds.clear(); hoverId = null; tip.style.display = "none";
  fitMode = "width";
  resetStages(); resetChat();
  placeholder.style.display = "none";
  $("zoombar").hidden = false; $("detailsToggle").hidden = false;

  if (doc.multipage) {
    multipage = true; totalPages = doc.page_count; currentPage = 1;
    parsedPages = 0; pendingPage = null;
    pageW = doc.page_w; pageH = doc.page_h;
    pipelineInfo = { detector: "Docling layout model", recognizer: "layout + OCR" };
    showNavigator(true);
    hideSpinner();
    loadPdfPage(1);          // page image shows at once; blocks fill in as parsing reaches it
    pollStatus();            // progress bar + enable Q&A as pages finish
  } else {
    multipage = false; showNavigator(false);
    pageW = doc.page_w; pageH = doc.page_h;
    img.onload = () => { layoutPage(); hideSpinner(); openParseWS(); };
    img.src = `/doc/${docId}/page.png?t=${Date.now()}`;
  }
}

function showSpinner(t) { spinnerText.textContent = t || "Working…"; spinner.hidden = false; }
function hideSpinner() { spinner.hidden = true; }
function showError(msg) {
  hideSpinner();
  placeholder.style.display = "flex";
  placeholder.innerHTML = '<div class="hero"><div class="hero-badge" style="color:var(--warn);border-color:var(--warn)">couldn\'t open</div>' +
    '<h2>That didn\'t work</h2><p>' + escapeHtml(msg) + '</p>' +
    '<div class="hero-cta"><button class="btn ghost" id="errRetry">Try again</button></div></div>';
  const r = $("errRetry"); if (r) r.onclick = () => location.reload();
}

function sizeCanvas() {
  if (!img.clientWidth) return;
  canvas.width = img.clientWidth;
  canvas.height = img.clientHeight;
  scale = img.clientWidth / pageW;
  redraw();
}

// ---- page fit / zoom (keeps a tall page fully viewable instead of cut off) ----
function layoutPage() {
  if (!img.naturalWidth) return;
  if (fitMode === "custom" && zoomW) { applyZoom(); return; }
  const cw = stageWrap.clientWidth - 4;
  const ch = (stageWrap.clientHeight || (window.innerHeight - 240)) - 4;
  zoomW = (fitMode === "page") ? Math.max(160, Math.min(cw, ch * (pageW / pageH)))
                               : Math.max(160, cw);        // default: fit width -> readable
  applyZoom();
}

function zoomBy(f) { fitMode = "custom"; zoomW = Math.max(140, (zoomW || img.clientWidth) * f); applyZoom(); }

function applyZoom() {
  placeholder.style.display = "none";
  img.style.width = zoomW + "px";
  img.style.maxWidth = "none";
  sizeCanvas();
  $("zoomLabel").textContent = Math.round(zoomW / (img.naturalWidth || pageW) * 100) + "%";
}

// ---- single-image parse WebSocket ----
function openParseWS() {
  if (parseWS) try { parseWS.close(); } catch {}
  const proto = location.protocol === "https:" ? "wss" : "ws";
  parseWS = new WebSocket(`${proto}://${location.host}/ws/parse/${docId}`);
  parseWS.onmessage = (e) => handleParse(JSON.parse(e.data));
}

function handleParse(ev) {
  switch (ev.type) {
    case "info":
      pipelineInfo = ev;
      pageW = ev.page_w; pageH = ev.page_h; scale = img.clientWidth / pageW;
      break;
    case "stage":
      setStage(ev.stage, ev.status === "start" ? "active" : "done");
      break;
    case "block":
      blocks.set(ev.id, { id: ev.id, type: ev.block_type, bbox: ev.bbox, recognized: false, order: null });
      updateStats(); redraw();
      break;
    case "recognized": {
      const b = blocks.get(ev.id);
      if (b) { b.recognized = true; b.type = ev.block_type; b.content = ev.content; }
      updateStats(); redraw();
      break;
    }
    case "order":
      ev.order.forEach((id, i) => { const b = blocks.get(id); if (b) b.order = i; });
      redraw();
      break;
    case "result":
      $("markdown").innerHTML = renderMarkdown(ev.markdown);
      $("mdMeta").textContent = `· ${blocks.size} blocks`;
      break;
    case "indexed":
      ["structure", "recognition", "relation"].forEach((s) => setStage(s, "done"));
      enableAsk(`${ev.chunks} chunks indexed`);
      break;
    case "error":
      stats.textContent = "error: " + ev.error;
      break;
  }
}

// ---- multi-page PDF ----
function showNavigator(on) { $("nav").hidden = !on; }
function updateNavigator() {
  $("pageLabel").textContent = `${currentPage} / ${totalPages}`;
  $("jumpPage").value = currentPage; $("jumpPage").max = totalPages;
}

// poll background-parse progress; enable Q&A once the first pages are indexed
async function pollStatus() {
  if (!multipage) return;
  let s;
  try { s = await (await fetch(`/doc/${docId}/status`)).json(); }
  catch { setTimeout(pollStatus, 2000); return; }
  if (s.status === "error") { $("askMode").textContent = "· parse error: " + (s.error || "failed"); return; }
  parsedPages = s.parsed_pages || 0;
  const prog = `${parsedPages}/${s.page_count} pages`;
  if (s.chunks > 0 && $("askInput").disabled) enableAsk(`${prog} parsed`);
  $("askMode").textContent = "· " + (s.status === "ready" ? `${s.page_count} pages ready` : `parsing… ${prog}`);
  if (pendingPage !== null && parsedPages >= pendingPage) { const p = pendingPage; pendingPage = null; loadPdfPage(p); }
  if (s.status === "parsing") setTimeout(pollStatus, 1500);
}

async function loadPdfPage(n, onReady) {
  if (!multipage) return;
  n = Math.max(1, Math.min(totalPages, n));
  currentPage = n; updateNavigator();
  blocks = new Map(); highlightIds.clear(); hoverId = null; tip.style.display = "none";
  resetStages();
  img.onload = async () => {
    try {
      const data = await (await fetch(`/doc/${docId}/page/${n}`)).json();
      if (data.status === "pending") {                 // not parsed yet — show image, await blocks
        pendingPage = n; pageW = data.page_w; pageH = data.page_h; blocks = new Map();
        $("markdown").innerHTML = '<span class="muted">— this page is still being parsed… —</span>';
        $("mdMeta").textContent = `· page ${n} (parsing…)`;
        ["structure", "recognition", "relation"].forEach((st) => setStage(st, "active"));
        layoutPage();
        return;
      }
      renderPdfPage(data);
      if (onReady) onReady();
    } catch { stats.textContent = "error loading page " + n; }
  };
  img.src = `/doc/${docId}/page/${n}.png?t=${Date.now()}`;
}

function renderPdfPage(data) {
  pageW = data.page_w; pageH = data.page_h; scale = img.clientWidth / pageW;
  blocks = new Map();
  for (const bd of data.blocks) {
    blocks.set(bd.id, { id: bd.id, type: bd.type, bbox: bd.bbox, recognized: true, order: bd.order, content: bd.content });
  }
  ["structure", "recognition", "relation"].forEach((s) => setStage(s, "done"));
  $("markdown").innerHTML = data.markdown ? renderMarkdown(data.markdown)
    : '<span class="muted">— no extractable text on this page —</span>';
  $("mdMeta").textContent = `· page ${data.page} · ${data.blocks.length} blocks`;
  updateStats(); layoutPage();
}

// ---- canvas drawing ----
function redraw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (const b of blocks.values()) {
    const [x0, y0, x1, y1] = b.bbox.map((v) => v * scale);
    const c = colorFor(b.type);
    const hot = highlightIds.has(b.id) || b.id === hoverId;
    ctx.lineWidth = b.recognized ? 2.5 : 1.5;
    ctx.strokeStyle = c;
    ctx.fillStyle = c + (hot ? "44" : b.recognized ? "26" : "12");
    ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    if (hot) {
      ctx.save();
      ctx.shadowColor = c; ctx.shadowBlur = 16;
      ctx.lineWidth = 3.5; ctx.strokeStyle = "#ffffff";
      ctx.strokeRect(x0 - 1.5, y0 - 1.5, (x1 - x0) + 3, (y1 - y0) + 3);
      ctx.restore();
    }
    ctx.font = "11px ui-monospace, monospace";
    ctx.fillStyle = c;
    ctx.fillText(b.type, x0 + 3, y0 + 12);
    if (b.order !== null && b.order !== undefined) {
      const r = 9;
      ctx.beginPath();
      ctx.arc(x1 - r - 2, y0 + r + 2, r, 0, 2 * Math.PI);
      ctx.fillStyle = "#04101f"; ctx.fill();
      ctx.strokeStyle = c; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.fillStyle = c; ctx.font = "bold 11px ui-monospace, monospace";
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(b.order + 1, x1 - r - 2, y0 + r + 2);
      ctx.textAlign = "start"; ctx.textBaseline = "alphabetic";
    }
  }
}

function onCanvasHover(e) {
  if (!blocks.size) return;
  const rect = canvas.getBoundingClientRect();
  const sx = (canvas.width / rect.width) || 1, sy = (canvas.height / rect.height) || 1;
  const x = (e.clientX - rect.left) * sx / scale, y = (e.clientY - rect.top) * sy / scale;
  let hit = null, best = Infinity;
  for (const b of blocks.values()) {
    const [x0, y0, x1, y1] = b.bbox;
    if (x >= x0 && x <= x1 && y >= y0 && y <= y1) {
      const a = (x1 - x0) * (y1 - y0);
      if (a < best) { best = a; hit = b; }
    }
  }
  if (hit) {
    hoverId = hit.id;
    const ord = (hit.order !== null && hit.order !== undefined) ? ` · reading #${hit.order + 1}` : "";
    tip.innerHTML = `<b style="color:${colorFor(hit.type)}">${hit.type}</b>${ord}<br>` +
      escapeHtml((hit.content || "(not recognized yet)").slice(0, 220));
    tip.style.display = "block";
    tip.style.left = Math.min(e.clientX + 14, window.innerWidth - 340) + "px";
    tip.style.top = (e.clientY + 14) + "px";
    canvas.style.cursor = "pointer";
  } else {
    hoverId = null; tip.style.display = "none"; canvas.style.cursor = "default";
  }
  redraw();
}

function scrollToBlock(id) {
  const b = blocks.get(id);
  if (!b) return;
  const yMid = ((b.bbox[1] + b.bbox[3]) / 2) * scale;
  stageWrap.scrollTo({ top: Math.max(0, yMid - stageWrap.clientHeight / 2), behavior: "smooth" });
}

function updateStats() {
  let rec = 0;
  for (const b of blocks.values()) if (b.recognized) rec++;
  stats.textContent = `${blocks.size} blocks · ${rec} recognized` +
    (pipelineInfo.detector ? ` · detector: ${pipelineInfo.detector} · recognizer: ${pipelineInfo.recognizer}` : "");
}

function setStage(stage, cls) { const el = $("st-" + stage); if (el) el.className = "light " + cls; }
function resetStages() {
  ["structure", "recognition", "relation"].forEach((s) => setStage(s, ""));
  stats.textContent = "parsing…";
  $("markdown").innerHTML = '<span class="muted">— parsing… —</span>';
  $("mdMeta").textContent = "";
}

// ---- ask / agent loop ----
function enableAsk(label) {
  $("askInput").disabled = false; $("askBtn").disabled = false;
  $("askMode").textContent = "· " + label;
  if (askWS) try { askWS.close(); } catch {}
  const proto = location.protocol === "https:" ? "wss" : "ws";
  askWS = new WebSocket(`${proto}://${location.host}/ws/ask/${docId}`);
  askWS.onmessage = (e) => handleAsk(JSON.parse(e.data));
  renderSuggestions();
}

const SUGGESTIONS = [
  "What was the year-over-year change in operating profit?",
  "What is the operating margin?",
  "Summarize the key financial figures",
  "What does it say about the outlook?",
];
function renderSuggestions() {
  const old = $("suggestions"); if (old) old.remove();
  const s = document.createElement("div"); s.id = "suggestions"; s.className = "suggestions";
  s.innerHTML = '<span class="sugg-label">Try asking:</span>';
  SUGGESTIONS.forEach((q) => {
    const c = document.createElement("button"); c.type = "button"; c.className = "sugg"; c.textContent = q;
    c.onclick = () => askQuestion(q);
    s.appendChild(c);
  });
  $("messages").appendChild(s);
}

let curTrace = null, curBot = null;
function onAsk(e) { e.preventDefault(); submitQuestion($("askInput").value); }
function askQuestion(q) { submitQuestion(q); }

function submitQuestion(q) {
  q = (q || "").trim();
  if (!q || !askWS || askWS.readyState !== 1) return;
  const sug = $("suggestions"); if (sug) sug.remove();          // clear prompts on first ask
  addMsg("user", escapeHtml(q));
  curBot = addMsg("bot", "");
  const thinking = document.createElement("div"); thinking.className = "thinking muted"; thinking.textContent = "Thinking…";
  curBot.appendChild(thinking); curBot._thinking = thinking;
  const toggle = document.createElement("span"); toggle.className = "trace-toggle"; toggle.textContent = "steps ▸";
  toggle.onclick = () => {
    const m = toggle.closest(".msg.bot"); const on = m.classList.toggle("show-trace");
    toggle.textContent = on ? "steps ▾" : "steps ▸";
  };
  curTrace = document.createElement("div"); curTrace.className = "trace";
  curBot.appendChild(toggle); curBot.appendChild(curTrace);
  askWS.send(JSON.stringify({ question: q }));
  $("askInput").value = "";
}

function handleAsk(ev) {
  if (ev.type === "agent_start") {
    traceLine(`▶ ${ev.mode}`);
  } else if (ev.type === "agent_node" && ev.node === "supervise") {
    traceLine(`🧭 route → ${ev.task}`);
    if (curBot._thinking) curBot._thinking.textContent =
      ev.task === "qa" ? "Searching the document…" : "Analyzing the figures…";
  } else if (ev.type === "agent_node" && ev.node === "retrieve") {
    traceLine(`🔍 retrieve · attempt ${ev.attempt} · ${ev.k} hits`);
  } else if (ev.type === "agent_node" && ev.node === "grade") {
    traceLine(`⚖ grade → ${ev.verdict}`, "node-grade " + ev.verdict);
  } else if (ev.type === "agent_node" && ev.node === "rewrite") {
    traceLine(`✏ rewrite → "${escapeHtml(ev.query)}"`);
  } else if (ev.type === "agent_node" && ev.node === "calculate") {
    traceLine(`🧮 calculate → ${escapeHtml(ev.expr)} = ${ev.result}`);
    curBot._calc = ev;                                       // surfaced as a badge under the answer
  } else if (ev.type === "agent_node" && ev.node === "verify") {
    const u = ev.unverified || [];
    traceLine(u.length ? `✅ verify → flagged ${u.join(", ")}` : "✅ verify → all figures cited");
    addVerifyBadge(curBot, u);                              // verify runs AFTER the answer is shown
  } else if (ev.type === "agent_answer") {
    if (curBot._thinking) { curBot._thinking.remove(); curBot._thinking = null; }
    const ans = document.createElement("div"); ans.className = "answer";
    ans.innerHTML = renderAnswer(ev.answer);
    curBot.insertBefore(ans, curBot.firstChild);            // answer above the steps toggle
    wireCitations(ans, ev.sources || []);
    renderAnalystBadges(ans, curBot);                       // 🧮 computed · ✓ verified
    if (ev.sources && ev.sources.length) { renderEvidence(curBot, ev.sources); renderSources(curBot, ev.sources); }
    scrollMsgs();
  } else if (ev.type === "error") {
    traceLine("⚠ " + escapeHtml(ev.error));
  }
}

// Make the analyst's work visible: a "computed" chip (formula + exact result) and a
// "verified" chip, shown right under the answer.
function renderAnalystBadges(ansEl, bot) {
  const wrap = document.createElement("div"); wrap.className = "analyst-badges";
  ansEl.insertAdjacentElement("afterend", wrap);
  bot._badges = wrap;
  if (bot._calc) {
    const b = document.createElement("span"); b.className = "abadge calc";
    b.innerHTML = `🧮 computed <code>${escapeHtml(String(bot._calc.expr))} = ${fmtNum(bot._calc.result)}</code>`;
    wrap.appendChild(b);
  }
}
// verify arrives after the answer; append its chip to the badge row (skip the "all clear"
// chip for plain text answers with no figures to report).
function addVerifyBadge(bot, unverified) {
  const wrap = bot._badges;
  if (!wrap || (!unverified.length && !bot._calc)) return;
  const b = document.createElement("span"); b.className = "abadge " + (unverified.length ? "warn" : "ok");
  b.textContent = unverified.length
    ? `⚠ ${unverified.length} unverified figure${unverified.length > 1 ? "s" : ""}`
    : "✓ figures verified";
  wrap.appendChild(b);
}
function fmtNum(n) { return (typeof n === "number" && !Number.isInteger(n)) ? n.toFixed(2) : n; }

function renderSources(bot, sources) {
  sourceRows.clear();
  const s = document.createElement("div"); s.className = "sources";
  const lbl = document.createElement("b");
  lbl.textContent = multipage ? "Sources — click to open that page" : "Sources — hover to locate on the page";
  s.appendChild(lbl);
  const onPage = [];
  sources.forEach((x) => {
    const row = document.createElement("div"); row.className = "src";
    const tag = x.type ? ` <span class="tag">${escapeHtml(x.type)}</span>` : "";
    row.innerHTML = `<b>p${x.page} · ${escapeHtml(x.heading)}</b>${tag} — ${escapeHtml(x.snippet)}…`;
    if (x.block_id !== null && x.block_id !== undefined) {
      sourceRows.set(x.page + ":" + x.block_id, row);        // page→answer bidirectional link
      row.classList.add("linked");
      const here = !multipage || x.page === currentPage;
      if (here) onPage.push(x.block_id);
      row.onmouseenter = () => { if (here) { highlightIds.clear(); highlightIds.add(x.block_id); redraw(); scrollToBlock(x.block_id); } };
      row.onmouseleave = () => { if (here) { highlightIds.clear(); redraw(); } };
      row.onclick = () => gotoSource(x);
    }
    s.appendChild(row);
  });
  bot.appendChild(s);
  if (onPage.length) {
    bot.onmouseenter = () => { highlightIds.clear(); onPage.forEach((b) => highlightIds.add(b)); redraw(); };
    bot.onmouseleave = () => { highlightIds.clear(); redraw(); };
  }
}

// ---- active citations ----
function renderAnswer(text) {
  const h = escapeHtml(text).replace(/\n/g, "<br>");
  return h.replace(/\[\s*(?:page|pg|p)\.?\s*(\d+)\s*\]/gi,
    (_m, n) => `<span class="cite" data-page="${n}">p${n}</span>`);
}

function jumpToPageBlock(page, blockId) {
  const hl = () => { if (blockId != null) { highlightIds.clear(); highlightIds.add(blockId); redraw(); scrollToBlock(blockId); } };
  if (multipage && page !== currentPage) loadPdfPage(page, hl); else hl();
}
function gotoSource(x) { jumpToPageBlock(x.page, x.block_id); }

function wireCitations(container, sources) {
  container.querySelectorAll(".cite").forEach((el) => {
    const pg = parseInt(el.dataset.page);
    const src = sources.find((s) => s.page === pg && s.block_id != null);
    el.onclick = () => jumpToPageBlock(pg, src ? src.block_id : null);
  });
}

function renderEvidence(bot, sources) {
  const pages = [...new Set(sources.map((s) => s.page))];
  if (!multipage && pages.length <= 1) return;
  const ev = document.createElement("div"); ev.className = "evidence";
  ev.innerHTML = '<span class="ev-label">cited pages:</span>';
  pages.forEach((p) => {
    const src = sources.find((s) => s.page === p && s.block_id != null);
    const c = document.createElement("span"); c.className = "ev-page"; c.textContent = "p" + p;
    c.onclick = () => jumpToPageBlock(p, src ? src.block_id : null);
    ev.appendChild(c);
  });
  bot.appendChild(ev);
}

// page → answer: click a cited block on the page, flash its source row in the chat
function onCanvasClick() {
  if (hoverId == null) return;
  const row = sourceRows.get(currentPage + ":" + hoverId);
  if (row) { row.classList.add("flash"); row.scrollIntoView({ block: "nearest", behavior: "smooth" }); setTimeout(() => row.classList.remove("flash"), 1200); }
}

function traceLine(text, cls = "") { const d = document.createElement("div"); if (cls) d.className = cls; d.textContent = text; curTrace.appendChild(d); scrollMsgs(); }
function addMsg(role, html) { const d = document.createElement("div"); d.className = "msg " + role; d.innerHTML = html; $("messages").appendChild(d); scrollMsgs(); return d; }
function scrollMsgs() { const m = $("messages"); m.scrollTop = m.scrollHeight; }
function resetChat() { $("messages").innerHTML = ""; $("askInput").disabled = true; $("askBtn").disabled = true; $("askMode").textContent = ""; }

// ---- tiny markdown renderer ----
function renderMarkdown(md) {
  return md.split(/\n\n+/).map((b) => {
    b = b.trim();
    if (!b) return "";
    if (b.startsWith("## ")) return `<h3>${escapeHtml(b.slice(3))}</h3>`;
    if (b.startsWith("$$")) return `<div class="formula">${escapeHtml(b.replace(/\$\$/g, "").trim())}</div>`;
    if (b.startsWith("|")) return renderTable(b);
    return `<p>${escapeHtml(b)}</p>`;
  }).join("");
}
function renderTable(b) {
  const rows = b.split("\n").filter((r) => r.trim().startsWith("|"));
  const cells = rows.map((r) => r.split("|").slice(1, -1).map((c) => c.trim()));
  const body = cells.filter((r) => !r.every((c) => /^-+$/.test(c)));
  return "<table>" + body.map((r, i) =>
    "<tr>" + r.map((c) => i === 0 ? `<th>${escapeHtml(c)}</th>` : `<td>${escapeHtml(c)}</td>`).join("") + "</tr>"
  ).join("") + "</table>";
}
function escapeHtml(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
