// app.js — EduRAG frontend. Talks to the FastAPI backend over /api/*.
// No build step, no framework: plain DOM + fetch, kept deliberately simple
// so it's easy to read top to bottom and easy to extend.

const state = {
  documents: [],       // [{name, chunks}]
  qaHistory: [],        // [{question, answer, low_confidence, results}]
  notesHistory: [],     // [{source, mode, content}]
};

const el = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// Upload: drag-and-drop + click-to-browse, both funnel into uploadFiles()
// ---------------------------------------------------------------------------

const dropzone = el("dropzone");
const fileInput = el("fileInput");

dropzone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) uploadFiles(fileInput.files);
  fileInput.value = ""; // allow re-selecting the same file later
});

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  })
);

["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
  })
);

dropzone.addEventListener("drop", (e) => {
  const files = e.dataTransfer.files;
  if (files.length) uploadFiles(files);
});

async function uploadFiles(fileList) {
  const invalid = [...fileList].filter((f) => !/\.(pdf|docx|txt)$/i.test(f.name));
  if (invalid.length) {
    showToast(`Skipping unsupported file(s): ${invalid.map((f) => f.name).join(", ")}`);
  }
  const valid = [...fileList].filter((f) => /\.(pdf|docx|txt)$/i.test(f.name));
  if (!valid.length) return;

  const formData = new FormData();
  valid.forEach((f) => formData.append("files", f));

  dropzone.querySelector(".dropzone-label").textContent = "Indexing…";
  try {
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    if (!res.ok) throw new Error((await res.json()).detail || "Upload failed");
    const data = await res.json();
    state.documents = data.documents;
    onDocumentsChanged();
  } catch (err) {
    showToast(`Upload failed: ${err.message}`);
  } finally {
    dropzone.querySelector(".dropzone-label").textContent = "Drop PDF, DOCX or TXT files";
  }
}

el("sampleBtn").addEventListener("click", async () => {
  try {
    const res = await fetch("/api/sample", { method: "POST" });
    const data = await res.json();
    state.documents = data.documents;
    onDocumentsChanged();
  } catch (err) {
    showToast("Could not load the sample textbook.");
  }
});

function onDocumentsChanged() {
  renderDocList();
  renderStats();
  renderNotesSourceOptions();

  const hasDocs = state.documents.length > 0;
  el("emptyState").hidden = hasDocs;
  el("workspace").hidden = !hasDocs;
  el("statsPanel").hidden = !hasDocs;
  el("exportBtn").disabled = !hasDocs;
  el("resetBtn").disabled = !hasDocs;
}

function renderDocList() {
  const list = el("docList");
  list.innerHTML = "";
  for (const doc of state.documents) {
    const li = document.createElement("li");
    li.className = "doc-item";
    li.innerHTML = `<span class="doc-item-name" title="${escapeHtml(doc.name)}">${escapeHtml(doc.name)}</span>
                     <span class="doc-item-count">${doc.chunks}</span>`;
    list.appendChild(li);
  }
}

// ---------------------------------------------------------------------------
// Session stats
// ---------------------------------------------------------------------------

function renderStats() {
  const totalChunks = state.documents.reduce((sum, d) => sum + d.chunks, 0);
  el("statChunks").textContent = totalChunks;
  el("statQuestions").textContent = state.qaHistory.length;

  if (state.qaHistory.length) {
    const scored = state.qaHistory.filter((h) => h.results.length);
    const avg = scored.length
      ? scored.reduce((sum, h) => sum + h.results[0].score, 0) / scored.length
      : 0;
    el("statConfidence").textContent = avg.toFixed(2);
  } else {
    el("statConfidence").textContent = "—";
  }
}

// ---------------------------------------------------------------------------
// Settings panel
// ---------------------------------------------------------------------------

const settingsToggle = el("settingsToggle");
const settingsBody = el("settingsBody");
settingsToggle.addEventListener("click", () => {
  const isOpen = !settingsBody.hidden;
  settingsBody.hidden = isOpen;
  settingsToggle.classList.toggle("open", !isOpen);
});

const useLlmToggle = el("useLlmToggle");
const apiKeyInput = el("apiKeyInput");
useLlmToggle.addEventListener("change", () => {
  apiKeyInput.hidden = !useLlmToggle.checked;
});

const topKSlider = el("topKSlider");
topKSlider.addEventListener("input", () => {
  el("topKValue").textContent = topKSlider.value;
});

function currentSettings() {
  return {
    use_llm: useLlmToggle.checked,
    api_key: apiKeyInput.value || null,
    top_k: parseInt(topKSlider.value, 10),
  };
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    el(`tab-${tab.dataset.tab}`).classList.add("active");
    if (tab.dataset.tab === "insights") loadInsights();
  });
});

// ---------------------------------------------------------------------------
// Insights tab — corpus-level analysis (pandas/seaborn/matplotlib, server-side)
// ---------------------------------------------------------------------------

async function loadInsights() {
  const body = el("insightsBody");
  body.innerHTML = skeletonHtml();

  try {
    const res = await fetch("/api/insights");
    if (!res.ok) throw new Error((await res.json()).detail || "Request failed");
    const data = await res.json();

    const rows = data.documents
      .map((d) => `<div class="stat"><dt>${escapeHtml(d.name)}</dt><dd>${d.words} words</dd></div>`)
      .join("");

    body.innerHTML = `
      <div class="insights-stats">
        <div class="stat"><dt>Total words</dt><dd>${data.total_words}</dd></div>
        <div class="stat"><dt>Unique words</dt><dd>${data.unique_words}</dd></div>
        ${rows}
      </div>
      <img class="insights-chart" src="${data.chart}" alt="Most frequent terms across loaded source(s)">
    `;
  } catch (err) {
    body.innerHTML = `<div class="transcript-hint">Could not load insights: ${escapeHtml(err.message)}</div>`;
  }
}

// ---------------------------------------------------------------------------
// Ask tab
// ---------------------------------------------------------------------------

el("askForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = el("askInput");
  const question = input.value.trim();
  if (!question) return;
  input.value = "";

  const transcript = el("qaTranscript");
  transcript.querySelector(".transcript-hint")?.remove();

  const entry = document.createElement("div");
  entry.className = "qa-entry";
  entry.innerHTML = `<div class="qa-question">${escapeHtml(question)}</div>` + skeletonHtml();
  transcript.appendChild(entry);
  transcript.scrollTop = transcript.scrollHeight;

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, ...currentSettings() }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || "Request failed");
    const data = await res.json();

    state.qaHistory.push({ question, ...data });
    entry.querySelector(".skeleton").outerHTML = answerCardHtml({ question, ...data });
    renderStats();
  } catch (err) {
    entry.querySelector(".skeleton").outerHTML = `<div class="answer-card"><p class="answer-text">Something went wrong: ${escapeHtml(err.message)}</p></div>`;
  }
  transcript.scrollTop = transcript.scrollHeight;
});

function answerCardHtml({ question, answer, low_confidence, results }) {
  const warning = low_confidence
    ? `<div class="answer-warning">⚠ Low match confidence — may not be well covered in the loaded source(s)</div>`
    : "";

  // citation-tab + its detail panel are siblings, matched up at click time
  // by DOM position (see the delegated click handler below) — no IDs needed.
  // Each detail panel leads with the single best-matching sentence (the
  // "key excerpt"), then the full passage below in muted text for context
  // — instead of dumping the whole chunk and making you find the relevant
  // line yourself.
  const citations = results
    .map((r, i) => {
      const loc = r.page ? `${r.source} · p.${r.page}` : r.source;
      return `<button class="citation-tab" type="button">[${i + 1}] ${escapeHtml(loc)} · <span class="citation-score">${r.score.toFixed(2)}</span></button>
              <div class="citation-detail">
                <p class="citation-key-sentence">${highlightMatches(r.highlight, question)}</p>
                <p class="citation-full-text">${highlightMatches(r.text, question)}</p>
              </div>`;
    })
    .join("");

  return `<div class="answer-card">
    ${warning}
    <p class="answer-text">${highlightMatches(answer, question)}</p>
    ${results.length ? `<div class="citations">${citations}</div>` : ""}
  </div>`;
}

// event delegation for citation tabs (they're added dynamically)
el("qaTranscript").addEventListener("click", (e) => {
  const btn = e.target.closest(".citation-tab");
  if (!btn) return;
  const detail = btn.nextElementSibling;
  const isOpen = detail.classList.toggle("open");
  btn.classList.toggle("open", isOpen);
});

// ---------------------------------------------------------------------------
// Notes tab
// ---------------------------------------------------------------------------

function renderNotesSourceOptions() {
  const select = el("notesSource");
  select.innerHTML = "";
  if (state.documents.length > 1) {
    const allOpt = document.createElement("option");
    allOpt.value = "__all__";
    allOpt.textContent = "All loaded documents";
    select.appendChild(allOpt);
  }
  for (const doc of state.documents) {
    const opt = document.createElement("option");
    opt.value = doc.name;
    opt.textContent = doc.name;
    select.appendChild(opt);
  }
}

let notesMode = "summary";
document.querySelectorAll(".segmented-option").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".segmented-option").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    notesMode = btn.dataset.mode;
  });
});

el("generateBtn").addEventListener("click", async () => {
  const source = el("notesSource").value;
  if (!source) return;

  const transcript = el("notesTranscript");
  transcript.querySelector(".transcript-hint")?.remove();

  const placeholder = document.createElement("div");
  placeholder.innerHTML = skeletonHtml();
  transcript.appendChild(placeholder);
  transcript.scrollTop = transcript.scrollHeight;

  try {
    const res = await fetch("/api/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source, mode: notesMode, ...currentSettings() }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || "Request failed");
    const data = await res.json();

    state.notesHistory.push({ source, mode: notesMode, content: data.content });
    placeholder.innerHTML = noteCardHtml(source, notesMode, data.content);
  } catch (err) {
    placeholder.innerHTML = `<div class="note-card"><p class="note-card-body">Something went wrong: ${escapeHtml(err.message)}</p></div>`;
  }
});

function noteCardHtml(source, mode, content) {
  const label = mode === "notes" ? "Bullet notes" : "Summary";
  const sourceLabel = source === "__all__" ? "all loaded documents" : source;
  return `<div class="note-card">
    <p class="note-card-title">${label} — ${escapeHtml(sourceLabel)}</p>
    <p class="note-card-body">${escapeHtml(content)}</p>
  </div>`;
}

// ---------------------------------------------------------------------------
// Export / reset
// ---------------------------------------------------------------------------

el("exportBtn").addEventListener("click", () => {
  const parts = [];
  for (const h of state.qaHistory) {
    const sources = [...new Set(h.results.map((r) => r.source))].join(", ") || "none";
    parts.push(`## Q: ${h.question}\n\n${h.answer}\n\n*Sources: ${sources}*`);
  }
  for (const n of state.notesHistory) {
    parts.push(`## ${n.mode === "notes" ? "Bullet notes" : "Summary"} — ${n.source}\n\n${n.content}`);
  }
  const blob = new Blob([parts.join("\n\n---\n\n")], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "edurag_notes.md";
  a.click();
  URL.revokeObjectURL(url);
});

el("resetBtn").addEventListener("click", async () => {
  if (!confirm("Clear all loaded documents and this session's history?")) return;
  await fetch("/api/reset", { method: "POST" });
  state.documents = [];
  state.qaHistory = [];
  state.notesHistory = [];
  el("qaTranscript").innerHTML = `<div class="transcript-hint">Ask anything covered in the loaded source(s). Answers cite exactly where they came from.</div>`;
  el("notesTranscript").innerHTML = `<div class="transcript-hint">Generate a short summary or exam-ready bullet notes from any loaded source.</div>`;
  onDocumentsChanged();
});

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function skeletonHtml() {
  return `<div class="skeleton">
    <div class="skeleton-line w-80"></div>
    <div class="skeleton-line w-60"></div>
    <div class="skeleton-line w-40"></div>
  </div>`;
}

function showToast(message) {
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3500);
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// Words too common to be worth highlighting even though they're in the
// question — "what", "is", "the", etc. Not exhaustive, just enough to
// keep highlights meaningful instead of lighting up every filler word.
const HIGHLIGHT_STOPWORDS = new Set([
  "the", "is", "a", "an", "of", "to", "in", "and", "or", "for", "on", "with",
  "what", "how", "why", "are", "was", "were", "be", "been", "being", "that",
  "this", "it", "as", "by", "at", "from", "which", "who", "whom", "does",
  "do", "did", "can", "could", "would", "should",
]);

/**
 * Highlight every word from `query` wherever it appears in `text` — the
 * same idea as Google bolding your search terms inside a result snippet.
 * Escapes `text` first (so this is safe to drop straight into innerHTML),
 * then wraps whole-word, case-insensitive matches in <mark>.
 */
function highlightMatches(text, query) {
  const escaped = escapeHtml(text);
  const terms = [...new Set(
    query.toLowerCase().split(/\W+/).filter((w) => w.length > 2 && !HIGHLIGHT_STOPWORDS.has(w))
  )];
  if (!terms.length) return escaped;

  const escapedTerms = terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const pattern = new RegExp(`\\b(${escapedTerms.join("|")})\\b`, "gi");
  return escaped.replace(pattern, '<mark class="hl">$1</mark>');
}

// initial state
onDocumentsChanged();
