"""
main.py
-------
FastAPI backend for EduRAG. All the ML logic lives in rag_core.py; this
file is the API layer that wraps it — upload documents, ask questions,
generate summaries.

This keeps ONE in-memory session (documents + index) for the whole
running server, which is the right amount of complexity for a local,
single-user project like this one. A production, multi-user version
would key this state per-session (e.g. a session ID from the frontend,
or a proper database) instead of one global dict — noted in the README
as the natural next step.

Run with:  uvicorn main:app --reload
Then open: http://localhost:8000
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import io

from rag_core import (
    load_pdf_pages, load_docx_text, chunk_document, build_index, retrieve, generate_answer,
    summarize_extractive, notes_extractive, summarize_with_llm, best_sentence,
)

app = FastAPI(title="EduRAG API")

# ---------------------------------------------------------------------------
# In-memory session state. See module docstring for why a global dict is
# the right call here, and what to change for a multi-user deployment.
# ---------------------------------------------------------------------------
STATE = {
    "documents": {},   # source filename -> full raw text (for summarization)
    "chunks": [],       # retrieval units: {text, source, page}
    "vectorizer": None,
    "matrix": None,
}


def _rebuild_index():
    """Re-fit the TF-IDF index over every chunk currently loaded. Called
    after any upload/reset so retrieval always reflects the full session."""
    if STATE["chunks"]:
        STATE["vectorizer"], STATE["matrix"] = build_index([c["text"] for c in STATE["chunks"]])
    else:
        STATE["vectorizer"], STATE["matrix"] = None, None


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str
    top_k: int = 3
    use_llm: bool = False
    api_key: str | None = None


class SummarizeRequest(BaseModel):
    source: str  # a filename, or "__all__" for every loaded document combined
    mode: str    # "summary" or "notes"
    use_llm: bool = False
    api_key: str | None = None


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.post("/api/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    added = []
    for f in files:
        raw = await f.read()
        name = f.filename

        if name.lower().endswith(".pdf"):
            pages = load_pdf_pages(io.BytesIO(raw))
            STATE["documents"][name] = "\n".join(pages)
            new_chunks = chunk_document(pages, source=name, track_pages=True)
        elif name.lower().endswith(".docx"):
            text = load_docx_text(io.BytesIO(raw))
            STATE["documents"][name] = text
            new_chunks = chunk_document([text], source=name, track_pages=False)
        elif name.lower().endswith(".txt"):
            text = raw.decode("utf-8", errors="replace")
            STATE["documents"][name] = text
            new_chunks = chunk_document([text], source=name, track_pages=False)
        else:
            raise HTTPException(400, f"Unsupported file type: {name}. Use PDF, DOCX, or TXT.")

        STATE["chunks"].extend(new_chunks)
        added.append({"name": name, "chunks": len(new_chunks)})

    _rebuild_index()
    return {"added": added, "documents": _document_summaries()}


@app.post("/api/sample")
def load_sample():
    """Load the built-in demo textbook — the one-click 'try it now' path."""
    name = "sample_textbook.txt"
    if name not in STATE["documents"]:
        with open("sample_textbook.txt") as fh:
            text = fh.read()
        STATE["documents"][name] = text
        STATE["chunks"].extend(chunk_document([text], source=name, track_pages=False))
        _rebuild_index()
    return {"documents": _document_summaries()}


@app.post("/api/reset")
def reset_session():
    STATE["documents"].clear()
    STATE["chunks"].clear()
    STATE["vectorizer"], STATE["matrix"] = None, None
    return {"ok": True}


@app.get("/api/documents")
def get_documents():
    return {"documents": _document_summaries()}


def _document_summaries():
    counts = {}
    for c in STATE["chunks"]:
        counts[c["source"]] = counts.get(c["source"], 0) + 1
    return [{"name": name, "chunks": counts.get(name, 0)} for name in STATE["documents"]]


@app.get("/api/insights")
def insights():
    """
    A small data-analysis view over whatever's loaded, built with the
    same pandas/seaborn/matplotlib stack used for classic data analysis
    work — cleaning raw text into a word frequency table (pandas), then
    charting it (seaborn/matplotlib). Deliberately separate from the RAG
    pipeline itself: this is "what does this corpus contain," not
    "answer a question about it."
    """
    if not STATE["documents"]:
        raise HTTPException(400, "No documents loaded yet.")

    import base64
    import io
    import re
    from collections import Counter

    import matplotlib
    matplotlib.use("Agg")  # headless — no display available on a server
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    # --- data cleaning: raw text -> a clean word frequency table ---
    # lowercase, keep only alphabetic tokens of length >= 4, and drop a
    # small stopword list so the chart shows actual content words instead
    # of "the/and/that" dominating the counts.
    STOPWORDS = {
        "this", "that", "with", "from", "have", "which", "their", "these",
        "those", "when", "where", "what", "will", "would", "could", "should",
        "there", "been", "being", "also", "such", "into", "than", "then",
        "each", "more", "some", "other", "were", "they", "them", "your",
    }
    all_text = " ".join(STATE["documents"].values()).lower()
    words = [w for w in re.findall(r"[a-z]{4,}", all_text) if w not in STOPWORDS]

    if not words:
        raise HTTPException(400, "Not enough text to analyze yet.")

    freq = pd.Series(Counter(words)).sort_values(ascending=False).head(12)

    # --- chart: seaborn barplot, styled to match the app's dark theme,
    # rendered server-side, sent to the frontend as an embedded image ---
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("#161A22")
    ax.set_facecolor("#161A22")

    sns.barplot(x=freq.values, y=freq.index, ax=ax, color="#6C8CFF")
    ax.set_xlabel("Occurrences", color="#9CA3B5")
    ax.set_ylabel("")  # the index (words) is self-explanatory; "None" looks like a bug
    ax.set_title("Most frequent terms across loaded source(s)", color="#ECEEF4", fontsize=12)
    ax.tick_params(colors="#9CA3B5")
    for spine in ax.spines.values():
        spine.set_color("#262B39")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    chart_b64 = base64.b64encode(buf.read()).decode("utf-8")

    # --- a couple of quick pandas-computed stats, shown alongside the chart ---
    doc_word_counts = pd.Series({name: len(text.split()) for name, text in STATE["documents"].items()})

    return {
        "chart": f"data:image/png;base64,{chart_b64}",
        "documents": [{"name": name, "words": int(count)} for name, count in doc_word_counts.items()],
        "total_words": int(doc_word_counts.sum()),
        "unique_words": len(set(words)),
    }


@app.post("/api/ask")
def ask(req: AskRequest):
    if not STATE["chunks"]:
        raise HTTPException(400, "No documents loaded yet.")

    results = retrieve(req.question, STATE["vectorizer"], STATE["matrix"], STATE["chunks"], top_k=req.top_k)
    for r in results:
        r["highlight"] = best_sentence(r["text"], req.question)

    answer, low_confidence = generate_answer(req.question, results, use_llm=req.use_llm, api_key=req.api_key)

    return {
        "answer": answer,
        "low_confidence": low_confidence,
        "results": results,  # [{text, source, page, score, highlight}]
    }


@app.post("/api/summarize")
def summarize(req: SummarizeRequest):
    if not STATE["documents"]:
        raise HTTPException(400, "No documents loaded yet.")

    if req.source == "__all__":
        target_text = "\n\n".join(STATE["documents"].values())
    elif req.source in STATE["documents"]:
        target_text = STATE["documents"][req.source]
    else:
        raise HTTPException(404, f"Unknown document: {req.source}")

    if req.use_llm and req.api_key:
        style = "notes" if req.mode == "notes" else "summary"
        content = summarize_with_llm(target_text, req.api_key, style=style)
    elif req.mode == "notes":
        content = "\n".join(f"- {p}" for p in notes_extractive(target_text, num_points=10))
    else:
        content = summarize_extractive(target_text, num_sentences=6)

    return {"content": content}


# ---------------------------------------------------------------------------
# Serve the frontend
# ---------------------------------------------------------------------------
app.mount("/assets", StaticFiles(directory="static"), name="assets")


@app.get("/")
def serve_index():
    return FileResponse("static/index.html")
