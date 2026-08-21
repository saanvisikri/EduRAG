"""
rag_core.py
------------
The actual "RAG" logic, with zero UI code, so you can read it top to bottom
and understand every step.

RAG = Retrieval-Augmented Generation. Three jobs:
    1. CHUNK the textbook(s) into small passages, tagged with where they
       came from (file + page number)
    2. RETRIEVE the passages most relevant to a question
    3. GENERATE an answer using those passages as grounding, and flag it
       when the match is weak instead of confidently guessing

This version uses TF-IDF + cosine similarity for retrieval instead of
neural embeddings. Why: it needs no internet connection, no model download,
and no GPU — it runs instantly on any laptop — while teaching the exact
same retrieval concept you'd use with embeddings later. Swapping it for
sentence-transformers + FAISS is described in the README as a next step.
"""

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Below this TF-IDF similarity score, we don't trust the match enough to
# present it as a confident answer. Tuned empirically for short queries
# against TF-IDF — see README for why this number, not a "real" constant.
LOW_CONFIDENCE_THRESHOLD = 0.08


def load_pdf_pages(file) -> list[str]:
    """Extract text from a PDF, one string per page, so page numbers can
    be attached to every chunk later."""
    import pdfplumber
    with pdfplumber.open(file) as pdf:
        return [page.extract_text() or "" for page in pdf.pages]


def load_docx_text(file) -> str:
    """
    Extract text from a Word document. DOCX has no fixed 'pages' the way a
    PDF does (page breaks depend on the reader's font/zoom, not the file
    itself), so this returns one block of text rather than a per-page list
    — callers should chunk it with track_pages=False, the same way a plain
    .txt file is handled.
    """
    import docx
    document = docx.Document(file)
    return "\n".join(p.text for p in document.paragraphs if p.text.strip())


def chunk_document(pages: list[str], source: str, track_pages: bool = True,
                    chunk_size: int = 150, overlap: int = 40) -> list[dict]:
    """
    Chunk a document page by page, so every chunk remembers which file (and,
    for PDFs, which page) it came from — that's what lets the app cite its
    source instead of just handing back an anonymous blob of text.

    track_pages=False is for plain .txt uploads, which have no real page
    boundaries — we chunk the single block of text but don't fabricate a
    page number for it.
    """
    chunks = []
    step = max(chunk_size - overlap, 1)  # avoid infinite loop if overlap >= chunk_size

    for page_num, page_text in enumerate(pages, start=1):
        words = page_text.split()
        if not words:
            continue
        start = 0
        while start < len(words):
            end = start + chunk_size
            text = " ".join(words[start:end])
            chunks.append({
                "text": text,
                "source": source,
                "page": page_num if track_pages else None,
            })
            start += step

    return chunks


def build_index(texts: list[str]):
    """
    Turn every chunk into a TF-IDF vector — a row of numbers where each
    column is a word, and the value says "how important is this word to
    this chunk, relative to the rest of the corpus."

    Returns the fitted vectorizer (needed later to vectorize the query
    the same way) and the matrix of chunk vectors.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def retrieve(query: str, vectorizer, matrix, chunks: list[dict], top_k: int = 3) -> list[dict]:
    """
    Vectorize the question with the SAME vectorizer used on the chunks,
    then rank every chunk by cosine similarity to the question.

    Returns the top matching chunk dicts (text/source/page), each with a
    "score" key added, best first.
    """
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).flatten()
    top_indices = scores.argsort()[::-1][:top_k]

    results = []
    for i in top_indices:
        if scores[i] > 0:
            entry = dict(chunks[i])
            entry["score"] = float(scores[i])
            results.append(entry)
    return results


def best_sentence(text: str, query: str) -> str:
    """
    Within a single retrieved chunk, find the ONE sentence most relevant to
    the question — same TF-IDF + cosine similarity idea used for retrieval
    and summarization, just applied at sentence granularity instead of
    chunk or document granularity. This is what lets the UI show a short
    "key excerpt" instead of making the reader scan a whole paragraph to
    find the relevant line themselves.

    Falls back to the first sentence (or the raw text) if there's nothing
    meaningful to compare — a chunk with only one sentence, or one so short
    TF-IDF has no vocabulary to work with.
    """
    sentences = split_sentences(text)
    if not sentences:
        return text
    if len(sentences) == 1:
        return sentences[0]

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(sentences + [query])
    except ValueError:
        return sentences[0]

    sentence_vectors = matrix[:-1]
    query_vector = matrix[-1]
    scores = cosine_similarity(query_vector, sentence_vectors).flatten()
    return sentences[int(scores.argmax())]


def _cite(chunk: dict) -> str:
    """Format a human-readable citation like 'chapter3.pdf, page 4'."""
    if chunk.get("page"):
        return f"{chunk['source']}, page {chunk['page']}"
    return chunk["source"]


def generate_answer(query: str, retrieved: list[dict],
                     use_llm: bool = False, api_key: str | None = None) -> tuple[str, bool]:
    """
    Turn retrieved chunks into an actual answer.

    Two modes:
      - Extractive (default, no API key needed): pull out the single
        sentence in the best-matching chunk that's most relevant to the
        question (via best_sentence, above), and lead with that — the
        full chunk stays available via its citation, but isn't dumped
        into the answer itself.
      - LLM-generated (optional): send the retrieved chunks + question to
        Claude, and let it write a proper answer grounded in that context,
        citing sources. This is the "G" in RAG — generation on top of
        retrieval.

    Returns (answer_text, low_confidence) — low_confidence is True when
    the best match is weak enough that the answer shouldn't be trusted
    blindly. The UI uses this to show a warning instead of a confident
    wrong answer.
    """
    if not retrieved:
        return "I couldn't find anything relevant to that question in this text.", True

    low_confidence = retrieved[0]["score"] < LOW_CONFIDENCE_THRESHOLD

    if not use_llm or not api_key:
        best = retrieved[0]
        key_sentence = best_sentence(best["text"], query)
        answer = f"{key_sentence}\n\n— {_cite(best)} · similarity {best['score']:.2f}"
        return answer, low_confidence

    import anthropic

    context = "\n\n---\n\n".join(f"[{_cite(r)}]\n{r['text']}" for r in retrieved)
    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        "Answer the question using ONLY the context below, which is taken "
        "from a textbook. Cite the source (given in brackets) inline when "
        "you use it. If the context doesn't contain the answer, say so "
        "clearly instead of guessing.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text, low_confidence


# ---------------------------------------------------------------------------
# Summarization / notes — a second use of the same TF-IDF machinery above,
# just pointed at "which sentences matter" instead of "which chunks match
# a question".
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """
    A lightweight sentence splitter — no NLTK/punkt download needed (this
    environment has no internet access to fetch that model), just a regex
    that's good enough for well-formatted textbook prose: split after
    ./!/? when followed by a space and a capital letter.
    """
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?]) +(?=[A-Z])", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]  # drop stray fragments


def _top_sentences(text: str, count: int) -> list[str]:
    """
    Score every sentence by how much distinctive, high-TF-IDF vocabulary
    it contains, keep the top `count`, and return them in their ORIGINAL
    order — that's what keeps an extractive summary readable instead of
    a jumbled bag of the "most important" sentences.
    """
    sentences = split_sentences(text)
    if len(sentences) <= count:
        return sentences

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(sentences)
    scores = matrix.sum(axis=1).A1  # total TF-IDF weight per sentence

    keep = sorted(scores.argsort()[::-1][:count])  # top N indices, back in reading order
    return [sentences[i] for i in keep]


def summarize_extractive(text: str, num_sentences: int = 6) -> str:
    """A short paragraph summary: the highest-scoring sentences, stitched
    back together in original order. No LLM, no internet, instant."""
    top = _top_sentences(text, num_sentences)
    return " ".join(top) if top else "Not enough text to summarize."


def notes_extractive(text: str, num_points: int = 10) -> list[str]:
    """Same scoring, more sentences, returned as a flat list for bullets."""
    top = _top_sentences(text, num_points)
    return top


def summarize_with_llm(text: str, api_key: str, style: str = "summary") -> str:
    """
    LLM-generated summary or notes — reads more naturally than the
    extractive version because it can paraphrase instead of just
    lifting sentences. style="notes" asks for exam-style bullet points;
    style="summary" asks for a short coherent paragraph.
    """
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    if style == "notes":
        instruction = (
            "Turn the following textbook content into concise, exam-ready "
            "bullet-point revision notes. Group points by topic if more "
            "than one topic is covered."
        )
    else:
        instruction = (
            "Write a concise, coherent summary (5-8 sentences) of the "
            "following textbook content, covering its key ideas."
        )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        messages=[{"role": "user", "content": f"{instruction}\n\n{text}"}],
    )
    return response.content[0].text
