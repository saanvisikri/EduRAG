# EduRAG

## Why is this helpful

So basically the problem started with me using ChatGPT/Claude to study for exams. It's genuinely helpful, but there's one issue — the wording it gives you is never the wording your teacher actually uses. And in exams, especially theory-heavy subjects, you lose marks if you don't use the technical/textbook language, even if your answer is conceptually correct.

The other problem is just how time consuming it is, if I have a question and the answer is buried somewhere in a 200-page textbook, I don't want to sit and skim through the whole thing to find one paragraph. That's basically what this project fixes.

EduRAG lets me drop in my own textbook (PDF or Word doc, both work), ask it a question in plain English, and it pulls the actual answer from *my* book — not an AI-generated explanation. So the language matches what's actually written in the material I'm being tested on, and I don't have to go hunting for it page by page.

**It's not trying to replace ChatGPT/Claude for studying**, it's more like a companion — use it to find the exact textbook wording fast, then use Claude/GPT normally if you want a broader explanation. Also added an option to plug in your own Claude API key if you want it to generate better answers/summaries instead of just pulling raw text — that part's optional though, it works fully offline without it too.

## What it actually does

- **Drag and drop your files** — PDF, DOCX, TXT, doesn't matter, and you can add more than one at a time. So say I have my textbook + my own handwritten notes typed up, I can throw both in and it'll search across all of them together.
- **Ask a question, get an actual answer with the source** — not just "here's a chunk of text, find your answer in it." It picks out the exact sentence that answers your question first, and if you want the full paragraph around it for context, you just click the citation and it's right there. Also tells you which file and which page it came from, so you can literally go check the book if you want.
- **Highlights your keywords in the citation** — kind of like how Google bolds your search terms in the results. Makes it easy to scan.
- **Confidence warning** — if the match isn't actually good, it tells you instead of confidently giving you a wrong-ish answer. Learned pretty early on that a bad "confident" answer is worse than no answer.
- **Summarize / notes tab** — for when I don't want Q&A, I just want a quick summary of a chapter or exam-style bullet points to revise from.
- **Insights tab** — uses pandas + data viz (seaborn/matplotlib). It shows a word-frequency breakdown of your document as a chart, so you get a visual sense of what your file is actually made of and how the retrieval is behaving, not just numbers in a sidebar.
- **Session stats in the sidebar** — chunks indexed, how many questions you've asked, average confidence, so you can see it's actually working in real time.
- **Export as a study sheet** — whatever you did in the session (Q&A + summaries) downloads as one markdown file so you're not stuck re-doing it.
- **Two modes: extractive or Claude-generated** — extractive is the default (offline, free, just pulls from your book). If you add a Claude API key, it can generate proper answers/summaries instead. One toggle switches both Q&A and the summarizer.

## How it's built (quick architecture note)

I split this into a proper backend + frontend instead of doing it all in Streamlit, mainly because I wanted actual practice building something closer to a real product, not just a demo script.

- `main.py` is the API layer — handles the file uploads, keeps track of the session, and has the routes (`/api/upload`, `/api/ask`, `/api/summarize`, `/api/sample`, `/api/reset`).
- `rag_core.py` is where the actual RAG logic lives — chunking, TF-IDF indexing, retrieval, generation. It is also expandable if i want to expand this to a mobile app later because the core logic stays the same.
- `static/` is just plain HTML/CSS/JS.


The backend serves the frontend directly too, so it's one process.

## How to run it

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then just open **http://localhost:8000** — drag in a file (or hit "Try the sample textbook" if you don't want to upload your own right away), and start asking questions from there.

## Why TF-IDF and not embeddings

I went with TF-IDF + cosine similarity instead of neural embeddings, and this was intentional, not because I didn't know embeddings exist.

- No model to download, no GPU needed, runs completely offline once installed — useful when you're revising and works offline.
- It's also fully explainable — the similarity score literally just means "how much overlapping vocabulary is there between your question and this passage." Nothing hidden inside a black box.

The real limitation is TF-IDF matches on exact words, not meaning. So if you ask "What is a CNN used for?" and the book only ever says "CNNs are...", the plural/singular mismatch alone can make it feel vague. This is honestly the textbook example of why the field eventually moved from keyword search (TF-IDF/BM25) to embedding-based semantic search. If I extend this later, that's the first thing I'd fix.

## What I'd add if I kept working on this if looking to expand

- Swap TF-IDF for `sentence-transformers` embeddings so it matches on meaning, not just exact words — same architecture, just smarter matching underneath.
- A proper vector DB (FAISS/ChromaDB) once someone's loading thousands of chunks across multiple books instead of just brute-force searching everything.
- Right now the backend holds one shared session in memory, which is fine for a local demo/submission but obviously wouldn't survive multiple users at once. A real version would need per-user sessions with an actual database. Not an oversight, just out of scope for what I needed this for.
- Streaming answers — FastAPI supports it, so the Claude-generated answers could type out token by token instead of appearing all at once.

---

I have tested this with my own study material and it performed well for the same, I dropped pdfs and word files and asked questions about the text and got the answers along with a summary.
