"""
ReviewRadar AI - AI-Powered Customer Review Analysis
Runs fully locally: Streamlit (UI) + Ollama/llama3.2:3b (LLM)
+ Sentence-Transformers (embeddings) + ChromaDB (vector store).
"""

import json
import re
import uuid

import chromadb
import ollama
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
MODEL = "llama3.2:3b"                 # local Ollama model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast sentence-embedding model
MAX_PROMPT_CHARS = 6000               # how much review text we send to the LLM at once
BATCH_SIZE = 5                        # reviews classified per LLM call
SENTIMENTS = ["Positive", "Neutral", "Negative"]

st.set_page_config(page_title="ReviewRadar AI", page_icon="🔎", layout="wide")


class ReviewRadarError(Exception):
    """An error with a friendly message that we can show directly to the user."""


# ----------------------------------------------------------------------
# Ollama helpers
# ----------------------------------------------------------------------
def _model_name(model_info):
    """Read a model name from an Ollama list entry (works with old and new ollama versions)."""
    for key in ("model", "name"):
        try:
            value = model_info[key]
            if value:
                return value
        except (KeyError, TypeError, AttributeError):
            pass
    return ""


def check_ollama():
    """Return (ok, message). Checks that Ollama is running and the model is downloaded."""
    try:
        response = ollama.list()
    except Exception:
        return False, "Ollama is not running. Open the Ollama app or run `ollama serve` in a terminal."

    try:
        models = response["models"]
    except Exception:
        models = getattr(response, "models", [])

    names = [_model_name(m) for m in models]
    if MODEL not in names:
        return False, f"Model `{MODEL}` is not downloaded. Run: `ollama pull {MODEL}`"
    return True, f"Ollama is running and `{MODEL}` is ready."


def chat(prompt, system="You are a helpful assistant.", json_mode=False):
    """Send one prompt to the local LLM and return its text answer."""
    try:
        kwargs = {"format": "json"} if json_mode else {}
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.2, "num_ctx": 4096},
            **kwargs,
        )
        return response["message"]["content"]
    except ollama.ResponseError as e:
        if getattr(e, "status_code", None) == 404:
            raise ReviewRadarError(f"Model `{MODEL}` is not available. Run: `ollama pull {MODEL}`")
        raise ReviewRadarError(f"Ollama returned an error: {e}")
    except Exception:
        raise ReviewRadarError(
            "Could not talk to Ollama. Make sure it is running (open the Ollama app or run `ollama serve`)."
        )


def parse_json(text):
    """Turn the LLM's answer into a dict. Returns {} if it is not valid JSON."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {}


# ----------------------------------------------------------------------
# PDF + text processing
# ----------------------------------------------------------------------
def extract_text_from_pdf(uploaded_file):
    """Extract text from an uploaded PDF using PyPDF2. Returns (text, page_count)."""
    try:
        reader = PdfReader(uploaded_file)
        if reader.is_encrypted:
            reader.decrypt("")  # try an empty password
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception:
        raise ReviewRadarError("This file could not be read. It may be an invalid, corrupted or password-protected PDF.")

    text = "\n".join(pages).strip()
    if not text:
        raise ReviewRadarError("No text found in this PDF. It may be empty or a scanned image (needs OCR).")
    return text, len(pages)


def split_into_reviews(text):
    """Split raw text into individual reviews (blank-line separated, or one per line)."""
    text = text.replace("\r", "\n")
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) <= 1:  # no blank lines -> assume one review per line
        blocks = [line.strip() for line in text.split("\n") if line.strip()]

    reviews = []
    for block in blocks:
        block = re.sub(r"\s+", " ", block)  # join wrapped lines
        block = re.sub(r"^(review\s*)?#?\d+\s*[\.\):\-]\s*", "", block, flags=re.IGNORECASE)  # remove "1." / "Review 2:"
        if len(block) >= 10:
            reviews.append(block)
    return reviews


def build_review_sample(reviews, max_chars=MAX_PROMPT_CHARS):
    """Pick evenly spaced reviews so the prompt fits the model's context window."""
    total = sum(len(r) for r in reviews)
    if total <= max_chars:
        chosen = reviews
    else:
        step = total / max_chars
        chosen = reviews[:: max(1, int(step) + 1)]
    lines, used = [], 0
    for r in chosen:
        r = r[:600]
        if used + len(r) > max_chars:
            break
        lines.append(f"- {r}")
        used += len(r)
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Sentiment analysis (LLM)
# ----------------------------------------------------------------------
def normalize_label(raw):
    """Convert whatever the LLM wrote into Positive / Neutral / Negative."""
    s = str(raw).lower()
    if "pos" in s:
        return "Positive"
    if "neg" in s:
        return "Negative"
    return "Neutral"


def classify_single(review):
    """Classify one review (used as a fallback if a batch answer is incomplete)."""
    answer = chat(
        "Classify this customer review as exactly one word: Positive, Neutral, or Negative.\n\n"
        f"Review: {review[:500]}\n\nAnswer:",
        system="You are a sentiment classifier.",
    )
    return normalize_label(answer)


def classify_batch(batch):
    """Classify a small batch of reviews in one LLM call."""
    numbered = "\n".join(f"{i + 1}. {r[:500]}" for i, r in enumerate(batch))
    prompt = (
        "Classify each customer review as Positive, Neutral, or Negative.\n"
        'Reply ONLY with JSON, e.g. {"1": "Positive", "2": "Negative"}.\n\n'
        f"Reviews:\n{numbered}"
    )
    data = parse_json(chat(prompt, system="You are a sentiment classifier.", json_mode=True))

    labels = []
    for i, review in enumerate(batch):
        raw = data.get(str(i + 1)) if isinstance(data, dict) else None
        labels.append(normalize_label(raw) if raw is not None else classify_single(review))
    return labels


def classify_reviews(reviews, progress_callback=None):
    """Classify all reviews in batches and return a list of labels."""
    labels = []
    for start in range(0, len(reviews), BATCH_SIZE):
        labels.extend(classify_batch(reviews[start:start + BATCH_SIZE]))
        if progress_callback:
            progress_callback(min(len(labels), len(reviews)), len(reviews))
    return labels


def overall_sentiment(counts):
    """Work out the overall sentiment from the sentiment counts."""
    pos, neu, neg = counts["Positive"], counts["Neutral"], counts["Negative"]
    if pos > neg and pos >= neu:
        return "😊 Positive"
    if neg > pos and neg >= neu:
        return "😞 Negative"
    if pos == neg and pos > neu:
        return "😐 Mixed"
    return "😐 Neutral"


# ----------------------------------------------------------------------
# Full AI analysis (LLM)
# ----------------------------------------------------------------------
def as_list(value):
    """Make sure a value from the LLM is a list of plain strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        value = [value]
    items = []
    for item in value:
        if isinstance(item, dict):
            item = " - ".join(str(v) for v in item.values())
        if str(item).strip():
            items.append(str(item).strip())
    return items


def as_text(value):
    """Make sure a value from the LLM is a plain string."""
    return " ".join(as_list(value))


def analyze_reviews(reviews, counts):
    """Ask the LLM for the full analysis (summary, points, topics, suggestions)."""
    sample = build_review_sample(reviews)
    prompt = (
        "Analyze the customer reviews below. Use ONLY information found in the reviews.\n"
        "Reply ONLY with JSON using exactly these keys:\n"
        '  "overall_sentiment": one sentence describing the overall sentiment,\n'
        '  "summary": a short 3-4 sentence summary of all reviews,\n'
        '  "positive_points": list of short strings,\n'
        '  "negative_points": list of short strings,\n'
        '  "common_complaints": list of short strings,\n'
        '  "topics": list of objects like {"topic": "short name", "description": "what customers say about it"} '
        "- only topics that customers actually discuss,\n"
        '  "satisfaction_summary": 2-3 sentences about customer satisfaction,\n'
        '  "suggestions": list of concrete suggestions for improvement.\n\n'
        f"Sentiment counts: {counts['Positive']} positive, {counts['Neutral']} neutral, "
        f"{counts['Negative']} negative.\n\nReviews:\n{sample}"
    )
    data = parse_json(chat(prompt, system="You are an expert customer review analyst.", json_mode=True))
    if not data:
        raise ReviewRadarError("The AI returned an answer that could not be understood. Please try again.")

    topics = []
    for t in data.get("topics", []) or []:
        if isinstance(t, dict):
            topics.append({"topic": str(t.get("topic", "")).strip(), "description": as_text(t.get("description", ""))})
        elif str(t).strip():
            topics.append({"topic": str(t).strip(), "description": ""})

    return {
        "overall_sentiment": as_text(data.get("overall_sentiment")),
        "summary": as_text(data.get("summary")),
        "positive_points": as_list(data.get("positive_points")),
        "negative_points": as_list(data.get("negative_points")),
        "common_complaints": as_list(data.get("common_complaints")),
        "topics": [t for t in topics if t["topic"]],
        "satisfaction_summary": as_text(data.get("satisfaction_summary")),
        "suggestions": as_list(data.get("suggestions")),
    }


# ----------------------------------------------------------------------
# RAG: embeddings (Sentence-Transformers) + vector store (ChromaDB)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~90 MB)...")
def load_embedder():
    """Load the sentence-embedding model once and reuse it."""
    return SentenceTransformer(EMBEDDING_MODEL)


def chunk_review(review, size=800):
    """Keep short reviews whole; split very long ones into smaller chunks."""
    return [review[i:i + size] for i in range(0, len(review), size)]


def build_index(df):
    """Embed every review chunk and store it in an in-memory ChromaDB collection."""
    try:
        embedder = load_embedder()
    except Exception as e:
        raise ReviewRadarError(f"Could not load the embedding model ({e}). The first run needs internet to download it.")

    try:
        texts, metadatas = [], []
        for idx, row in df.iterrows():
            for chunk in chunk_review(row["Review"]):
                texts.append(chunk)
                metadatas.append({"review_id": int(idx), "sentiment": row["Sentiment"]})

        # Remove the previous collection (if any) and make a fresh one
        client = st.session_state.get("chroma_client") or chromadb.EphemeralClient()
        old = st.session_state.get("collection_name")
        if old:
            try:
                client.delete_collection(old)
            except Exception:
                pass
        name = "reviews_" + uuid.uuid4().hex[:8]
        collection = client.create_collection(name)

        embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
        for start in range(0, len(texts), 500):
            end = start + 500
            collection.add(
                ids=[f"chunk_{i}" for i in range(start, min(end, len(texts)))],
                documents=texts[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )
    except ReviewRadarError:
        raise
    except Exception as e:
        raise ReviewRadarError(f"ChromaDB error while indexing reviews: {e}")

    st.session_state["chroma_client"] = client
    st.session_state["collection_name"] = name
    st.session_state["collection"] = collection


def answer_question(question, top_k):
    """Retrieve relevant reviews from ChromaDB, then let the LLM answer from them."""
    collection = st.session_state["collection"]
    try:
        embedder = load_embedder()
        query_embedding = embedder.encode([question], show_progress_bar=False).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, collection.count()),
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]
    except Exception as e:
        raise ReviewRadarError(f"ChromaDB error while searching reviews: {e}")

    if not docs:
        raise ReviewRadarError("No relevant reviews were found.")

    context = "\n".join(f"{i + 1}. [{m['sentiment']}] {d}" for i, (d, m) in enumerate(zip(docs, metas)))
    prompt = (
        "Answer the question using ONLY the customer reviews below. "
        "If the reviews do not contain the answer, say so. Be clear and concise.\n\n"
        f"Reviews:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    answer = chat(prompt, system="You answer questions about customer reviews.")
    return answer, list(zip(docs, metas))


# ----------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------
def show_error(e):
    """Show a friendly error message."""
    if isinstance(e, ReviewRadarError):
        st.error(f"❌ {e}")
    else:
        st.error(f"❌ Unexpected error: {e}")


def bullet_card(title, items, empty="Nothing reported."):
    """A bordered card with a title and bullet points."""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if items:
            st.markdown("\n".join(f"- {i}" for i in items))
        else:
            st.caption(empty)


def set_question(q):
    """Button callback: put an example question into the text box."""
    st.session_state["question_input"] = q


# ----------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------
def page_analyze(max_reviews):
    """Page 1: get reviews from a PDF or pasted text, then run the analysis."""
    st.subheader("📥 Add your reviews")
    method = st.radio("Input method", ["📄 Upload a PDF", "✍️ Paste reviews"], horizontal=True)

    raw_text = ""
    if method == "📄 Upload a PDF":
        uploaded = st.file_uploader("Upload a PDF with customer reviews", type=["pdf"])
        if uploaded is not None:
            try:
                raw_text, pages = extract_text_from_pdf(uploaded)
                st.success(f"✅ Extracted {len(raw_text):,} characters from {pages} page(s).")
            except ReviewRadarError as e:
                show_error(e)
    else:
        raw_text = st.text_area(
            "Paste reviews here (one review per line, or separate reviews with a blank line)",
            height=250,
            key="pasted_text",
        )

    st.caption("Tip: put each review on its own line or separate reviews with a blank line.")

    if not st.button("🚀 Analyze Reviews", type="primary"):
        return

    # ---- Validate input ----
    if method == "📄 Upload a PDF" and uploaded is None:
        st.warning("⚠️ Please upload a PDF file first.")
        return
    if not raw_text.strip():
        st.warning("⚠️ There is no text to analyze. Please add some reviews.")
        return

    reviews = split_into_reviews(raw_text)
    if not reviews:
        st.warning("⚠️ No reviews could be found in the text.")
        return

    ok, message = check_ollama()
    if not ok:
        st.error(f"❌ {message}")
        return

    if len(reviews) > max_reviews:
        st.info(f"Found {len(reviews)} reviews. Analyzing the first {max_reviews} (change this in the sidebar).")
        reviews = reviews[:max_reviews]

    # ---- Run the analysis ----
    try:
        progress = st.progress(0.0, text="Classifying sentiment...")

        def update(done, total):
            progress.progress(done / total, text=f"Classifying sentiment... {done}/{total}")

        labels = classify_reviews(reviews, update)
        df = pd.DataFrame({"Review": reviews, "Sentiment": labels})
        counts = {s: int((df["Sentiment"] == s).sum()) for s in SENTIMENTS}

        with st.spinner("🤖 Generating AI insights..."):
            analysis = analyze_reviews(reviews, counts)

        with st.spinner("🔍 Building search index..."):
            build_index(df)
       

        progress.empty()
        st.session_state["df"] = df
        st.session_state["analysis"] = analysis
        st.success(f"✅ Analysis complete for {len(df)} reviews! Open **📊 Dashboard** or **💬 Ask ReviewRadar** in the sidebar.")
    except Exception as e:
        show_error(e)


def page_dashboard():
    """Page 2: metrics, summary, topics and insights."""
    st.subheader("📊 Dashboard")
    if "df" not in st.session_state:
        st.info("No data yet. Go to **📥 Analyze Reviews** and analyze some reviews first.")
        return

    df, analysis = st.session_state["df"], st.session_state["analysis"]
    counts = {s: int((df["Sentiment"] == s).sum()) for s in SENTIMENTS}

    # Metric cards
    cols = st.columns(5)
    cols[0].metric("Total Reviews", len(df))
    cols[1].metric("Positive 😊", counts["Positive"])
    cols[2].metric("Neutral 😐", counts["Neutral"])
    cols[3].metric("Negative 😞", counts["Negative"])
    cols[4].metric("Overall Sentiment", overall_sentiment(counts))

    with st.container(border=True):
        st.markdown("**🧠 AI-Generated Summary**")
        st.write(analysis["summary"] or "No summary was generated.")
        if analysis["overall_sentiment"]:
            st.caption(f"Overall sentiment: {analysis['overall_sentiment']}")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Sentiment", "🧩 Main Topics", "✅ Insights", "📝 All Reviews"])

    with tab1:
        chart_df = pd.DataFrame({"Reviews": [counts[s] for s in SENTIMENTS]}, index=SENTIMENTS)
        st.bar_chart(chart_df)
        for s in SENTIMENTS:
            pct = counts[s] / len(df) * 100
            st.write(f"**{s}:** {counts[s]} reviews ({pct:.0f}%)")

    with tab2:
        if not analysis["topics"]:
            st.info("The AI did not identify any topics.")
        for i in range(0, len(analysis["topics"]), 2):
            cols = st.columns(2)
            for col, topic in zip(cols, analysis["topics"][i:i + 2]):
                with col, st.container(border=True):
                    st.markdown(f"**{topic['topic']}**")
                    st.caption(topic["description"] or "—")

    with tab3:
        left, right = st.columns(2)
        with left:
            bullet_card("👍 Positive Points", analysis["positive_points"])
        with right:
            bullet_card("👎 Negative Points", analysis["negative_points"])
        with st.expander("⚠️ Common Complaints", expanded=True):
            st.markdown("\n".join(f"- {c}" for c in analysis["common_complaints"]) or "None reported.")
        with st.expander("😊 Customer Satisfaction Summary", expanded=True):
            st.write(analysis["satisfaction_summary"] or "Not available.")
        with st.expander("🛠️ Suggestions for Improvement", expanded=True):
            st.markdown("\n".join(f"- {s}" for s in analysis["suggestions"]) or "None reported.")

    with tab4:
        choice = st.selectbox("Filter by sentiment", ["All"] + SENTIMENTS)
        view = df if choice == "All" else df[df["Sentiment"] == choice]
        st.dataframe(view, use_container_width=True, hide_index=True)


def page_ask(top_k):
    """Page 3: RAG question answering."""
    st.subheader("💬 Ask ReviewRadar")
    if "collection" not in st.session_state:
        st.info("No data yet. Go to **📥 Analyze Reviews** and analyze some reviews first.")
        return

    st.write("Ask anything about your reviews. Answers come from the most relevant reviews found in ChromaDB.")

    examples = [
        "What are customers complaining about?",
        "What do customers like most?",
        "Why are customers dissatisfied?",
        "What should the company improve?",
        "Summarize the delivery-related complaints.",
    ]
    st.caption("Try an example:")
    cols = st.columns(len(examples))
    for col, q in zip(cols, examples):
        col.button(q, on_click=set_question, args=(q,), use_container_width=True)

    question = st.text_input("Your question", key="question_input")

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("⚠️ Please type a question first.")
            return
        ok, message = check_ollama()
        if not ok:
            st.error(f"❌ {message}")
            return
        try:
            with st.spinner("Searching reviews and thinking..."):
                answer, sources = answer_question(question.strip(), top_k)
            with st.container(border=True):
                st.markdown("**🤖 Answer**")
                st.write(answer)
            with st.expander(f"📚 Reviews used to answer ({len(sources)})"):
                for doc, meta in sources:
                    st.markdown(f"**[{meta['sentiment']}]** {doc}")
        except Exception as e:
            show_error(e)


# ----------------------------------------------------------------------
# Main app
# ----------------------------------------------------------------------
def main():
    st.title("🔎 ReviewRadar AI")
    st.subheader("AI-Powered Customer Review Analysis")

    # ---- Sidebar: navigation + options ----
    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Go to", ["📥 Analyze Reviews", "📊 Dashboard", "💬 Ask ReviewRadar"], label_visibility="collapsed")

        st.divider()
        st.header("Options")
        max_reviews = st.slider("Max reviews to analyze", 5, 100, 30,
                                help="More reviews take longer because a local model is used.")
        top_k = st.slider("Reviews retrieved per question", 3, 15, 6)

        st.divider()
        st.header("Status")
        ok, message = check_ollama()
        (st.success if ok else st.error)(message)

        if st.button("🗑️ Clear results"):
            for key in ["df", "analysis", "collection", "collection_name", "chroma_client", "question_input"]:
                st.session_state.pop(key, None)
            st.rerun()

    if page.startswith("📥"):
        page_analyze(max_reviews)
    elif page.startswith("📊"):
        page_dashboard()
    else:
        page_ask(top_k)


if __name__ == "__main__":
    main()
