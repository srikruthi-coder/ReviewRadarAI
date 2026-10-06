# 🔎 ReviewRadar AI

**AI-Powered Customer Review Analysis** — upload a PDF of customer reviews or paste them in, and get sentiment, topics, a summary, improvement suggestions, and a chat-style Q&A. Everything runs **locally** with Ollama (no paid APIs, no OpenAI).

## Features

- Upload a PDF (PyPDF2) or paste reviews manually
- Sentiment classification of every review: Positive / Neutral / Negative
- AI-generated summary, positive points, negative points, common complaints, satisfaction summary and suggestions
- Topics discovered by the AI from your actual reviews (nothing hard-coded)
- Dashboard with metrics, sentiment chart, topic cards, and a filterable review table
- **💬 Ask ReviewRadar**: RAG question answering using Sentence-Transformers embeddings + ChromaDB retrieval + Llama 3.2
- Error handling for missing input, invalid/empty PDFs, Ollama not running, missing model, ChromaDB problems, and unexpected errors

## Required software

- Python 3.12
- [Ollama](https://ollama.com) with the `llama3.2:3b` model
- VS Code (optional, any editor works)

## Installation

```bash
# 1. Go into the project folder
cd ReviewRadarAI

# 2. (Recommended) create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install the Python packages
pip install -r requirements.txt
```

## How to install Ollama

1. Go to <https://ollama.com/download> and download the installer for your OS (Windows, macOS, Linux).
2. Install it and start the Ollama app. On Linux you can also run `ollama serve`.
3. Check it works: `ollama --version`

## How to download llama3.2:3b

```bash
ollama pull llama3.2:3b
```

Verify with `ollama list` — you should see `llama3.2:3b`.

## How to run

```bash
streamlit run app.py
```

Your browser opens at <http://localhost:8501>.

> **Note:** The first run downloads the small embedding model `all-MiniLM-L6-v2` (~90 MB) from the internet once. After that it is cached and everything works offline.

## How to use

1. **📥 Analyze Reviews** — upload a PDF or paste reviews, then click **Analyze Reviews**.
2. **📊 Dashboard** — see the metrics, summary, topics, and insights.
3. **💬 Ask ReviewRadar** — ask questions about your reviews.

**Input format tip:** put each review on its own line, or separate reviews with a blank line. Numbering like `1.` or `Review 2:` is removed automatically. Scanned (image-only) PDFs contain no text and can't be read.

Because a local 3B model is used, analysis time depends on your computer. Use the **Max reviews to analyze** slider in the sidebar to keep it quick.

## Example questions to ask the AI

- What are customers complaining about?
- What do customers like most?
- Why are customers dissatisfied?
- What should the company improve?
- Summarize the delivery-related complaints.
- What do customers say about the price?
- Are customers happy with customer service?

## Troubleshooting

| Problem | Fix |
|---|---|
| "Ollama is not running" | Open the Ollama app or run `ollama serve` |
| "Model llama3.2:3b is not downloaded" | Run `ollama pull llama3.2:3b` |
| "Could not load the embedding model" | Connect to the internet once so it can download |
| PDF has no text | It's probably a scanned image; use a text-based PDF |

## Project structure

```
ReviewRadarAI/
├── app.py
├── requirements.txt
└── README.md
```
