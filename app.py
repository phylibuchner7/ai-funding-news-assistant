# ============================================================
# AI FUNDING NEWS ASSISTANT
# ============================================================
# A retrieval-augmented generation (RAG) chatbot that answers
# questions about recent AI and startup funding news. Content is
# scraped from TechCrunch's Fundraising section, chunked, embedded,
# and stored in a FAISS index for retrieval. Answers are generated
# using OpenAI's API, grounded only in retrieved article content.
#
# Ranking questions ("most", "least", etc.) are routed to a
# separate structured lookup table instead of RAG retrieval, since
# retrieval only searches a handful of chunks and cannot reliably
# answer comparison questions across the full dataset.
# ============================================================

import os
import re
import time
import numpy as np
import requests
from bs4 import BeautifulSoup
import faiss
from openai import OpenAI
import gradio as gr

# ============================================================
# SETUP
# ============================================================
# On Render, environment variables are set in the dashboard
# rather than loaded from Colab Secrets.
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

client = OpenAI(api_key=openai_api_key)

BASE_URL = "https://techcrunch.com/category/fundraising/"

def get_openai_embeddings(texts):
    """
    Generates embeddings using OpenAI's API instead of a local
    sentence-transformers model, to avoid the memory overhead of
    loading a full model (and PyTorch) on Render's free tier.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings)

# ============================================================
# STEP 1: Scrape Article Links
# ============================================================
def get_article_links(url, max_articles=20):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    articles = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        text = link.get_text(strip=True)
        if "techcrunch.com/2026/" in href or "techcrunch.com/2025/" in href:
            if text and len(text) > 15:
                articles.append({"title": text, "url": href})

    seen = set()
    unique_articles = []
    for a in articles:
        if a["url"] not in seen:
            seen.add(a["url"])
            unique_articles.append(a)

    return unique_articles[:max_articles]


# ============================================================
# STEP 2: Fetch Full Article Text
# ============================================================
def get_article_text(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        content_container = soup.find("div", class_="entry-content")
        if content_container:
            paragraphs = content_container.find_all("p")
        else:
            paragraphs = soup.find_all("p")

        text = " ".join(p.get_text(strip=True) for p in paragraphs)
        return text
    except Exception as e:
        print(f"Could not fetch {url}: {e}")
        return ""


# ============================================================
# STEP 3: Chunk Article Text
# ============================================================
def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ============================================================
# STEP 4: Extract Funding Amounts for the Ranking Table
# ============================================================
def extract_funding_amount(text):
    match = re.search(r'\$(\d+\.?\d*)\s*(billion|B|million|M)', text, re.IGNORECASE)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2).lower()
    return amount * 1000 if unit.startswith('b') else amount


# Manual corrections for titles where automatic extraction grabbed
# the wrong number (a valuation instead of the actual raise, or an
# initial ask instead of the final settled amount).
MANUAL_CORRECTIONS = {
    "AI coding startup Cognition reportedly already in talks to raise at $40B valuation": None,
    "Lovable confirms new $13.3B valuation, raises another $400M": 400.0,
    "Databricks wanted to raise $1B, investors wanted $15B. It settled on $5B at a $190B valuation.": 5000.0,
    "Bending Spoons to buy Airtable for $1.28B": None,
}


# ============================================================
# STEP 5: BUILD PIPELINE: Runs once at startup to produce all_chunks,
# the FAISS index, and funding_table.
# ============================================================
def build_pipeline():
    print("Scraping TechCrunch fundraising articles...")
    articles = get_article_links(BASE_URL, max_articles=20)

    for article in articles:
        article["text"] = get_article_text(article["url"])
        time.sleep(1)

    print("Chunking article text...")
    all_chunks = []
    for article in articles:
        for chunk in chunk_text(article["text"]):
            all_chunks.append({
                "text": chunk,
                "title": article["title"],
                "url": article["url"]
            })

    print("Generating embeddings and building FAISS index...")
    chunk_texts = [c["text"] for c in all_chunks]
    embeddings = get_openai_embeddings(chunk_texts)
    embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print("Building structured funding table...")
    funding_table = []
    for article in articles:
        amount = extract_funding_amount(article["title"])
        if amount is not None:
            funding_table.append({
                "title": article["title"],
                "amount_millions": amount,
                "url": article["url"]
            })

    for entry in funding_table:
        if entry["title"] in MANUAL_CORRECTIONS:
            corrected = MANUAL_CORRECTIONS[entry["title"]]
            entry["amount_millions"] = corrected  # may be set to None to exclude

    funding_table = [e for e in funding_table if e["amount_millions"] is not None]
    funding_table.sort(key=lambda x: x["amount_millions"], reverse=True)

    print(f"Pipeline built: {len(all_chunks)} chunks, {len(funding_table)} funding entries.")
    return all_chunks, index, funding_table

# Run once when the app starts up
all_chunks, index, funding_table = build_pipeline()

# ============================================================
# STEP 6: Retrieval + Question Answering
# ============================================================
def retrieve_relevant_chunks(question, top_k=4):
    question_embedding = get_openai_embeddings([question])
    question_embedding = np.ascontiguousarray(question_embedding.astype(np.float32))
    distances, indices = index.search(question_embedding, top_k)
    return [all_chunks[i] for i in indices[0]]


def answer_question(question):
    relevant_chunks = retrieve_relevant_chunks(question)
    context = "\n\n".join(
        f"Source: {chunk['title']}\n{chunk['text']}"
        for chunk in relevant_chunks
    )

    prompt = (
        "You are an assistant that answers questions about recent "
        "AI and startup funding news. Using ONLY the context below, "
        "answer the question. If the context does not contain the "
        "answer, respond with exactly: NOT_FOUND. "
        "If the question asks you to compare, rank, or find the "
        "'most' or 'least' across companies, note that your answer "
        "is based only on the limited set of articles retrieved, "
        "not a full comparison of all available funding news.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer = response.choices[0].message.content.strip()

    if answer == "NOT_FOUND":
        friendly_message = (
            "I don't have enough information in the current articles "
            "to answer that. Try asking about a specific company, "
            "funding amount, or investor instead."
        )
        return friendly_message, []

    sources = list(set(chunk["url"] for chunk in relevant_chunks))
    return answer, sources


# ============================================================
# STEP 7: Route Ranking Questions to the Structured Table
# ============================================================
def is_ranking_question(question):
    """
    Routes to the structured funding table ONLY for pure amount
    rankings across ALL companies (e.g., "which company raised
    the most"). Any question with an added category, topic, or
    time filter (AI voice, recently, this month, etc.) is excluded,
    since the funding table has no category or date data and
    cannot answer filtered questions accurately.
    """
    question_lower = question.lower()

    ranking_keywords = [
        "most", "least", "largest", "smallest",
        "highest", "lowest", "biggest", "top", "bottom"
    ]
    funding_keywords = [
        "funding", "raised", "raise", "round", "money", "amount"
    ]
    exclusion_keywords = [
        "who is", "who's", "ceo", "founder", "founded",
        "when", "where", "why", "how many employees",
        "headquartered", "based in", "investor", "led by",
        "recent", "recently", "this week", "this month",
        "space", "sector", "industry", "category",
        "voice", "audio", "hardware", "chip", "defense",
        "healthcare", "fintech", "energy"
    ]

    has_ranking_word = any(kw in question_lower for kw in ranking_keywords)
    has_funding_word = any(kw in question_lower for kw in funding_keywords)
    has_exclusion_word = any(kw in question_lower for kw in exclusion_keywords)

    return has_ranking_word and has_funding_word and not has_exclusion_word

# ============================================================
# STEP 8: Gradio Chat Interface
# ============================================================
demo = gr.ChatInterface(
    fn=chatbot_response,
    title="AI Funding News Assistant",
    description=(
        "Ask about recent AI and startup funding news. "
        "Answers are grounded in real, recently scraped articles "
        "from TechCrunch's Fundraising section, with sources included."
    ),
    examples=[
        "What was the largest funding round this week?",
        "Which companies raised money for AI hardware?",
        "What did Databricks raise?",
    ],
)

if __name__ == "__main__":
    # Render provides a PORT environment variable; default to 7860
    # (Gradio's standard port) if running locally.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
