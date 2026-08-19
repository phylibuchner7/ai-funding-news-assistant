# ai-funding-news-assistant
A retrieval-augmented generation (RAG) chatbot that answers questions about recent AI and startup funding news. Answers are grounded in real, recently scraped articles, with sources included, rather than relying on an LLM's built-in knowledge, which is only accurate up to its training cutoff.

## What It Does

- Scrapes recent articles from TechCrunch's Fundraising section
- Chunks and embeds the content using a Hugging Face `sentence-transformers` model
- Stores embeddings in a FAISS vector index for fast retrieval
- Answers questions using OpenAI's API, grounded only in retrieved article content
- Detects ranking questions ("most," "least," "largest," "smallest") and routes them to a separate, accurate structured data table instead of relying on retrieval alone, since retrieval only searches a handful of chunks and cannot reliably answer comparison questions across a full dataset
- Presents everything through a Gradio chat interface

## Tech Stack

- Python
- BeautifulSoup (scraping)
- Hugging Face `sentence-transformers` (embeddings)
- FAISS (vector search)
- OpenAI API (answer generation)
- Gradio (chat interface)
- Render (deployment)

## Example Questions

- "What was the largest funding round this week?"
- "Which companies raised money for AI hardware?"
- "What did Databricks raise?"

## Known Limitation

Standard RAG retrieval is well-suited to lookup-style questions but struggles with ranking or comparison questions, since it only retrieves a small number of relevant chunks rather than scanning the entire dataset. This project addresses that by extracting structured funding data separately and routing ranking questions to that data directly, rather than relying on the language model to compare a partial set of retrieved articles.

## Roadmap

- Scheduled data refresh and a weekly email digest of new funding news
