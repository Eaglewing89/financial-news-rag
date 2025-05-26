# Financial News RAG

> **Automate. Analyze. Act.**

A modern, AI-powered backend for financial news retrieval, search, and analysis—built for teams who need actionable insights, not information overload.

```mermaid
graph TD
    orchestrator["FinancialNewsRAG"]
    news_fetcher["Fetch"]
    text_processor["Process"]
    embeddings_generator["Embed"]
    semantic_similarity["Search"]

    subgraph core_functionality[" "]
        direction LR
        news_fetcher --> text_processor --> embeddings_generator --> semantic_similarity
    end
    orchestrator --> core_functionality

    style orchestrator fill:#98cbf5,stroke:#3182CE,color:#333333
    style news_fetcher fill:#E0E0E0,stroke:#B0B0B0,color:#333333
    style text_processor fill:#c4b2ed,stroke:#8A6FDF,color:#333333
    style embeddings_generator fill:#87d6b6,stroke:#4D9A7F,color:#333333
    style semantic_similarity fill:#B2DFDB,stroke:#80CBC4,color:#333333
```

---

## 🚀 What is Financial News RAG?

**Financial News RAG** (Retrieval Augmented Generation) is a Python library and backend system that streamlines the process of fetching, processing, embedding, and searching financial news articles. It leverages advanced NLP and LLMs to help you:

- **Automate** news collection from sources like EODHD
- **Transform** raw news into structured, searchable data
- **Search** semantically for relevant articles using vector embeddings
- **Re-rank** results with LLMs for maximum relevance

**Stop sifting. Start finding.**

---

## 🔑 Key Features

- **End-to-end pipeline:** From news ingestion to semantic search
- **LLM-powered re-ranking:** Uses Gemini for smarter results
- **Configurable & extensible:** API keys, models, and storage are all customizable
- **Production-ready:** Built for integration into financial dashboards, research tools, and analytics platforms

---

## 📦 Installation

See the [Installation Guide](docs/installation.md) for full instructions, including API key setup.

```bash
pip install financial-news-rag
```

---

## ⚡ Quick Start

```python
from financial_news_rag import FinancialNewsRAG

# Initialize with your config (see docs/configuration.md)
rag = FinancialNewsRAG()

# Article Storage Pipeline
rag.fetch_and_store_articles(
        tag="MERGERS AND ACQUISITIONS", 
        from_date="2025-05-26", 
        to_date="2025-04-26", 
        limit=1000
    )
rag.process_articles_by_status(status='PENDING')
rag.embed_processed_articles()

# Article Search
query = "Which billion-dollar M&A deals were announced in the tech sector in the last 30 days, and what were the valuation multiples?"
results = orchestrator.search_articles(
        query=query, 
        n_results=50,
        rerank=True
    )
```

See the [Usage Guide](./docs/usage_guide.md) for a full walkthrough and more examples.

Or check out the [Notebook Example](./examples/financial_news_rag_example.ipynb) for an interactive tutorial. 

---

## 📚 Documentation

- [Project Overview & Architecture](./docs/index.md)
- [Installation Guide](./docs/installation.md)
- [Usage Guide](./docs/usage_guide.md)
- [Configuration Reference](./docs/configuration.md)
- [API Reference](./docs/api_reference/index.md)
- [Testing Guide](./docs/testing.md)
- [Development Guide](./docs/development.md)

---

## Method Flowcharts

### Article Storage Pipeline

This flowchart illustrates the article storage pipeline.

```mermaid
flowchart LR
    eod[EODHDClient]
    rdb1[(Article<br>Manager)]
    rdb2[(Article<br>Manager)]
    rdb3[(Article<br>Manager)]
    txt[TextProcessor]
    vdb[(ChromaDB<br>Manager)]
    emb[Embeddings<br>Generator]

    subgraph Fetch
        direction TB
        eod-- store article --> rdb1
    end
    subgraph Process
        direction TB
        rdb2<-- store processed<br>status update --> txt
    end
    subgraph Embed
        direction TB
        rdb3<-- status update --> emb
    end
    Fetch --> Process --> Embed<-- link<br>databases --> vdb

    style eod fill:#c4b2ed,stroke:#8A6FDF,color:#333333
    style rdb1 fill:#87d6b6,stroke:#4D9A7F,color:#333333
    style rdb2 fill:#87d6b6,stroke:#4D9A7F,color:#333333
    style rdb3 fill:#87d6b6,stroke:#4D9A7F,color:#333333
    style txt fill:#B2DFDB,stroke:#80CBC4,color:#333333
    style vdb fill:#a8e2ed,stroke:#4A9BA8,color:#333333
    style emb fill:#f7e4a6,stroke:#FFC107,color:#333333
```

### Article Search

This flowchart shows the steps involved in searching for articles based on a user query.

```mermaid
flowchart LR
    usr[User Query]
    rdb[(Article<br>Manager)]
    vdb[(ChromaDB<br>Manager)]
    rer[ReRanker]
    emb[Embeddings<br>Generator]
    out[Articles]

    subgraph FinancialNewsRAG
        direction LR
        emb --- vdb --- rdb
        rdb --- rer
    end
    usr --> emb
    rdb --> out
    rer --> out

    style usr fill:#F5F5F5,stroke:#A9A9A9,color:#333333
    style rdb fill:#87d6b6,stroke:#4D9A7F,color:#333333
    style vdb fill:#a8e2ed,stroke:#4A9BA8,color:#333333
    style rer fill:#f0aaaa,stroke:#D32F2F,color:#333333
    style emb fill:#f7e4a6,stroke:#FFC107,color:#333333
    style out fill:#90CAF9,stroke:#42A5F5,color:#333333
```

---

**Financial News RAG: Built for speed, scale, and smarter financial decisions.**
