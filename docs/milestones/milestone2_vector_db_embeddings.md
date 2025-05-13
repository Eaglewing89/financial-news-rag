# Milestone 2: Vector Database and Embeddings

## Goal: Integrate the vector database, generate embeddings for news articles, and store them for semantic search.

## Issues:

- **Issue #5: Integrate ChromaDB**

  - **Description**: Set up and configure ChromaDB as the vector database for storing news article embeddings.
  - **Labels**: `enhancement`, `database`, `vector-db`
  - **References**: Core Objective 4, Tech Stack

- **Issue #6: Implement Embedding Generation**

  - **Description**: Select a suitable sentence transformer model (e.g., from Hugging Face) and implement the logic to generate embeddings for the processed news articles.
  - **Labels**: `enhancement`, `machine-learning`, `nlp`, `embeddings`
  - **References**: Core Objective 3, Tech Stack

- **Issue #7: Store Articles and Embeddings in ChromaDB**
  - **Description**: Develop scripts to take processed articles and their generated embeddings and store them effectively in ChromaDB. This includes handling metadata and ensuring data integrity.
  - **Labels**: `enhancement`, `database`, `data-pipeline`
  - **References**: Core Objective 4
