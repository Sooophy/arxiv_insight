# Overview

ArxivInsight is a retrieval-augmented scientific summarization tool that uses Large Language Models (LLMs) to extract insights from recent arXiv papers.

This project enables users to query, filter, and summarize academic abstracts — supporting both single-document summarization and multi-document synthesis via Retrieval-Augmented Generation (RAG). Powered by Qwen-3, semantic chunking, BGE embeddings, FAISS similarity search, and structured prompt engineering.

## Features

LLM-based summarization using Qwen-3
Summarize or extract research questions, methods, and contributions from abstracts using task-specific prompts.

RAG-enabled multi-document synthesis
Retrieve semantically relevant chunks from multiple papers to enhance context and output quality.

User interface with Streamlit
Customize category filters, keywords, number of papers, task type, and toggle between classic vs. RAG-based generation.

Plug-and-play architecture
Modular code for chunking, embedding, indexing, retrieval, and LLM prompting. Easily extensible.


## Models & Tools
Component	Technology
Embedding Model	BAAI/bge-small-en-v1.5
LLM	Qwen/Qwen3-1.7B-Base
Vector Search	faiss-cpu
Frontend	Streamlit
NLP Toolkit	sentence-transformers, transformers

## Example Use Cases

Summarize recent pose estimation methods from `cs.CV` arXiv papers

Extract structured insight from multiple abstracts across `cs.LG, cs.AI`

Use RAG to trace emerging trends by combining context across dozens of abstracts

## How to Run

1. Install dependencies
    ```
    pip install -r requirements.txt
    ```

2. Start the app

    ```
    streamlit run app.py
    ```

3. In your browser

    - Enter a keyword (e.g., "pose estimation")

    - Select category, paper amount, task type

    - Enable RAG mode if desired

    - Click Fetch and Summarize

## Sample Output (Structured Summary with RAG)
**Research Question**: How to jointly model individual and parallel 3D lines for better structural understanding?

**Method**: Introduces RiemanLine, a minimal representation on Riemannian manifolds combining vanishing point direction and orthogonal subspaces.

**Contribution**: Reduces parameter space for parallel lines, improves pose estimation accuracy, and integrates into a unified bundle adjustment framework.
