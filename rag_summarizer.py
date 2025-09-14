from retriever import RAGRetriever
from summarizer import summarizer  # Qwen summarizer pipeline
from utils import clean_text

def build_rag_prompt(query: str, retrieved_chunks: list, task: str = "Summarize") -> str:
    """
    Construct a prompt for LLM using retrieved chunks.
    """
    intro = (
        "You are a helpful assistant analyzing scientific papers. "
        "Based on the following retrieved context, answer the task below.\n\n"
    )

    context = "\n\n".join([f"Context {i+1}:\n{chunk}" for i, (chunk, _) in enumerate(retrieved_chunks)])

    task_map = {
        "Summarize": "Summarize the overall approach described.",
        "Extract Research Question": "What is the research question these papers are addressing?",
        "Extract Method": "What methods or techniques are being proposed?",
        "Extract Contribution": "What are the main contributions discussed?",
        "Structured Summary (All)": (
            "Write a structured summary answering:\n"
            "- Research Question\n"
            "- Method\n"
            "- Contribution"
        )
    }

    task_instruction = task_map.get(task, task_map["Summarize"])

    return f"{intro}{context}\n\nTask: {task_instruction}"

def rag_answer(query: str, all_chunks: list, top_k: int = 5, task: str = "Summarize") -> str:
    """
    Perform RAG-based generation using top-k retrieved chunks.
    """
    retriever = RAGRetriever()
    retriever.build_index(all_chunks)

    retrieved = retriever.search(query, top_k=top_k)

    rag_prompt = build_rag_prompt(query, retrieved, task)
    rag_prompt = clean_text(rag_prompt)  # optional cleaning

    result = summarizer(rag_prompt, return_full_text=False)
    return result[0]["generated_text"].strip()
