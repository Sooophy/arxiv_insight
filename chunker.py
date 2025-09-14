# chunker.py

from typing import List
import re

def simple_sentence_split(text: str) -> List[str]:
    """
    Very basic sentence splitter using punctuation.
    """
    text = re.sub(r"\s+", " ", text)
    return re.split(r'(?<=[.!?]) +', text)

def chunk_text(text: str, chunk_size: int = 2) -> List[str]:
    """
    Splits text into overlapping chunks of N sentences.

    Args:
        text (str): The abstract or full text to split.
        chunk_size (int): Number of sentences per chunk.

    Returns:
        List[str]: List of chunks (as strings)
    """
    sentences = simple_sentence_split(text)
    chunks = []

    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i + chunk_size]).strip()
        if len(chunk) > 0:
            chunks.append(chunk)

    return chunks

# Test
if __name__ == "__main__":
    abstract = (
        "Large language models are powerful tools. "
        "They have shown impressive performance on various NLP tasks. "
        "However, their ability to reason across multiple documents is limited. "
        "This paper proposes a retrieval-augmented method for multi-document question answering."
    )

    print("Chunks:")
    for c in chunk_text(abstract, chunk_size=2):
        print("-", c)
