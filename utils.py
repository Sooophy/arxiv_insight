import re

def clean_text(text: str) -> str:
    """
    Clean up LaTeX commands and formatting artifacts from arXiv abstracts.
    """
    text = re.sub(r"\\\\?text[a-z]+\{(.*?)\}", r"\\1", text)  # common LaTeX text formatting
    text = re.sub(r"\${1,2}.*?\${1,2}", "", text)
    text = re.sub(r"\\\\?[a-zA-Z]+\{.*?\}", "", text)
    text = text.replace("\\", "")
    text = re.sub(r"\s+", " ", text)

    return text.strip()
