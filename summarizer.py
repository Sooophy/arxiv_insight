from transformers import pipeline
from utils import clean_text


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text: str, task: str = "Summarize") -> str:
    cleaned_text = clean_text(text)

    task_prompt_map = {
        "Summarize": "Summarize the following scientific abstract in 2-3 sentences, focusing on the method and contribution.",
        "Extract Research Question": "What is the main research question addressed in the following abstract?",
        "Extract Method": "What method or approach is proposed in the following abstract?",
        "Extract Contribution": "What is the main contribution of the following research?",
        "Structured Summary (All)": (
            "Please extract the following items from the abstract:\n"
            "1. Research Question\n"
            "2. Method\n"
            "3. Main Contribution\n\n"
            "Format each answer as a numbered bullet point.\n"
        )
    }

    length_settings = {
        "Summarize": (40, 120),
        "Extract Research Question": (20, 50),
        "Extract Method": (30, 80),
        "Extract Contribution": (30, 80),
        "Structured Summary (All)": (60, 180)
    }

    prompt = task_prompt_map.get(task, task_prompt_map["Summarize"])
    min_len, max_len = length_settings.get(task, (40, 120))

    input_text = prompt + "\n\n" + cleaned_text

    summary = summarizer(
        input_text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        early_stopping=True
    )

    return summary[0]["summary_text"]



if __name__ == "__main__":
    input_text = (
        "Large language models (LLMs) have shown significant promise across many NLP tasks. "
        "However, applying them to complex tasks such as scientific summarization remains challenging. "
        "In this paper, we propose a new approach that combines supervised fine-tuning with retrieval-based prompting. "
        "Our method improves factual consistency and enables better citation grounding."
    )

    print("Original:")
    print(input_text)
    print("\nSummary:")
    print(summarize_text(input_text))
