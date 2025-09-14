from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from utils import clean_text


MODEL_ID = "Qwen/Qwen3-1.7B-Base"
# MODEL_ID = "Qwen/Qwen3-4B-Base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", trust_remote_code=True)


summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def summarize_text(text: str, task: str = "Summarize") -> str:
    cleaned_text = clean_text(text)

    task_prompt_map = {
        "Summarize": "Summarize the following scientific abstract in 2-3 sentences, focusing on the method and contribution.",
        "Extract Research Question": "What is the main research question addressed in the following abstract?",
        "Extract Method": "What method or approach is proposed in the following abstract?",
        "Extract Contribution": "What is the main contribution of the following research?",
        "Structured Summary (All)": (
            "Please read the abstract and provide the following:\n"
            "- A short description of the main research question.\n"
            "- A brief explanation of the method used.\n"
            "- A summary of the main contribution.\n"
            "Format each answer as a numbered bullet point.\n"        
        )
    }

    prompt = task_prompt_map.get(task, task_prompt_map["Summarize"])

    input_text = prompt + "\n\n" + cleaned_text

    summary = summarizer(
        input_text,
        return_full_text=False
    )

    return summary[0]["generated_text"].strip()



if __name__ == "__main__":
    input_text = (
        "Large language models are powerful tools. "
        "They have shown impressive performance on various NLP tasks. "
        "However, their ability to reason across multiple documents is limited. "
        "This paper proposes a retrieval-augmented method for multi-document question answering."
    )

    print("Original:")
    print(input_text)
    print("\nSummary:")
    print(summarize_text(input_text))
