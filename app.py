# app.py

import streamlit as st
from arxiv_scraper import get_recent_arxiv_papers, keyword_filter, ARXIV_CATEGORIES

from summarizer import summarize_text

# --- Page config ---
st.set_page_config(page_title="ArxivInsight", layout="wide")
st.title("ArxivInsight: LLM-Powered Paper Summarizer")
st.markdown("Enter a keyword and select a category to fetch and summarize recent arXiv papers.")

# --- Inputs ---
query = st.text_input("Keyword (exact phrase match)", value="pose estimation")

category_label = st.selectbox("Select arXiv Category", options=list(ARXIV_CATEGORIES.keys()), index=0)
category_code = ARXIV_CATEGORIES[category_label]

max_fetch = st.slider("Number of papers to fetch per category", min_value=10, max_value=200, value=100, step=10)
max_display = st.slider("Max number of matching papers to display", min_value=1, max_value=20, value=5)

task_type = st.selectbox(
    "Choose LLM Task",
    options=[
        "Summarize",
        "Extract Research Question",
        "Extract Method",
        "Extract Contribution",
        "Structured Summary (All)"
    ]
)

# --- Helper for multiple categories ---
MULTI_CATEGORY_DEFAULTS = ["cs.CV", "cs.LG", "cs.CL", "cs.AI", "stat.ML"]

def get_combined_category_papers(categories, max_per_cat=100):
    combined = []
    for cat in categories:
        papers = get_recent_arxiv_papers(max_results=max_per_cat, category=cat)
        combined.extend(papers)
    return combined

# --- Run ---
if st.button("Fetch and Summarize"):
    with st.spinner("Searching arXiv and summarizing..."):
        if category_code is None:
            # All Fields: pull from multiple sub-categories
            all_papers = get_combined_category_papers(MULTI_CATEGORY_DEFAULTS, max_per_cat=max_fetch)
        else:
            # Specific category
            all_papers = get_recent_arxiv_papers(max_results=max_fetch, category=category_code)

        matched_papers = keyword_filter(all_papers, query, top_k=max_display)
        total_matches = len(matched_papers)

        if total_matches == 0:
            st.warning("No matching papers found for that keyword.")
        else:
            st.success(f"Found {total_matches} matching paper(s).")
            for paper in matched_papers:
                st.markdown("---")
                st.subheader(paper["title"])
                st.markdown(f"**Authors:** {', '.join(paper['authors'])}")
                st.markdown(f"**Published:** {paper['published']} | [Link]({paper['link']})")
                st.markdown("**Abstract:**")
                st.write(paper["summary"])

                with st.expander("LLM Summary", expanded=True):
                    summary = summarize_text(paper["summary"], task=task_type)
                    st.success(summary)
