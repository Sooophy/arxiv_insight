import streamlit as st
from arxiv_scraper import get_recent_arxiv_papers, keyword_filter, ARXIV_CATEGORIES
from summarizer import summarize_text
from rag_summarizer import rag_answer
from chunker import chunk_text

st.set_page_config(page_title="ArxivInsight", layout="wide")
st.title("ArxivInsight: LLM-Powered Paper Summarizer")

# --- UI Inputs ---
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

use_rag = st.checkbox("Use Retrieval-Augmented Generation (RAG)", value=False)

# --- Main Logic ---
if st.button("Fetch and Summarize"):
    with st.spinner("Fetching papers..."):
        if category_code is None:
            from retriever import MULTI_CATEGORY_DEFAULTS  # list of fallback categories
            all_papers = []
            for cat in MULTI_CATEGORY_DEFAULTS:
                all_papers += get_recent_arxiv_papers(max_results=max_fetch, category=cat)
        else:
            all_papers = get_recent_arxiv_papers(max_results=max_fetch, category=category_code)

        matched_papers = keyword_filter(all_papers, query, top_k=max_display)

        if not matched_papers:
            st.warning("No matching papers found.")
        else:
            st.success(f"Found {len(matched_papers)} matching paper(s).")

            # Prepare all chunks for RAG
            all_chunks = []
            for paper in matched_papers:
                chunks = chunk_text(paper["summary"], chunk_size=2)
                all_chunks.extend(chunks)

            for paper in matched_papers:
                st.markdown("---")
                st.subheader(paper["title"])
                st.markdown(f"**Authors:** {', '.join(paper['authors'])}")
                st.markdown(f"**Published:** {paper['published']} | [Link]({paper['link']})")
                st.markdown("**Abstract:**")
                st.write(paper["summary"])

                with st.expander("LLM Output", expanded=True):
                    if use_rag:
                        st.markdown("*Using RAG over multiple papers...*")
                        response = rag_answer(query, all_chunks, top_k=5, task=task_type)
                    else:
                        response = summarize_text(paper["summary"], task=task_type)
                    st.success(response)