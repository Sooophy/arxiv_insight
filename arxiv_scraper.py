import feedparser
from datetime import datetime, timedelta

def get_recent_arxiv_papers(max_results=50, category="cs.CV", days_back=7):
    """
    Fetch recent arXiv papers from a specific category.

    Parameters:
        max_results (int): Max number of papers to fetch
        category (str): arXiv category, e.g. 'cs.CV' or 'cs.LG'
        days_back (int): How many days back to include

    Returns:
        List[dict]: List of papers with metadata
    """
    base_url = "http://export.arxiv.org/api/query?"
    search_query = (
        f"search_query=cat:{category}"
        f"&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    )

    feed = feedparser.parse(base_url + search_query)
    papers = []

    for entry in feed.entries:
        published_date = datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%SZ")
        if published_date < datetime.utcnow() - timedelta(days=days_back):
            continue

        papers.append({
            "title": entry.title.strip().replace("\n", " "),
            "summary": entry.summary.strip().replace("\n", " "),
            "authors": [author.name for author in entry.authors],
            "published": published_date.strftime("%Y-%m-%d"),
            "link": entry.link
        })

    return papers


def keyword_filter(papers, keyword, top_k=5):
    """
    Filters papers whose title or abstract contains the exact keyword phrase (case-insensitive).

    Parameters:
        papers (List[dict]): List of paper dicts
        keyword (str): Phrase to match in title or abstract
        top_k (int): Number of papers to keep

    Returns:
        List[dict]: Filtered list of papers
    """
    keyword = keyword.lower()
    filtered = [
        p for p in papers
        if keyword in p["title"].lower() or keyword in p["summary"].lower()
    ]
    return filtered[:top_k]

ARXIV_CATEGORIES = {
    "All Fields": None,
    "Computer Vision (cs.CV)": "cs.CV",
    "Machine Learning (cs.LG)": "cs.LG",
    "Artificial Intelligence (cs.AI)": "cs.AI",
    "Computation and Language (cs.CL)": "cs.CL",
    "Robotics (cs.RO)": "cs.RO",
    "Neural and Evolutionary Computing (cs.NE)": "cs.NE",
    "Statistical ML (stat.ML)": "stat.ML",
    "Mathematics of ML (math.OC)": "math.OC"
}

# Test run
if __name__ == "__main__":
    raw = get_recent_arxiv_papers(max_results=50, category="cs.CV")
    filtered = keyword_filter(raw, "pose estimation", top_k=5)
    for p in filtered:
        print(f"\n📰 {p['title']} ({p['published']})")
        print(f"👥 {', '.join(p['authors'])}")
        print(f"🔗 {p['link']}")
        print(f"📄 Abstract:\n{p['summary'][:300]}...")
