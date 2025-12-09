#The following code is assited by Chatgpt to create a text analysis of James Harden's 2018 season and playoff performance
#Prompt 

########
#  I’m doing a text-analysis assignment and I need code that matches the instructions. Please generate two Python 
# files. First, download_wiki.py that uses the mediawiki Python package to download exactly these Wikipedia pages: “James Harden”,
# “2017–18 Houston Rockets season”, and “2018 Western Conference Finals”; clean the text by removing bracketed citations like [1] 
# and extra newlines; then save them to a JSONL file called harden_2018_season_vs_playoff.jsonl, one JSON object per line with "title"
#  and "content". Second, text_analysis.py that loads that JSONL, tokenizes with a custom stopword list, builds bag-of-words per document,
#  prints top 15 words per document, finds words frequent in the season page but not the WCF page (and vice versa), computes cosine similarity
#  between the three page pairs, showing a text bar for the score, and extracts top 15 proper names (capitalized words) per document. Both files 
# must end with if __name__ == "__main__": main() so they can be run directly. Don’t use pandas or numpy. Add brief comments noting that some functions 
# (like cosine and JSONL saving) were learned from AI/online sources.



import os
import json
import re
from collections import Counter
from math import sqrt
from typing import Dict, List

from mediawiki import MediaWiki     # type: ignore
import matplotlib.pyplot as plt     # type: ignore


JSONL_NAME = "harden_2018_season_vs_playoff.jsonl"
TITLES = [
    "James Harden",
    "2017–18 Houston Rockets season",
    "2018 Western Conference Finals",
]
SEASON = "2017–18 Houston Rockets season"
WCF    = "2018 Western Conference Finals"

STOP = {
    "the","and","a","an","of","for","to","in","on","at","as","by","from","with","it",
    "its","is","are","was","were","be","been","being","that","this","these","those",
    "or","not","no","but","so","if","into","than","then","their","his","her","they",
    "them","he","she","we","you","your","our","i","over","after","before","during",
    "within","without","between","about","also","such","there","here","up","down"
}

_CIT = re.compile(r"\[.*?\]")
_NL  = re.compile(r"\n+")

#Follwing gets the current directory of this file

HERE = os.path.dirname(os.path.abspath(__file__))

def find_jsonl() -> str:
    """Return the path to the JSONL file, checking current and parent directory.

    Checks two candidate paths (local folder and parent folder). If neither exists,
    returns the default expected location in the current folder.
    """
    candidates = [
        os.path.join(HERE, JSONL_NAME),
        os.path.join(HERE, "..", JSONL_NAME),
    ]
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.exists(p):
            return p
    # default to current folder if not found
    return os.path.join(HERE, JSONL_NAME)

# Following function cleans the text by removing citations and replacing newlines with spaces.

def clean_text(text: str) -> str:
    """Remove bracketed citations like [1] and collapse repeated newlines into spaces.

    Returns cleaned text. Empty input returns an empty string.
    """
    if not text:
        return ""
    text = _CIT.sub("", text)
    text = _NL.sub(" ", text)
    return text.strip()

def fetch_page(title: str):
    """Fetch a Wikipedia page object using the MediaWiki package."""
    wiki = MediaWiki()
    return wiki.page(title)

def harvest_titles(titles: List[str], preview_chars: int = 600) -> Dict[str, str]:
    """
    Download Wikipedia pages, print short previews, clean text, and return a dict.
     Args:
        titles: list of Wikipedia page titles to fetch.
        preview_chars: number of characters to preview in console.
    Returns:
        Dict mapping page title → cleaned text.
    """
    docs: Dict[str, str] = {}
    for t in titles:
        page = fetch_page(t)
        print(page.title)
        print(page.content[:preview_chars], "...\n")
        docs[page.title] = clean_text(page.content)
    return docs

def save_jsonl(docs: Dict[str, str], path: str):
    """
    Save documents to a JSONL file, one JSON object per line.

    Note:
        JSONL-saving logic was learned from AI/online tutorials.
    """
    with open(path, "w", encoding="utf-8") as f:
        for title, content in docs.items():
            f.write(json.dumps({"title": title, "content": content}, ensure_ascii=False) + "\n")

def ensure_jsonl(path: str):
    """
    Ensure the JSONL exists; if not, download pages and write them to disk.
    """

    if os.path.exists(path):
        print(f"[info] using existing JSONL at {path}")
        return
    print("[info] JSONL not found, downloading from Wikipedia with mediawiki...")
    docs = harvest_titles(TITLES, preview_chars=800)
    save_jsonl(docs, path)
    print(f"[info] saved {path} with {len(docs)} docs.")

#Following function loads JSON lines from a file and returns a dictionary of titles and their corresponding content

def load_jsonl(path: str):
     
    """
    Load a JSONL file and return a dict of title → content strings.
    """

    docs = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            title = obj.get("title", "").strip()
            if title:
                docs[title] = obj.get("content", "")
    return docs

def tokens(text):

    """
    Tokenize text into lowercase words, removing punctuation and custom stopwords.
    """
    out = []
    for w in text.split():
        w = w.strip(".,!?;:\"()[]{}<>''``“”’").lower()
        if w and w not in STOP and len(w) > 1:
            out.append(w)
    return out

def bow(text):
    """
    Build a bag-of-words Counter from text tokens.
    """
    return Counter(tokens(text))

def top_n(c, n=15):
    """
    Return the top-n most common elements from a Counter.
    """
    return c.most_common(n)

def unique_freq(a, b, thr=5):
    """
    Return items frequent in dict a but not in b, based on a threshold.
    """
    items = [(w, ca) for w, ca in a.items() if ca >= thr and b.get(w, 0) < thr]
    items.sort(key=lambda x: x[1], reverse=True)
    return items

def cosine(a, b):
    """
    Compute cosine similarity between two bag-of-words Counters.
    """
    if not a or not b:
        return 0.0
    dot = sum(a[w]*b.get(w,0) for w in a)
    na  = sqrt(sum(v*v for v in a.values()))
    nb  = sqrt(sum(v*v for v in b.values()))
    return 0.0 if na==0 or nb==0 else dot/(na*nb)

def proper_names(text, k=15):
    """
    Extract top-k capitalized words (length ≥ 3) interpreted as proper names.
    """
    c = Counter()
    for w in text.split():
        w = w.strip(".,!?;:\"()[]{}<>''``“”’")
        if w.istitle() and len(w) >= 3:
            c[w] += 1
    return c.most_common(k)

def bar(x, scale=20):
    """
    Return a simple unicode bar proportional to similarity score.
    """
    return "█" * int(round(x * scale))

# Following function ensures that the figures directory exists

def ensure_fig_dir():
    """
    Ensure that a ./figures directory exists; create it if needed and return path.
    """
    figdir = os.path.join(HERE, "figures")
    if not os.path.exists(figdir):
        os.makedirs(figdir)
    return figdir

def slugify(s: str) -> str:
    """
    Convert a title string into a filesystem-safe slug for filenames.
    """
    return (
        s.lower()
         .replace(" ", "_")
         .replace("–", "-")
         .replace("—", "-")
         .replace("/", "_")
         .replace("(", "")
         .replace(")", "")
    )

def shorten(s: str, maxlen=30) -> str:
    """
    Shorten string for display if longer than maxlen.
    """
    return s if len(s) <= maxlen else s[:maxlen-3] + "..."

def plot_top_words_per_doc(bows, top=15):
    """
    Generate horizontal bar charts of top words per document and save to ./figures.
    """
    figdir = ensure_fig_dir()
    for title, counter in bows.items():
        common = counter.most_common(top)
        if not common:
            continue
        words  = [w for w, _ in common][::-1]
        counts = [c for _, c in common][::-1]
        plt.figure(figsize=(8,5))
        plt.barh(words, counts)
        plt.title(f"Top {top} words: {title}")
        plt.xlabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, f"topwords_{slugify(title)}.png"), dpi=150)
        plt.close()

def plot_season_vs_wcf(bows, thr=5, top=20):
    """
    Plot words frequent in the SEASON doc but not WCF (and vice versa).
    """
    if SEASON not in bows or WCF not in bows:
        return
    figdir = ensure_fig_dir()
    season = bows[SEASON]
    wcf    = bows[WCF]

    season_unique = [(w, c) for w, c in season.items() if c >= thr and wcf.get(w, 0) < thr]
    wcf_unique    = [(w, c) for w, c in wcf.items() if c >= thr and season.get(w, 0) < thr]

    season_unique.sort(key=lambda x: x[1], reverse=True)
    wcf_unique.sort(key=lambda x: x[1], reverse=True)

    season_unique = season_unique[:top]
    wcf_unique    = wcf_unique[:top]

    if season_unique:
        words  = [w for w, _ in season_unique][::-1]
        counts = [c for _, c in season_unique][::-1]
        plt.figure(figsize=(8,5))
        plt.barh(words, counts)
        plt.title("Words common in SEASON but not WCF")
        plt.xlabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, "season_unique.png"), dpi=150)
        plt.close()

    if wcf_unique:
        words  = [w for w, _ in wcf_unique][::-1]
        counts = [c for _, c in wcf_unique][::-1]
        plt.figure(figsize=(8,5))
        plt.barh(words, counts)
        plt.title("Words common in WCF but not SEASON")
        plt.xlabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, "wcf_unique.png"), dpi=150)
        plt.close()

def plot_cosine_heatmap(bows):
    """
    Plot a cosine similarity heatmap comparing all documents.
    """
    figdir = ensure_fig_dir()
    titles = list(bows.keys())
    n = len(titles)
    if n == 0:
        return
    mat = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            mat[i][j] = cosine(bows[titles[i]], bows[titles[j]])
    plt.figure(figsize=(5 + n*0.4, 5 + n*0.4))
    plt.imshow(mat, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="Cosine similarity")
    plt.xticks(range(n), [shorten(t) for t in titles], rotation=45, ha="right")
    plt.yticks(range(n), [shorten(t) for t in titles])
    plt.title("Document similarity (cosine)")
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "cosine_heatmap.png"), dpi=150)
    plt.close()

def plot_proper_names_all(docs):
    """
    Plot bar charts of the top proper names for each document.
    """
    figdir = ensure_fig_dir()
    for title, text in docs.items():
        names = proper_names(text, 15)
        if not names:
            continue
        labels = [n for n, _ in names][::-1]
        counts = [c for _, c in names][::-1]
        plt.figure(figsize=(8,5))
        plt.barh(labels, counts)
        plt.title(f"Top names in {title}")
        plt.xlabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, f"names_{slugify(title)}.png"), dpi=150)
        plt.close()

#Following is the main function that runs the entire analysis

def main():
    """
    Plot bar charts of the top proper names for each document.
    """
    # 1) find / make JSONL
    jsonl_path = find_jsonl()
    ensure_jsonl(jsonl_path)

    # 2) load docs
    docs = load_jsonl(jsonl_path)
    if not docs:
        print("[error] Loaded 0 documents.")
        return

    # 3) text analysis 
    bows = {t: bow(txt) for t, txt in docs.items()}

    print("\n=== Top Words per Document ===")
    for t, c in bows.items():
        print(f"\n{t}")
        for w, n in top_n(c, 15):
            print(f"  {w:<18} {n}")

    if SEASON in bows and WCF in bows:
        print("\n=== Frequent in SEASON but not WCF (thr=5) ===")
        for w, n in unique_freq(bows[SEASON], bows[WCF], 5)[:20]:
            print(f"  {w:<18} {n}")

        print("\n=== Frequent in WCF but not SEASON (thr=5) ===")
        for w, n in unique_freq(bows[WCF], bows[SEASON], 5)[:20]:
            print(f"  {w:<18} {n}")

    print("\n=== Cosine Similarity ===")
    pairs = [("James Harden", SEASON), ("James Harden", WCF), (SEASON, WCF)]
    for a, b in pairs:
        if a in bows and b in bows:
            s = cosine(bows[a], bows[b])
            print(f"  {a} ↔ {b}\n    {s:.3f} {bar(s)}")

    print("\n=== Proper Names (top 15) ===")
    for t, txt in docs.items():
        print(f"\n{t}")
        for name, n in proper_names(txt, 15):
            print(f"  {name:<18} {n}")

    # 4) charts
    print("\n[info] generating charts in ./figures ...")
    plot_top_words_per_doc(bows)
    plot_season_vs_wcf(bows)
    plot_cosine_heatmap(bows)
    plot_proper_names_all(docs)
    print("[info] done. check the 'figures' folder.")

if __name__ == "__main__":
    main()
