#This code is a text anlysis of 2017-2018 James Harden Wikipedia Article and also the 2018 Western Conference Finals.

#Counter is recommended by chatgpt for word counting
from collections import Counter
from math import sqrt
import json, os

#This explains which WCF it is and the JSONL and the Season
JSONL = "harden_2018_season_vs_playoff.jsonl"
SEASON = "2017-18 Houston Rockets season"
WCF    = "2018 Western Conference Finals"

# This is a stop list provided by chatgpt to remove common words from analysis
STOP = {
    "the","and","a","an","of","for","to","in","on","at","as","by","from","with","it",
    "its","is","are","was","were","be","been","being","that","this","these","those",
    "or","not","no","but","so","if","into","than","then","their","his","her","they",
    "them","he","she","we","you","your","our","i","over","after","before","during",
    "within","without","between","about","also","such","there","here","up","down"
}

def load_jsonl(path=JSONL):
    """Load a JSONL file and return a dictionary mapping title → content.

    Reads UTF-8 to support special characters in Wikipedia text.
    Each line is parsed as a JSON object containing "title" and "content".
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
    """Tokenize text into cleaned, lowercase words.

    Steps:
        - Split on whitespace
        - Remove punctuation characters
        - Convert to lowercase
        - Remove stopwords and single-character tokens
    Returns:
        A list of valid tokens.
    """
    out = []
    for w in text.split():
        w = w.strip(".,!?;:\"()[]{}<>''``“”’").lower()
        if w and w not in STOP and len(w) > 1:
            out.append(w)
    return out

def bow(text):
    """Return a bag-of-words Counter for the given text using tokenized words."""
    return Counter(tokens(text))

def top_n(c, n=15):
    """Return the top n most common words from a Counter."""
    return c.most_common(n)

def unique_freq(a, b, thr=5):
    """Find words frequent in document A but not in B.

    Args:
        a, b: Counter objects
        thr: minimum count threshold

    Returns:
        List of (word, count) pairs sorted by frequency.
    """
    items = [(w, ca) for w, ca in a.items() if ca >= thr and b.get(w, 0) < thr]
    items.sort(key=lambda x: x[1], reverse=True)
    return items

def cosine(a, b):
    """Compute cosine similarity between two bag-of-words Counters.

    Formula:
        dot(a, b) / (||a|| * ||b||)

    Notes:
        - dot product sums shared word contributions
        - learned from online/AI examples
    """
    if not a or not b: return 0.0
    dot = sum(a[w]*b.get(w,0) for w in a)
    na  = sqrt(sum(v*v for v in a.values()))
    nb  = sqrt(sum(v*v for v in b.values()))
    return 0.0 if na==0 or nb==0 else dot/(na*nb)

def proper_names(text, k=15):
    """Return the top-k capitalized words interpreted as proper names.

    A word qualifies if:
        - It begins with an uppercase letter (Python's istitle())
        - Has length ≥ 3
    """
    c = Counter()
    for w in text.split():
        w = w.strip(".,!?;:\"()[]{}<>''``“”’")
        if w.istitle() and len(w) >= 3:
            c[w] += 1
    return c.most_common(k)

def bar(x, scale=20):
    """Return a simple bar made of unicode blocks scaled to x (0–1 range)."""
    return "█" * int(round(x*scale))

def analyze(docs):
    """Run the full textual analysis workflow.

    Steps:
        1. Build bag-of-words for all documents.
        2. Print top frequent words per document.
        3. Print words frequent in Season but not WCF (and vice versa).
        4. Compute cosine similarity for key document pairs.
        5. Extract and print top proper names.
    """
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

def main():
    """Load JSONL file and run the Harden season vs WCF text analysis.

    Ensures:
        - JSONL exists
        - At least one document is successfully loaded
    Then calls analyze().
    """
    if not os.path.exists(JSONL):
        print("[error] JSONL not found. Run Part 1 first.")
        return
    docs = load_jsonl(JSONL)
    if not docs:
        print("[error] Loaded 0 documents.")
        return
    analyze(docs)

if __name__ == "__main__":
    main()
