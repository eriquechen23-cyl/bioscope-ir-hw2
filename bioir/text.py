"""One tokenizer shared by distribution, embeddings and retrieval."""
import re
from .porter import stem

# A short, explicit and versioned teaching stoplist, not a downloaded corpus.
STOPWORDS = frozenset("a an and are as at be been being but by can could did do does for from had has have he her hers him his how i if in into is it its may more most no not of on or our ours out s she should so some such t than that the their them then there these they this those through to under up was we were what when where which while who will with would you your".split())
TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:[-‐‑–][A-Za-z0-9]+)*|\d+(?:\.\d+)?", re.ASCII)


def normalize(value: str) -> str:
    return value.lower().translate(str.maketrans({"‐": "-", "‑": "-", "–": "-"}))


def token_spans(text: str):
    return [(normalize(m.group()), m.start(), m.end()) for m in TOKEN_PATTERN.finditer(text)]


def tokenize(text: str, remove_stopwords: bool = False, stemming: bool = False):
    words = [w for w, _, _ in token_spans(text) if not w[0].isdigit()]
    if remove_stopwords:
        words = [w for w in words if w not in STOPWORDS]
    return [stem(w) for w in words] if stemming else words


def document_text(record: dict, include_title: bool = False) -> str:
    return (record["title"] + "\n\n" if include_title else "") + record["abstract"]


def sentences(records, remove_stopwords=False, stemming=False, include_title=False):
    result = []
    for record in records:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", document_text(record, include_title)):
            tokens = tokenize(sentence, remove_stopwords, stemming)
            if tokens:
                result.append(tokens)
    return result
