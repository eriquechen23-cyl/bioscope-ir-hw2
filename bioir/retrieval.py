"""Dynamic-programming Levenshtein distance and positional token retrieval."""
import html
from collections import Counter
from .porter import stem
from .text import document_text, token_spans, tokenize


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def distance_matrix(a: str, b: str):
    if max(len(a), len(b)) > 40:
        raise ValueError("教學矩陣限制為每字最多 40 字元。")
    matrix = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        matrix[i][0] = i
    for j in range(len(b) + 1):
        matrix[0][j] = j
    for i, x in enumerate(a, 1):
        for j, y in enumerate(b, 1):
            matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1]+(x != y))
    return matrix


def suggestions(word: str, vocabulary: Counter, max_distance: int = 2, limit: int = 5):
    candidates = [(term, edit_distance(word, term), count) for term, count in vocabulary.items()
                  if abs(len(word) - len(term)) <= max_distance and term != word]
    return sorted((c for c in candidates if c[1] <= max_distance), key=lambda c: (c[1], -c[2], c[0]))[:limit]


def search(records, query, mode="exact", max_distance=1, require_all=True, include_title=False):
    terms = list(dict.fromkeys(tokenize(query)))
    if not terms:
        return []
    if len(terms) > 10 or any(len(t) > 50 for t in terms):
        raise ValueError("請使用最多 10 個查詢詞，每個詞最多 50 字元。")
    if mode not in {"exact", "stem", "fuzzy"}:
        raise ValueError("Unknown search mode")
    docs = [(record, document_text(record, include_title)) for record in records]
    vocabulary = {w for _, text in docs for w in tokenize(text)}
    matches = {}
    for term in terms:
        if mode == "exact":
            matches[term] = {term}
        elif mode == "stem":
            matches[term] = {w for w in vocabulary if stem(w) == stem(term)}
        else:
            matches[term] = {w for w in vocabulary if abs(len(w) - len(term)) <= max_distance and edit_distance(w, term) <= max_distance}
    results = []
    for record, text in docs:
        spans, matched_terms = [], set()
        for token_index, (word, start, end) in enumerate(token_spans(text)):
            matched = [term for term in terms if word in matches[term]]
            if matched:
                matched_terms.update(matched)
                spans.append({"word": word, "start": start, "end": end, "token_index": token_index, "query_terms": matched})
        if matched_terms and (not require_all or len(matched_terms) == len(terms)):
            results.append({"article": record, "text": text, "spans": spans, "matched_terms": len(matched_terms), "hits": len(spans)})
    return sorted(results, key=lambda r: (-r["matched_terms"], -r["hits"], r["article"]["pmid"]))


def highlight(text: str, spans: list[dict]) -> str:
    output, position = [], 0
    for span in sorted(spans, key=lambda s: s["start"]):
        start, end = span["start"], span["end"]
        if start < position:
            continue
        output.extend((html.escape(text[position:start]), "<mark>", html.escape(text[start:end]), "</mark>"))
        position = end
    output.append(html.escape(text[position:]))
    return "".join(output).replace("\n", "<br>")
