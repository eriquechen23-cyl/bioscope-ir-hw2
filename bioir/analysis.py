from collections import Counter
import hashlib
import numpy as np
import pandas as pd
from .porter import stem
from .text import document_text, tokenize, sentences


def frequencies(records, remove_stopwords=False, include_title=False):
    raw = Counter(w for r in records for w in tokenize(document_text(r, include_title), remove_stopwords))
    stemmed = Counter()
    for word, count in raw.items():
        stemmed[stem(word)] += count
    return raw, stemmed


def frequency_table(counts):
    return pd.DataFrame([{"rank": i, "term": term, "frequency": count} for i, (term, count)
                         in enumerate(sorted(counts.items(), key=lambda x: (-x[1], x[0])), 1)], columns=["rank", "term", "frequency"])


def zipf_fit(table):
    if len(table) < 2:
        return None
    x, y = np.log10(table["rank"]), np.log10(table["frequency"])
    slope, intercept = np.polyfit(x, y, 1)
    residual = float(np.sum((y - (slope * x + intercept)) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return {"slope": float(slope), "intercept": float(intercept), "r_squared": 1 - residual / total if total else 0.0}


def stable_hash(word):
    return int.from_bytes(hashlib.sha256(word.encode("utf-8")).digest()[:4], "little")


def train_embeddings(records, architecture="Skip-gram", vector_size=100, window=5,
                     min_count=3, epochs=10, remove_stopwords=False, stemming=False, include_title=False,
                     seed=42, compute_loss=False, callbacks=()):
    from gensim.models import Word2Vec
    if architecture not in {"CBOW", "Skip-gram"}:
        raise ValueError("模型必須為 CBOW 或 Skip-gram。")
    if min(vector_size, window, min_count, epochs) < 1:
        raise ValueError("維度、window、min_count 與 epochs 必須為正整數。")
    corpus = sentences(records, remove_stopwords, stemming, include_title)
    counts = Counter(word for sentence in corpus for word in sentence)
    if sum(v >= min_count for v in counts.values()) < 2:
        raise ValueError("可訓練詞彙不足，請降低 min_count 或增加文章數。")
    return Word2Vec(sentences=corpus, sg=int(architecture == "Skip-gram"), vector_size=vector_size,
                    window=window, min_count=min_count, epochs=epochs, workers=1, seed=seed,
                    hashfxn=stable_hash, negative=5, compute_loss=compute_loss, callbacks=callbacks)


def project_vectors(model, words):
    vectors = np.array([model.wv[word] for word in words])
    centered = vectors - vectors.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ vt[:2].T
    return pd.DataFrame({"word": words, "PC1": coordinates[:, 0], "PC2": coordinates[:, 1]})
