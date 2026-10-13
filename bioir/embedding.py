"""Reproducible Part 2 Word2Vec experiments shared by reports and Streamlit."""
from collections import Counter
import hashlib
import json
from time import perf_counter

import numpy as np
import pandas as pd

from .text import sentences

QUERY_WORDS = ["glp-1", "semaglutide", "liraglutide", "insulin", "glucose", "diabetes", "obesity", "weight", "receptor", "agonist"]
RELATED_PAIRS = [("semaglutide", "liraglutide"), ("insulin", "glucose"), ("diabetes", "obesity"), ("obesity", "weight"), ("receptor", "agonist")]
RARE_WORDS = ["tumor", "diagnosis", "neuropathy", "retinopathy", "gastroparesis", "nephropathy", "cardiomyopathy"]


def corpus_fingerprint(records):
    """Fingerprint the ordered abstract text used by the fixed experiment."""
    payload = json.dumps([r["abstract"] for r in records], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def training_samples(tokens, window):
    """Illustrate a full fixed window without min_count, subsampling or shrinking."""
    if window < 1:
        raise ValueError("window 必須至少為 1。")
    cbow, skipgram = [], []
    for i, target in enumerate(tokens):
        context = [tokens[j] for j in range(max(0, i-window), min(len(tokens), i+window+1)) if j != i]
        if context:
            cbow.append({"context": context, "target": target})
            skipgram.extend({"input": target, "target": word} for word in context)
    return cbow, skipgram


def vocabulary_statistics(records, min_count=3):
    corpus = sentences(records)
    counts = Counter(word for sentence in corpus for word in sentence)
    return {"documents": len(records), "sentences": len(corpus), "tokens": sum(counts.values()),
            "vocabulary": len(counts), "retained_vocabulary": sum(n >= min_count for n in counts.values()),
            "counts": counts}


def train_run(records, architecture, window, epochs=20, seed=42, vector_size=100, min_count=3):
    from gensim.models.callbacks import CallbackAny2Vec
    from .analysis import train_embeddings

    class LossHistory(CallbackAny2Vec):
        def __init__(self):
            self.rows = []
            self.previous = 0.0

        def on_epoch_end(self, model):
            current = float(model.get_latest_training_loss())
            self.rows.append({"epoch": len(self.rows)+1, "cumulative_loss": current,
                              "epoch_loss": current-self.previous})
            self.previous = current

    history = LossHistory()
    start = perf_counter()
    model = train_embeddings(records, architecture, vector_size, window, min_count, epochs,
                             seed=seed, compute_loss=True, callbacks=[history])
    elapsed = perf_counter()-start
    corpus = [[w for w in sentence if w in model.wv] for sentence in sentences(records)]
    cbow_examples = sum(len(sentence) for sentence in corpus if len(sentence) > 1)
    skipgram_pairs = sum(min(i, window)+min(len(sentence)-i-1, window)
                         for sentence in corpus for i in range(len(sentence)))
    meta = {"architecture": architecture, "window": window, "vector_size": vector_size,
            "min_count": min_count, "epochs": epochs, "seed": seed, "workers": 1,
            "negative": 5, "sample": 0.001, "shrink_windows": True, "cbow_mean": 1,
            "alpha": 0.025, "min_alpha": 0.0001, "hs": 0, "ns_exponent": 0.75,
            "remove_stopwords": False, "stemming": False, "include_title": False,
            "vocabulary": len(model.wv), "elapsed_seconds": elapsed, "loss": history.rows,
            "nominal_cbow_examples": cbow_examples, "nominal_skipgram_pairs": skipgram_pairs}
    return model, meta


def neighbor_table(model, terms, topn=5):
    rows = []
    for word in dict.fromkeys(terms):
        if word not in model.wv:
            rows.append({"word": word, "neighbor": "", "rank": 0, "cosine": np.nan,
                         "frequency": 0, "neighbor_frequency": 0, "status": "OOV"})
            continue
        for rank, (neighbor, similarity) in enumerate(model.wv.most_similar(word, topn=min(topn, len(model.wv)-1)), 1):
            rows.append({"word": word, "neighbor": neighbor, "rank": rank, "cosine": float(similarity),
                         "frequency": model.wv.get_vecattr(word, "count"),
                         "neighbor_frequency": model.wv.get_vecattr(neighbor, "count"), "status": "ok"})
    return pd.DataFrame(rows, columns=["word", "neighbor", "rank", "cosine", "frequency", "neighbor_frequency", "status"])


def pair_table(model, pairs):
    rows = []
    for left, right in pairs:
        present = left in model.wv and right in model.wv
        rows.append({"left": left, "right": right, "cosine": float(model.wv.similarity(left, right)) if present else np.nan,
                     "left_frequency": model.wv.get_vecattr(left, "count") if left in model.wv else 0,
                     "right_frequency": model.wv.get_vecattr(right, "count") if right in model.wv else 0,
                     "status": "ok" if present else "OOV"})
    return pd.DataFrame(rows)


def project_embeddings(model, words, method="PCA", seed=42):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE, trustworthiness
    selected = list(dict.fromkeys(words))
    if method not in {"PCA", "t-SNE"}:
        raise ValueError("降維方法必須為 PCA 或 t-SNE。")
    if len(selected) < (3 if method == "t-SNE" else 2) or any(w not in model.wv for w in selected):
        raise ValueError("PCA 至少需要 2 詞，t-SNE 至少 3 詞，且全部詞須存在於模型。")
    vectors = np.asarray([model.wv[w] for w in selected], dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(vectors).all() or vectors.shape[1] < 2:
        raise ValueError("向量必須有限、非零，且至少 2 維。")
    normalized = vectors / norms
    meta = {"method": method, "seed": seed, "words": len(selected), "l2_normalized": True}
    if method == "PCA":
        estimator = PCA(n_components=2, svd_solver="full")
        coordinates = estimator.fit_transform(normalized)
        meta["explained_variance_ratio"] = estimator.explained_variance_ratio_.tolist()
    else:
        perplexity = min(15.0, max(2.0, (len(selected)-1)/3))
        estimator = TSNE(n_components=2, perplexity=perplexity, random_state=seed, metric="cosine",
                         init="random", learning_rate="auto", max_iter=1000, method="exact")
        coordinates = estimator.fit_transform(normalized)
        meta.update({"perplexity": perplexity, "metric": "cosine", "init": "random", "max_iter": 1000,
                     "learning_rate": "auto", "kl_divergence": float(estimator.kl_divergence_)})
    if len(selected) >= 3:
        k = min(5, (len(selected)-1)//2)
        meta["trustworthiness_k"] = k
        meta["trustworthiness"] = float(trustworthiness(normalized, coordinates, n_neighbors=k, metric="cosine"))
    return pd.DataFrame({"word": selected, "x": coordinates[:, 0], "y": coordinates[:, 1]}), meta


def compare_neighbors(left, right):
    rows = []
    for word in sorted(set(left.word) | set(right.word)):
        a = set(left[(left.word == word) & (left.status == "ok")].neighbor)
        b = set(right[(right.word == word) & (right.status == "ok")].neighbor)
        present = bool(a and b)
        rows.append({"word": word, "overlap_at_5": len(a & b)/5 if present else np.nan,
                     "shared_neighbors": ", ".join(sorted(a & b)), "status": "ok" if present else "OOV"})
    return pd.DataFrame(rows)
