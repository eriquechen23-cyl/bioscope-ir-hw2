"""Required Project 2 experiments on an unchanged abstract collection.

A -> B -> C -> D is a cumulative pipeline. Numeric-only tokens are excluded
in every condition; hyphens inside biomedical terms are preserved throughout.
"""
from collections import Counter
import re
import numpy as np
import pandas as pd

from .analysis import frequency_table
from .porter import stem
from .text import STOPWORDS, TOKEN_PATTERN, normalize

CONDITIONS = {
    "A": "Basic: tokenize + lowercase, punctuation retained",
    "B": "A + remove standalone punctuation",
    "C": "B + remove stopwords",
    "D": "C + Porter stemming",
}
BASIC_PATTERN = re.compile(TOKEN_PATTERN.pattern + r"|[^\w\s]", re.ASCII)


def preprocess(text: str, condition: str) -> list[str]:
    if condition not in CONDITIONS:
        raise ValueError("Condition must be A, B, C or D")
    pattern = BASIC_PATTERN if condition == "A" else TOKEN_PATTERN
    # Exclude pure numbers consistently. Non-ASCII letters not captured by the
    # English tokenizer are not reclassified as punctuation in condition A.
    tokens = [normalize(m.group()) for m in pattern.finditer(text)
              if not m.group()[0].isdigit() and
              (m.group()[0].isascii() or not m.group()[0].isalnum())]
    if condition in {"C", "D"}:
        tokens = [word for word in tokens if word not in STOPWORDS]
    return [stem(word) for word in tokens] if condition == "D" else tokens


def term_statistics(records: list[dict], condition="B"):
    if not records:
        raise ValueError("At least one document is required")
    document_tf = [Counter(preprocess(r["abstract"], condition)) for r in records]
    cf, df = Counter(), Counter()
    for counts in document_tf:
        cf.update(counts)
        df.update(counts.keys())
    table = frequency_table(cf).rename(columns={"frequency": "CF"})
    table["DF"] = [df[t] for t in table["term"]]
    table["IDF"] = [float(np.log(len(records) / df[t])) for t in table["term"]]
    table["CF_per_DF"] = table["CF"] / table["DF"]
    summary = {"condition": condition, "documents": len(records), "tokens": sum(cf.values()),
               "vocabulary": len(cf), "average_tokens_per_document": sum(cf.values()) / len(records),
               "hapax_terms": sum(c == 1 for c in cf.values())}
    return table, summary, document_tf


def fit_zipf(table: pd.DataFrame) -> dict:
    if len(table) < 2:
        return {"slope": None, "intercept": None, "exponent": None, "r_squared": None, "rmse": None}
    x = np.log10(table["rank"].to_numpy(dtype=float))
    y = np.log10(table["CF"].to_numpy(dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (intercept + slope * x)
    ss_total = float(np.sum((y - y.mean()) ** 2))
    return {"slope": float(slope), "intercept": float(intercept), "exponent": float(-slope),
            "r_squared": float(1 - np.sum(residuals ** 2) / ss_total) if ss_total > 1e-14 else None,
            "rmse": float(np.sqrt(np.mean(residuals ** 2)))}


def regional_fits(table: pd.DataFrame) -> pd.DataFrame:
    n = len(table)
    head_end = max(1, int(np.ceil(n * .1)))
    tail_start = max(head_end, int(np.floor(n * .9)))
    rows = []
    for region, part in [("High (first 10%)", table.iloc[:head_end]),
                         ("Middle (next 80%)", table.iloc[head_end:tail_start]),
                         ("Low (last 10%)", table.iloc[tail_start:])]:
        if not part.empty:
            rows.append({"region": region, "rank_start": int(part["rank"].min()),
                         "rank_end": int(part["rank"].max()), "terms": len(part),
                         "distinct_frequencies": int(part["CF"].nunique()), **fit_zipf(part)})
    return pd.DataFrame(rows)


def run_experiments(records):
    conditions, summaries = {}, []
    for code in CONDITIONS:
        table, summary, _ = term_statistics(records, code)
        fit = fit_zipf(table)
        summaries.append({**summary, **fit})
        conditions[code] = {"terms": table, "fit": fit, "regions": regional_fits(table)}
    return {"summary": pd.DataFrame(summaries), "conditions": conditions}


def select_comparison_terms(table: pd.DataFrame, count=25):
    """Predeclared mixture of common and biomedical terms, not chosen by IDF."""
    preferred = ["glp-1", "glucagon-like", "peptide-1", "receptor", "agonists", "insulin", "glucose",
                 "diabetes", "obesity", "semaglutide", "liraglutide", "exenatide", "cardiovascular",
                 "patients", "treatment"]
    chosen = list(dict.fromkeys(table["term"].head(10).tolist() + preferred))
    chosen = [word for word in chosen if word in set(table["term"])]
    for word in table["term"]:
        if len(chosen) >= count:
            break
        if word not in chosen:
            chosen.append(word)
    order = {word: i for i, word in enumerate(chosen[:count])}
    return table[table["term"].isin(order)].assign(_order=lambda d: d["term"].map(order)).sort_values("_order").drop(columns="_order").reset_index(drop=True)
