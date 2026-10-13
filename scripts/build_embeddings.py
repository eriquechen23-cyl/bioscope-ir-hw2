"""Run Part 2: python -m scripts.build_embeddings (fixed original corpus)."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from bioir.embedding import (QUERY_WORDS, RELATED_PAIRS, RARE_WORDS, corpus_fingerprint,
                             vocabulary_statistics, train_run, neighbor_table, pair_table,
                             project_embeddings, compare_neighbors)
from bioir.pubmed import load_corpus

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_SHA256 = "0aeec800f63dc342abc1fe836be4e11adff879f096972482c57e0e7dde0eaf67"
EXTRA_PROJECTION_WORDS = ["blood", "glycemic", "body", "loss", "agonists", "exenatide", "dulaglutide", "secretion", "appetite", "gastric", "emptying", "pancreatic"]


def build_results(records, output, epochs=20):
    """Train six conditions and persist their actual outputs; no external models."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stats = vocabulary_statistics(records)
    counts = stats.pop("counts")
    vocabulary = pd.DataFrame([{"word": w, "frequency": n, "retained": n >= 3}
                               for w, n in sorted(counts.items(), key=lambda item: (-item[1], item[0]))])
    vocabulary.to_csv(output / "vocabulary.csv", index=False)
    runs, neighbors, pairs, projections, losses = [], [], [], [], []
    neighbor_frames, vectors = {}, {}
    projection_words = [w for w in QUERY_WORDS + EXTRA_PROJECTION_WORDS if counts[w] >= 3]
    for architecture in ["CBOW", "Skip-gram"]:
        for window, seed in [(2, 42), (5, 42), (5, 7)]:
            name = f"{'cbow' if architecture == 'CBOW' else 'skipgram'}_w{window}_s{seed}"
            print(f"Training {name} ({epochs} epochs)...", flush=True)
            model, meta = train_run(records, architecture, window, epochs=epochs, seed=seed)
            meta["model"] = name
            meta["projections"] = []
            for row in meta["loss"]:
                losses.append({"model": name, **row})
            table = neighbor_table(model, QUERY_WORDS + RARE_WORDS)
            table.insert(0, "model", name)
            neighbor_frames[name] = table
            neighbors.append(table)
            pair_frame = pair_table(model, RELATED_PAIRS)
            pair_frame.insert(0, "model", name)
            pairs.append(pair_frame)
            if seed == 42:
                for method in ["PCA", "t-SNE"]:
                    frame, projection_meta = project_embeddings(model, projection_words, method)
                    frame.insert(0, "method", method)
                    frame.insert(0, "model", name)
                    projections.append(frame)
                    meta["projections"].append(projection_meta)
            vectors[f"{name}_words"] = np.asarray(model.wv.index_to_key)
            vectors[f"{name}_vectors"] = model.wv.vectors
            runs.append(meta)
            print(f"  {meta['elapsed_seconds']:.2f}s; vocabulary={meta['vocabulary']}", flush=True)
    pd.concat(neighbors, ignore_index=True).to_csv(output / "neighbors.csv", index=False)
    pd.concat(pairs, ignore_index=True).to_csv(output / "pairs.csv", index=False)
    pd.concat(projections, ignore_index=True).to_csv(output / "projections.csv", index=False)
    pd.DataFrame(losses).to_csv(output / "loss.csv", index=False)
    pd.DataFrame([{k: v for k, v in run.items() if k not in {"loss", "projections"}} for run in runs]).to_csv(output / "summary.csv", index=False)
    window_frames, seed_frames = [], []
    for prefix in ["cbow", "skipgram"]:
        change = compare_neighbors(neighbor_frames[f"{prefix}_w2_s42"], neighbor_frames[f"{prefix}_w5_s42"])
        change.insert(0, "architecture", prefix)
        window_frames.append(change)
        stable = compare_neighbors(neighbor_frames[f"{prefix}_w5_s42"], neighbor_frames[f"{prefix}_w5_s7"])
        stable.insert(0, "architecture", prefix)
        stable["frequency"] = stable.word.map(counts)
        stable["group"] = stable.word.map(lambda w: "query" if w in QUERY_WORDS else "rare (3-20)")
        seed_frames.append(stable)
    pd.concat(window_frames, ignore_index=True).to_csv(output / "window_comparison.csv", index=False)
    pd.concat(seed_frames, ignore_index=True).to_csv(output / "stability.csv", index=False)
    np.savez_compressed(output / "vectors.npz", **vectors)
    manifest = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
                "corpus_sha256": corpus_fingerprint(records), "corpus_statistics": stats,
                "versions": {name: version(name) for name in ["gensim", "numpy", "pandas", "scikit-learn"]},
                "python": platform.python_version(), "platform": platform.platform(), "query_words": QUERY_WORDS,
                "related_pairs": RELATED_PAIRS, "rare_words": RARE_WORDS, "projection_words": projection_words,
                "runs": runs, "notes": ["Loss is an epoch increment of Gensim cumulative SGNS loss, not comparable quality across architectures.",
                "Nominal samples use min_count-filtered fixed windows BEFORE subsampling and dynamic-window shrinking.",
                "Overlap@5 = size of intersection / 5; not semantic accuracy.",
                "Projection coordinates use L2-normalized vectors; separate fits cannot be compared by absolute coordinates."]}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    from bioir.embedding_report import write_findings
    write_findings(output)
    return manifest


def render_charts(output):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from adjustText import adjust_text
    output = Path(output)
    data = pd.read_csv(output / "projections.csv")
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    for window in [2, 5]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
        for col, (prefix, color) in enumerate([("cbow", "#007f78"), ("skipgram", "#bd642c")]):
            for row, method in enumerate(["PCA", "t-SNE"]):
                ax = axes[row, col]
                frame = data[(data.model == f"{prefix}_w{window}_s42") & (data.method == method)]
                ax.scatter(frame.x, frame.y, color=color, s=24)
                texts = [ax.text(point.x, point.y, point.word, fontsize=8) for point in frame.itertuples()]
                ax.margins(.18)
                ax.set(title=f"{prefix.upper()} / {method} / window={window}", xlabel="Component 1", ylabel="Component 2")
                ax.grid(alpha=.15)
                np.random.seed(42)
                adjust_text(texts, x=frame.x.to_numpy(), y=frame.y.to_numpy(), ax=ax, iter_lim=250,
                            arrowprops={"arrowstyle": "-", "color": "#8b98ac", "lw": .5})
        fig.suptitle("Same selected words; normalized vectors; independent projections", fontsize=12)
        for extension in ["png", "svg"]:
            fig.savefig(output / f"projections_w{window}.{extension}", dpi=180)
        plt.close(fig)
    loss = pd.read_csv(output / "loss.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained")
    for ax, prefix in zip(axes, ["cbow", "skipgram"]):
        for window in [2, 5]:
            rows = loss[loss.model == f"{prefix}_w{window}_s42"]
            ax.plot(rows.epoch, rows.epoch_loss, marker=".", label=f"window={window}")
        ax.set(title=f"{prefix.upper()} (seed=42)", xlabel="Epoch", ylabel="Epoch loss increment")
        ax.legend()
        ax.grid(alpha=.15)
    fig.savefig(output / "training_loss.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    path = ROOT / "data/glp1_1000.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("Original GLP-1 snapshot SHA-256 mismatch; refusing to replace fixed results.")
    output = ROOT / "reports/embedding"
    manifest = build_results(load_corpus(path), output, epochs=args.epochs)
    manifest["source_file_sha256"] = ORIGINAL_SHA256
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    render_charts(output)
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
