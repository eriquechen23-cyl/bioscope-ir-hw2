"""Part 2 behavioral tests: samples, real training, OOV and projections."""
import numpy as np
import pytest


def records():
    return [{"title": "unused", "abstract": "Insulin reduces glucose. Semaglutide treats obesity and diabetes."}] * 20


def test_sample_direction_and_sentence_boundaries():
    from bioir.embedding import training_samples
    cbow, sg = training_samples(["a", "b", "c"], 1)
    assert cbow == [{"context": ["b"], "target": "a"}, {"context": ["a", "c"], "target": "b"}, {"context": ["b"], "target": "c"}]
    assert sg == [{"input": "a", "target": "b"}, {"input": "b", "target": "a"}, {"input": "b", "target": "c"}, {"input": "c", "target": "b"}]
    assert training_samples(["alone"], 2) == ([], [])
    with pytest.raises(ValueError):
        training_samples(["a", "b"], 0)


def test_vocabulary_counts_and_discarded_terms():
    from bioir.embedding import vocabulary_statistics
    stats = vocabulary_statistics([{"abstract": "Insulin insulin GLP-1. Rare."}], 2)
    assert stats["documents"] == 1 and stats["sentences"] == 2
    assert stats["tokens"] == 4 and stats["vocabulary"] == 3
    assert stats["retained_vocabulary"] == 1


@pytest.mark.parametrize("architecture", ["CBOW", "Skip-gram"])
def test_real_training_loss_neighbors_and_cosine(architecture):
    from bioir.embedding import train_run, neighbor_table, pair_table
    model, meta = train_run(records(), architecture, 2, epochs=3, min_count=1, vector_size=6)
    assert model.sg == int(architecture == "Skip-gram") and meta["seed"] == 42
    assert len(meta["loss"]) == 3 and meta["elapsed_seconds"] > 0
    assert all(row["epoch_loss"] >= 0 for row in meta["loss"])
    assert meta["loss"][-1]["cumulative_loss"] > 0
    neighbors = neighbor_table(model, ["insulin", "missing"])
    insulin = neighbors[neighbors.word == "insulin"]
    assert len(insulin) == 5 and "insulin" not in insulin.neighbor.tolist()
    assert neighbors[neighbors.word == "missing"].iloc[0].status == "OOV"
    pairs = pair_table(model, [("insulin", "glucose"), ("glucose", "insulin"), ("insulin", "missing")])
    assert pairs.iloc[0].cosine == pytest.approx(pairs.iloc[1].cosine)
    assert pairs.iloc[2].status == "OOV"


def test_projection_bounds_dedup_and_reproducibility():
    from bioir.embedding import train_run, project_embeddings
    model, _ = train_run(records(), "CBOW", 2, epochs=2, min_count=1, vector_size=6)
    words = model.wv.index_to_key[:6]
    for method in ["PCA", "t-SNE"]:
        frame, meta = project_embeddings(model, words + [words[0]], method)
        assert len(frame) == 6 and np.isfinite(frame[["x", "y"]].to_numpy()).all()
        assert meta["method"] == method
    first, _ = project_embeddings(model, words, "t-SNE")
    second, _ = project_embeddings(model, words, "t-SNE")
    np.testing.assert_allclose(first[["x", "y"]], second[["x", "y"]])
    with pytest.raises(ValueError):
        project_embeddings(model, words[:2], "t-SNE")
    with pytest.raises(ValueError):
        project_embeddings(model, ["missing", words[0]], "PCA")
    with pytest.raises(ValueError):
        train_run(records(), "typo", 2)


def test_neighbor_overlap_uses_fixed_top_five_denominator():
    import pandas as pd
    from bioir.embedding import compare_neighbors
    left = pd.DataFrame({"word": ["q"]*5, "neighbor": ["a", "b", "c", "d", "e"], "status": ["ok"]*5})
    right = pd.DataFrame({"word": ["q"]*5, "neighbor": ["a", "b", "x", "y", "z"], "status": ["ok"]*5})
    result = compare_neighbors(left, right)
    assert result.iloc[0].overlap_at_5 == pytest.approx(0.4)
    assert result.iloc[0].shared_neighbors == "a, b"


def test_pipeline_has_six_conditions_and_verifiable_artifacts(tmp_path):
    import json
    from bioir.embedding import QUERY_WORDS, RELATED_PAIRS, corpus_fingerprint
    from scripts.build_embeddings import build_results
    texts = [" ".join(QUERY_WORDS) + ". cancer tumor diagnosis treatment rareterm."] * 10
    data = [{"title": "unused", "abstract": text} for text in texts]
    manifest = build_results(data, tmp_path, epochs=1)
    assert len(manifest["runs"]) == 6
    assert manifest["corpus_sha256"] == corpus_fingerprint(data)
    assert manifest["query_words"] == QUERY_WORDS and len(RELATED_PAIRS) >= 5
    assert {r["window"] for r in manifest["runs"]} == {2, 5}
    assert {r["seed"] for r in manifest["runs"]} == {7, 42}
    for name in ["summary.csv", "loss.csv", "neighbors.csv", "pairs.csv", "projections.csv", "window_comparison.csv", "stability.csv", "vocabulary.csv", "vectors.npz"]:
        assert (tmp_path / name).stat().st_size > 0
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["projection_words"]
    report = (tmp_path / "findings_zh.md").read_text(encoding="utf-8")
    assert all(f"RQ{i}" in report for i in range(1, 6))
    assert "Prediction target" in report and "低頻" in report


def test_corpus_fingerprint_tracks_abstracts_not_metadata():
    from bioir.embedding import corpus_fingerprint
    original = [{"abstract": "Text", "title": "Old"}]
    assert corpus_fingerprint(original) == corpus_fingerprint([{"abstract": "Text", "title": "New"}])
    assert corpus_fingerprint(original) != corpus_fingerprint([{"abstract": "Different", "title": "Old"}])
