from collections import Counter
import pytest
from bioir.porter import stem, measure
from bioir.text import tokenize, token_spans
from bioir.retrieval import edit_distance, distance_matrix, search, highlight, suggestions
from bioir.analysis import frequencies, frequency_table, zipf_fit, train_embeddings, project_vectors


@pytest.mark.parametrize("word,expected", [
    ("caresses", "caress"), ("ponies", "poni"), ("ties", "ti"), ("cats", "cat"),
    ("feed", "feed"), ("agreed", "agre"), ("plastered", "plaster"), ("bled", "bled"),
    ("motoring", "motor"), ("sing", "sing"), ("conflated", "conflat"), ("troubled", "troubl"),
    ("hopping", "hop"), ("falling", "fall"), ("filing", "file"), ("happy", "happi"),
    ("sky", "sky"), ("relational", "relat"), ("conditional", "condit"), ("vietnamization", "vietnam"),
    ("triplicate", "triplic"), ("formative", "form"), ("electriciti", "electr"),
    ("revival", "reviv"), ("probate", "probat"), ("controll", "control"), ("GLP-1", "glp-1"),
])
def test_porter_reference_examples(word, expected):
    assert stem(word) == expected


def test_measure_and_token_positions():
    assert [measure(w) for w in ("tree", "trouble", "troubles")] == [0, 1, 2]
    text = "GLP‑1 patients: 10.5% <tag>"
    assert tokenize(text) == ["glp-1", "patients", "tag"]
    assert tokenize("The GLP-1 patients", True, True) == ["glp-1", "patient"]
    for word, start, end in token_spans(text):
        assert start < end and text[start:end]


@pytest.mark.parametrize("a,b,d", [("", "", 0), ("", "abc", 3), ("kitten", "sitting", 3),
                                      ("semaglutde", "semaglutide", 1), ("glp-1", "glp-2", 1)])
def test_edit_distance(a, b, d):
    assert edit_distance(a, b) == edit_distance(b, a) == d
    assert distance_matrix(a, b)[-1][-1] == d


def sample_records():
    return [{"pmid": "1", "title": "insulin study", "abstract": "Patients treated with semaglutide. GLP-1 <script>alert(1)</script> & patients."},
            {"pmid": "2", "title": "Other study", "abstract": "A patient received insulin and GLP-1."}]


def test_positional_search_modes_and_escaping():
    records = sample_records()
    assert not search(records, "insulin")[:1][0]["article"]["pmid"] == "1"
    assert len(search(records, "patient", mode="stem")) == 2
    assert len(search(records, "semaglutde", mode="fuzzy")) == 1
    assert search(records, "semaglutide insulin", require_all=True) == []
    assert len(search(records, "semaglutide insulin", require_all=False)) == 2
    result = search(records, "patients")[0]
    for span in result["spans"]:
        assert result["text"][span["start"]:span["end"]].lower() == span["word"]
    markup = highlight(result["text"], result["spans"])
    assert "<script>" not in markup and "&lt;script&gt;" in markup
    assert markup.count("<mark>") == 2
    assert len(search(records, "insulin", include_title=True)) == 2


def test_suggestions_and_frequency_conservation():
    assert suggestions("cot", Counter({"cat": 5, "cut": 9}))[0][0] == "cut"
    raw, stems = frequencies(sample_records())
    assert sum(raw.values()) == sum(stems.values())
    assert len(stems) < len(raw)
    table = frequency_table(Counter({"b": 10, "a": 10, "c": 2}))
    assert table.iloc[0]["term"] == "a"
    assert zipf_fit(table)["slope"] < 0


@pytest.mark.parametrize("architecture", ["CBOW", "Skip-gram"])
def test_embeddings_train_on_actual_tokens(architecture):
    model = train_embeddings(sample_records() * 5, architecture=architecture, vector_size=10, min_count=1, epochs=5)
    assert "glp-1" in model.wv
    assert model.sg == (architecture == "Skip-gram")
    projection = project_vectors(model, ["glp-1", "insulin", "semaglutide"])
    assert projection.shape == (3, 3)


def test_original_porter_against_independent_reference():
    from pathlib import Path
    from nltk.stem import PorterStemmer
    from bioir.pubmed import load_corpus
    path = Path("data/glp1_1000.jsonl")
    if not path.exists():
        pytest.skip("Corpus has not yet been downloaded")
    reference = PorterStemmer(mode=PorterStemmer.ORIGINAL_ALGORITHM)
    vocabulary = {w for r in load_corpus(path) for w in tokenize(r["abstract"]) if w.isalpha()}
    assert [(w, stem(w), reference.stem(w)) for w in sorted(vocabulary) if stem(w) != reference.stem(w)] == []
