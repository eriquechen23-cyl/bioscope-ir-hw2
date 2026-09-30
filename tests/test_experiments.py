import math
import pandas as pd
import pytest
from bioir.experiments import preprocess, term_statistics, fit_zipf, regional_fits


def test_cumulative_preprocessing_preserves_biomedical_hyphens():
    text = "The GLP-1 receptors, running! 2026"
    assert preprocess(text, "A") == ["the", "glp-1", "receptors", ",", "running", "!"]
    assert preprocess(text, "B") == ["the", "glp-1", "receptors", "running"]
    assert preprocess(text, "C") == ["glp-1", "receptors", "running"]
    assert preprocess(text, "D") == ["glp-1", "receptor", "run"]


def test_cf_df_idf_distinguish_repetition_and_document_coverage():
    documents = [{"abstract": "Insulin insulin glucose."}, {"abstract": "GLP-1 glucose."}]
    table, summary, tf = term_statistics(documents)
    terms = table.set_index("term")
    assert summary["tokens"] == 5 and summary["vocabulary"] == 3
    assert summary["average_tokens_per_document"] == 2.5
    assert terms.loc["insulin", "CF"] == terms.loc["glucose", "CF"] == 2
    assert terms.loc["insulin", "DF"] == 1 and terms.loc["glucose", "DF"] == 2
    assert terms.loc["insulin", "IDF"] == pytest.approx(math.log(2))
    assert terms.loc["glucose", "IDF"] == 0
    assert tf[0]["insulin"] == 2 and tf[1]["insulin"] == 0


def test_known_power_law_and_constant_tail():
    exact = pd.DataFrame({"rank": range(1, 101), "CF": [1000/r for r in range(1, 101)]})
    fit = fit_zipf(exact)
    assert fit["slope"] == pytest.approx(-1)
    assert fit["intercept"] == pytest.approx(3)
    assert fit["exponent"] == pytest.approx(1)
    assert fit["r_squared"] == pytest.approx(1)
    assert fit["rmse"] == pytest.approx(0, abs=1e-12)
    regions = regional_fits(exact)
    assert list(regions["rank_start"]) == [1, 11, 91]
    assert list(regions["rank_end"]) == [10, 90, 100]
    tail = pd.DataFrame({"rank": [98, 99, 100], "CF": [1, 1, 1]})
    assert fit_zipf(tail)["r_squared"] is None


def test_empty_documents_and_single_term_are_handled():
    table, summary, _ = term_statistics([{"abstract": "1234"}])
    assert table.empty and summary["tokens"] == 0
    assert fit_zipf(table)["slope"] is None
    with pytest.raises(ValueError):
        term_statistics([])
