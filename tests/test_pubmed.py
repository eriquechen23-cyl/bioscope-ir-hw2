import hashlib
import json
from pathlib import Path
import pytest
from bioir.pubmed import PubMedClient, corpus_bytes, load_corpus, parse_articles, parse_pmids, validate_records

XML = '''<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID><Article>
<ArticleTitle>A <i>GLP-1</i> study</ArticleTitle><Journal><Title>Journal</Title><JournalIssue><PubDate><Year>2025</Year></PubDate></JournalIssue></Journal>
<Abstract><AbstractText Label="METHODS">First <b>part</b>.</AbstractText><AbstractText Label="RESULTS">Second part.</AbstractText><CopyrightInformation>Author rights.</CopyrightInformation></Abstract>
</Article></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">10.123/test</ArticleId></ArticleIdList></PubmedData></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID>124</PMID><Article><ArticleTitle>No abstract</ArticleTitle></Article></MedlineCitation></PubmedArticle></PubmedArticleSet>'''


def test_xml_structured_abstract():
    records = parse_articles(XML)
    assert len(records) == 1
    assert records[0]["abstract"] == "METHODS: First part.\n\nRESULTS: Second part."
    assert records[0]["title"] == "A GLP-1 study"
    assert records[0]["copyright"] == "Author rights."
    assert records[0]["doi"] == "10.123/test"


def test_pmid_validation():
    assert parse_pmids("123, 123\nhttps://pubmed.ncbi.nlm.nih.gov/456/") == ["123", "456"]
    for value in ("abc 123", "", "0", "12x34"):
        with pytest.raises(ValueError):
            parse_pmids(value)
    with pytest.raises(ValueError):
        validate_records(parse_articles(XML) * 2)


def test_one_thousand_article_limit(monkeypatch):
    ids = [str(i) for i in range(1, 1001)]
    assert parse_pmids("\n".join(ids)) == ids
    records = [{"pmid": pmid, "title": "GLP-1", "abstract": "An abstract."} for pmid in ids]
    assert len(validate_records(records)) == 1000
    with pytest.raises(ValueError, match="1,000"):
        parse_pmids("\n".join(ids + ["1001"]))
    with pytest.raises(ValueError, match="1,000"):
        validate_records(records + [{"pmid": "1001", "title": "GLP-1", "abstract": "An abstract."}])
    client = PubMedClient()
    def unexpected_request(*args, **kwargs):
        pytest.fail("Invalid count must be rejected before an NCBI request")
    monkeypatch.setattr(client, "_request", unexpected_request)
    with pytest.raises(ValueError, match="1,000"):
        client.search("GLP-1", 1001)


def test_search_refills_missing_and_deduplicates(monkeypatch):
    client = PubMedClient()
    class Response:
        def __init__(self, ids): self.ids = ids
        def json(self): return {"esearchresult": {"count": "4", "idlist": self.ids, "querytranslation": "glp-1"}}
    def fake_request(endpoint, params):
        return Response(["1", "2"] if params["retstart"] == 0 else ["2", "3"])
    def fake_fetch(ids):
        return [{"pmid": p, "title": "Title", "abstract": "Abstract"} for p in ids if p != "1"]
    monkeypatch.setattr(client, "_request", fake_request)
    monkeypatch.setattr(client, "fetch_pmids", fake_fetch)
    records, metadata = client.search("glp-1", 2)
    assert [r["pmid"] for r in records] == ["2", "3"]
    assert metadata["actual_count"] == 2
    with pytest.raises(ValueError, match="3"):
        client.search("glp-1", 3)


def test_bundled_corpus_integrity():
    path = Path("data/glp1_1000.jsonl")
    assert path.exists(), "The project must ship with its real corpus"
    records = load_corpus(path)
    assert len(records) == len({r["pmid"] for r in records}) == 1000
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]
    assert path.with_suffix(".pmids.txt").read_text().splitlines() == [r["pmid"] for r in records]
