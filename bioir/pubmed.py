"""NCBI E-utilities ingestion; no network calls at import or app startup."""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from defusedxml import ElementTree as ET

DEFAULT_QUERY = '("GLP-1"[Title/Abstract] OR "glucagon-like peptide-1"[Title/Abstract]) AND hasabstract AND english[Language]'
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_LOCK = threading.Lock()
_LAST_REQUEST = 0.0


def parse_pmids(text: str) -> list[str]:
    """Accept only PMID numbers or PubMed URLs; do not silently eat bad input."""
    text = re.sub(r"https?://pubmed\.ncbi\.nlm\.nih\.gov/(\d+)/?", r"\1", text)
    values = [v for v in re.split(r"[\s,;]+", text.strip()) if v]
    if not values or any(not re.fullmatch(r"[1-9]\d{0,8}", v) for v in values):
        raise ValueError("請輸入以空白、逗號或換行分隔的 PMID，或 PubMed 文章網址。")
    values = list(dict.fromkeys(values))
    if len(values) > 1000:
        raise ValueError("一次最多 1,000 個不同 PMID。")
    return values


def _text(node) -> str:
    return "" if node is None else " ".join("".join(node.itertext()).split())


def parse_articles(xml: str) -> list[dict]:
    root = ET.fromstring(xml)
    errors = root.findall(".//ERROR")
    if errors:
        raise ValueError("NCBI: " + "; ".join(_text(e) for e in errors))
    records = []
    for entry in root.findall(".//PubmedArticle"):
        citation = entry.find("MedlineCitation")
        article = citation.find("Article")
        pmid = _text(citation.find("PMID"))
        title = _text(article.find("ArticleTitle"))
        parts = []
        for node in article.findall("Abstract/AbstractText"):
            value = _text(node)
            if value:
                label = node.get("Label", "")
                parts.append(f"{label}: {value}" if label else value)
        abstract = "\n\n".join(parts)
        if not pmid or not title or not abstract:
            continue
        doi = next((_text(n) for n in entry.findall("PubmedData/ArticleIdList/ArticleId")
                    if n.get("IdType") == "doi"), "")
        date = article.find("Journal/JournalIssue/PubDate")
        date_string = " ".join(_text(n) for n in date) if date is not None else ""
        records.append({
            "pmid": pmid, "title": title, "abstract": abstract,
            "journal": _text(article.find("Journal/Title")), "date": date_string,
            "doi": doi, "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "copyright": _text(article.find("Abstract/CopyrightInformation")),
        })
    return records


def validate_records(records: list[dict]) -> list[dict]:
    if not isinstance(records, list) or not 1 <= len(records) <= 1000:
        raise ValueError("資料必須包含 1 至 1,000 篇文章。")
    seen = set()
    cleaned = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("每篇文章必須是 JSON 物件。")
        pmid = str(record.get("pmid", ""))
        if not re.fullmatch(r"[1-9]\d{0,8}", pmid) or pmid in seen:
            raise ValueError("PMID 不合法或重複。")
        for field in ("title", "abstract"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"PMID {pmid} 缺少 {field}。")
        seen.add(pmid)
        cleaned.append({"pmid": pmid, "title": record["title"], "abstract": record["abstract"],
                        **{field: str(record.get(field, "")) for field in ("journal", "date", "doi", "copyright")},
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"})
    return cleaned


def corpus_bytes(records: list[dict]) -> bytes:
    return ("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n").encode("utf-8")


def load_corpus(path: Path) -> list[dict]:
    return validate_records([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


class PubMedClient:
    def __init__(self, email: str = "", api_key: str = ""):
        self.session = requests.Session()
        self.identity = {"tool": "bioscope_ir_hw2"}
        if email:
            self.identity["email"] = email
        if api_key:
            self.identity["api_key"] = api_key

    def _request(self, endpoint: str, params: dict):
        global _LAST_REQUEST
        for attempt in range(5):
            # Shared across this app process, including multiple user sessions.
            with _LOCK:
                time.sleep(max(0, 0.36 - (time.monotonic() - _LAST_REQUEST)))
                _LAST_REQUEST = time.monotonic()
            try:
                response = self.session.get(f"{BASE_URL}/{endpoint}.fcgi",
                    params={"db": "pubmed", **self.identity, **params}, timeout=(15, 90))
                if response.status_code == 429 or response.status_code >= 500:
                    time.sleep(min(2 ** attempt, 16))
                    continue
                response.raise_for_status()
                return response
            except requests.RequestException:
                if attempt == 4:
                    # Avoid leaking request URLs containing API keys in UI/logs.
                    raise RuntimeError("無法連線 NCBI，請稍後再試。") from None
                time.sleep(2 ** attempt)
        raise RuntimeError("NCBI 暫時限流或無法使用，請稍後再試。")

    def fetch_pmids(self, pmids: list[str], progress=None) -> list[dict]:
        pmids = parse_pmids(" ".join(pmids))
        found = {}
        for offset in range(0, len(pmids), 100):
            response = self._request("efetch", {"id": ",".join(pmids[offset:offset + 100]), "retmode": "xml"})
            for record in parse_articles(response.text):
                if record["pmid"] in pmids:
                    found[record["pmid"]] = record
            if progress:
                progress(min(offset + 100, len(pmids)), len(pmids))
        return [found[p] for p in pmids if p in found]

    def search(self, query: str, count: int = 1000, progress=None) -> tuple[list[dict], dict]:
        if not query.strip() or not 1 <= count <= 1000:
            raise ValueError("請輸入查詢式，篇數須介於 1 到 1,000。")
        records, seen, start, total = [], set(), 0, None
        translated = ""
        while len(records) < count and start < 9999:
            batch = min(200, max(100, count - len(records)), 9999 - start)
            payload = self._request("esearch", {"term": query, "retmode": "json", "sort": "relevance",
                                               "retstart": start, "retmax": batch}).json()
            result = payload.get("esearchresult", {})
            if "error" in payload or "ERROR" in result or result.get("errorlist"):
                raise ValueError("PubMed 無法解析查詢式，請確認關鍵字與欄位。")
            total = int(result.get("count", 0))
            translated = result.get("querytranslation", "")
            ids = result.get("idlist", [])
            if not ids:
                break
            for record in self.fetch_pmids(ids):
                if record["pmid"] not in seen:
                    records.append(record)
                    seen.add(record["pmid"])
                    if len(records) == count:
                        break
            start += len(ids)
            if progress:
                progress(len(records), count)
            if start >= total:
                break
        if len(records) < count:
            raise ValueError(f"僅找到 {len(records)} 篇有摘要的不重複文章，未達 {count} 篇。請減少篇數或擴大查詢。")
        metadata = {"query": query, "query_translation": translated, "sort": "relevance",
                    "pubmed_matches": total, "requested_count": count, "actual_count": len(records),
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source": BASE_URL, "sha256": hashlib.sha256(corpus_bytes(records)).hexdigest()}
        return records, metadata
