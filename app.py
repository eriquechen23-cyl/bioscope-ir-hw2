from __future__ import annotations

import hashlib
import html
import io
import json
import math
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from bioir.analysis import frequencies, frequency_table, zipf_fit, train_embeddings, project_vectors
from bioir.porter import stem
from bioir.pubmed import DEFAULT_QUERY, PubMedClient, corpus_bytes, load_corpus, parse_pmids, validate_records
from bioir.retrieval import search, suggestions, highlight, distance_matrix
from bioir.text import document_text, tokenize
from bioir.assignment_ui import render_assignment

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="BioScope · GLP-1 文獻實驗室", page_icon="🧬", layout="wide")
st.markdown("""<style>
.block-container {max-width:1380px;padding-top:4.5rem}
h1 {letter-spacing:-.04em} h2,h3 {letter-spacing:-.025em}
.eyebrow {color:#007f78;font-size:.78rem;font-weight:750;letter-spacing:.18em}
.intro {color:#60738a;font-size:1.08rem;max-width:850px;line-height:1.8}
[data-testid="stMetric"] {background:white;border:1px solid #dce6ef;border-radius:14px;padding:18px 20px}
mark {background:#c5f2dc;color:#123c33;padding:1px 3px;border-radius:3px}
.abstract {line-height:1.9;font-size:1rem;background:white;border:1px solid #dce6ef;border-radius:12px;padding:22px}
</style>""", unsafe_allow_html=True)


@st.cache_data
def bundled_corpus():
    path = ROOT / "data/glp1_1000.jsonl"
    return load_corpus(path) if path.exists() else []


@st.cache_data(max_entries=12)
def compute_frequencies(records, remove_stopwords, include_title):
    return frequencies(records, remove_stopwords, include_title)


def downloads(records, prefix="corpus"):
    a, b, c = st.columns(3)
    a.download_button("下載摘要 JSONL", corpus_bytes(records), f"{prefix}.jsonl", "application/jsonl")
    b.download_button("下載 PMID 清單", "\n".join(r["pmid"] for r in records), f"{prefix}.pmids.txt", "text/plain")
    c.download_button("下載 CSV", pd.DataFrame(records).to_csv(index=False).encode("utf-8-sig"), f"{prefix}.csv", "text/csv")


def article_detail(record):
    st.subheader(record["title"])
    st.caption(f"PMID {record['pmid']} · {record.get('journal', '')} · {record.get('date', '')}")
    st.markdown('<div class="abstract">' + html.escape(record["abstract"]).replace("\n", "<br>") + "</div>", unsafe_allow_html=True)
    st.link_button("在 PubMed 閱讀原始紀錄 ↗", record["url"])
    if record.get("copyright"):
        st.caption(record["copyright"])


with st.sidebar:
    st.markdown("### 🧬 BioScope")
    st.caption("BIOMEDICAL INFORMATION RETRIEVAL\n\nHomework 02 · GLP-1")
    st.divider()
    source = st.radio("資料來源", ["預載 GLP-1", "本次自訂資料"], key="source")
    all_records = bundled_corpus() if source == "預載 GLP-1" else st.session_state.get("custom_records", [])
    if all_records:
        n = st.slider("分析篇數", 10, len(all_records), len(all_records)) if len(all_records) > 10 else len(all_records)
    else:
        n = 0
    remove_stopwords = st.checkbox("移除英文停用詞", value=False)
    include_title = st.checkbox("將標題納入分析", value=False)
    st.caption("預設保留停用詞、只分析摘要。GLP-1 等連字號詞彙會保留；純數字不納入詞頻。")
    st.divider()
    st.caption("資料來源：NCBI PubMed\n\nPorter 1980 · Word2Vec · Levenshtein")

records = all_records[:n]
st.markdown('<div class="eyebrow">PUBMED / GLP-1 / TEXT MINING</div>', unsafe_allow_html=True)
st.title("GLP-1 文獻實驗室")
st.markdown('<p class="intro">從 1,000 篇摘要開始，觀察詞頻分布、比較詞幹化，探索詞向量與文獻檢索。</p>', unsafe_allow_html=True)
page = st.radio("工作區", ["作業分析", "文獻資料", "Zipf 與 Porter", "文獻搜尋", "Word2Vec", "方法與展示"], horizontal=True, label_visibility="collapsed")
st.divider()

if records:
    raw, stemmed = compute_frequencies(records, False if page == "作業分析" else remove_stopwords, False if page == "作業分析" else include_title)
    cols = st.columns(4)
    cols[0].metric("分析文章", f"{len(records):,}", help="所有載入文章均有 PMID 與非空摘要")
    cols[1].metric("總詞數", f"{sum(raw.values()):,}")
    cols[2].metric("原始詞彙", f"{len(raw):,}")
    cols[3].metric("Porter（含停用詞）" if page == "作業分析" else "Porter 詞彙", f"{len(stemmed):,}", f"{(len(stemmed)/len(raw)-1)*100:.1f}%" if raw else None, delta_color="inverse", help="此處直接對上方原始詞彙做 stemming；下方 D 條件先去停用詞再 stemming。" if page == "作業分析" else None)
    st.write("")
else:
    raw, stemmed = Counter(), Counter()
    st.info("尚未載入資料。請在下方預備 PubMed 摘要，或切換回預載 GLP-1。")

if page == "作業分析" and records:
    render_assignment(records, ROOT)

elif page == "文獻資料":
    if records:
        st.subheader("摘要資料集")
        if source == "預載 GLP-1":
            manifest_path = ROOT / "data/glp1_1000.manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                st.caption(f"GLP-1 · English · 有摘要 · PubMed relevance 排序 · 快照 {manifest['retrieved_at_utc'][:10]}")
                with st.expander("查看資料來源與查詢紀錄"):
                    st.json(manifest)
        elif "custom_metadata" in st.session_state:
            with st.expander("查看本次查詢紀錄"):
                st.json(st.session_state.custom_metadata)
        table = pd.DataFrame(records)[["pmid", "title", "journal", "date", "url"]]
        st.dataframe(table, hide_index=True, width="stretch", height=330, column_config={"url": st.column_config.LinkColumn("PubMed")})
        downloads(records, "glp1" if source == "預載 GLP-1" else "custom_pubmed")
        selected = st.selectbox("閱讀摘要", options=range(len(records)), format_func=lambda i: f"{records[i]['pmid']} · {records[i]['title'][:105]}")
        article_detail(records[selected])
    st.divider()
    st.subheader("預備另一組 PubMed 摘要")
    st.caption("新資料只保存在本次瀏覽器工作階段。若要下次仍可讀取，請下載後上傳 JSONL，或使用命令列預備資料並提交至 GitHub。")
    mode = st.selectbox("匯入方式", ["PubMed 關鍵字", "PMID 清單", "上傳 JSONL"])
    if mode == "上傳 JSONL":
        uploaded = st.file_uploader("上傳此網站匯出的摘要 JSONL", type=["jsonl"])
        if st.button("載入檔案", disabled=uploaded is None):
            try:
                data = validate_records([json.loads(line) for line in uploaded.getvalue().decode("utf-8-sig").splitlines() if line.strip()])
                st.session_state.custom_records = data
                st.session_state.custom_metadata = {"source": uploaded.name, "actual_count": len(data)}
                st.success(f"已載入 {len(data)} 篇。請在側欄選擇「本次自訂資料」。")
            except (ValueError, UnicodeError) as error:
                st.error(str(error))
    else:
        with st.form("fetch"):
            query = st.text_area("PubMed 查詢式" if mode == "PubMed 關鍵字" else "PMID（空白、逗號、換行分隔）", value=DEFAULT_QUERY if mode == "PubMed 關鍵字" else "")
            count = st.number_input("目標篇數", 10, 1000, 1000, step=10) if mode == "PubMed 關鍵字" else None
            email = st.text_input("NCBI 聯絡 email（選填）", help="僅傳給 NCBI 作為 API 聯絡資訊，不写入資料集。")
            fetch = st.form_submit_button("下載 PubMed 摘要", type="primary")
        if fetch:
            bar = st.progress(0, text="連接 NCBI PubMed…")
            try:
                client = PubMedClient(email=email)
                progress = lambda done, total: bar.progress(min(done / total, 1.0), text=f"已處理 {done:,} / {total:,}")
                if mode == "PubMed 關鍵字":
                    data, metadata = client.search(query, int(count), progress)
                else:
                    ids = parse_pmids(query)
                    data = client.fetch_pmids(ids, progress)
                    missing = [p for p in ids if p not in {r["pmid"] for r in data}]
                    metadata = {"source": "PMID list", "requested_count": len(ids), "actual_count": len(data), "missing_pmids": missing}
                    if missing:
                        st.warning("沒有摘要或查無紀錄的 PMID：" + ", ".join(missing))
                validate_records(data)
                st.session_state.custom_records = data
                st.session_state.custom_metadata = metadata
                st.success(f"已取得 {len(data):,} 篇摘要。請在側欄切換為「本次自訂資料」。")
                st.download_button("先保存這次摘要 JSONL", corpus_bytes(data), "pubmed_custom.jsonl")
            except (ValueError, RuntimeError) as error:
                st.error(str(error))
            finally:
                bar.empty()

elif page == "Zipf 與 Porter" and records:
    st.subheader("詞頻的長尾，詞幹化的差異")
    st.caption("所有詞彙依頻次遞減排序；同頻時依字母排序。比較相同語料、相同停用詞與標題設定。")
    raw_table, stem_table = frequency_table(raw), frequency_table(stemmed)
    fig = go.Figure()
    for table, label, color in [(raw_table, "原始 token", "#007f78"), (stem_table, "Porter stem", "#ec8f48")]:
        fig.add_trace(go.Scatter(x=table["rank"], y=table["frequency"], text=table["term"], name=label, mode="lines", line={"color": color}, hovertemplate="%{text}<br>rank=%{x}<br>frequency=%{y}<extra>%{fullData.name}</extra>"))
    if not raw_table.empty:
        fig.add_trace(go.Scatter(x=raw_table["rank"], y=raw_table.iloc[0]["frequency"] / raw_table["rank"], name="參考 f(1) / rank", line={"dash": "dot", "color": "#9aaabd"}))
    fig.update_layout(xaxis_type="log", yaxis_type="log", xaxis_title="Rank（log）", yaxis_title="Frequency（log）", height=460, legend={"orientation": "h"}, margin={"t": 30})
    st.plotly_chart(fig, width="stretch")
    stats = []
    for table, label in [(raw_table, "原始"), (stem_table, "Porter")]:
        fitted = zipf_fit(table)
        if fitted:
            stats.append({"處理方式": label, "詞彙數": len(table), "log-log slope": round(fitted["slope"], 3), "R²": round(fitted["r_squared"], 3)})
    st.dataframe(pd.DataFrame(stats), hide_index=True, width="stretch")
    st.caption("OLS 使用所有排名，Zipf 參考斜率為 −1；有限語料的長尾與停用詞會影響斜率，R² 並非檢索品質評分。")
    a, b = st.columns(2)
    a.markdown("#### 原始高頻詞")
    a.dataframe(raw_table.head(40), hide_index=True, width="stretch")
    b.markdown("#### 詞幹化後高頻詞")
    b.dataframe(stem_table.head(40), hide_index=True, width="stretch")
    a.download_button("下載完整原始詞頻", raw_table.to_csv(index=False).encode("utf-8-sig"), "raw_frequencies.csv")
    b.download_button("下載完整 Porter 詞頻", stem_table.to_csv(index=False).encode("utf-8-sig"), "porter_frequencies.csv")
    st.subheader("Porter 轉換示例")
    example = st.text_input("輸入英文單字或句子", "patients treated with receptor agonists")
    st.dataframe(pd.DataFrame([{"token": w, "stem": stem(w)} for w in tokenize(example)]), hide_index=True)

elif page == "文獻搜尋" and records:
    st.subheader("在摘要中找到關鍵字")
    query = st.text_input("查詢關鍵字", "semaglutide", max_chars=300, key="search_query")
    a, b, c = st.columns(3)
    mode_label = a.selectbox("匹配方式", ["完整詞匹配", "Porter 詞幹匹配", "近似匹配（編輯距離）"])
    mode = {"完整詞匹配": "exact", "Porter 詞幹匹配": "stem", "近似匹配（編輯距離）": "fuzzy"}[mode_label]
    threshold = b.selectbox("最大編輯距離", [1, 2], disabled=mode != "fuzzy")
    require_all = c.checkbox("包含所有查詢詞（AND）", value=True)
    st.caption("搜尋使用原文 token，不刪除停用詞。排序：命中查詢詞數、命中次數、PMID；不是臨床相關性排名。")
    query_tokens = tokenize(query)
    vocabulary = Counter(w for r in records for w in tokenize(document_text(r, include_title)))
    if len(query_tokens) <= 10 and all(len(t) <= 50 for t in query_tokens):
        for word in dict.fromkeys(query_tokens):
            if word not in vocabulary:
                candidates = suggestions(word, vocabulary)
                if candidates:
                    st.info(f"「{word}」未在語料中出現。候選字：" + "、".join(f"{w}（距離 {d}，頻次 {f}）" for w, d, f in candidates))
                    chosen = st.selectbox(f"選擇 {word} 的替代字", [c[0] for c in candidates], key=f"fix_{word}")
                    def correct_query(original, replacement):
                        st.session_state.search_query = " ".join(replacement if t == original else t for t in tokenize(st.session_state.search_query))
                    st.button(f"以 {chosen} 搜尋", key=f"apply_{word}", on_click=correct_query, args=(word, chosen))
        effective_query = " ".join(query_tokens)
        if effective_query != " ".join(tokenize(query)):
            st.caption(f"本次校正後查詢：{effective_query}")
        results = search(records, effective_query, mode, threshold, require_all, include_title)
        st.markdown(f"**{len(results):,} 篇命中文獻**")
        if results:
            pages = math.ceil(len(results) / 10)
            current = st.number_input("結果頁碼", 1, pages, 1, key=f"search_page_{hashlib.sha256((effective_query+mode+str(threshold)+str(require_all)+str(n)+source+str(include_title)).encode()).hexdigest()[:16]}")
            for result in results[(current - 1) * 10:current * 10]:
                article = result["article"]
                with st.expander(f"PMID {article['pmid']} · {article['title']} · {result['hits']} 次命中"):
                    st.markdown('<div class="abstract">' + highlight(result["text"], result["spans"]) + '</div>', unsafe_allow_html=True)
                    st.caption("位置為所選分析原文的 Python 字元索引（0 起算，[start, end)）；token_index 包含數字 token。")
                    st.dataframe(pd.DataFrame(result["spans"]), hide_index=True, width="stretch")
                    st.link_button("PubMed 原始紀錄 ↗", article["url"])
            st.download_button("下載命中文獻與位置", json.dumps(results, ensure_ascii=False, indent=2), "search_results.json", "application/json")
        else:
            st.info("沒有符合文章。可減少查詢詞、改為 OR，或使用詞幹／近似匹配。")
    else:
        st.warning("請使用最多 10 個詞，每詞最多 50 字元。")
    with st.expander("查看 Levenshtein 動態規劃矩陣"):
        a = st.text_input("字串 A", "semaglutde", max_chars=40)
        b = st.text_input("字串 B", "semaglutide", max_chars=40)
        matrix = distance_matrix(a, b)
        st.metric("Edit distance", matrix[-1][-1])
        st.dataframe(pd.DataFrame(matrix, index=["∅"] + [f"{i}:{c}" for i, c in enumerate(a)], columns=["∅"] + [f"{i}:{c}" for i, c in enumerate(b)]))

elif page == "Word2Vec" and records:
    st.subheader("讓同一主題的詞彙形成向量")
    st.caption("CBOW：上下文預測中心詞；Skip-gram：中心詞預測上下文。僅使用目前所選文獻訓練，不使用外部預訓練向量。")
    with st.form("embedding_training"):
        a, b, c = st.columns(3)
        architecture = a.selectbox("模型", ["Skip-gram", "CBOW"])
        window = b.slider("Window", 2, 10, 5)
        min_count = c.slider("min_count", 1, 10, 3)
        vector_size = a.selectbox("向量維度", [50, 100, 200], index=1)
        epochs = b.slider("Epochs", 5, 30, 10)
        use_stems = c.checkbox("先做 Porter stemming")
        train = st.form_submit_button("訓練 Word2Vec", type="primary")
    data_key = hashlib.sha256(corpus_bytes(records) + str((remove_stopwords, include_title)).encode()).hexdigest()
    if train:
        with st.spinner("正在訓練詞向量…"):
            try:
                model = train_embeddings(records, architecture, vector_size, window, min_count, epochs, remove_stopwords, use_stems, include_title)
                st.session_state.embedding = {"model": model, "data_key": data_key, "stemming": use_stems,
                    "settings": {"architecture": architecture, "vector_size": vector_size, "window": window,
                                 "min_count": min_count, "epochs": epochs, "seed": 42, "workers": 1,
                                 "stemming": use_stems, "remove_stopwords": remove_stopwords,
                                 "include_title": include_title, "articles": len(records), "data_key": data_key}}
            except ValueError as error:
                st.error(str(error))
    trained = st.session_state.get("embedding")
    if trained and trained["data_key"] != data_key:
        st.info("資料或前處理已變更，請重新訓練以使用目前語料。")
    elif trained:
        model = trained["model"]
        st.success(f"模型已就緒 · {len(model.wv):,} 個詞向量")
        st.json(trained["settings"], expanded=False)
        term = st.text_input("探索相近詞", "semaglutide")
        terms = tokenize(term, stemming=trained["stemming"])
        if len(terms) != 1:
            st.info("請輸入一個英文 token。")
        elif terms[0] not in model.wv:
            st.warning("此詞未在模型詞彙中；可降低 min_count 或換詞。")
            st.write("詞彙範例：" + "、".join(model.wv.index_to_key[:25]))
        else:
            neighbors = model.wv.most_similar(terms[0], topn=min(15, len(model.wv) - 1))
            a, b = st.columns([1, 2])
            a.dataframe(pd.DataFrame(neighbors, columns=["word", "cosine similarity"]), hide_index=True, width="stretch")
            projected = project_vectors(model, [terms[0]] + [w for w, _ in neighbors])
            fig = px.scatter(projected, x="PC1", y="PC2", text="word", color_discrete_sequence=["#007f78"])
            fig.update_traces(textposition="top center", marker_size=10)
            b.plotly_chart(fig, width="stretch")
            st.caption("PCA 是這組詞向量的二維投影；詞距不等同醫學證據或因果關係。小型單主題語料的相似詞可能不穩定。")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            vectors = pd.DataFrame(model.wv.vectors, index=model.wv.index_to_key)
            archive.writestr("vectors.csv", vectors.to_csv(index_label="word"))
            archive.writestr("settings.json", json.dumps(trained["settings"], indent=2))
        st.download_button("下載詞向量與訓練參數", buffer.getvalue(), "word2vec_vectors.zip", "application/zip")
    else:
        st.info("選擇參數後按「訓練 Word2Vec」。模型只在需要時訓練，讓首頁保持快速載入。")

elif page == "方法與展示":
    st.subheader("作業功能與示範路線")
    st.markdown("""
1. **文獻資料**：確認 1,000 個不同 PMID、摘要與查詢來源；下載清單與文章。
2. **Zipf 與 Porter**：比較原始與 stemming 後詞頻、詞彙數及 log-log 分布；切換停用詞觀察差異。
3. **文獻搜尋**：查 `semaglutide`；再查 `semaglutde` 示範拼字候選與近似匹配。展開結果查看命中位置。
4. **Word2Vec**：以相同語料分別訓練 Skip-gram / CBOW，查 `semaglutide`、`insulin`、`obesity`，比較相近詞。

**演算法與實作**

- Tokenizer 使用英文詞與連字號規則，統一大小寫；保留 `glp-1`，排除純數字。句子以標點與換行近似切分。
- Porter 1980 suffix stripping 由 `bioir/porter.py` 實作。英文字母 token 執行規則；GLP-1 等混合 token 保持原樣。
- 詞頻使用 Counter；Zipf 曲線以 log10(rank) / log10(frequency) 做 OLS，參考曲線為 f(1)/rank。
- Levenshtein 使用動態規劃，插入、刪除、替換成本均為 1。候選字依距離、語料頻次、字母排序。
- Word2Vec 使用 **Gensim**（不是自行實作神經網路訓練器），negative sampling=5、seed=42、workers=1，並使用穩定 hash 初始化。
- 所有比較共用目前選取的語料與前處理設定。Word2Vec 欄位變更後要再按訓練；已訓練參數顯示於模型下方。

**資料與重現性**

預載快照記錄查詢、排序、UTC 時間與 SHA-256；即時 PubMed 查詢會隨索引更新而改變。摘要內容權利屬原作者或出版者，逐篇保留來源及可取得的權利聲明。

**部署**

GitHub 保存程式與 JSONL，Streamlit Community Cloud 執行 `app.py`。部署入口與操作步驟見專案 README。雲端暫存與本機 session 資料不視為永久保存。
""")
    st.link_button("Porter 原始規則", "https://tartarus.org/martin/PorterStemmer/def.txt")
    st.link_button("Gensim Word2Vec 文件", "https://radimrehurek.com/gensim/models/word2vec.html")

st.divider()
st.caption("BioScope · Information Retrieval HW2 · NCBI PubMed abstracts · GLP-1")
