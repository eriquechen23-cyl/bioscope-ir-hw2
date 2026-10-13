"""October 6 Part 2 assignment: fixed, verified experiment results."""
import io
import json
from pathlib import Path
import zipfile

import pandas as pd
import plotly.express as px
import streamlit as st

from .embedding import corpus_fingerprint, training_samples
from .text import sentences

BACKGROUND = pd.DataFrame([
    {"Property": "Input", "CBOW": "周圍 context words 的平均向量", "Skip-Gram": "中心詞向量"},
    {"Property": "Prediction target", "CBOW": "中心詞", "Skip-Gram": "各個周圍 context words"},
    {"Property": "Training examples", "CBOW": "一組上下文 → 一個中心詞", "Skip-Gram": "一個中心詞 → 多組上下文配對"},
    {"Property": "Training speed", "CBOW": "通常較快；實測見上表", "Skip-Gram": "通常較慢；配對較多"},
    {"Property": "Rare words", "CBOW": "上下文平均可能稀釋稀有詞訊號", "Skip-Gram": "每次出現獨立預測多個鄰詞"},
    {"Property": "Expected behavior", "CBOW": "平滑上下文，常見詞與替代詞", "Skip-Gram": "更多局部共現／領域詞關係；仍受資料量限制"},
])


@st.cache_data(max_entries=2)
def load_fixed_results(root, revision):
    folder = Path(root) / "reports/embedding"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    tables = {name: pd.read_csv(folder / f"{name}.csv") for name in
              ["summary", "loss", "neighbors", "pairs", "projections", "window_comparison", "stability", "vocabulary"]}
    return manifest, tables


def render_embedding_assignment(records, root):
    st.subheader("Project 2 · Part 2 詞嵌入比較")
    folder = root / "reports/embedding"
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        st.info("固定實驗尚未建立；請執行 python -m scripts.build_embeddings。")
        return
    manifest, tables = load_fixed_results(str(root), manifest_path.stat().st_mtime_ns)
    if corpus_fingerprint(records) != manifest["corpus_sha256"]:
        st.info("固定實驗使用原始 1,000 篇摘要；目前選取的資料不同。切回預載 GLP-1 並選擇 1,000 篇查看正式比較，或到 Word2Vec 訓練目前資料。")
        return
    st.caption("2026-10-06 作業 · 同一語料比較 CBOW / Skip-Gram。正式結果只分析摘要、保留停用詞、不 stemming，與側欄的探索設定分開。")
    stats = manifest["corpus_statistics"]
    st.success(f"已完成 6 個模型 · {stats['sentences']:,} 個句子 · min_count=3 後 {stats['retained_vocabulary']:,} 個詞向量")
    summary = tables["summary"]
    st.dataframe(summary[["model", "architecture", "window", "seed", "vector_size", "epochs", "vocabulary", "elapsed_seconds"]], hide_index=True, width="stretch")
    st.caption(f"100 維、{int(summary.epochs.iloc[0])} epochs、negative=5、workers=1；seed=42 主實驗，seed=7 穩定性檢查。秒數為本機單次量測，包含建詞表，非跨平台速度排名。")
    tabs = st.tabs(["架構與樣本", "近鄰與詞對", "PCA / t-SNE", "參數與穩定性", "RQ1–RQ5 與下載"])
    with tabs[0]:
        st.dataframe(BACKGROUND, hide_index=True, width="stretch")
        st.markdown("#### 從摘要生成訓練樣本")
        corpus = sentences(records)
        index = st.selectbox("示範句子", range(min(30, len(corpus))), format_func=lambda i: " ".join(corpus[i])[:100])
        window = st.selectbox("示範 context window", [2, 5], index=1)
        tokens = corpus[index]
        st.code(" ".join(tokens))
        cbow, skipgram = training_samples(tokens, window)
        a, b = st.columns(2)
        a.markdown("**CBOW：上下文 → 中心詞**")
        a.dataframe(pd.DataFrame(cbow, columns=["context", "target"]).assign(context=lambda d: d.context.map(" / ".join)), hide_index=True, width="stretch")
        b.markdown("**Skip-Gram：中心詞 → 上下文詞**")
        b.dataframe(pd.DataFrame(skipgram), hide_index=True, width="stretch", height=350)
        st.caption("示例是固定 window 的原始 tokens。實際 Gensim 訓練另有 min_count 過濾、sample=0.001 高頻詞 subsampling、shrink_windows=True 隨機有效視窗，因此不是實際更新次數。")
        st.markdown("#### 高頻詞與詞彙表")
        st.dataframe(tables["vocabulary"].head(30), hide_index=True, width="stretch")
        st.dataframe(summary[["model", "nominal_cbow_examples", "nominal_skipgram_pairs"]], hide_index=True, width="stretch")
        st.caption("nominal：min_count 過濾後、尚未 subsampling／縮小視窗的每 epoch 樣本上限；CBOW 只計有上下文的中心位置。")
    with tabs[1]:
        term = st.selectbox("比較指定詞（10 詞）", manifest["query_words"], index=1)
        base_neighbors = tables["neighbors"][tables["neighbors"].model.isin(["cbow_w5_s42", "skipgram_w5_s42"])]
        for col, model in zip(st.columns(2), ["cbow_w5_s42", "skipgram_w5_s42"]):
            col.markdown(f"**{model} · Top 5**")
            col.dataframe(base_neighbors[(base_neighbors.model == model) & (base_neighbors.word == term)][["neighbor", "cosine", "frequency", "neighbor_frequency"]], hide_index=True, width="stretch")
        st.markdown("#### 至少 5 組相關詞對 · 原始 100 維 cosine")
        base_pairs = tables["pairs"][tables["pairs"].model.isin(["cbow_w5_s42", "skipgram_w5_s42"])]
        st.dataframe(base_pairs, hide_index=True, width="stretch")
        with st.expander("完整 10 詞 × 2 模型 × Top 5 與低頻詞"):
            st.dataframe(base_neighbors, hide_index=True, width="stretch")
        st.caption("cosine 衡量學到的共現方向，非醫學因果或語意準確率。OOV 以狀態與空值表示，不補造向量。")
    with tabs[2]:
        projection_window = st.selectbox("投影模型 window", [5, 2])
        for method in ["PCA", "t-SNE"]:
            st.markdown(f"#### {method} · 相同 {len(manifest['projection_words'])} 詞")
            for col, prefix in zip(st.columns(2), ["cbow", "skipgram"]):
                name = f"{prefix}_w{projection_window}_s42"
                data = tables["projections"]
                frame = data[(data.model == name) & (data.method == method)]
                fig = px.scatter(frame, x="x", y="y", text="word", title=name,
                                 color_discrete_sequence=["#007f78" if prefix == "cbow" else "#bd642c"])
                fig.update_traces(textposition="top center", marker_size=9)
                fig.update_layout(height=430, xaxis_title=f"{method} 1", yaxis_title=f"{method} 2")
                col.plotly_chart(fig, width="stretch", key=f"embedding-{name}-{method}")
                run = next(r for r in manifest["runs"] if r["model"] == name)
                meta = next(p for p in run["projections"] if p["method"] == method)
                if method == "PCA":
                    col.caption(f"兩軸解釋變異 {sum(meta['explained_variance_ratio']):.1%}；trustworthiness(k=5)={meta['trustworthiness']:.3f}")
                else:
                    col.caption(f"perplexity={meta['perplexity']:g}；cosine metric；seed=42；trustworthiness(k=5)={meta['trustworthiness']:.3f}")
        st.caption("投影前 L2 正規化。PCA 為線性全域投影；t-SNE 強調局部鄰近。各圖獨立擬合，軸與群間距離不能跨圖直接比較；trustworthiness 只量測投影保留鄰近程度，非語意品質。可互動縮放與 hover 讀取標籤。")
    with tabs[3]:
        st.markdown("#### 每 epoch 訓練損失")
        loss = tables["loss"][tables["loss"].model.str.endswith("s42")]
        fig = px.line(loss, x="epoch", y="epoch_loss", color="model", markers=True)
        st.plotly_chart(fig, width="stretch")
        st.caption("epoch_loss = 本 epoch 結束的累積 loss − 上次累積 loss；負採樣任務與更新量不同，不能以 CBOW／Skip-Gram 的 loss 大小直接排名語意品質。")
        st.markdown("#### RQ4 · window 2 vs 5（其他參數相同）")
        st.dataframe(tables["window_comparison"], hide_index=True, width="stretch")
        st.dataframe(tables["pairs"][tables["pairs"].model.str.endswith("s42")], hide_index=True, width="stretch")
        st.markdown("#### RQ5 · seed 42 vs 7 的穩定性")
        st.dataframe(tables["stability"], hide_index=True, width="stretch")
        st.caption("overlap_at_5 = 兩次 Top 5 的共同詞數 / 5；rare 組為預先選定、頻次 3–20 的 7 個醫學詞。只測兩個 seed，非充分統計證據。")
    with tabs[4]:
        findings = folder / "findings_zh.md"
        if findings.exists():
            st.markdown(findings.read_text(encoding="utf-8"))
            st.download_button("下載 Part 2 完整問題回答", findings.read_bytes(), "part2_findings_zh.md", "text/markdown")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(folder.iterdir()):
                if file.suffix in {".csv", ".json", ".md", ".png", ".svg"}:
                    archive.writestr(file.name, file.read_bytes())
        st.download_button("下載 Part 2 實驗資料與圖表", buffer.getvalue(), "part2_experiments.zip", "application/zip")
        st.download_button("下載 6 個模型完整詞向量 NPZ", (folder / "vectors.npz").read_bytes(), "part2_vectors.npz", "application/octet-stream")
        st.json(manifest, expanded=False)
