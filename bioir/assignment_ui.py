"""Streamlit view for the September 29 required assignment."""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from .experiments import CONDITIONS, run_experiments, select_comparison_terms, term_statistics


@st.cache_data(max_entries=8)
def cached_experiments(records):
    return run_experiments(records)


def render_assignment(records, root: Path):
    st.subheader("Project 2 · 必做實驗")
    st.caption("使用目前所選文章的摘要；A–D 的前處理固定如下，不受側欄停用詞／標題選項影響。正式報告使用原始 1,000 篇快照。Optional Challenge（兩領域比較）省略。")
    experiment = cached_experiments(records)
    summary = experiment["summary"]
    st.markdown("**A** 分詞、小寫、保留獨立標點 → **B** 移除獨立標點 → **C** 再移除停用詞 → **D** 再做 Porter stemming。全部條件排除純數字，保留 GLP-1 等詞內連字號。")
    st.dataframe(summary, hide_index=True, width="stretch")
    st.caption("tokens = 全語料詞次；vocabulary = 不同詞彙數；average_tokens_per_document = tokens / documents。迴歸使用 log10；IDF 使用 ln(N/DF)。")
    st.download_button("下載四條件比較", summary.to_csv(index=False).encode("utf-8-sig"), "preprocessing_comparison.csv")
    condition = st.selectbox("檢視實驗條件", list(CONDITIONS), index=1, format_func=lambda key: f"{key} · {CONDITIONS[key]}")
    result = experiment["conditions"][condition]
    table, fitted = result["terms"], result["fit"]
    if len(table) < 2:
        st.info("可分析詞彙不足，請選擇更多文章。")
        return
    left, right = st.columns(2)
    rank_plot = go.Figure(go.Scatter(x=table["rank"], y=table["CF"], mode="lines", line_color="#007f78", name="Observed"))
    rank_plot.update_layout(title="Experiment 1 · Rank vs CF", xaxis_title="Rank (linear)", yaxis_title="Collection frequency (linear)", height=360)
    left.plotly_chart(rank_plot, width="stretch")
    log_plot = go.Figure()
    log_plot.add_trace(go.Scatter(x=table["rank"], y=table["CF"], mode="lines", name="Observed", line_color="#007f78"))
    log_plot.add_trace(go.Scatter(x=table["rank"], y=10 ** fitted["intercept"] * table["rank"] ** fitted["slope"], name="OLS fit", line={"color": "#e48443", "dash": "dash"}))
    log_plot.update_layout(title="Experiment 2 · Log-log & regression", xaxis_type="log", yaxis_type="log", xaxis_title="Rank", yaxis_title="CF", height=360, legend={"orientation": "h"})
    right.plotly_chart(log_plot, width="stretch")
    st.dataframe(pd.DataFrame([fitted]), hide_index=True, width="stretch")
    st.caption("log10(CF) = intercept + slope × log10(rank)；Zipf exponent k = −slope；RMSE 在 log10(CF) 單位上計算。")
    overlay = go.Figure()
    for key, values in experiment["conditions"].items():
        terms = values["terms"]
        overlay.add_trace(go.Scatter(x=terms["rank"], y=terms["CF"], name=key, mode="lines"))
    overlay.update_layout(title="A–D 前處理分布比較", xaxis_type="log", yaxis_type="log", xaxis_title="Rank", yaxis_title="CF", height=370)
    st.plotly_chart(overlay, width="stretch")
    st.markdown("#### 全域排名的三區比較")
    st.dataframe(result["regions"], hide_index=True, width="stretch")
    st.caption("預先固定前 10%、中間 80%、後 10%；保留全域 rank。若 CF 完全相同，R² 為未定義；低 RMSE 的水平平台不代表符合 k≈1。不同區間的 R² 不能單獨用來排名擬合品質。")
    st.markdown("#### Top 50 CF terms")
    st.dataframe(table.head(50), hide_index=True, width="stretch", height=460)
    st.download_button("下載完整 CF / DF / IDF", table.to_csv(index=False).encode("utf-8-sig"), f"condition_{condition}_terms.csv")
    st.markdown("#### CF vs DF 與 IDF（至少 20 個詞）")
    comparison = select_comparison_terms(table)
    st.dataframe(comparison, hide_index=True, width="stretch")
    st.markdown("CF 表示詞次總量，DF 表示出現於多少篇文件；反覆出現在少數文章中的詞可以有高 CF、低 DF。IDF 依 **DF** 計算，不能只從 CF 判定。")
    st.markdown("#### 單篇 TF-IDF 示範")
    chosen = st.selectbox("選擇文件（PMID）", range(len(records)), format_func=lambda i: records[i]["pmid"], key="tfidf_document")
    _, _, counts = term_statistics([records[chosen]], condition)
    document = table[table["term"].isin(counts[0])].copy()
    document["TF"] = document["term"].map(counts[0])
    document["TF_IDF"] = document["TF"] * document["IDF"]
    st.dataframe(document[["term", "TF", "CF", "DF", "IDF", "TF_IDF"]].sort_values("TF_IDF", ascending=False).head(20), hide_index=True, width="stretch")
    st.caption("TF 採此篇原始出現次數；IDF 仍以目前整個選取語料計算，未做長度正規化。")
    st.markdown("#### 正式作業報告（原始 1,000 篇）")
    for filename, label, mime in [("project2_report.pdf", "下載完整報告 PDF", "application/pdf"),
                                  ("executive_summary.pdf", "下載一頁 Executive Summary", "application/pdf"),
                                  ("project2_report.html", "下載離線 HTML 報告", "text/html")]:
        path = root / "reports/assignment" / filename
        if path.exists():
            st.download_button(label, path.read_bytes(), filename, mime)
    conclusions = root / "reports/assignment/findings_zh.md"
    if conclusions.exists():
        with st.expander("研究問題 RQ1–RQ5 與 Final Challenge（固定快照結果）", expanded=False):
            st.markdown(conclusions.read_text(encoding="utf-8"))
