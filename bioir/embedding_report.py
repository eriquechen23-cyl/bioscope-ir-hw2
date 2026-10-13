"""Generate evidence-based Part 2 answers directly from experiment artifacts."""
import json
from pathlib import Path
import pandas as pd


def markdown_table(frame):
    def cell(value):
        if isinstance(value, float):
            return "OOV / undefined" if pd.isna(value) else f"{value:.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"]*len(frame.columns)) + " |"] +
                      ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)])


def write_findings(output):
    output = Path(output)
    m = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    stats = m["corpus_statistics"]
    tables = {name: pd.read_csv(output / f"{name}.csv") for name in ["neighbors", "pairs", "window_comparison", "stability", "vocabulary"]}
    base = [r for r in m["runs"] if r["window"] == 5 and r["seed"] == 42]
    cbow, sg = base
    pairs = tables["pairs"][tables["pairs"].model.isin([r["model"] for r in base])]
    pair_wide = pairs.pivot(index=["left", "right"], columns="model", values="cosine").reset_index()
    windows = tables["window_comparison"][tables["window_comparison"].word.isin(m["query_words"])].groupby("architecture").overlap_at_5.mean()
    stability = tables["stability"].groupby(["architecture", "group"]).overlap_at_5.mean()
    hapax = int((tables["vocabulary"].frequency == 1).sum())
    neighbor_rows = []
    for model in [r["model"] for r in base]:
        for word in m["query_words"]:
            selected = tables["neighbors"][(tables["neighbors"].model == model) & (tables["neighbors"].word == word)]
            neighbor_rows.append({"model": model, "word": word, "Top 5 (cosine)": "; ".join(
                f"{row.neighbor} ({row.cosine:.3f})" if row.status == "ok" else "OOV" for row in selected.itertuples())})
    loss_rows = [{"model": r["model"], "seconds": r["elapsed_seconds"],
                  "epoch_1_loss": r["loss"][0]["epoch_loss"], "last_epoch_loss": r["loss"][-1]["epoch_loss"],
                  "cumulative_loss": r["loss"][-1]["cumulative_loss"]} for r in base]
    projection_rows = []
    projection_comparisons = []
    for r in base:
        score = {p["method"]: p["trustworthiness"] for p in r["projections"]}
        relation = "高於" if score["t-SNE"] > score["PCA"] else "低於或等於"
        projection_comparisons.append(f"{r['architecture']} 的 t-SNE trustworthiness {relation} PCA")
        for p in r["projections"]:
            projection_rows.append({"model": r["model"], "method": p["method"],
                                    "PCA_two_axis_variance": sum(p["explained_variance_ratio"]) if p["method"] == "PCA" else float("nan"),
                                    "trustworthiness_k5": p["trustworthiness"]})
    examples = []
    gastroparesis_count = int(tables["vocabulary"].set_index("word").frequency.get("gastroparesis", 0))
    for model in [r["model"] for r in base]:
        for word in ["semaglutide", "insulin", "gastroparesis"]:
            selected = tables["neighbors"][(tables["neighbors"].model == model) & (tables["neighbors"].word == word) & (tables["neighbors"].status == "ok")].head(3)
            examples.append(f"- {model} / {word}：" + "、".join(f"{row.neighbor} ({row.cosine:.3f})" for row in selected.itertuples()))
    text = f"""# Project 2 Part 2：CBOW 與 Skip-Gram 實驗與問題回答

## 中央問題：1,000 篇科學摘要能學到有意義的詞表示嗎？

可以得到部分符合領域語境的共現關係，但不足以證明普遍、穩定的語意表示。相近詞可以是替代詞、品牌、搭配詞或共同研究主題；cosine、低 loss 和二維群聚均不等於語意準確率。本報告的數值由實際模型輸出產生，未使用外部預訓練向量。

## 語料、前處理與可重現設定

原始 GLP-1 PubMed 快照：{stats['documents']:,} 篇、{stats['sentences']:,} 個近似句子、{stats['tokens']:,} tokens、{stats['vocabulary']:,} 詞；min_count=3 後保留 {stats['retained_vocabulary']:,} 詞。英文小寫，去除獨立標點與純數字，保留 glp-1 等連字號詞；保留停用詞、不 stemming、只用摘要，句界不跨文章。句界採標點／換行近似，並非完整生醫分句器。

基準：{cbow['vector_size']} 維、window=5、min_count=3、{cbow['epochs']} epochs、negative=5、sample=0.001、workers=1、seed=42、SHA-256 穩定 hash、CBOW mean=1、alpha=0.025 → min_alpha=0.0001、shrink_windows=True。四主模型是兩架構 × window 2/5；另以 seed=7 重訓兩個 window=5 模型。完整版本、參數、語料 fingerprint 與每 epoch loss 均在 manifest.json。跨套件／平台可能有數值差異。

## 背景比較表（PDF 第三頁）

| Property | CBOW | Skip-Gram |
| --- | --- | --- |
| Input | 周圍詞的平均向量 | 中心詞向量 |
| Prediction target | 中心詞 | 周圍各個 context words |
| Training examples | context bag → target；每個有效中心位置一筆 | target → context；每個鄰詞各一組配對 |
| Training speed | 本機基準 {cbow['elapsed_seconds']:.2f} 秒 | 本機基準 {sg['elapsed_seconds']:.2f} 秒 |
| Rare words | 多詞平均會稀釋個別稀有語境訊號 | 稀有中心詞每次出現可產生多次上下文預測 |
| Expected behavior | 較快、平滑上下文；可能聚集替代詞 | 較多局部共現／搭配詞；通常計算量較高 |

window=5、min_count 過濾後每 epoch 的固定視窗**名義上限**：CBOW {cbow['nominal_cbow_examples']:,} 筆中心樣本，Skip-Gram {sg['nominal_skipgram_pairs']:,} 組配對。這些數量先於 subsampling 與動態 window，不能當成實際 SGD 更新次數。速度是同機單次訓練（含建詞表），未做多次計時，不作硬體無關 benchmark。

## RQ1：CBOW 與 Skip-Gram 有何差異？

兩架構的輸入／預測方向與樣本量如上。此次 Skip-Gram 基準耗時約為 CBOW 的 {sg['elapsed_seconds']/cbow['elapsed_seconds']:.2f} 倍；訓練 loss 均記錄 epoch 結束的累積值及相鄰差分。

{markdown_table(pd.DataFrame(loss_rows))}

loss 是 Gensim SGNS 負採樣目標的累積量；每 epoch 損失增量下降可以反映該訓練目標的變化，但更新量、視窗及架構不同，不能用原始 loss 大小直接宣稱 CBOW 語意較好。需同時分析近鄰、詞對、低頻詞與穩定性。

{chr(10).join(examples)}

在此固定 GLP-1 快照中，CBOW 的 semaglutide 近鄰包含多個藥名；Skip-Gram 的近鄰出現品牌／局部搭配詞。insulin 的近鄰也呈現架構差異。低頻案例 gastroparesis 出現 {gastroparesis_count} 次，結果仍可能反映少數文章的共現。這些是此領域實測現象，不是任意小語料必有的結果。

## RQ2：詞嵌入是否具有語意意義？

以下五組預先選定詞對，在**原始 100 維**中計算 cosine；不從二維座標反推相似度。

{markdown_table(pair_wide)}

semaglutide/liraglutide 反映藥名共享語境；insulin/glucose 反映代謝研究關聯；diabetes/obesity 為共同疾病主題；obesity/weight 為疾病與研究指標搭配；receptor/agonist 反映受體與作用詞的共同語境。五組有正向相似度並不等於同義詞辨識成功；一些詞對未進 Top 5，且不同模型分數未校準。最有力的質性證據是多個可解釋近鄰，而非只選最大 cosine。

至少十個指定詞、兩模型各 Top 5 如下；低頻詞結果另在 neighbors.csv。

{markdown_table(pd.DataFrame(neighbor_rows))}

## RQ3：PCA 與 t-SNE 視覺化有何差異？

所有投影使用相同 {len(m['projection_words'])} 詞及 L2 正規化向量。PCA 是線性全域變異投影，兩軸只保留部分變異；t-SNE 用 cosine 鄰近、perplexity={cbow['projections'][1]['perplexity']:g}、random init、seed=42、1,000 iterations，強調局部鄰近。

{markdown_table(pd.DataFrame(projection_rows))}

在基準圖中，可觀察藥名、diabetes/obesity 與 gastric/emptying 等局部關係；PCA 和 t-SNE 的群位置、分離度與鄰近關係不同。此次{'；'.join(projection_comparisons)}。較高的 trustworthiness 表示對這批選詞的原向量鄰近保留較多，**並不評估原向量是否有正確語意**。PCA 丟失變異，t-SNE 受 perplexity、選詞及隨機初始化影響，群間距離與面積不宜解讀。不同圖的座標獨立擬合，不能以位置移動大小跨圖量測參數效應。

## RQ4：模型參數如何影響結果？

固定語料、前處理、維度、epochs、seed 與 min_count，只比較 window=2 和 5。十個指定詞的 Top-5 平均重疊率為 CBOW {windows['cbow']:.1%}、Skip-Gram {windows['skipgram']:.1%}。overlap@5 = 共同詞數 / 5；較低代表鄰居改變較多，不代表品質變差。

window=2 較重視近距離搭配，window=5 納入更多周圍主題詞；此為上下文機制的解釋，須以表中實際變化核對，不能假定每詞都遵守此模式。完整逐詞重疊、四模型詞對與兩組投影分別在 window_comparison.csv、pairs.csv、projections_w2.png、projections_w5.png。投影重新擬合，觀察群聚變化只能作輔助；100 維近鄰與 cosine 是主要比較依據。

## RQ5：小型語料的限制為何？

{stats['vocabulary']:,} 詞中有 {hapax:,} 個只出現一次（{hapax/stats['vocabulary']:.1%}）；min_count=3 排除 {stats['vocabulary']-stats['retained_vocabulary']:,} 個詞。低頻術語可能完全 OOV，或只學到少數文章的局部語境；連字號、縮寫、單複數與詞形變化又會分散計數。單一 GLP-1 領域資料不能代表其他科學領域，也未做人工標註或下游任務評価。

固定 window=5，seed 42 與 7 的十個常用查詢詞近鄰平均重疊：CBOW {stability.get(('cbow','query'), float('nan')):.1%}、Skip-Gram {stability.get(('skipgram','query'), float('nan')):.1%}；七個低頻醫學詞（頻次 3–20）：CBOW {stability.get(('cbow','rare (3-20)'), float('nan')):.1%}、Skip-Gram {stability.get(('skipgram','rare (3-20)'), float('nan')):.1%}。

{markdown_table(tables['stability'][tables['stability'].group == 'rare (3-20)'][['architecture','word','frequency','overlap_at_5','shared_neighbors','status']])}

兩個 seed 只提供有限穩定性檢查；若要建立可信的泛化結論，應增加及多樣化語料、執行更多 seed、加入領域詞義人工評估與獨立下游驗證。這不是 1,000 篇永遠不足，而是本次證據範圍有限。

## 為什麼 Skip-Gram 特別適合低頻或領域專業詞？

稀有詞作為中心詞時，Skip-Gram 對每個 context word 產生獨立預測訊號；CBOW 先平均上下文，個別稀有詞訊號可能被其他詞稀釋。此次七低頻詞的 seed 重疊結果是支持觀察，但不是語意準確率或統計顯著的優勢；例如 tumor 的近鄰仍混入難以解釋的詞。低頻詞被 min_count 排除後，兩模型都沒有向量，Skip-Gram 無法補救未觀測語境。

## 實作、交付與參考

- Word2Vec 使用 Gensim 真實訓練，並非自寫神經網路；context 樣本生成、比較、loss callback 與實驗輸出由本專案實作。
- vectors.npz 保存六個模型全部輸入詞向量；不含輸出權重或 optimizer 狀態，不能用來繼續訓練。可重算 cosine、近鄰與投影。
- 訓練樣本示例使用固定 window；Gensim 的高頻 subsampling 和 dynamic window 會改變實際樣本。沒有把名義樣本數誤稱實際訓練 updates。
- [GitHub](https://github.com/eriquechen23-cyl/bioscope-ir-hw2) · [Streamlit](https://bioscope-glp1-ir.streamlit.app/)
- [Mikolov et al. 2013, Efficient Estimation](https://arxiv.org/abs/1301.3781) · [Distributed Representations](https://arxiv.org/abs/1310.4546)
- [Gensim Word2Vec 文件](https://radimrehurek.com/gensim/models/word2vec.html) · [scikit-learn t-SNE 文件](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
- 作業依據：2026-10-06 Part 2 PDF，已核對三頁；正式分析以保存快照為準。
"""
    (output / "findings_zh.md").write_text(text, encoding="utf-8")
    return text
