# Project 2 Part 2：CBOW 與 Skip-Gram 實驗與問題回答

## 中央問題：1,000 篇科學摘要能學到有意義的詞表示嗎？

可以得到部分符合領域語境的共現關係，但不足以證明普遍、穩定的語意表示。相近詞可以是替代詞、品牌、搭配詞或共同研究主題；cosine、低 loss 和二維群聚均不等於語意準確率。本報告的數值由實際模型輸出產生，未使用外部預訓練向量。

## 語料、前處理與可重現設定

原始 GLP-1 PubMed 快照：1,000 篇、7,267 個近似句子、161,515 tokens、10,691 詞；min_count=3 後保留 4,672 詞。英文小寫，去除獨立標點與純數字，保留 glp-1 等連字號詞；保留停用詞、不 stemming、只用摘要，句界不跨文章。句界採標點／換行近似，並非完整生醫分句器。

基準：100 維、window=5、min_count=3、20 epochs、negative=5、sample=0.001、workers=1、seed=42、SHA-256 穩定 hash、CBOW mean=1、alpha=0.025 → min_alpha=0.0001、shrink_windows=True。四主模型是兩架構 × window 2/5；另以 seed=7 重訓兩個 window=5 模型。完整版本、參數、語料 fingerprint 與每 epoch loss 均在 manifest.json。跨套件／平台可能有數值差異。

## 背景比較表（PDF 第三頁）

| Property | CBOW | Skip-Gram |
| --- | --- | --- |
| Input | 周圍詞的平均向量 | 中心詞向量 |
| Prediction target | 中心詞 | 周圍各個 context words |
| Training examples | context bag → target；每個有效中心位置一筆 | target → context；每個鄰詞各一組配對 |
| Training speed | 本機基準 2.33 秒 | 本機基準 6.96 秒 |
| Rare words | 多詞平均會稀釋個別稀有語境訊號 | 稀有中心詞每次出現可產生多次上下文預測 |
| Expected behavior | 較快、平滑上下文；可能聚集替代詞 | 較多局部共現／搭配詞；通常計算量較高 |

window=5、min_count 過濾後每 epoch 的固定視窗**名義上限**：CBOW 153,898 筆中心樣本，Skip-Gram 1,321,598 組配對。這些數量先於 subsampling 與動態 window，不能當成實際 SGD 更新次數。速度是同機單次訓練（含建詞表），未做多次計時，不作硬體無關 benchmark。

## RQ1：CBOW 與 Skip-Gram 有何差異？

兩架構的輸入／預測方向與樣本量如上。此次 Skip-Gram 基準耗時約為 CBOW 的 2.98 倍；訓練 loss 均記錄 epoch 結束的累積值及相鄰差分。

| model | seconds | epoch_1_loss | last_epoch_loss | cumulative_loss |
| --- | --- | --- | --- | --- |
| cbow_w5_s42 | 2.3330 | 416820.7500 | 176248.0000 | 4862793.5000 |
| skipgram_w5_s42 | 6.9563 | 1880853.8750 | 790784.0000 | 20920880.0000 |

loss 是 Gensim SGNS 負採樣目標的累積量；每 epoch 損失增量下降可以反映該訓練目標的變化，但更新量、視窗及架構不同，不能用原始 loss 大小直接宣稱 CBOW 語意較好。需同時分析近鄰、詞對、低頻詞與穩定性。

- cbow_w5_s42 / semaglutide：tirzepatide (0.852)、dulaglutide (0.824)、liraglutide (0.823)
- cbow_w5_s42 / insulin：glucose-stimulated (0.653)、stimulating (0.628)、glucagon (0.624)
- cbow_w5_s42 / gastroparesis：persist (0.819)、drug-induced (0.816)、shared (0.814)
- skipgram_w5_s42 / semaglutide：dulaglutide (0.673)、rybelsus (0.669)、tirzepatide (0.659)
- skipgram_w5_s42 / insulin：resistance (0.625)、glucose-induced (0.572)、augmentation (0.547)
- skipgram_w5_s42 / gastroparesis：regurgitation (0.877)、peri-operatively (0.841)、aspiration (0.826)

在此固定 GLP-1 快照中，CBOW 的 semaglutide 近鄰包含多個藥名；Skip-Gram 的近鄰出現品牌／局部搭配詞。insulin 的近鄰也呈現架構差異。低頻案例 gastroparesis 出現 4 次，結果仍可能反映少數文章的共現。這些是此領域實測現象，不是任意小語料必有的結果。

## RQ2：詞嵌入是否具有語意意義？

以下五組預先選定詞對，在**原始 100 維**中計算 cosine；不從二維座標反推相似度。

| left | right | cbow_w5_s42 | skipgram_w5_s42 |
| --- | --- | --- | --- |
| diabetes | obesity | 0.6106 | 0.5220 |
| insulin | glucose | 0.4495 | 0.4126 |
| obesity | weight | 0.2799 | 0.3341 |
| receptor | agonist | 0.3263 | 0.5258 |
| semaglutide | liraglutide | 0.8228 | 0.4810 |

semaglutide/liraglutide 反映藥名共享語境；insulin/glucose 反映代謝研究關聯；diabetes/obesity 為共同疾病主題；obesity/weight 為疾病與研究指標搭配；receptor/agonist 反映受體與作用詞的共同語境。五組有正向相似度並不等於同義詞辨識成功；一些詞對未進 Top 5，且不同模型分數未校準。最有力的質性證據是多個可解釋近鄰，而非只選最大 cosine。

至少十個指定詞、兩模型各 Top 5 如下；低頻詞結果另在 neighbors.csv。

| model | word | Top 5 (cosine) |
| --- | --- | --- |
| cbow_w5_s42 | glp-1 | glucagon (0.615); biased (0.537); bitter (0.537); gpr (0.533); insulinotropic (0.532) |
| cbow_w5_s42 | semaglutide | tirzepatide (0.852); dulaglutide (0.824); liraglutide (0.823); exenatide (0.795); albiglutide (0.738) |
| cbow_w5_s42 | liraglutide | dulaglutide (0.843); exenatide (0.839); semaglutide (0.823); lixisenatide (0.785); tirzepatide (0.755) |
| cbow_w5_s42 | insulin | glucose-stimulated (0.653); stimulating (0.628); glucagon (0.624); diurnal (0.600); postprandial (0.579) |
| cbow_w5_s42 | glucose | sugar (0.751); postprandial (0.707); lipid (0.688); carbohydrate (0.671); blood (0.665) |
| cbow_w5_s42 | diabetes | wild (0.714); mellitus (0.653); treating (0.641); ii (0.632); nafld (0.618) |
| cbow_w5_s42 | obesity | mellitus (0.810); managing (0.754); type (0.751); treating (0.749); nafld (0.746) |
| cbow_w5_s42 | weight | hair (0.766); temperature (0.660); ketone (0.614); meaningful (0.580); glycaemic (0.573) |
| cbow_w5_s42 | receptor | glp-1r (0.738); biased (0.665); full (0.659); dual (0.641); analogs (0.605) |
| cbow_w5_s42 | agonist | agonism (0.773); analog (0.740); long-acting (0.728); gcgr (0.726); antagonist (0.715) |
| skipgram_w5_s42 | glp-1 | unlabelled (0.666); peptide-2 (0.615); elicits (0.611); atropine (0.603); centrally (0.596) |
| skipgram_w5_s42 | semaglutide | dulaglutide (0.673); rybelsus (0.669); tirzepatide (0.659); ozempic (0.655); wegovy (0.653) |
| skipgram_w5_s42 | liraglutide | exenatide (0.681); dulaglutide (0.646); saxenda (0.623); lixisenatide (0.613); degludec (0.589) |
| skipgram_w5_s42 | insulin | resistance (0.625); glucose-induced (0.572); augmentation (0.547); stimulating (0.531); sensitivity (0.515) |
| skipgram_w5_s42 | glucose | tolerance (0.680); carbohydrate (0.629); sugar (0.619); lipid (0.615); blood-glucose (0.608) |
| skipgram_w5_s42 | diabetes | mellitus (0.804); type (0.702); dm (0.691); revolutionized (0.679); dmt (0.671) |
| skipgram_w5_s42 | obesity | landscape (0.588); intensification (0.577); sid (0.576); osa (0.571); revolutionized (0.565) |
| skipgram_w5_s42 | weight | loss (0.799); gain (0.711); meaningful (0.622); reduction (0.589); glycated (0.582) |
| skipgram_w5_s42 | receptor | weak (0.627); co-agonists (0.623); unlabelled (0.607); triple (0.584); marketed (0.579) |
| skipgram_w5_s42 | agonist | gcgr (0.639); agonists (0.597); analogue (0.596); cck-1 (0.593); tiab (0.588) |

## RQ3：PCA 與 t-SNE 視覺化有何差異？

所有投影使用相同 22 詞及 L2 正規化向量。PCA 是線性全域變異投影，兩軸只保留部分變異；t-SNE 用 cosine 鄰近、perplexity=7、random init、seed=42、1,000 iterations，強調局部鄰近。

| model | method | PCA_two_axis_variance | trustworthiness_k5 |
| --- | --- | --- | --- |
| cbow_w5_s42 | PCA | 0.3911 | 0.8825 |
| cbow_w5_s42 | t-SNE | OOV / undefined | 0.9117 |
| skipgram_w5_s42 | PCA | 0.2827 | 0.8435 |
| skipgram_w5_s42 | t-SNE | OOV / undefined | 0.8838 |

在基準圖中，可觀察藥名、diabetes/obesity 與 gastric/emptying 等局部關係；PCA 和 t-SNE 的群位置、分離度與鄰近關係不同。此次CBOW 的 t-SNE trustworthiness 高於 PCA；Skip-gram 的 t-SNE trustworthiness 高於 PCA。較高的 trustworthiness 表示對這批選詞的原向量鄰近保留較多，**並不評估原向量是否有正確語意**。PCA 丟失變異，t-SNE 受 perplexity、選詞及隨機初始化影響，群間距離與面積不宜解讀。不同圖的座標獨立擬合，不能以位置移動大小跨圖量測參數效應。

## RQ4：模型參數如何影響結果？

固定語料、前處理、維度、epochs、seed 與 min_count，只比較 window=2 和 5。十個指定詞的 Top-5 平均重疊率為 CBOW 66.0%、Skip-Gram 50.0%。overlap@5 = 共同詞數 / 5；較低代表鄰居改變較多，不代表品質變差。

window=2 較重視近距離搭配，window=5 納入更多周圍主題詞；此為上下文機制的解釋，須以表中實際變化核對，不能假定每詞都遵守此模式。完整逐詞重疊、四模型詞對與兩組投影分別在 window_comparison.csv、pairs.csv、projections_w2.png、projections_w5.png。投影重新擬合，觀察群聚變化只能作輔助；100 維近鄰與 cosine 是主要比較依據。

## RQ5：小型語料的限制為何？

10,691 詞中有 4,434 個只出現一次（41.5%）；min_count=3 排除 6,019 個詞。低頻術語可能完全 OOV，或只學到少數文章的局部語境；連字號、縮寫、單複數與詞形變化又會分散計數。單一 GLP-1 領域資料不能代表其他科學領域，也未做人工標註或下游任務評価。

固定 window=5，seed 42 與 7 的十個常用查詢詞近鄰平均重疊：CBOW 68.0%、Skip-Gram 66.0%；七個低頻醫學詞（頻次 3–20）：CBOW 57.1%、Skip-Gram 68.6%。

| architecture | word | frequency | overlap_at_5 | shared_neighbors | status |
| --- | --- | --- | --- | --- | --- |
| cbow | cardiomyopathy | 5 | 0.4000 | nephropathy, retinopathy | ok |
| cbow | diagnosis | 11 | 0.6000 | collaborative, evidence-based, timely | ok |
| cbow | gastroparesis | 4 | 0.4000 | drug-induced, persist | ok |
| cbow | nephropathy | 4 | 0.4000 | cardiomyopathy, degeneration | ok |
| cbow | neuropathy | 10 | 0.8000 | anterior, mi, nonfatal, optic | ok |
| cbow | retinopathy | 13 | 0.8000 | cardiomyopathy, dr, eye, nephropathy | ok |
| cbow | tumor | 11 | 0.6000 | interleukin-6, necrosis, tnf | ok |
| skipgram | cardiomyopathy | 5 | 0.6000 | arrhythmias, cardiomyopathies, eye | ok |
| skipgram | diagnosis | 11 | 0.4000 | acnes, bipolar | ok |
| skipgram | gastroparesis | 4 | 0.8000 | aspiration, delayed, peri-operatively, regurgitation | ok |
| skipgram | nephropathy | 4 | 0.6000 | comorbidity, dyslipidaemia, microvascular | ok |
| skipgram | neuropathy | 10 | 1.0000 | anterior, ischemic, non-arteritic, nonarteritic, optic | ok |
| skipgram | retinopathy | 13 | 0.6000 | end-stage, eye, nonfatal | ok |
| skipgram | tumor | 11 | 0.8000 | interleukin-6, pcsk, tnf, zone | ok |

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
