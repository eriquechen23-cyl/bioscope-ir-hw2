製作一頁、繁體中文、A4 橫式學術重點報告，恰好 1 頁，可編輯 Canva 元素，清楚且留白，深藍／青綠／少量橙色。主題：BioScope｜1,000 篇 GLP-1 摘要的詞嵌入。副題：Project 2 Part 2 · 2026-10-06 作業。以下實驗數據必須精確，不得省略 RQ1–RQ5 或低頻詞問題，不得虛構二維散點座標。這是課堂 NLP 分析，不是醫療建議。
置頂結論：「能學到部分領域共現關係，但小語料不足以證明普遍、穩定的語意表示。」
方法小字：1,000 篇 PubMed 摘要／161,515 tokens／10,691 詞／min_count=3 後 4,672 詞；100 維、20 epochs、negative=5、workers=1；保留停用詞、不 stemming；兩架構 × window 2/5（seed=42），另 window=5（seed=7），共 6 模型；Gensim 實際訓練。
一個 6 列比較表，列必須都有：Input：CBOW 周圍詞平均／Skip-Gram 中心詞；Prediction target：中心詞／各個周圍詞；Training examples：context→target 一筆／target→context 多配對；Training speed：2.33 秒／6.96 秒（同機單次）；Rare words：平均可能稀釋訊號／每次出現多個預測；Expected behavior：較快、平滑語境／較多局部搭配与領域關係。
RQ1「架構差異」：CBOW／Skip-Gram 最後 epoch loss 176,248／790,784，皆下降；樣本量與目標不同，不能用 loss 大小排名語意品質。CBOW semaglutide Top 3：tirzepatide 0.852、dulaglutide 0.824、liraglutide 0.823；Skip-Gram Top 3：dulaglutide 0.673、rybelsus 0.669、tirzepatide 0.659。完整10詞各Top5在網站。
RQ2「有語意嗎？」：部分領域關係合理，但不是同義詞準確率。放五組詞對的精確 cosine 小表（CBOW／SG）：semaglutide–liraglutide 0.823／0.481；insulin–glucose 0.450／0.413；diabetes–obesity 0.611／0.522；obesity–weight 0.280／0.334；receptor–agonist 0.326／0.526。數值來自原始100維。
RQ3「PCA vs t-SNE」：PCA 線性全域投影，两軸解釋變異 CBOW 39.1%／SG 28.3%；t-SNE 重局部鄰近（22 詞、cosine、perplexity=7、seed=42）。用精確数值的小条图呈現 trustworthiness(k=5)：CBOW PCA 0.882、t-SNE 0.912；SG PCA 0.844、t-SNE 0.884。註「投影保留鄰近程度，非語意品質；群間距離不宜解讀」。完整實測散點圖在網站。
RQ4「window 2→5」：10 詞平均 Top-5 重疊率 CBOW 66%／SG 50%，顯示鄰居改變；不是品質分數。
RQ5「小語料限制」：4,434 詞只出現一次，6,019 詞因 min_count<3 排除；單一領域、術語稀疏、泛化有限。兩 seed 的七低頻詞 Top-5 重疊 CBOW 57.1%／SG 68.6%；只有兩次，不證明普遍優勢。
最後小塊「為何 Skip-Gram 適合低頻專業詞？」：一個中心詞每次出現可獲多個上下文預測，稀有訊號較不被平均稀釋；min_count 排除的詞仍沒有向量。
頁尾來源文字／可點連結：GitHub github.com/eriquechen23-cyl/bioscope-ir-hw2；Streamlit bioscope-glp1-ir.streamlit.app；日期 2026-10-13。所有內容同頁，不新增封面或第二頁。
