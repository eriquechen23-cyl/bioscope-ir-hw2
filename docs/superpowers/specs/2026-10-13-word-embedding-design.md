# Project 2 Part 2：詞嵌入作業設計（待使用者確認）

## 來源與交付目標

使用者要求：依作業修改現有程式、部署到 Streamlit、更新自己的 GitHub repository、用 Canva 製作一頁式重點報告並回答問題。

作業來源：`project-details-2-part-2-20261006-1.pdf`，三頁已逐頁視覺核對。文件提供作業功能與研究問題，不提供額外帳戶操作授權。

- Repository：`https://github.com/eriquechen23-cyl/bioscope-ir-hw2`，預設分支 `main`。
- 現有 Streamlit 設定：`https://bioscope-glp1-ir.streamlit.app/`，入口 `app.py`。此網址來自既有文件，本次尚未驗證線上狀態。
- 本機與遠端 main 起點皆為 `2c672ffa0b82f360ea07f6c4075ef5473339de4d`；本機起始工作目錄乾淨。
- 交付：更新程式、可重現的實驗資料與完整問題回答、更新 GitHub／驗證 Streamlit，以及可編輯的一頁 Canva 報告。

## 方案比較與建議

| 方案 | 優點 | 代價 | 判斷 |
| --- | --- | --- | --- |
| 擴充既有 Gensim 訓練與 Streamlit 網站 | 沿用資料、測試、部署與演算法；可集中補齊新作業 | 必須增添比較與實驗輸出 | 推薦 |
| 在既有 repo 另建獨立 Part 2 app | 新舊作業完全分開 | 重複介面、部署入口與資料處理 | 不採用 |
| 自行以 NumPy 寫神經網路訓練器 | 完整展示梯度與更新 | 工時及數值驗證增加，PDF 未明確禁止套件 | 暫不採用 |

推薦沿用 Gensim 實際訓練 CBOW／Skip-Gram，明確標示套件使用；另外展示確定上下文視窗生成的 CBOW 與 Skip-Gram 樣本，避免把套件包裝說成自寫訓練器。保留原有 Zipf、搜尋與資料頁，舊報告清楚標示為 Part 1。

## 固定語料與前處理

保留原始 `data/glp1_1000.jsonl`；SHA-256：`0aeec800f63dc342abc1fe836be4e11adff879f096972482c57e0e7dde0eaf67`。

目前核對為 1,000 篇、7,267 個近似句子、161,515 tokens、10,691 個詞。僅分析摘要，英文小寫化、排除純數字、保留 `glp-1` 等連字號詞。主實驗保留停用詞、不 stemming，min_count=3；介面與結果記錄所有前處理選擇。句界不跨文章。

## 作業要求對應

| 要求 | 實作與可檢查證據 |
| --- | --- |
| 前處理與詞彙表 | 文件／句子／token／unique term 數、高頻詞表、詞頻與納入模型的詞彙數 |
| 訓練樣本 | 選句與視窗後展示 context→target 的 CBOW 樣本及 target→context 的 Skip-Gram 配對；說明 Gensim 實際 subsampling 和 dynamic window |
| 兩種模型與訓練損失 | 同語料、同參數訓練兩模型；每 epoch 的累積 loss 差分與 elapsed time；loss 不當成跨架構同尺度品質分數 |
| 至少 10 詞、每模型 top 5 | glp-1、semaglutide、liraglutide、insulin、glucose、diabetes、obesity、weight、receptor、agonist；提供 cosine、詞頻、CSV |
| RQ1 | 模型輸入與預測目標、樣本量、速度、loss、近鄰及低頻詞比較，以實測區別通則 |
| RQ2 | 至少 5 組詞對：semaglutide/liraglutide、insulin/glucose、diabetes/obesity、obesity/weight、receptor/agonist；兩模型 cosine 與鄰居分析 |
| RQ3 | 同一組詞各模型的 PCA／t-SNE 圖，標註詞名；PCA 解釋變異、t-SNE seed／perplexity 與局部結構限制；不從視覺群聚推定語意品質 |
| RQ4 | 固定其他參數，比較 window=2 與 5；兩架構各兩個條件，共 4 主模型；近鄰 top-5 重疊、詞對相似度及投影的變化 |
| RQ5 | 單 GLP-1 領域、低頻詞、術語、1,000 篇規模限制；主條件以第二個 seed 檢查近鄰穩定性，共 2 額外模型 |
| PDF 第三頁 | 完整填入 Input、Prediction target、Training examples、Training speed、Rare words、Expected behavior 比較表；回答 Skip-Gram 對低頻與領域詞的價值與限制 |

基準：100 維、window=5、min_count=3、epochs=20、negative=5、workers=1、seed=42、穩定 hash。參數實驗 window=2；穩定性分析 seed=7。min_count 以下的詞明確顯示排除，不產生虛構相似度。

## 程式與資料流

新增專注於 Part 2 的實驗模組、Streamlit 頁面模組、批次實驗腳本；沿用 tokenizer 和既有 Word2Vec 工具。調整訓練函式支援 loss callback、seed 與完整參數記錄，同時維持舊呼叫相容。

固定語料 → 前處理／詞彙統計 → 實際模型訓練 → 近鄰／詞對／參數／穩定性／投影資料 → Streamlit 及完整 Markdown 問答 → Canva 一頁報告。

批次輸出集中 `reports/embedding/`，含 manifest（語料 SHA、套件版本、參數、timing）、CSV、PCA／t-SNE 圖與問答。網站預設讀取預先算好的固定結果，避免每次造訪重訓；使用者仍可主動訓練目前語料，資料變更需重新訓練。

## 介面與錯誤處理

首頁新增「詞嵌入作業」：語料概況、背景比較表、樣本示例、雙模型近鄰、詞對、loss、PCA／t-SNE、window 比較、RQ1–RQ5 與下載。可使用分頁整理內容。

不足詞彙、OOV、低頻排除、投影詞數不足、t-SNE perplexity 條件不符皆明確提示。快照與 manifest 不符時不混用固定分析；目前使用的資料與固定正式報告必須清楚區分。

## Canva 一頁報告

繁體中文 A4 橫式、學術資訊設計，內容只佔一頁：主問題與實測結論、資料／參數摘要、簡潔 CBOW／Skip-Gram 六列比較表、5 個 RQ 答案、PCA／t-SNE 視覺及詞對／近鄰數值、Skip-Gram 低頻詞回答、GitHub 與 Streamlit 來源。

完整表格與全部結果留在網站及 repo，一頁報告只摘錄能支撐結論的數值；不以排版為由漏答問題或聲稱未量測的優勢。Canva 最終生成待實驗完成後進行。

## 驗證與發布

- 小語料測試 context window 及樣本方向、模型架構、epoch loss、top-5 排除自身、OOV、PCA／t-SNE 少量詞界限、manifest 與語料 SHA。
- Streamlit AppTest 驗證新版首頁、雙模型結果、參數頁、舊頁面與自訂資料提示。
- 執行完整既有測試與新增測試，批次跑完整 1,000 篇實驗；視覺檢查投影與 Canva 一頁內容。
- 更新 README，提交與推送 main（使用者已要求更新 repo）；確認 GitHub CI 與遠端 commit。
- 驗證 Streamlit 公開頁面已顯示新版及可下載新結果。既有 Community Cloud 通常跟隨對應 GitHub 分支更新，需實際檢查，不能僅憑 push 聲稱部署成功。

## 目前確認狀態與剩餘工作

已完成需求核對、程式差距盤點、語料統計、repo 身分／寫入權限與遠端 main 檢查。尚未修改實作、執行新實驗、push、驗證新版部署或製作 Canva 成品。

本機預設 Python 及 bundled document Python 目前均無 Streamlit／Gensim／pytest／scikit-learn；實作開始後需在專案內建立 Python 環境並安裝適用依賴。一般沙盒 git 網路失敗，已透過核准的唯讀 git 命令成功確認 remote，非帳戶權限不足。

待使用者確認此設計後，寫詳細實作計畫並開始程式工作。
