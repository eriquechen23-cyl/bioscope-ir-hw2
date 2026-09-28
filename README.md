# 🧬 BioScope · GLP-1 文獻實驗室

Information Retrieval HW2 的 Streamlit 專案。預備 **1,000 篇有摘要的 GLP-1 PubMed 文章**，用同一語料展示詞頻、Porter stemming、Word2Vec 與文獻檢索。

## 網站與 GitHub

- Repository：https://github.com/eriquechen23-cyl/bioscope-ir-hw2
- 部署方式：GitHub 存放程式與資料，**Streamlit Community Cloud** 執行網站。GitHub Pages 無法執行這個 Python 應用程式。
- 網站入口：`app.py`；Python 建議 **3.12**。

## 功能

| 工作區 | 可展示的內容 |
| --- | --- |
| 文獻資料 | 10–1,000 篇分析、PMID／標題／期刊／摘要／PubMed 連結、JSONL／CSV／PMID 匯出 |
| Zipf 與 Porter | 原始與詞幹化詞頻、log-log 圖、參考 1/r 曲線、斜率與 R²、轉換示例 |
| 文獻搜尋 | 完整 token、Porter、編輯距離 1–2 的近似匹配、AND／OR、拼字候選、原文位置標示 |
| Word2Vec | CBOW／Skip-gram、window／維度／min_count／epochs、相近詞、PCA、向量與參數下載 |
| 方法與展示 | 演算法說明、實作來源、作業示範流程 |

Porter 1980 與 Levenshtein 由專案程式實作；Word2Vec 使用 Gensim。測試中的 NLTK 僅用於獨立比對 Porter，不是網站執行依賴。不需下載 NLTK 語料。

## 本機執行

先安裝 Python 3.12。在專案資料夾執行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

macOS / Linux：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

開啟終端機顯示的本機網址。預載資料可直接使用；首頁不會連線 PubMed 或自動訓練模型。

## 部署至 Streamlit Community Cloud

1. 確認上述 repository 的 `main` 分支包含 `app.py`、`requirements.txt`、`bioir/`、`data/`、`.streamlit/config.toml`。
2. 到 [Streamlit Community Cloud](https://share.streamlit.io/) 登入，選 **Create app → Yup, I have an app**。
3. Repository：`eriquechen23-cyl/bioscope-ir-hw2`；Branch：`main`；Main file path：`app.py`。
4. Advanced settings 選 Python **3.12**，按 Deploy。預載資料與一般查詢不需要 API key。
5. 等待建置完成，開啟平台提供的 `https://….streamlit.app` 網址；可在其他電腦或手機使用。

後續 push 到相同分支會更新網站。雲端主機可以休眠，重新開啟可能需要等待喚醒。正式網站網址以部署成功後平台顯示的結果為準。

[官方部署文件](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

## 重新預備 PubMed 資料

預設查詢：

```text
("GLP-1"[Title/Abstract] OR "glucagon-like peptide-1"[Title/Abstract]) AND hasabstract AND english[Language]
```

以 PubMed relevance 排序抓取，略過沒有摘要或重複的紀錄並補足至目標。只取有摘要文章，不下載全文。若無法取得指定篇數，命令失敗且不覆蓋現有資料。

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_corpus --count 1000
# 自訂主題及輸出檔
.\.venv\Scripts\python.exe -m scripts.prepare_corpus --query 'semaglutide AND hasabstract' --count 100 --output data/semaglutide.jsonl
# 指定 PMID，檔案可用空白、逗號或換行分隔
.\.venv\Scripts\python.exe -m scripts.prepare_corpus --pmids my_pmids.txt --output data/my_corpus.jsonl
```

`NCBI_EMAIL`、`NCBI_API_KEY` 為選填的環境變數。勿把 key 寫入程式或提交 GitHub。程式每秒最多約 2.8 個請求，同一程序的請求共用限速器，遇到限流或暫時性失敗會退避重試。使用官方 ESearch 與 EFetch，每批取 100 篇。

輸出包括：

- `glp1_1000.jsonl`：每行一篇，欄位為 pmid、title、abstract、journal、date、doi、url、copyright。
- `glp1_1000.pmids.txt`：1,000 個實際保存的 PMID。
- `glp1_1000.manifest.json`：查詢、排序、下載時間、實際篇數、SHA-256。

若想把新語料變成所有訪客的預設資料，更新預載檔案並一併提交三個檔案。網站的即時查詢與上傳只存在當前 session，不會直接改寫 GitHub；重新啟動或斷線後請重新上傳保存的 JSONL。

## 方法與限制

- 預設只處理摘要，保留停用詞；可在側欄切換。英文小寫化，GLP-1 等連字號 token 保留，純數字排除。英文 stoplist 位於 `bioir/text.py`。
- Porter 模組遵照 1980 原始規則；非純英文字母 token 保持不變。stemming 是字尾化簡，不是醫學同義詞辨識。
- Zipf OLS 使用所有詞頻排名；斜率和 R² 不表示搜尋品質。詞幹化前後總 token 數守恆。
- 搜尋依原文 token 操作，停用詞設定不影響搜尋。命中位置為選定原文的 0-based 字元索引 `[start,end)`。
- Word2Vec 使用句子內上下文（標點／換行近似分句）。seed=42、workers=1、穩定 hash；不同套件版本或平台仍可能有數值差異。小型語料中的詞向量只作教學示範。
- PubMed 索引與 relevance 排序會變動；重現同一份結果應使用 repository 保存的資料快照。
- PubMed abstracts 的權利屬原作者／出版者，保留 PMID、原始連結及可取得的版權聲明；本專案沒有對摘要重新授予開放授權。

## 驗證

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

涵蓋公開 Porter 範例與真實語料逐詞 NLTK 原始模式比對、編輯距離、命中位置與 escaping、XML 結構化摘要、補足篇數、PMID 去重、資料 SHA-256、兩種 Word2Vec 模型與 Streamlit 主要頁面操作。

架構與里程碑報告在 [`docs/architecture.md`](docs/architecture.md)、[`docs/reports/`](docs/reports/)。

## 參考

- [Porter 1980 原始規則](https://tartarus.org/martin/PorterStemmer/def.txt)
- [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)
- [NCBI E-utilities 使用說明](https://www.ncbi.nlm.nih.gov/books/NBK25497/)
