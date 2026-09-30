"""Rebuild the required Project 2 reports: python -m scripts.build_assignment."""
import base64
import hashlib
import html
import json
from pathlib import Path
import re
import os
# Keep report-generation caches inside the project on restricted hosts.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "tmp/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from bioir.pubmed import load_corpus
from bioir.experiments import CONDITIONS, run_experiments, select_comparison_terms, term_statistics
from bioir.text import STOPWORDS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/assignment"
ORIGINAL_SHA256 = "0aeec800f63dc342abc1fe836be4e11adff879f096972482c57e0e7dde0eaf67"
SOURCES = [
    ("Course specification, 2026-09-29", "project-2-details-20260929-1.pdf (local document)"),
    ("Manning, Raghavan & Schutze (2008), Zipf's law", "https://nlp.stanford.edu/IR-book/html/htmledition/zipfs-law-modeling-the-distribution-of-terms-1.html"),
    ("Manning et al., Inverse document frequency", "https://nlp.stanford.edu/IR-book/html/htmledition/inverse-document-frequency-1.html"),
    ("Manning et al., Index compression", "https://nlp.stanford.edu/IR-book/html/htmledition/index-compression-1.html"),
    ("Clauset, Shalizi & Newman (2009), Power-law distributions in empirical data", "https://arxiv.org/abs/0706.1062"),
    ("Porter (1980), An algorithm for suffix stripping", "https://tartarus.org/martin/PorterStemmer/def.txt"),
    ("NCBI E-utilities", "https://www.ncbi.nlm.nih.gov/books/NBK25497/"),
]


def formatted(frame):
    frame = frame.copy()
    for col in frame.columns:
        if pd.api.types.is_float_dtype(frame[col]):
            frame[col] = frame[col].map(lambda v: "undefined" if pd.isna(v) else f"{v:.4f}")
    return frame


def charts(experiments):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    baseline = experiments["conditions"]["B"]
    t, fit = baseline["terms"], baseline["fit"]
    for kind in ("baseline_distribution", "preprocessing_and_residuals"):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        if kind == "baseline_distribution":
            ax[0].plot(t["rank"], t["CF"], color="#007f78")
            ax[0].set(title="Experiment 1: rank vs CF", xlabel="Rank (linear)", ylabel="CF (linear)")
            ax[1].loglog(t["rank"], t["CF"], label="Observed B", color="#007f78")
            ax[1].loglog(t["rank"], 10**fit["intercept"]*t["rank"]**fit["slope"], "--", label="Full-range OLS", color="#db7c36")
            ax[1].loglog(t["rank"], t.iloc[0]["CF"]/t["rank"], ":", label="f(1)/rank reference", color="#8b98ac")
            ax[1].set(title="Experiment 2: log-log", xlabel="Rank (log)", ylabel="CF (log)")
            ax[1].legend(fontsize=8)
        else:
            for code, color in zip(CONDITIONS, ["#8095a8", "#007f78", "#db7c36", "#6554a5"]):
                terms = experiments["conditions"][code]["terms"]
                ax[0].loglog(terms["rank"], terms["CF"], color=color, label=code)
            ax[0].set(title="Preprocessing comparison", xlabel="Rank (log)", ylabel="CF (log)")
            ax[0].legend()
            residual = np.log10(t["CF"]) - (fit["intercept"]+fit["slope"]*np.log10(t["rank"]))
            ax[1].semilogx(t["rank"], residual, color="#007f78")
            ax[1].axhline(0, linestyle="--", color="#8b98ac")
            ax[1].set(title="Baseline OLS residuals", xlabel="Rank (log)", ylabel="Observed - fitted log10(CF)")
        for axis in ax:
            axis.grid(alpha=.18)
        for extension in ("png", "svg"):
            fig.savefig(OUT / f"{kind}.{extension}", dpi=180, bbox_inches="tight")
        plt.close(fig)


def pdf(path, pages, compact=False):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", fontName="Helvetica-Bold", fontSize=19, leading=23, spaceAfter=12, textColor=colors.HexColor("#16354c")))
    styles.add(ParagraphStyle(name="ReportBody", fontSize=9.5 if compact else 10, leading=12.8 if compact else 14, spaceAfter=9))
    styles.add(ParagraphStyle(name="Cell", fontSize=8, leading=9.5, wordWrap="CJK"))
    story = []
    for i, page in enumerate(pages):
        if i: story.append(PageBreak())
        story.append(Paragraph(html.escape(page["title"]), styles["ReportTitle"]))
        for item in page["items"]:
            if isinstance(item, str):
                story.append(Paragraph(html.escape(item), styles["ReportBody"]))
            elif isinstance(item, pd.DataFrame):
                frame = formatted(item)
                cells = [[Paragraph(html.escape(str(c)), styles["Cell"]) for c in frame.columns]]
                cells += [[Paragraph(html.escape(str(v)), styles["Cell"]) for v in row] for row in frame.itertuples(index=False, name=None)]
                widths = [503/len(frame.columns)] * len(frame.columns)
                if "term" in frame.columns:
                    widths = [383/(len(frame.columns)-1)] * len(frame.columns)
                    widths[list(frame.columns).index("term")] = 120
                table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
                padding = 1 if len(frame) >= 50 else 2
                table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#dceee9")),
                    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f4f7fa")]),
                    ("VALIGN", (0,0), (-1,-1), "TOP"), ("TOPPADDING", (0,0), (-1,-1), padding),
                    ("BOTTOMPADDING", (0,0), (-1,-1), padding)]))
                story += [table, Spacer(1,12)]
            elif isinstance(item, Path):
                from PIL import Image as PILImage
                with PILImage.open(item) as image: width, height = image.size
                story += [Image(str(item), width=503, height=503*height/width), Spacer(1,10)]
    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#60738a"))
        canvas.drawString(46,29,"BioScope | Project 2 | 1,000 GLP-1 abstracts | 2026-09-30")
        canvas.drawRightString(549,29,str(doc.page))
    SimpleDocTemplate(str(path), pagesize=A4, rightMargin=46, leftMargin=46, topMargin=43, bottomMargin=50,
                      title=pages[0]["title"], author="BioScope Project 2").build(story,onFirstPage=footer,onLaterPages=footer)


def html_report(pages):
    sections = []
    for page in pages:
        content = [f"<h2>{html.escape(page['title'])}</h2>"]
        for item in page["items"]:
            if isinstance(item,str): content.append(f"<p>{html.escape(item)}</p>")
            elif isinstance(item,pd.DataFrame): content.append(formatted(item).to_html(index=False,border=0,escape=True))
            elif isinstance(item,Path): content.append(f'<img alt="{item.stem}" src="data:image/png;base64,{base64.b64encode(item.read_bytes()).decode()}">')
        sections.append("<section>"+"".join(content)+"</section>")
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GLP-1 Project 2</title><style>body{font:16px/1.7 system-ui;color:#18364a;background:#edf3f7;margin:0}main{max-width:980px;margin:32px auto;background:white;padding:40px}h1,h2{color:#007f78}section{padding:12px 0 32px;border-bottom:1px solid #dce6ef}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;text-align:left;border-bottom:1px solid #dce6ef}th{background:#e1f0eb}tr:nth-child(even){background:#f6f8fa}img{width:100%}p{overflow-wrap:anywhere}@media print{main{margin:0;padding:0}section{break-before:page}}</style><main><h1>Zipf analysis of 1,000 GLP-1 abstracts</h1><p>Parts I-IX, RQ1-RQ5 and Final Challenge. Optional cross-domain challenge omitted.</p>'+"".join(sections)+"</main></html>"


def main():
    path=ROOT/"data/glp1_1000.jsonl"
    checksum=hashlib.sha256(path.read_bytes()).hexdigest()
    if checksum!=ORIGINAL_SHA256: raise ValueError("Original corpus changed; refusing to replace the assignment dataset.")
    records=load_corpus(path)
    assert len(records)==len({r['pmid'] for r in records})==1000
    manifest=json.loads(path.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    experiments=run_experiments(records)
    OUT.mkdir(parents=True,exist_ok=True)
    summary=experiments['summary']
    summary.to_csv(OUT/'preprocessing_comparison.csv',index=False)
    for code,result in experiments['conditions'].items():
        result['terms'].to_csv(OUT/f'condition_{code}_terms.csv',index=False)
        result['terms'].head(50).to_csv(OUT/f'condition_{code}_top50.csv',index=False)
        result['regions'].to_csv(OUT/f'condition_{code}_regions.csv',index=False)
    baseline=experiments['conditions']['B']
    b,c,d=[summary.set_index('condition').loc[code] for code in 'BCD']
    comp=select_comparison_terms(baseline['terms'])
    comp.to_csv(OUT/'cf_df_idf_25_terms.csv',index=False)
    _,_,counts=term_statistics(records,'B')
    pd.DataFrame([{'pmid':r['pmid'],'term':w,'TF':tf} for r,counter in zip(records,counts) for w,tf in counter.items()]).to_csv(OUT/'document_tf.csv',index=False)
    demo_index=max(range(len(records)),key=lambda i:counts[i]['semaglutide'])
    demo=baseline['terms'][baseline['terms']['term'].isin(counts[demo_index])].copy()
    demo['TF']=demo['term'].map(counts[demo_index]); demo['TF_IDF']=demo['TF']*demo['IDF']
    demo=demo.sort_values(['TF_IDF','term'],ascending=[False,True])
    demo.to_csv(OUT/'example_document_tfidf.csv',index=False)
    high,middle=baseline['regions'].iloc[0],baseline['regions'].iloc[1]
    terms=baseline['terms'].set_index('term')
    recurrent=baseline['terms'].query('CF >= 20 and DF <= 10').sort_values('CF_per_DF',ascending=False).iloc[0]
    reduction=(1-c['tokens']/b['tokens'])*100
    vocab_reduction=(1-d['vocabulary']/c['vocabulary'])*100
    charts(experiments)
    implications=(
        f"Zipf's Law matters to retrieval because it describes an uneven allocation of evidence and storage. In this collection, {int(b['vocabulary']):,} vocabulary entries account for {int(b['tokens']):,} token occurrences, but {int(b['hapax_terms']):,} terms occur only once. A system designed around a uniform vocabulary would misrepresent both common queries and rare terminology. The useful engineering observation is this imbalance, rather than an assumption that every rank obeys an exact inverse law. [Manning et al., Zipf's law.]\n\n"
        f"Stopword removal reduces tokens by {reduction:.1f}% in our experiment. The word 'the' appears in {int(terms.loc['the','DF'])} documents, so its presence rarely distinguishes one abstract from another. However, removal is a retrieval policy, not an automatic quality improvement. Negation and phrase structure can matter, especially in biomedical queries. Our fixed teaching stoplist removes 'not'; a production system should evaluate that choice using relevance judgments. We did not measure precision, recall or ranking quality, so fewer tokens cannot be interpreted as better retrieval.\n\n"
        "An inverted index stores a dictionary entry for each retained term and a postings list for the documents containing it. Many singleton entries can make dictionary overhead substantial, while widespread terms require long document lists. Repeated occurrences in one document increase CF and positional information without increasing DF. Thus dictionary size, document postings and positional storage have different drivers. Sorted document identifiers allow gap encoding; integer encoders can represent many small gaps compactly. Actual compression depends on ordering and implementation, and this project reports counts rather than measured index bytes. [Manning et al., Index compression.]\n\n"
        f"TF-IDF combines local repetition with collection-wide rarity. Here 'glp-1' occurs in {int(terms.loc['glp-1','DF'])} of 1,000 abstracts, giving IDF {terms.loc['glp-1','IDF']:.3f}, whereas 'semaglutide' has IDF {terms.loc['semaglutide','IDF']:.3f}. A biologically central topic word can consequently distinguish documents less than a narrower drug name. High CF alone is insufficient to justify downweighting: IDF is determined by DF. Zipf describes global frequency imbalance, while TF-IDF uses document distribution to turn that imbalance into a weighting decision. Rare terms still require scrutiny because spelling variants and noise also receive large IDF values. [Manning et al., Inverse document frequency.]"
    )
    word_count=len(re.findall(r"\b[\w'-]+\b",implications)); assert 300<=word_count<=500,word_count
    fit_answer=f"The full-range B fit is log10(CF) = {b['intercept']:.4f} {b['slope']:+.4f} log10(rank): k={b['exponent']:.4f}, R2={b['r_squared']:.4f}, RMSE={b['rmse']:.4f} log10 units. The corpus shows a strongly skewed, approximately Zipf-like pattern, but the full-range exponent is above the ideal k=1 and residuals are structured. This is descriptive evidence of approximate scaling, not proof of an exact power law."
    regions_answer=f"Under predetermined rank bands, the head (ranks 1-{int(high['rank_end'])}) has k={high['exponent']:.4f}, R2={high['r_squared']:.4f}, RMSE={high['rmse']:.4f}. The middle has k={middle['exponent']:.4f}. The last 10% consists entirely of CF=1 terms: slope and RMSE are zero, but R2 is undefined. The head best matches an ideal inverse-rank relationship: slope closest to -1, small residual error and no constant-frequency degeneracy. Different rank ranges have different variance; R2 alone is not a fair ranking of regions."
    final_challenge=f"An approximate empirical law is useful when it guides robust choices without claiming exact predictions. Our head exponent {high['exponent']:.3f} and whole-vocabulary exponent {b['exponent']:.3f} differ, yet both plots show concentration in a few terms and a sparse tail. That motivates separate treatment of frequent postings and rare dictionary entries, monitoring common queries and using document-aware weights. It does not give an exact storage budget or guarantee relevance. A sound system measures its own corpus, checks residuals and preprocessing sensitivity, and evaluates changes on retrieval tasks. Zipf is a testable design heuristic whose value survives deviations from k=1; its limitations show where measurement must replace extrapolation."
    pages=[
      {'title':'1 | Corpus and preprocessing','items':[
        'Project 2 - Biomedical Information Retrieval. Analysis date: 2026-09-30. Scope: Parts I-IX, research questions, Final Challenge and a separate one-page Executive Summary. Optional two-domain comparison omitted.',
        f"The original frozen corpus contains 1,000 distinct PMID records with nonempty English-filtered abstracts, retrieved {manifest['retrieved_at_utc']}. Source: NCBI PubMed ESearch/EFetch, relevance ordering. Only abstracts are analyzed; titles and journals are excluded. English status follows the PubMed language filter, not an independent language detector.",
        'Query: '+manifest['query'],'Original SHA-256: '+checksum,
        f"Tokenization recognizes English words and internal biomedical hyphens (GLP-1 stays one term). Pure numbers are excluded consistently. A retains standalone punctuation; B drops those tokens; C additionally removes the fixed {len(STOPWORDS)}-term stoplist; D additionally applies the project's Porter 1980 implementation. Greek/non-English letters are not analyzed. Structured abstract section labels remain in the text.",
        summary[['condition','documents','tokens','vocabulary','average_tokens_per_document','hapax_terms']],
        'The single-topic relevance-selected snapshot is not a random sample of PubMed. Punctuation handling preserves intraword hyphens. The tokenizer and stoplist are versioned in code; their choices constrain generalization.']},
      {'title':'2 | Top 50 collection-frequency terms','items':[
        'Condition B: CF sums per-document TF; DF counts documents containing the term. Ranking uses descending CF, then alphabetical tie-breaking.',
        baseline['terms'].head(50)[['rank','term','CF','DF']]]},
      {'title':'3 | Rank-frequency and log-log analysis','items':[
        OUT/'baseline_distribution.png',
        'The linear chart exposes head concentration but compresses the sparse tail. The separate log-log chart reveals the whole range. OLS uses every term; f(1)/rank is only a k=1 reference curve.',
        pd.DataFrame([baseline['fit']]),fit_answer,
        'R2 is not sufficient evidence: logarithms change the error model, neighboring ranks are dependent, equal frequencies form steps, and the finite corpus truncates the tail. Different distributions may look linear over restricted ranges. Residuals, sensitivity analysis and formal comparisons are needed for a stronger claim. This assignment uses descriptive rank regression, not maximum-likelihood fitting of a frequency probability distribution; their exponents must not be conflated. [Clauset et al., 2009.]',
        'OLS definitions: x=log10(rank), y=log10(CF); slope=sum((x-x_mean)(y-y_mean))/sum((x-x_mean)^2); intercept=y_mean-slope*x_mean; k=-slope; R2=1-SSE/SST; RMSE=sqrt(SSE/number_of_terms).']},
      {'title':'4 | Preprocessing and rank regions','items':[
        OUT/'preprocessing_and_residuals.png',summary[['condition','vocabulary','exponent','r_squared','rmse']],
        f"B removes {int(summary.iloc[0]['tokens']-b['tokens']):,} punctuation occurrences. C removes {int(b['tokens']-c['tokens']):,} stopwords ({reduction:.1f}%), lowering k from {b['exponent']:.3f} to {c['exponent']:.3f}. D keeps C's token count but reduces vocabulary by {vocab_reduction:.1f}% and raises k to {d['exponent']:.3f}. Preprocessing alters head mass and tail length, not merely labels.",
        'Top five: '+'; '.join(code+': '+', '.join(result['terms']['term'].head(5)) for code,result in experiments['conditions'].items()),
        baseline['regions'][['region','rank_start','rank_end','exponent','r_squared','rmse']],regions_answer]},
      {'title':'5 | CF versus DF','items':[
        'These 25 terms mix the top ten CF terms with predefined biomedical terms, filled to 25 in CF order. They were not selected to maximize IDF contrast. CF/DF is the average count within documents containing the term.',
        comp[['term','CF','DF','CF_per_DF']],
        f"Repetition can yield high CF but relatively low DF. '{recurrent['term']}' has CF={int(recurrent['CF'])}, DF={int(recurrent['DF'])}, averaging {recurrent['CF_per_DF']:.2f} mentions per containing document. This illustrative example was selected by highest CF/DF among terms with CF>=20 and DF<=10.",
        'DF (or DF/N) measures document coverage. CF measures occurrence volume. Retrieval needs both: CF describes commonness and positional workload; DF describes document spread, postings length and IDF. Neither measure alone establishes clinical importance or relevance.']},
      {'title':'6 | IDF and a TF-IDF example','items':[
        'IDF(t)=ln(N/DF(t)), N=1,000, with natural logarithms and no smoothing. There are no zero-DF vocabulary entries. Changing log base rescales IDF but preserves its ordering. This table includes more than ten terms.',
        comp[['term','CF','DF','IDF']].head(15),
        f"'the' has IDF={terms.loc['the','IDF']:.4f}; 'glp-1' has IDF={terms.loc['glp-1','IDF']:.4f}; 'semaglutide' has IDF={terms.loc['semaglutide','IDF']:.4f}. Terms found almost everywhere provide little discrimination inside the collection. Topic specificity makes GLP-1 common here even though it might distinguish unrelated domains.",
        f"PMID {records[demo_index]['pmid']}, chosen as the document with the most 'semaglutide' occurrences, illustrates raw TF-IDF=TF*IDF. These are its top ten weights using the same corpus IDF, without length normalization.",
        demo[['term','TF','DF','IDF','TF_IDF']].head(10),
        'Zipf describes concentrated collection counts. TF-IDF addresses an associated retrieval problem by downweighting widespread terms while preserving local repetition. High CF does not imply high DF: IDF uses measured DF, not Zipf rank. [Manning et al., Inverse document frequency.]']},
      {'title':'7 | Information Retrieval implications','items':[
        f'Part IX discussion ({word_count} words; required range: 300-500).',*implications.split('\n\n'),
        'Final Challenge - why an approximate law remains useful',final_challenge]},
      {'title':'8 | Research answers and reproducibility','items':[
        'RQ1 - Approximate Zipf-like structure is visible, particularly in the high-frequency head. The full distribution is not exactly k=1; residual curvature and a singleton plateau contradict an exact universal fit.',
        f"RQ2 - Baseline B k={b['exponent']:.4f}; head-only k={high['exponent']:.4f}. State preprocessing and rank interval with every exponent.",
        'RQ3 - A-D exponents are '+', '.join(f'{row.exponent:.4f}' for row in summary.itertuples())+'. Stopwords change head mass; stemming merges variants and shortens the tail. A smaller vocabulary does not guarantee better fitting or retrieval.',
        'RQ4 - CF sums TF; DF counts documents with nonzero TF. Repetition increases CF without necessarily changing DF. The 25-term table demonstrates these distinct distributional properties.',
        'RQ5 - Frequency imbalance informs stopword policy, dictionary/postings tradeoffs and document-aware weights. Relevance testing and measured index storage remain separate experiments.',
        'Reproduce: install requirements-report.txt; run python -m scripts.build_assignment. The generator checks the original SHA-256 first. Outputs include all terms for A-D, Top 50 tables, region fits, CF/DF/IDF, nonzero document TF and a TF-IDF example. No analysis-time downloads. Existing Word2Vec/search demos are extras, not substitutes for these experiments.',
        'References (the local course PDF is not redistributed):',*[f'[{i}] {name}. {url}' for i,(name,url) in enumerate(SOURCES,1)]]},
    ]
    executive=[{'title':'Executive Summary | GLP-1 & Zipf','items':[
        'What did analyzing 1,000 scientific abstracts teach us beyond the textbook?',
        f"Dataset and method. The original 1,000 unique-PMID GLP-1 abstracts yield {int(b['tokens']):,} baseline tokens and {int(b['vocabulary']):,} terms ({b['average_tokens_per_document']:.3f} tokens per document). Four cumulative conditions isolate punctuation, stopwords and stemming; titles are excluded.",
        f"Finding 1 - The law is approximate and range-dependent. The log-log curve looks nearly linear, yet residuals are curved and {int(b['hapax_terms']):,} terms occur once. The final 10% is a flat CF=1 plateau. Seeing that plateau explains why a textbook straight line is an idealization rather than a model of every observed rank.",
        f"Finding 2 - There is no context-free exponent. Baseline slope={b['slope']:.4f}, intercept={b['intercept']:.4f}, k={b['exponent']:.4f}, R2={b['r_squared']:.4f}, log10 RMSE={b['rmse']:.4f}. The first 10% has k={high['exponent']:.4f}, R2={high['r_squared']:.4f}, and best approximates k=1 under our bands. High R2 alone cannot establish a power law; the constant tail even has zero RMSE without meaningful R2.",
        f"Finding 3 - Preprocessing changes the experiment. Removing stopwords reduces tokens by {reduction:.1f}% and changes k to {c['exponent']:.4f}. Stemming preserves those tokens but reduces vocabulary to {int(d['vocabulary']):,}, raising k to {d['exponent']:.4f}. Preprocessing is part of the result, not merely preparation before analysis.",
        f"Finding 4 - Topic importance differs from retrieval discrimination. GLP-1 occurs in {int(terms.loc['glp-1','DF'])}/1,000 abstracts (IDF {terms.loc['glp-1','IDF']:.3f}); semaglutide in {int(terms.loc['semaglutide','DF'])} (IDF {terms.loc['semaglutide','IDF']:.3f}). A central topic word can weakly distinguish documents in a focused collection. CF measures repetition; DF measures document coverage and determines IDF.",
        'IR consequence. A frequent head and sparse tail motivate different treatment of long postings and dictionary overhead, careful stopword policy, compression and TF-IDF. We measured distributions, not retrieval quality: word removal and weighting still require evaluation on relevant queries. The single-domain, relevance-selected sample also constrains generalization.',
        'Conclusion. Zipf is useful as an empirical guide to workload and weighting even when k differs from one. These results replace an assumed universal line with a measured, preprocessing-sensitive description. The full report provides tables, sources and reproducibility. Optional cross-domain comparison omitted.'
    ]}]
    pdf(OUT/'project2_report.pdf',pages)
    pdf(OUT/'executive_summary.pdf',executive,compact=True)
    (OUT/'project2_report.html').write_text(html_report(pages),encoding='utf-8')
    (OUT/'part_ix_discussion.txt').write_text(implications,encoding='utf-8')
    findings=f'''## RQ1 / RQ2：是否符合 Zipf？指數是多少？
主要條件 B：slope={b['slope']:.4f}、intercept={b['intercept']:.4f}、k={b['exponent']:.4f}、R²={b['r_squared']:.4f}、log10 RMSE={b['rmse']:.4f}。
近似符合 Zipf 長尾，但全域 k 不是 1，不能宣稱完全符合。前 10% k={high['exponent']:.4f} 較接近 1；尾端 10% 全部 CF=1，R² 未定義。

## RQ3：前處理影響
移除停用詞使 token 減少 {reduction:.1f}%，k 降為 {c['exponent']:.4f}。再 stemming 保持 token 總數、詞彙降到 {int(d['vocabulary']):,}，k 變為 {d['exponent']:.4f}。詞彙較少不等於檢索品質較好。

## RQ4：CF / DF / IDF
CF 是整體詞次，DF 是文件數。同篇重複增加 CF，不一定增加 DF。GLP-1 出現於 {int(terms.loc['glp-1','DF'])} 篇，IDF={terms.loc['glp-1','IDF']:.3f}；semaglutide 出現於 {int(terms.loc['semaglutide','DF'])} 篇，IDF={terms.loc['semaglutide','IDF']:.3f}。主題重要性與區辨力不同。

## RQ5 / Final Challenge
少數詞占很多詞次、很多罕見詞增加詞典項目，能引導停用詞、倒排索引、壓縮與 TF-IDF 設計。實際效益仍需檢索任務與索引大小評估；本實驗沒有證明效能提升，也沒有做兩領域 Optional Challenge。
高 R² 不足以證明 power law，還要看 residuals、同頻平台、區間與替代模型。以上為原始 1,000 篇固定結果；網站選取子集時以即時計算表格為準。
'''
    (OUT/'findings_zh.md').write_text(findings,encoding='utf-8')
    (OUT/'analysis_manifest.json').write_text(json.dumps({'corpus':'data/glp1_1000.jsonl','sha256':checksum,'documents':1000,'analysis_date':'2026-09-30','baseline':'B','idf_log':'natural','regression_log':'log10','optional_challenge':'omitted','part_ix_words':word_count,'tfidf_example_pmid':records[demo_index]['pmid'],'stopwords':sorted(STOPWORDS)},indent=2),encoding='utf-8')
    print(summary.to_string(index=False)); print(f'Part IX: {word_count} words; reports: {OUT}')


if __name__=='__main__': main()
