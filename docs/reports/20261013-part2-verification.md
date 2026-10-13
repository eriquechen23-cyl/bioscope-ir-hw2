# Part 2 verification — 2026-10-13

Approved scope: extend the existing GLP-1 Streamlit app for the October 6 word-embedding assignment, update the existing GitHub deployment and create a one-page Canva report.

## Evidence

- Original 1,000-abstract snapshot remains unchanged: file SHA-256 `0aeec800f63dc342abc1fe836be4e11adff879f096972482c57e0e7dde0eaf67`.
- Six models trained on actual abstracts: two architectures × windows 2/5 at seed 42, plus two baseline seed-7 repeats. 100 dimensions, 20 epochs, min_count 3; 4,672 retained words.
- Full Windows Python 3.12.14 suite: **59 passed in 16.96s** after review fixes. This covers actual training, losses, samples, OOV, projections, pipeline and all Streamlit workspaces, including interactive training and custom/subset data handling.
- Windows sandbox AppTest socket creation hangs in `asyncio` / `socket._fallback_socketpair`. Stack trace identified the environment limitation; the same tests with local socket access ran successfully. Stalled diagnostic processes were terminated by their verified PIDs.
- Independent read-only review found **no Important/Critical issues**. It recomputed all 102 query/rare-word Top-5 lists from saved vectors, neighbor/pair cosines (errors <=1.8e-7), window/seed overlaps, PCA coordinates and projection trustworthiness.
- Reviewer suggestions applied: UI epoch label reads actual artifact settings; projection-comparison wording and gastroparesis frequency are derived from data.
- Local browser verified default Part 2 workspace, 1,000 / 161,515 / 10,691 / 4,672 metrics, six model summary, training examples and zero Streamlit exceptions.
- Matplotlib PCA/t-SNE labels visually checked after collision adjustment. Same 22 selected words and normalized vectors are used in every projection. Loss charts and complete RQ1–RQ5/background/rare-term answers are included.
- Canva generated design `DAHX3LgwbkU`, A4 landscape 1123×794, **one page**. All scientific numbers were checked. Generated Chinese typos and source URLs were corrected; user explicitly approved the displayed preview, and the draft was committed. A post-save content read confirmed RQ1–RQ5, both correct URLs and no identified typos.

## Interpretation boundaries

Raw loss across architectures is not a semantic quality score. Overlap@5 is neighbor stability, not accuracy. PCA/t-SNE cluster appearance and trustworthiness measure projections, not the truth of biomedical semantics. Two seeds and one GLP-1 domain do not demonstrate generalization.

## Publication

Code and experiments are ready for the authorized main-branch update. Live deployment and CI evidence will be added after remote verification. The saved [Canva report](https://canva.link/hkj0gtudpvpkouv) is recorded in reports/embedding/canva_report.md.

Git CLI lacked a saved credential. Publication uses the user's existing authenticated GitHub connector, without requesting a new OAuth grant or altering repository permissions. The immutable blob API uploads the exact Git-index file contents; the assembled tree is checked against the local tree before a non-forced, expected-head-checked main update.
