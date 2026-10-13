# Word Embedding Part 2 Implementation Plan

> **For agentic workers:** Use executing-plans for inline execution in this session. User approved the design and requested updating the existing main deployment. Use a separate reviewer at the final code review checkpoint as required by requesting-code-review.

**Goal:** Complete the October 6 Part 2 assignment, publish the verified Streamlit update and create an editable one-page Canva report.

**Architecture:** Shared training and analysis functions generate fixed experiment artifacts. A separate Streamlit view reads those artifacts after checking the corpus fingerprint. The existing interactive Word2Vec and Part 1 pages remain available.

**Tech Stack:** Python 3.12, Gensim 4.4, NumPy, pandas, scikit-learn, Streamlit, Plotly, Matplotlib, Canva connector.

## Global constraints

- Original 1,000 GLP-1 abstract snapshot SHA-256 `0aeec800f63dc342abc1fe836be4e11adff879f096972482c57e0e7dde0eaf67`.
- Baseline: vector_size=100, window=5, min_count=3, epochs=20, negative=5, workers=1, seed=42, stable hash; keep stopwords, abstract only, no stemming.
- Both architectures at windows 2 and 5; repeat baseline with seed 7. Exactly six trained models.
- At least ten query words, top five neighbors per model, five related pairs, labeled PCA and t-SNE, six-property comparison table, RQ1–RQ5 and the rare-term question.
- Report observed outcomes; no semantic-quality claim from loss magnitude or 2D clustering alone.

## Task 1: Shared training and scientific analysis

Files: modify `bioir/analysis.py`; create `bioir/embedding.py`, `tests/test_embedding.py`; add scikit-learn to `requirements.txt`.

Interfaces: `train_embeddings(..., seed=42, compute_loss=False, callbacks=())` retains its existing return type; `training_samples(tokens, window) -> (cbow, skipgram)`; `vocabulary_statistics(records, min_count) -> dict`; `train_run(records, architecture, window, epochs=20, seed=42, vector_size=100, min_count=3) -> (model, metadata)`; `neighbor_table(model, terms, topn=5) -> DataFrame`; `pair_table(model, pairs) -> DataFrame`; `project_embeddings(model, words, method='PCA', seed=42) -> (DataFrame, metadata)`; `compare_neighbors(left, right) -> DataFrame`.

- [x] Create venv and install requirements; run `.venv/Scripts/python.exe -m pytest -q` to establish baseline.
- [x] Write failing tests for exact sample direction/boundaries, vocabulary counts, six-dimensional toy model with epoch loss, OOV and self exclusion, cosine symmetry, projection bounds and seed determinism.
- [x] Run `.venv/Scripts/python.exe -m pytest tests/test_embedding.py -q` and confirm missing-feature failures.
- [x] Implement the specified APIs; collect cumulative Gensim loss once per epoch and store differences. Reject invalid architectures and projection inputs; preserve defaults for existing calls.
- [x] Rerun targeted tests and existing algorithm tests, confirming real trained vectors and finite projections.

## Task 2: Fixed experiment pipeline and complete answers

Files: create `scripts/build_embeddings.py`, `bioir/embedding_report.py`, `reports/embedding/`; extend pipeline tests in `tests/test_embedding.py`.

Interfaces: `build_results(records, output, epochs=20) -> manifest`; output summary, loss, vocabulary, neighbors, pairs, projections, window_comparison, stability CSVs; `manifest.json` and `findings_zh.md`. Model IDs use `{cbow|skipgram}_w{2|5}_s{42|7}`; key columns `model`, `word`, `neighbor`, `rank`, `cosine`, `frequency`.

- [x] Write failing tests asserting all six conditions, required word/pair coverage, artifact schema and matching SHA.
- [x] Run targeted pipeline test before implementation.
- [x] Implement sequential six-model training, top-5 analysis, same selected words for every projection, top-5 intersection/5 comparisons, rare-term seed comparison, PNG/SVG scientific plots and JSON manifest with runtime versions.
- [x] Build with `.venv/Scripts/python.exe -m scripts.build_embeddings`; require original SHA before writing output. Record raw model vectors in a compact compressed archive for independent reproduction.
- [x] Read all result tables before composing `findings_zh.md`; answer five RQs plus background table and rare-term question with exact empirical examples and limitations.
- [x] Verify generated PNG charts visually and confirm every reported numeric claim against CSV/manifest.

## Task 3: Streamlit assignment view

Files: create `bioir/embedding_ui.py`; modify `app.py`, `tests/test_app.py`, `README.md`.

Interface: `render_embedding_assignment(records, root)` reads fingerprint-matched fixed results and renders the full assignment view. It never automatically trains a model on a visitor's page load.

- [x] Update AppTest to require the new default workspace, ten query terms, architecture comparison, RQ answers and working subset mismatch notice. Run tests to verify failure.
- [x] Add new default workspace `詞嵌入作業`, relabel old report as Part 1, and render overview, training examples, neighbors/pairs, loss, four projections, window and seed experiments, answers and downloads.
- [x] Preserve interactive training page and validate its existing flow.
- [x] Run complete test suite; start local Streamlit with `.venv/Scripts/python.exe -m streamlit run app.py --server.headless true`; verify browser renders charts and results.
- [x] Rewrite README lead around Part 2, reproducible commands, empirical findings and existing public deployment.

## Task 4: Review, publish and Canva delivery

Files: update design status, plan checkboxes; create `docs/reports/20261013-part2-verification.md`, `reports/embedding/canva_brief_zh.md` and final Canva link record.

- [x] Request code review of implementation against the approved design, correct important findings and rerun affected tests.
- [x] Commit only reviewed project files, push main without force, verify remote revision and GitHub Actions result.
- [x] Verify public Streamlit page contains Part 2, actual model tables and download artifacts; check downloadable data matches fixed outputs.
- [x] Compose concise A4 landscape Canva brief answering all questions using verified data. Call Canva create_design, follow async job polling policy when no widget is mounted; verify one-page design and text.
- [x] Save Canva edit link and deployment evidence; deliver all three final links and concise testing evidence.

## Plan review

Every approved requirement maps to a task: Task 1 supplies model/sample/analysis behavior, Task 2 supplies actual experiment evidence and all answers, Task 3 exposes the complete assignment, Task 4 publishes and creates the requested Canva report. No external approval is needed beyond the existing user instruction to update their repo and deploy.
