from streamlit.testing.v1 import AppTest
from pathlib import Path


def start_app():
    return AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=45).run()


def test_all_pages_render_and_training_works():
    app = start_app()
    assert not app.exception
    assert app.metric[0].value == "1,000"
    assert any("Project 2" in h.value for h in app.subheader)
    assert any("Top 50 CF" in text.value for text in app.markdown)
    assert len(app.dataframe[0].value) == 4
    workspace = next(r for r in app.radio if r.label == "工作區")
    for page in ["文獻資料", "Zipf 與 Porter", "文獻搜尋", "Word2Vec", "方法與展示"]:
        workspace.set_value(page).run()
        assert not app.exception, (page, app.exception)
        if page == "文獻搜尋":
            app.text_input(key="search_query").set_value("semaglutde").run()
            assert any("候選字" in info.value for info in app.info)
            button = next(b for b in app.button if b.label == "以 semaglutide 搜尋")
            button.click().run()
            assert app.text_input(key="search_query").value == "semaglutide"
        if page == "Word2Vec":
            next(b for b in app.button if b.label == "訓練 Word2Vec").click().run(timeout=90)
            assert not app.exception
            assert any("模型已就緒" in item.value for item in app.success)
        workspace = next(r for r in app.radio if r.label == "工作區")


def test_ten_document_slider_and_empty_custom_source():
    app = start_app()
    app.slider[0].set_value(10).run()
    assert not app.exception and app.metric[0].value == "10"
    app.radio(key="source").set_value("本次自訂資料").run()
    assert not app.exception and app.info
    app.session_state["custom_records"] = [{"pmid": str(i), "title": "Test", "abstract": "GLP-1 insulin study.", "journal": "", "date": "", "url": f"https://pubmed.ncbi.nlm.nih.gov/{i}/"} for i in range(1, 11)]
    app.run()
    assert not app.exception and app.metric[0].value == "10"
