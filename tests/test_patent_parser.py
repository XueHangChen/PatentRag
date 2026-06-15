from pathlib import Path

from patent_rag.domain.schemas import PatentType
from patent_rag.ingestion.parser import parse_patent_markdown
from patent_rag.ingestion.pipeline import parse_patent_directory


def test_parse_invention_sample() -> None:
    path = Path("patant/发明1.md")
    document = parse_patent_markdown(path, path.read_text(encoding="utf-8"))

    assert document.metadata.patent_id == "CN106859869A"
    assert document.metadata.patent_type == PatentType.INVENTION_APPLICATION
    assert document.metadata.title == "一种肿瘤内科抢救车"
    assert document.metadata.application_number == "201710170165.7"
    assert document.metadata.application_date == "2017.03.21"
    assert document.metadata.publication_date == "2017.06.20"
    assert document.metadata.applicants == ["南阳医学高等专科学校第一附属医院"]
    assert "程鹏" in document.metadata.inventors
    assert document.metadata.agency == "郑州知己知识产权代理有限公司 41132"
    assert "A61G3/00" in document.metadata.ipc_classes
    assert document.abstract and "微创超低温冷冻消融肿瘤" in document.abstract
    assert len(document.claims) >= 10
    assert any(section.name == "技术领域" for section in document.sections)


def test_parse_utility_model_sample() -> None:
    path = Path("patant/实用新型1.md")
    document = parse_patent_markdown(path, path.read_text(encoding="utf-8"))

    assert document.metadata.patent_id == "CN206539886U"
    assert document.metadata.patent_type == PatentType.UTILITY_MODEL
    assert document.metadata.title == "车载液氮罐固定架"
    assert document.metadata.application_number == "201720197538.5"
    assert document.metadata.publication_date == "2017.10.03"
    assert document.metadata.applicants == ["天津市宁河区医院"]
    assert "张志娟" in document.metadata.inventors
    assert "F17C13/08" in document.metadata.ipc_classes
    assert len(document.claims) == 6
    assert any(section.name == "背景技术" for section in document.sections)


def test_parse_current_directory_without_errors() -> None:
    documents, errors = parse_patent_directory(Path("patant"))

    assert errors == {}
    assert len(documents) == 20
    assert all(document.metadata.patent_id for document in documents)
    assert all(document.metadata.title for document in documents)
    assert all("知识产权局" not in str(document.metadata.title) for document in documents)
    assert all(document.claims for document in documents)


def test_parse_title_when_54_marker_has_no_markdown_heading() -> None:
    invention_6 = Path("patant/发明6.md")
    invention_7 = Path("patant/发明7.md")

    document_6 = parse_patent_markdown(invention_6, invention_6.read_text(encoding="utf-8"))
    document_7 = parse_patent_markdown(invention_7, invention_7.read_text(encoding="utf-8"))

    assert document_6.metadata.title == "一种新生儿科用保健护理装置"
    assert document_7.metadata.title == "一种心脏支架材料及其制备方法"
