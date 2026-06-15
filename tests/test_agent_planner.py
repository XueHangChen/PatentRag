from patent_rag.agent import PatentAgentPlanner, classify_intent, extract_patent_id


def test_classify_intent_detects_idea_analysis() -> None:
    assert classify_intent("我想设计一个低温样本运输装置，可以怎么改进？") == "idea_analysis"


def test_classify_intent_detects_patent_summary() -> None:
    assert classify_intent("总结 CN206539886U 这篇专利") == "patent_summary"


def test_classify_intent_defaults_to_patent_qa() -> None:
    assert classify_intent("低温样本运输时如何避免容器碰撞？") == "patent_qa"


def test_extract_patent_id_normalizes_spaces() -> None:
    assert extract_patent_id("请介绍 CN 206539886 U") == "CN206539886U"


def test_planner_creates_tool_sequence_for_summary() -> None:
    plan = PatentAgentPlanner().plan("总结 CN206539886U 这篇专利")

    assert plan.intent == "patent_summary"
    assert [step.tool_name for step in plan.steps] == [
        "search_patents",
        "summarize_patent",
        "graph_search",
        "graph_rag_answer",
    ]
