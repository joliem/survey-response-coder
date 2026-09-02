import pandas as pd

from demo_data import code_responses_demo


def _demo_row(primary="Old A", secondary="Old B"):
    return pd.DataFrame({
        "_true_theme": [primary], "_true_theme_2": [secondary],
        "_true_confidence": [0.88], "_true_valence_score": [2],
        "_true_valence_label": ["Negative"], "_true_emotion": ["Frustrated"],
    })


def test_demo_rename_and_merge_translate_precomputed_labels():
    renamed = [
        {"name": "New A", "_source_names": ["Old A"]},
        {"name": "New B", "_source_names": ["Old B"]},
    ]
    result = code_responses_demo(
        ["irrelevant text"], renamed, df=_demo_row(), multi_theme=True,
        include_valence=True, include_emotion=True,
    )[0]
    assert result["themes"] == ["New A", "New B"]
    assert result["confidence"] == 0.88
    assert result["label"] == "Negative"
    assert result["emotion"] == "Frustrated"

    merged = [{"name": "Combined", "_source_names": ["Old A", "Old B"]}]
    result = code_responses_demo(
        ["irrelevant text"], merged, df=_demo_row(), multi_theme=True,
    )[0]
    assert result["themes"] == ["Combined"]


def test_demo_split_uses_keywords_instead_of_claiming_precomputed_assignment():
    split = [
        {"name": "Waiting", "_source_names": ["Old A"], "_keywords": ["delay"]},
        {"name": "Rudeness", "_source_names": ["Old A"], "_keywords": ["rude"]},
    ]
    result = code_responses_demo(
        ["The customer service agent was extremely rude to me"],
        split, df=_demo_row(secondary=None),
    )[0]
    assert result["themes"] == ["Rudeness"]
