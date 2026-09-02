import base64
import io
import json
import unittest
from pathlib import Path

import pandas as pd

from report import generate_notebook, prepare_download_dataframe


class ReportTests(unittest.TestCase):
    def test_public_download_filenames_are_clean(self):
        app_source = (Path(__file__).parents[1] / "app.py").read_text()
        self.assertIn('file_name="coded_responses.csv"', app_source)
        self.assertIn('file_name="survey_analysis.ipynb"', app_source)
        self.assertNotIn('file_name="coded_responses_corrected.csv"', app_source)
        self.assertNotIn('file_name="survey_analysis_corrected.ipynb"', app_source)

    def test_theme_toggle_does_not_depend_on_a_hot_reloaded_helper_signature(self):
        app_source = (Path(__file__).parents[1] / "app.py").read_text()
        self.assertNotIn("unit=_chart_unit", app_source)
        self.assertIn('update_xaxes(title_text=f"Number of {_chart_unit}")', app_source)

    def test_download_schema_honors_multi_theme_and_sentiment_selections(self):
        base = pd.DataFrame({
            "response": ["one"], "theme": ["A"], "primary_theme": ["A"],
            "confidence": [0.9], "_true_theme": ["A"],
        })
        for multi_theme in (False, True):
            for valence in (False, True):
                for emotion in (False, True):
                    df = base.copy()
                    if valence:
                        df["valence_score"], df["valence_label"] = 4, "Positive"
                    if emotion:
                        df["emotion"] = "Satisfied"
                    result = prepare_download_dataframe(df, multi_theme)
                    self.assertEqual("theme_list" in result, multi_theme)
                    self.assertEqual("valence_score" in result, valence)
                    self.assertEqual("valence_label" in result, valence)
                    self.assertEqual("emotion" in result, emotion)
                    self.assertFalse(any(c.startswith("_") for c in result))
                    raw = generate_notebook(
                        result, "response", [], [{"name": "A", "description": "A"}],
                        include_valence=valence, include_emotion=emotion,
                    )
                    nb = json.loads(raw)
                    cell = next(
                        c for c in nb["cells"]
                        if c["cell_type"] == "code" and "_b64 =" in c["source"]
                    )
                    embedded = pd.read_csv(io.BytesIO(base64.b64decode(cell["source"].split('"""', 2)[1].strip())))
                    pd.testing.assert_frame_equal(embedded, result)

    def test_notebook_uses_report_screen_covariate_override(self):
        df = pd.DataFrame({
            "response": ["one", "two"], "primary_theme": ["A", "B"],
            "numeric_code": [1, 2],
        })
        raw = generate_notebook(
            df, "response", ["numeric_code"],
            [{"name": "A", "description": "A"}, {"name": "B", "description": "B"}],
            covariate_types={"numeric_code": "categorical"},
        ).decode()
        self.assertIn("Theme Mix by numeric_code", raw)
        self.assertNotIn("Welch ANOVA — numeric_code", raw)

    def test_alpha_thresholds_trim_unnecessary_trailing_zeroes(self):
        df = pd.DataFrame({
            "response": [str(i) for i in range(20)],
            "primary_theme": list("AB") * 10,
            **{f"group_{j}": list("xy") * 10 for j in range(5)},
        })
        raw = generate_notebook(
            df, "response", [f"group_{j}" for j in range(5)],
            [{"name": "A", "description": "A"}, {"name": "B", "description": "B"}],
        )
        nb = json.loads(raw)
        text = "\n".join(cell["source"] for cell in nb["cells"])
        self.assertIn("threshold is **α = 0.01**", text)
        self.assertIn("familywise α = 0.05", text)
        self.assertNotIn("α = 0.010", text)

    def test_notebook_embeds_exact_download_dataframe(self):
        df = pd.DataFrame({
            "response": ["one", "two"],
            "primary_theme": ["A", "B"],
            "group": ["x", "y"],
        })
        raw = generate_notebook(
            df, "response", ["group"],
            [{"name": "A", "description": "A"}, {"name": "B", "description": "B"}],
        )
        nb = json.loads(raw)
        cell = next(c for c in nb["cells"] if c["cell_type"] == "code" and "_b64 =" in c["source"])
        encoded = cell["source"].split('"""', 2)[1].strip()
        embedded = pd.read_csv(io.BytesIO(base64.b64decode(encoded)))
        pd.testing.assert_frame_equal(embedded, df)

    def test_notebook_uses_same_corrected_methods(self):
        df = pd.DataFrame({
            "response": [str(i) for i in range(40)],
            "primary_theme": list("AB") * 20,
            "group": [f"g{i % 10}" for i in range(40)],
        })
        text = generate_notebook(df, "response", ["group"], [
            {"name": "A", "description": "A"}, {"name": "B", "description": "B"}
        ]).decode()
        self.assertIn("Permutation chi-square", text)
        self.assertIn("normalize='index'", text)
        self.assertNotIn("p={_p:.4f}", text)
        self.assertIn("fig.update_layout(font=dict(size=16), title_font_size=20)", text)
        self.assertNotIn("font-family:", text)
        self.assertNotIn("font=dict(family=", text)
        self.assertIn("These are expected cell counts, not observed category totals", text)
        self.assertNotIn("Expected-count diagnostic: minimum", text)
        self.assertNotIn("p = <", text)
        self.assertNotIn("significant", text.lower())
        self.assertIn("Bonferroni-adjusted threshold", text)
        nb = json.loads(text)
        covariate_chart = next(
            cell["source"] for cell in nb["cells"]
            if cell["cell_type"] == "code" and "Theme Mix by group" in cell["source"]
        )
        self.assertIn("x=_pct.index.tolist()", covariate_chart)
        self.assertIn("'ticktext': _cat_labels", covariate_chart)
        self.assertNotIn("<br>", covariate_chart)
        self.assertNotIn("orientation='h'", covariate_chart)

    def test_notebook_matches_valence_and_emotion_chart_orders(self):
        df = pd.DataFrame({
            "response": [str(i) for i in range(20)],
            "primary_theme": list("AB") * 10,
            "valence_score": [2, 4] * 10,
            "valence_label": ["Negative", "Positive"] * 10,
            "emotion": ["Angry", "Relieved"] * 10,
        })
        text = generate_notebook(
            df, "response", [],
            [{"name": "A", "description": "A"}, {"name": "B", "description": "B"}],
            include_valence=True, include_emotion=True,
        ).decode()
        self.assertIn("_val_order = VALENCE_ORDER", text)
        self.assertIn("xaxis={'title': 'Number of Responses'", text)
        self.assertIn("text=_val_counts['Label']", text)
        self.assertIn("text=_emo_counts['Label']", text)
        self.assertIn("unadjusted per-test threshold; no cross-test correction", text)
        self.assertIn("'1 Very Negative','2 Negative','3 Neutral','4 Positive','5 Very Positive'", text)
        self.assertNotIn("_emo_counts = _emo_counts.iloc[::-1]", text)

    def test_multi_theme_notebook_matches_selected_distribution_view(self):
        df = pd.DataFrame({
            "response": ["one", "two"],
            "primary_theme": ["A", "B"],
            "theme_list": ["A | B", "B"],
        })
        taxonomy = [{"name": "A", "description": "A"}, {"name": "B", "description": "B"}]
        tags = generate_notebook(df, "response", [], taxonomy, theme_distribution_view="all_tags").decode()
        primary = generate_notebook(df, "response", [], taxonomy, theme_distribution_view="primary_themes").decode()
        self.assertIn("_unit = 'Theme Tags'", tags)
        self.assertIn("_unit = 'Responses'", primary)
        self.assertIn("df_analysis['primary_theme'].value_counts()", primary)


if __name__ == "__main__":
    unittest.main()
