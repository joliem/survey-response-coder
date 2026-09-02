import base64
import io
import json
import unittest

import pandas as pd

from report import generate_notebook, prepare_download_dataframe


class ReportTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
