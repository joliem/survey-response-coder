import base64
import io
import json
import unittest

import pandas as pd

from report import generate_notebook


class ReportTests(unittest.TestCase):
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
