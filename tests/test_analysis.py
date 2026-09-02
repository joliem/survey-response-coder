import unittest

import numpy as np
import pandas as pd

from analysis import (
    chi_square_summary,
    covariate_heatmap,
    covariate_stacked_bar,
    format_p_value,
    theme_over_time_line,
    trend_test_summary,
)


class AnalysisTests(unittest.TestCase):
    def test_sparse_table_uses_reproducible_permutation_test(self):
        df = pd.DataFrame({"group": list("ABCDEFGHIJ") * 4, "theme": list("XXYY") * 10})
        first = chi_square_summary(df, "theme", "group", n_permutations=999)
        second = chi_square_summary(df, "theme", "group", n_permutations=999)
        self.assertIn("Permutation", first["method"])
        self.assertEqual(first["p_value"], second["p_value"])
        self.assertGreater(first["expected_below_5_pct"], 20)

    def test_dense_table_uses_pearson(self):
        df = pd.DataFrame({
            "group": np.repeat(["A", "B"], 100),
            "theme": (["X"] * 50 + ["Y"] * 50) * 2,
        })
        result = chi_square_summary(df, "theme", "group")
        self.assertEqual(result["method"], "Pearson chi-square")

    def test_heatmap_percentages_sum_within_category(self):
        df = pd.DataFrame({"group": ["A", "A", "B", "B"], "theme": ["X", "Y", "X", "X"]})
        fig = covariate_heatmap(df, "theme", "group")
        np.testing.assert_allclose(np.asarray(fig.data[0].z).sum(axis=0), [100, 100])

    def test_stacked_bar_keeps_long_category_labels_horizontal_and_wrapped(self):
        df = pd.DataFrame({
            "group": ["Checking or savings account", "Payday loan, title loan, or personal loan"] * 2,
            "theme": ["X", "Y", "Y", "X"],
        })
        fig = covariate_stacked_bar(df, "theme", "group")
        self.assertTrue(all(trace.orientation == "h" for trace in fig.data))
        self.assertTrue(any("<br>" in label for label in fig.data[0].y))
        self.assertEqual(fig.layout.yaxis.autorange, "reversed")

    def test_time_chart_inserts_zero_theme_periods(self):
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-03-01"],
            "theme": ["A", "B"],
        })
        fig = theme_over_time_line(df, "theme", "date", "Month")
        self.assertTrue(all(len(trace.x) == 2 for trace in fig.data))
        self.assertTrue(any(0 in trace.y for trace in fig.data))

    def test_p_value_format_never_displays_zero(self):
        self.assertEqual(format_p_value(0.0001), "< 0.001")
        self.assertEqual(format_p_value(0.12349), "0.123")

    def test_trends_use_holm_adjustment(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="MS")
        rows = []
        for i, date in enumerate(dates):
            rows.extend({"date": date, "theme": "A"} for _ in range(i + 1))
            rows.extend({"date": date, "theme": "B"} for _ in range(10 - i))
        result = trend_test_summary(pd.DataFrame(rows), "theme", "date")
        self.assertIn("p-adj (Holm)", result.columns)
        self.assertTrue((result["p-adj (Holm)"] >= result["p-value"]).all())


if __name__ == "__main__":
    unittest.main()
