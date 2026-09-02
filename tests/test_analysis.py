import unittest

import numpy as np
import pandas as pd

from analysis import (
    chi_square_summary,
    covariate_heatmap,
    covariate_stacked_bar,
    emotion_distribution_chart,
    format_p_value,
    sentiment_by_theme_chart,
    sentiment_distribution_chart,
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

    def test_categorical_charts_use_single_line_truncated_labels_with_full_values(self):
        df = pd.DataFrame({
            "group": ["Checking or savings account", "Payday loan, title loan, or personal loan"] * 2,
            "theme": ["X", "Y", "Y", "X"],
        })
        fig = covariate_stacked_bar(df, "theme", "group")
        self.assertTrue(all(trace.orientation in (None, "v") for trace in fig.data))
        self.assertTrue(all("<br>" not in label for label in fig.layout.xaxis.ticktext))
        self.assertTrue(any(label.endswith("…") for label in fig.layout.xaxis.ticktext))
        self.assertIn("Payday loan, title loan, or personal loan", fig.data[0].x)
        self.assertEqual(fig.layout.xaxis.tickangle, -30)
        self.assertIsNone(fig.layout.font.family)

        heatmap = covariate_heatmap(df, "theme", "group")
        self.assertTrue(all("<br>" not in label for label in heatmap.layout.xaxis.ticktext))
        self.assertTrue(any(label.endswith("…") for label in heatmap.layout.xaxis.ticktext))

    def test_valence_charts_share_full_ordered_y_axis_labels(self):
        df = pd.DataFrame({
            "valence_label": ["Negative", "Positive"],
            "valence_score": [2, 4],
            "theme": ["A", "B"],
        })
        overall = sentiment_distribution_chart(df, "valence_label")
        self.assertEqual(overall.data[0].orientation, "h")
        self.assertEqual(list(overall.data[0].y), [
            "Very Negative", "Negative", "Neutral", "Positive", "Very Positive",
        ])
        self.assertEqual(list(overall.data[0].x), [0, 1, 0, 1, 0])

        by_theme = sentiment_by_theme_chart(df, "theme", "valence_score")
        self.assertEqual(list(by_theme.layout.yaxis.ticktext), [
            "1 Very Negative", "2 Negative", "3 Neutral", "4 Positive", "5 Very Positive",
        ])

    def test_emotion_overall_places_angry_at_bottom(self):
        df = pd.DataFrame({"emotion": ["Angry", "Worried", "Relieved"]})
        fig = emotion_distribution_chart(df, "emotion")
        self.assertEqual(list(fig.data[0].y), ["Angry", "Worried", "Relieved"])

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
