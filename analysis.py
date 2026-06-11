"""Data analysis and chart generation."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency, f_oneway, shapiro, kruskal, mannwhitneyu, linregress


def _style(fig: go.Figure) -> go.Figure:
    """Apply consistent font sizing to any Plotly figure."""
    fig.update_layout(font=dict(size=16), title_font_size=20)
    fig.update_xaxes(tickfont_size=15, title_font_size=16, automargin=True)
    fig.update_yaxes(tickfont_size=15, title_font_size=16, automargin=True)
    return fig


def theme_bar_chart(df: pd.DataFrame, theme_col: str, color_map: dict | None = None,
                    total: int | None = None) -> go.Figure:
    counts = df[theme_col].value_counts().reset_index()
    counts.columns = ["Theme", "Count"]
    denom = total if total else counts["Count"].sum()
    counts["Pct"] = (counts["Count"] / denom * 100).round(1)
    counts["Label"] = counts.apply(lambda r: f"{r['Count']} ({r['Pct']}%)", axis=1)
    counts = counts.sort_values("Count")  # ascending so largest ends up at top

    if color_map:
        bar_colors = [color_map.get(t, "#c9c9c9") for t in counts["Theme"]]
    else:
        palette = px.colors.qualitative.Pastel
        bar_colors = [palette[i % len(palette)] for i in range(len(counts))]
    fig = go.Figure(go.Bar(
        x=counts["Count"],
        y=counts["Theme"],
        orientation="h",
        text=counts["Label"],
        textposition="outside",
        textfont=dict(size=15),
        marker_color=bar_colors,
    ))
    fig.update_layout(
        title="Theme Distribution",
        showlegend=False,
        bargap=0.3,
        xaxis={
            "title": "Number of Responses",
            "range": [0, counts["Count"].max() * 1.4],
        },
        yaxis={"automargin": True},
        margin={"l": 0, "r": 20, "t": 60, "b": 40},
    )
    return _style(fig)


def _wrap_label(text: str, width: int = 16) -> str:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return "<br>".join(lines)


def covariate_heatmap(
    df: pd.DataFrame, theme_col: str, covariate_col: str,
    theme_order: list | None = None,
) -> go.Figure:
    # Themes as rows (y-axis), categories as columns (x-axis); normalize per column
    ct = pd.crosstab(df[theme_col], df[covariate_col], normalize="columns") * 100
    ct = ct.round(1)
    if theme_order:
        ct = ct.reindex([t for t in theme_order if t in ct.index])
    wrapped_x = [_wrap_label(str(c)) for c in ct.columns]
    fig = px.imshow(
        ct, text_auto=True, aspect="auto",
        color_continuous_scale="Blues",
        title=f"Theme Distribution by {covariate_col} (% of each category)",
        labels={"color": "% of category", "x": covariate_col, "y": "Theme"},
    )
    fig.update_xaxes(tickvals=list(range(len(ct.columns))), ticktext=wrapped_x, tickangle=-30)
    fig.update_layout(margin={"l": 10, "r": 10, "t": 60, "b": 160})
    return _style(fig)


def covariate_stacked_bar(
    df: pd.DataFrame, theme_col: str, covariate_col: str,
    color_map: dict | None = None,
    theme_order: list | None = None,
) -> go.Figure:
    ct = pd.crosstab(df[covariate_col], df[theme_col])
    pct = (ct.div(ct.sum(axis=1), axis=0) * 100).round(1)
    cat_labels = [_wrap_label(str(c), width=20) for c in pct.index]

    n_cats = len(pct)
    height = max(450, n_cats * 55 + 200)
    palette = px.colors.qualitative.Pastel
    # Rarest first so most-common ends up on top of the stack;
    # legendrank reverses the legend so most-common still appears first.
    themes = list(reversed(theme_order)) if theme_order else list(pct.columns)
    n_present = sum(1 for t in themes if t in pct.columns)

    fig = go.Figure()
    for i, theme in enumerate(themes):
        if theme not in pct.columns:
            continue
        color = color_map.get(theme) if color_map else palette[i % len(palette)]
        fig.add_trace(go.Bar(
            name=theme,
            x=cat_labels,
            y=pct[theme].tolist(),
            marker_color=color,
        ))
    fig.update_layout(
        barmode="stack",
        title=f"Theme Mix by {covariate_col}",
        yaxis_title="% of responses",
        yaxis={"range": [0, 101]},
        xaxis_title=covariate_col,
        xaxis={"automargin": True, "tickangle": -30},
        legend={"title": "Theme", "traceorder": "reversed"},
        height=height,
        margin={"l": 10, "r": 10, "t": 60, "b": 160},
    )
    return _style(fig)


def covariate_stacked_bar_vertical(df: pd.DataFrame, theme_col: str, covariate_col: str) -> go.Figure:
    """Original vertical version — kept for easy rollback."""
    ct = pd.crosstab(df[covariate_col], df[theme_col], normalize="index") * 100
    ct = ct.round(1).reset_index()
    ct_melted = ct.melt(id_vars=covariate_col, var_name="Theme", value_name="Percent")
    fig = px.bar(
        ct_melted, x=covariate_col, y="Percent",
        color="Theme", barmode="stack",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title=f"Theme Mix by {covariate_col}",
    )
    fig.update_layout(
        yaxis_title="% of responses",
        xaxis_title=covariate_col,
        legend_title="Theme",
        margin={"l": 10, "r": 10, "t": 60, "b": 60},
    )
    return _style(fig)


def chi_square_summary(df: pd.DataFrame, theme_col: str, covariate_col: str) -> dict:
    ct = pd.crosstab(df[covariate_col], df[theme_col])
    chi2, p, dof, _ = chi2_contingency(ct)
    n = ct.values.sum()
    cramers_v = (chi2 / (n * (min(ct.shape) - 1))) ** 0.5
    return {
        "chi2": round(chi2, 3),
        "p_value": round(p, 4),
        "dof": dof,
        "cramers_v": round(cramers_v, 3),
        "n": n,
        "significant": p < 0.05,
    }


def explode_themes(df: pd.DataFrame, theme_col: str = "theme") -> pd.DataFrame:
    """Expand pipe-separated multi-theme column so each theme gets its own row."""
    df = df.copy()
    df[theme_col] = df[theme_col].str.split(" | ", regex=False)
    return df.explode(theme_col)


VALENCE_ORDER = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
VALENCE_COLORS = ["#d62728", "#ff7f0e", "#aec7e8", "#98df8a", "#2ca02c"]

EMOTION_ORDER = [
    "Angry", "Frustrated", "Disappointed", "Worried",
    "Confused", "Neutral", "Relieved", "Satisfied",
]
EMOTION_COLORS = [
    "#d62728", "#ff7f0e", "#e377c2", "#9467bd",
    "#8c564b", "#aec7e8", "#98df8a", "#2ca02c",
]


def sentiment_distribution_chart(df: pd.DataFrame, label_col: str) -> go.Figure:
    order = [l for l in VALENCE_ORDER if l in df[label_col].values]
    colors = dict(zip(VALENCE_ORDER, VALENCE_COLORS))
    counts = df[label_col].value_counts().reindex(order).fillna(0).reset_index()
    counts.columns = ["Valence", "Count"]

    fig = go.Figure(go.Bar(
        x=counts["Valence"],
        y=counts["Count"],
        marker_color=[colors.get(v, "#aaa") for v in counts["Valence"]],
    ))
    fig.update_layout(
        title="Valence Distribution",
        showlegend=False,
        bargap=0.3,
        xaxis={
            "categoryorder": "array",
            "categoryarray": order,
            "automargin": True,
            "tickangle": 0,
            "title": "Valence",
        },
        yaxis_title="Count",
        margin={"l": 10, "r": 10, "t": 60, "b": 80},
    )
    return _style(fig)


def sentiment_by_theme_chart(
    df: pd.DataFrame, theme_col: str, score_col: str,
    color_map: dict | None = None,
    theme_order: list | None = None,
) -> go.Figure:
    cat_orders = {theme_col: theme_order} if theme_order else {}
    fig = px.box(
        df, x=theme_col, y=score_col, color=theme_col,
        **({"color_discrete_map": color_map} if color_map else {"color_discrete_sequence": px.colors.qualitative.Pastel}),
        category_orders=cat_orders,
        title="Valence Score by Theme  (1 = Very Negative → 5 = Very Positive)",
        points="outliers",
    )
    fig.update_layout(
        showlegend=False,
        yaxis={
            "tickvals": [1, 2, 3, 4, 5],
            "ticktext": ["1 Very Neg", "2 Neg", "3 Neutral", "4 Pos", "5 Very Pos"],
            "range": [0.5, 5.5],
        },
        xaxis={"automargin": True, "tickangle": -30},
        yaxis_title="Valence Score",
        xaxis_title="",
        margin={"l": 10, "r": 10, "t": 60, "b": 140},
    )
    return _style(fig)


def emotion_distribution_chart(df: pd.DataFrame, emotion_col: str) -> go.Figure:
    order = [e for e in EMOTION_ORDER if e in df[emotion_col].values]
    colors = dict(zip(EMOTION_ORDER, EMOTION_COLORS))
    counts = df[emotion_col].value_counts().reindex(order).fillna(0).reset_index()
    counts.columns = ["Emotion", "Count"]
    # Reverse so the first in EMOTION_ORDER ends up at top of horizontal chart
    counts = counts.iloc[::-1].reset_index(drop=True)

    fig = go.Figure(go.Bar(
        x=counts["Count"],
        y=counts["Emotion"],
        orientation="h",
        marker_color=[colors.get(e, "#aaa") for e in counts["Emotion"]],
    ))
    fig.update_layout(
        title="Emotion Distribution",
        showlegend=False,
        bargap=0.3,
        xaxis_title="Count",
        yaxis={"automargin": True},
        margin={"l": 0, "r": 10, "t": 60, "b": 40},
    )
    return _style(fig)


def emotion_by_theme_chart(
    df: pd.DataFrame, theme_col: str, emotion_col: str,
    color_map: dict | None = None,
    theme_order: list | None = None,
) -> go.Figure:
    # Denominator = ALL responses per theme (including unlabeled)
    theme_totals = df.groupby(theme_col).size()
    labeled = df[df[emotion_col].notna() & (df[emotion_col].str.strip() != "")]
    counts = pd.crosstab(labeled[theme_col], labeled[emotion_col])
    # Divide by total so bars reach only the fraction that were labeled
    pct = (counts.div(theme_totals, axis=0) * 100).round(1)

    # Horizontal bars: first item in y list = bottom, last = top.
    # Reverse theme_order so most common ends up at the top.
    if theme_order:
        themes_y = [t for t in reversed(theme_order) if t in pct.index]
    else:
        themes_y = list(pct.index)
    pct = pct.reindex(themes_y)

    emotion_colors_dict = dict(zip(EMOTION_ORDER, EMOTION_COLORS))
    emotions_present = [e for e in EMOTION_ORDER if e in pct.columns]

    fig = go.Figure()
    for emotion in emotions_present:
        fig.add_trace(go.Bar(
            name=emotion,
            y=themes_y,
            x=pct[emotion].tolist(),
            orientation="h",
            marker_color=emotion_colors_dict.get(emotion),
        ))
    fig.update_layout(
        barmode="stack",
        title="Emotion Mix by Theme",
        xaxis_title="% of all responses in theme",
        yaxis={"automargin": True},
        legend_title="Emotion",
        margin={"l": 10, "r": 10, "t": 60, "b": 40},
    )
    return _style(fig)


def emotion_by_theme_chart_vertical(df: pd.DataFrame, theme_col: str, emotion_col: str) -> go.Figure:
    """Original vertical version — kept for easy rollback."""
    df = df[df[emotion_col].notna() & (df[emotion_col].str.strip() != "")].copy()
    ct = pd.crosstab(df[theme_col], df[emotion_col], normalize="index") * 100
    ct = ct.round(1).reset_index()
    ct_melted = ct.melt(id_vars=theme_col, var_name="Emotion", value_name="Percent")
    colors = dict(zip(EMOTION_ORDER, EMOTION_COLORS))
    fig = px.bar(
        ct_melted, x=theme_col, y="Percent",
        color="Emotion", barmode="stack",
        color_discrete_map=colors,
        title="Emotion Mix by Theme",
        category_orders={"Emotion": EMOTION_ORDER},
    )
    fig.update_layout(
        yaxis_title="% of responses",
        xaxis_title="Theme",
        xaxis_tickangle=-20,
        margin={"l": 10, "r": 10, "t": 60, "b": 100},
    )
    return _style(fig)


_DATE_NAME_HINTS = ("date", "time", "timestamp", "created", "updated", "submitted", "recorded")


def detect_covariate_type(series: pd.Series) -> str:
    """Return 'date', 'numeric', or 'categorical' based on column content."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    sample = series.dropna().head(100)
    if sample.empty:
        return "categorical"
    # Use a lower parse threshold when the column name looks date-like
    name_lower = str(series.name).lower()
    name_is_date = any(hint in name_lower for hint in _DATE_NAME_HINTS)
    threshold = 0.75 if name_is_date else 0.95
    try:
        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= threshold:
            return "date"
    except Exception:
        pass
    return "categorical"


def theme_over_time_chart(
    df: pd.DataFrame, theme_col: str, date_col: str, granularity: str = "Month",
    color_map: dict | None = None,
    theme_order: list | None = None,
) -> go.Figure:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    period_code = {"Day": "D", "Week": "W", "Month": "M"}[granularity]
    df["_period"] = df[date_col].dt.to_period(period_code).dt.start_time
    grouped = df.groupby(["_period", theme_col]).size().reset_index(name="count")
    totals = grouped.groupby("_period")["count"].transform("sum")
    grouped["pct"] = (grouped["count"] / totals * 100).round(1)
    _order = list(reversed(theme_order)) if theme_order else None
    fig = px.area(
        grouped, x="_period", y="pct", color=theme_col,
        **({"color_discrete_map": color_map} if color_map else {"color_discrete_sequence": px.colors.qualitative.Pastel}),
        category_orders={theme_col: _order} if _order else {},
        title=f"Theme Proportion Over Time (by {granularity})",
        labels={"_period": granularity, "pct": "% of responses", theme_col: "Theme"},
    )
    # Reverse legend so most-common (top of stack) appears first
    _n = len(fig.data)
    for _i, _trace in enumerate(fig.data):
        _trace.legendrank = _n - _i
    fig.update_layout(
        yaxis_title="% of responses", xaxis_title=granularity,
        legend_title="Theme", margin={"l": 10, "r": 10, "t": 60, "b": 40},
    )
    return _style(fig)


def theme_over_time_line(
    df: pd.DataFrame, theme_col: str, date_col: str, granularity: str = "Month",
    color_map: dict | None = None,
    theme_order: list | None = None,
) -> go.Figure:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    period_code = {"Day": "D", "Week": "W", "Month": "M"}[granularity]
    df["_period"] = df[date_col].dt.to_period(period_code).dt.start_time
    grouped = df.groupby(["_period", theme_col]).size().reset_index(name="count")
    totals = grouped.groupby("_period")["count"].transform("sum")
    grouped["pct"] = (grouped["count"] / totals * 100).round(1)
    _order = list(reversed(theme_order)) if theme_order else None
    fig = px.line(
        grouped, x="_period", y="pct", color=theme_col,
        markers=True,
        **({"color_discrete_map": color_map} if color_map else {"color_discrete_sequence": px.colors.qualitative.Pastel}),
        category_orders={theme_col: _order} if _order else {},
        title=f"Theme Proportion Over Time (by {granularity})",
        labels={"_period": granularity, "pct": "% of responses", theme_col: "Theme"},
    )
    _n = len(fig.data)
    for _i, _trace in enumerate(fig.data):
        _trace.legendrank = _n - _i
    fig.update_layout(
        yaxis_title="% of responses", xaxis_title=granularity,
        legend_title="Theme", margin={"l": 10, "r": 10, "t": 60, "b": 40},
    )
    return _style(fig)


def confidence_box_chart(
    df: pd.DataFrame, theme_col: str,
    color_map: dict | None = None,
    theme_order: list | None = None,
) -> go.Figure:
    cat_orders = {theme_col: theme_order} if theme_order else {}
    fig = px.box(
        df, x=theme_col, y="confidence",
        color=theme_col,
        **({"color_discrete_map": color_map} if color_map else {"color_discrete_sequence": px.colors.qualitative.Pastel}),
        category_orders=cat_orders,
        title="Confidence Distribution by Theme",
        labels={"confidence": "Confidence (0 = uncertain, 1 = certain)", theme_col: ""},
        points="outliers",
    )
    fig.update_layout(
        showlegend=False,
        xaxis={"automargin": True, "tickangle": -30},
        yaxis_title="Confidence",
        xaxis_title="",
        margin={"l": 10, "r": 10, "t": 60, "b": 140},
    )
    return _style(fig)


def anova_box_chart(
    df: pd.DataFrame, theme_col: str, numeric_col: str,
    color_map: dict | None = None,
    theme_order: list | None = None,
) -> go.Figure:
    cat_orders = {theme_col: theme_order} if theme_order else {}
    fig = px.box(
        df, x=theme_col, y=numeric_col, color=theme_col,
        **({"color_discrete_map": color_map} if color_map else {"color_discrete_sequence": px.colors.qualitative.Pastel}),
        category_orders=cat_orders,
        title=f"Distribution of {numeric_col} by Theme",
        points="outliers",
    )
    fig.update_layout(
        showlegend=False,
        xaxis={"automargin": True, "tickangle": -30},
        yaxis_title=numeric_col,
        xaxis_title="",
        margin={"l": 10, "r": 10, "t": 60, "b": 140},
    )
    return _style(fig)


def anova_summary(df: pd.DataFrame, theme_col: str, numeric_col: str) -> dict:
    named_groups = [
        (name, g[numeric_col].dropna().values)
        for name, g in df.groupby(theme_col)
        if g[numeric_col].dropna().shape[0] > 1
    ]
    if len(named_groups) < 2:
        return None
    group_arrays = [g for _, g in named_groups]
    f_stat, p_value = f_oneway(*group_arrays)
    # Use only rows from analyzed groups so grand_mean and ss_total are consistent
    group_series = [pd.Series(g) for g in group_arrays]
    all_vals = pd.concat(group_series)
    grand_mean = all_vals.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in group_series)
    ss_total = ((all_vals - grand_mean) ** 2).sum()
    eta_sq = round(ss_between / ss_total, 3) if ss_total > 0 else 0.0
    return {
        "f_stat": round(f_stat, 3),
        "p_value": round(p_value, 4),
        "eta_squared": eta_sq,
        "n_groups": len(group_arrays),
        "significant": p_value < 0.05,
    }


def anova_posthoc(df: pd.DataFrame, theme_col: str, numeric_col: str) -> pd.DataFrame:
    """Bonferroni-corrected pairwise Welch t-tests as ANOVA post-hoc."""
    from scipy.stats import ttest_ind
    import itertools

    groups = {
        name: g[numeric_col].dropna().values
        for name, g in df.groupby(theme_col)
        if g[numeric_col].dropna().shape[0] > 1
    }
    pairs = list(itertools.combinations(sorted(groups), 2))
    if not pairs:
        return pd.DataFrame()

    rows = []
    for a, b in pairs:
        _, p = ttest_ind(groups[a], groups[b], equal_var=False)
        p_adj = min(1.0, p * len(pairs))
        rows.append({
            "Theme A": a,
            "Theme B": b,
            "Mean A": round(float(groups[a].mean()), 2),
            "Mean B": round(float(groups[b].mean()), 2),
            "Diff (A−B)": round(float(groups[a].mean() - groups[b].mean()), 2),
            "p-adj (Bonferroni)": round(p_adj, 4),
            "Sig.": "✓" if p_adj < 0.05 else "",
        })
    return pd.DataFrame(rows).sort_values("p-adj (Bonferroni)").reset_index(drop=True)


def normality_summary(df: pd.DataFrame, theme_col: str, numeric_col: str) -> dict:
    """
    Run Shapiro-Wilk per theme group and decide whether ANOVA or Kruskal-Wallis
    is more appropriate.

    Shapiro-Wilk requires 3–5000 observations; groups outside that range are
    skipped and noted.  The recommendation switches to Kruskal-Wallis if any
    testable group fails (p < 0.05).
    """
    rows = []
    any_fail = False
    for name, g in df.groupby(theme_col):
        vals = g[numeric_col].dropna().values
        n = len(vals)
        if n < 3:
            rows.append({"Theme": name, "N": n, "W": None, "p": None, "Normal?": "—", "Note": "< 3 obs"})
            continue
        sample = vals if n <= 5000 else vals[:5000]
        w, p = shapiro(sample)
        passes = p >= 0.05
        if not passes:
            any_fail = True
        note = f"sampled 5000/{n}" if n > 5000 else ""
        rows.append({
            "Theme": name,
            "N": n,
            "W": round(float(w), 4),
            "p": round(float(p), 4),
            "Normal?": "✓" if passes else "✗",
            "Note": note,
        })

    recommendation = "kruskal" if any_fail else "anova"
    return {"groups": rows, "recommendation": recommendation, "any_fail": any_fail}


def kruskal_summary(df: pd.DataFrame, theme_col: str, numeric_col: str) -> dict | None:
    """Kruskal-Wallis H test + epsilon-squared effect size."""
    named = [
        (name, g[numeric_col].dropna().values)
        for name, g in df.groupby(theme_col)
        if g[numeric_col].dropna().shape[0] > 1
    ]
    if len(named) < 2:
        return None
    arrays = [v for _, v in named]
    h, p = kruskal(*arrays)
    n_total = sum(len(a) for a in arrays)
    k = len(arrays)
    # Epsilon-squared: unbiased effect size for Kruskal-Wallis
    eps_sq = (h - k + 1) / (n_total - k) if n_total > k else 0.0
    eps_sq = round(max(0.0, eps_sq), 3)
    return {
        "h_stat": round(float(h), 3),
        "p_value": round(float(p), 4),
        "epsilon_squared": eps_sq,
        "n_groups": k,
        "significant": p < 0.05,
    }


def kruskal_posthoc(df: pd.DataFrame, theme_col: str, numeric_col: str) -> pd.DataFrame:
    """Bonferroni-corrected pairwise Mann-Whitney U tests as Kruskal-Wallis post-hoc."""
    import itertools
    groups = {
        name: g[numeric_col].dropna().values
        for name, g in df.groupby(theme_col)
        if g[numeric_col].dropna().shape[0] > 1
    }
    pairs = list(itertools.combinations(sorted(groups), 2))
    if not pairs:
        return pd.DataFrame()
    rows = []
    for a, b in pairs:
        _, p = mannwhitneyu(groups[a], groups[b], alternative="two-sided")
        p_adj = min(1.0, p * len(pairs))
        rows.append({
            "Theme A": a,
            "Theme B": b,
            "Median A": round(float(pd.Series(groups[a]).median()), 2),
            "Median B": round(float(pd.Series(groups[b]).median()), 2),
            "Diff (A−B)": round(float(pd.Series(groups[a]).median() - pd.Series(groups[b]).median()), 2),
            "p-adj (Bonferroni)": round(float(p_adj), 4),
            "Sig.": "✓" if p_adj < 0.05 else "",
        })
    return pd.DataFrame(rows).sort_values("p-adj (Bonferroni)").reset_index(drop=True)


def trend_test_summary(
    df: pd.DataFrame, theme_col: str, date_col: str, granularity: str = "Month"
) -> pd.DataFrame:
    """
    Per-theme linear trend test: regress proportion-per-period on time index.
    Returns a DataFrame sorted by absolute slope (strongest trends first).
    Flagged as exploratory — time series can have autocorrelation.
    """
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col])
    period_code = {"Day": "D", "Week": "W", "Month": "M"}[granularity]
    tmp["_period"] = tmp[date_col].dt.to_period(period_code).dt.start_time

    # Period totals from raw data
    period_totals = tmp.groupby("_period").size()
    periods = sorted(period_totals.index)
    if len(periods) < 3:
        return pd.DataFrame()

    # Pivot to (period × theme), zero-fill periods where a theme had no responses
    pivot = (
        tmp.groupby(["_period", theme_col]).size()
        .unstack(fill_value=0)
        .reindex(periods, fill_value=0)
    )
    pct_pivot = pivot.div(period_totals, axis=0) * 100
    period_index = {p: i for i, p in enumerate(periods)}

    rows = []
    for theme in pct_pivot.columns:
        pcts = pct_pivot[theme].values
        t_idx = list(range(len(periods)))
        res = linregress(t_idx, pcts)
        rows.append({
            "Theme": theme,
            "Slope (pp/period)": round(float(res.slope), 3),
            "Direction": "↑ Rising" if res.slope > 0 else "↓ Falling",
            "R²": round(float(res.rvalue ** 2), 3),
            "p-value": round(float(res.pvalue), 4),
            "Sig.": "✓" if res.pvalue < 0.05 else "",
            "Periods": len(periods),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("Slope (pp/period)", key=abs, ascending=False).reset_index(drop=True)


def _agreement_label(k: float) -> str:
    if k < 0:     return "Less than chance"
    if k < 0.20:  return "Slight"
    if k < 0.40:  return "Fair"
    if k < 0.60:  return "Moderate"
    if k < 0.80:  return "Substantial"
    return "Almost perfect"


def irr_cohen_kappa(rater1: list, rater2: list) -> dict:
    """Cohen's Kappa for two-rater nominal agreement (no external dependencies)."""
    from collections import Counter
    n = len(rater1)
    categories = sorted(set(rater1) | set(rater2))
    p_o = sum(a == b for a, b in zip(rater1, rater2)) / n
    c1, c2 = Counter(rater1), Counter(rater2)
    p_e = sum((c1[c] / n) * (c2[c] / n) for c in categories)
    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1.0 else (1.0 if p_o == 1.0 else 0.0)
    return {"kappa": round(kappa, 3), "interpretation": _agreement_label(kappa), "pct_agree": round(p_o * 100, 1)}


def irr_krippendorff_alpha(rater1: list, rater2: list) -> dict:
    """
    Krippendorff's Alpha for nominal data, 2 raters, no missing values.

    Uses pooled marginals rather than per-rater marginals, so it handles
    asymmetric label distributions better than Cohen's Kappa and generalises
    to >2 raters or ordinal/interval scales (not implemented here).
    Krippendorff recommends α ≥ 0.80 for reliable conclusions,
    0.67–0.80 for tentative conclusions.
    """
    categories = sorted(set(rater1) | set(rater2))
    n_units = len(rater1)

    # Coincidence matrix: each unit contributes 2 entries (one per direction)
    o: dict[tuple, float] = {(c, k): 0.0 for c in categories for k in categories}
    for a, b in zip(rater1, rater2):
        o[a, b] += 1.0
        o[b, a] += 1.0

    n = 2.0 * n_units  # total entries in coincidence matrix
    n_k = {k: sum(o[c, k] for c in categories) for k in categories}

    # Observed disagreement (nominal metric: d²=1 when c≠k)
    D_o = sum(o[c, k] for c in categories for k in categories if c != k) / n
    # Expected disagreement using pooled marginals
    D_e = (n * n - sum(n_k[k] ** 2 for k in categories)) / (n * (n - 1))

    if D_e == 0:
        alpha = 1.0 if D_o == 0 else 0.0
    else:
        alpha = 1.0 - D_o / D_e

    return {"alpha": round(alpha, 3), "interpretation": _agreement_label(alpha)}


def top_words_per_theme(
    df: pd.DataFrame, theme_col: str, text_col: str, top_n: int = 10,
    theme_order: list | None = None,
) -> dict[str, list[str]]:
    STOPWORDS = {
        "i", "me", "my", "the", "a", "an", "and", "or", "but", "in", "on", "at",
        "to", "for", "of", "with", "was", "is", "are", "be", "been", "have", "has",
        "had", "this", "that", "they", "their", "them", "it", "its", "not", "no",
        "by", "from", "as", "up", "about", "into", "through", "when", "which",
        "who", "will", "would", "could", "should", "do", "did", "does", "am",
        "were", "he", "she", "we", "you", "your", "his", "her", "our", "there",
        "then", "than", "so", "if", "also", "just", "more", "out", "after",
        "told", "called", "said", "even", "still", "since", "over", "back",
    }
    themes = theme_order if theme_order else list(df[theme_col].unique())
    result = {}
    for theme in themes:
        if theme not in df[theme_col].values:
            continue
        text = " ".join(df.loc[df[theme_col] == theme, text_col].dropna()).lower()
        words = [w.strip(".,!?;:\"'()[]") for w in text.split()]
        words = [w for w in words if len(w) > 3 and w not in STOPWORDS
                 and not all(c in "x/" for c in w)]
        freq = pd.Series(words).value_counts()
        result[theme] = freq.head(top_n).index.tolist()
    return result
