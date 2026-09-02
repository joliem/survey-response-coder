"""Generate a self-contained Jupyter notebook analysis report."""

import base64
import json
import uuid
import pandas as pd
from analysis import chi_square_summary


def _cid() -> str:
    return uuid.uuid4().hex[:8]


def _md(source: str) -> dict:
    return {"cell_type": "markdown", "id": _cid(), "metadata": {}, "source": source}


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _cid(),
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def _detect_type(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    name_lower = str(series.name).lower()
    date_hints = ("date", "time", "timestamp", "created", "updated", "submitted", "recorded")
    sample = series.dropna().head(100)
    if any(h in name_lower for h in date_hints) and not sample.empty:
        try:
            if pd.to_datetime(sample, errors="coerce").notna().mean() >= 0.75:
                return "date"
        except Exception:
            pass
    return "categorical"


def prepare_download_dataframe(
    coded_df: pd.DataFrame,
    multi_theme: bool,
    quotes: dict | None = None,
) -> pd.DataFrame:
    """Build the one canonical dataset used by both CSV and notebook downloads."""
    drop_cols = [c for c in coded_df.columns if c.startswith("_")]
    if not multi_theme and "theme" in coded_df.columns:
        drop_cols.append("theme")
    result = coded_df.drop(columns=drop_cols).copy()
    if multi_theme and "theme" in result.columns:
        result = result.rename(columns={"theme": "theme_list"})

    if quotes:
        role_by_row, excerpt_by_row = {}, {}
        for picks in quotes.values():
            for pick in picks:
                role_by_row[pick["row"]] = pick["role"]
                excerpt_by_row[pick["row"]] = pick["quote"]
        result["highlighted_quote"] = result.index.map(lambda i: role_by_row.get(i, ""))
        result["quote_excerpt"] = result.index.map(lambda i: excerpt_by_row.get(i, ""))
    return result


# ── Notebook-embedded versions of the _style helper and chart helpers ──────────
# Keep these in sync with analysis.py whenever that file changes.

_STYLE_FN = """\
def _style(fig):
    fig.update_layout(font=dict(family='Arial, sans-serif', size=16), title_font_size=20)
    fig.update_xaxes(tickfont_size=15, title_font_size=16, automargin=True)
    fig.update_yaxes(tickfont_size=15, title_font_size=16, automargin=True)
    return fig
"""

_WRAP_FN = """\
def _wrap_label(text, width=16):
    words = text.split()
    lines, cur = [], ''
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur); cur = w
        else:
            cur = (cur + ' ' + w).strip()
    if cur:
        lines.append(cur)
    return '<br>'.join(lines)
"""

_ASSOCIATION_FN = '''\
def association_test(data, category_col, theme_col, n_permutations=10000, random_seed=20260902):
    ct = pd.crosstab(data[category_col], data[theme_col])
    chi2, asymptotic_p, dof, expected = chi2_contingency(ct, correction=False)
    expected_min = float(expected.min()); below5 = float((expected < 5).mean() * 100)
    sparse = expected_min < 1 or below5 > 20
    if sparse:
        a = pd.Categorical(data[category_col], categories=ct.index).codes
        b = pd.Categorical(data[theme_col], categories=ct.columns).codes
        valid = (a >= 0) & (b >= 0); a, b = a[valid], b[valid]
        rng = np.random.default_rng(random_seed); extreme = 0
        for _ in range(n_permutations):
            sim = np.zeros(ct.shape, dtype=np.int64)
            np.add.at(sim, (a, rng.permutation(b)), 1)
            extreme += np.sum((sim - expected) ** 2 / expected) >= chi2 - 1e-12
        p = (extreme + 1) / (n_permutations + 1)
        method = f'Permutation chi-square ({n_permutations:,} permutations)'
    else:
        p = float(asymptotic_p); method = 'Pearson chi-square'
    r, k = ct.shape; n = int(ct.values.sum()); phi2 = chi2 / n
    phi2c = max(0.0, phi2 - ((k-1)*(r-1))/(n-1)) if n > 1 else 0.0
    rc = r - ((r-1)**2)/(n-1) if n > 1 else 1.0
    kc = k - ((k-1)**2)/(n-1) if n > 1 else 1.0
    denom = min(rc-1, kc-1); v = (phi2c/denom)**0.5 if denom > 0 else 0.0
    return {'table': ct, 'chi2': chi2, 'p_value': p, 'dof': dof,
            'cramers_v': v, 'n': n, 'method': method,
            'expected_min': expected_min, 'expected_below_5_pct': below5}
'''

_WELCH_FN = '''\
def welch_anova(groups):
    arrays = list(groups.values()); ns = np.array([len(v) for v in arrays], float)
    means = np.array([np.mean(v) for v in arrays]); variances = np.array([np.var(v, ddof=1) for v in arrays])
    weights = ns / variances; wsum = weights.sum(); mean = np.sum(weights * means) / wsum
    k = len(arrays); correction = np.sum(((1 - weights/wsum)**2) / (ns-1))
    stat = (np.sum(weights * (means-mean)**2)/(k-1)) / (1 + 2*(k-2)/(k*k-1)*correction)
    df2 = (k*k-1)/(3*correction); return stat, scipy_f.sf(stat, k-1, df2), k-1, df2
'''


def generate_notebook(
    coded_df: pd.DataFrame,
    text_col: str,
    covariate_cols: list,
    taxonomy: list,
    include_valence: bool = False,
    include_emotion: bool = False,
    theme_order: list | None = None,
    quotes: dict | None = None,
    covariate_types: dict | None = None,
) -> bytes:
    """Return .ipynb bytes for a self-contained analysis report."""
    _NONE = "None of the above"
    is_multi = "theme_list" in coded_df.columns
    has_conf = "confidence" in coded_df.columns
    n_resp = len(coded_df)
    _themed_df = coded_df[coded_df["primary_theme"] != _NONE]
    n_themes = _themed_df["primary_theme"].nunique()

    if not theme_order:
        theme_order = coded_df["primary_theme"].value_counts().index.tolist()
    theme_order_repr = repr(theme_order)

    b64 = base64.b64encode(coded_df.to_csv(index=False).encode()).decode()

    cells = []

    # ── Title & taxonomy ───────────────────────────────────────
    taxonomy_lines = "\n".join(f"- **{t['name']}**: {t['description']}" for t in taxonomy)
    theme_counts = _themed_df["primary_theme"].value_counts()
    top_theme = theme_counts.index[0]
    top_pct = round(theme_counts.iloc[0] / n_resp * 100, 1)
    cells.append(_md(
        "<style>\n"
        ".rendered_html, .jp-RenderedHTMLCommon, .jp-MarkdownOutput, "
        ".output_area, .output_area pre { font-family: Arial, sans-serif !important; }\n"
        "</style>\n\n"
        f"# Survey Response Coder — Analysis Report\n\n"
        f"**{n_resp:,} responses** · **{n_themes} themes**\n\n"
        f"Generated with the [Survey Response Coder](https://survey-response-coder.streamlit.app) "
        f"by [Jolie Martin](https://github.com/joliem/survey-response-coder).\n\n"
        f"This notebook is self-contained — the coded dataset is embedded in the cell below.\n"
        f"Run cells top-to-bottom; customize any cell as needed.\n\n"
        f"---\n\n"
        f"## Coding Taxonomy\n\n{taxonomy_lines}"
    ))

    # ── Representative quotes (verbatim) ────────────────────────
    if quotes:
        quote_lines = [
            "## Representative Quotes\n",
            "_Verbatim responses selected to capture each theme._\n",
        ]
        any_quotes = False
        for theme in (theme_order or list(quotes.keys())):
            picks = quotes.get(theme) or []
            if not picks:
                continue
            any_quotes = True
            quote_lines.append(f"### {theme}")
            for p in picks:
                excerpt = str(p.get("quote", "")).replace("\n", " ").strip()
                reason = str(p.get("reason", "")).strip()
                quote_lines.append(f"> {excerpt}\n")
                if reason:
                    quote_lines.append(f"_{reason}_\n")
        if any_quotes:
            cells.append(_md("\n".join(quote_lines)))

    # ── Setup & load ───────────────────────────────────────────
    cells.append(_code(
        "# Install if needed: pip install pandas plotly scipy\n"
        "import base64, io, itertools\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import plotly.express as px\n"
        "import plotly.graph_objects as go\n"
        "from scipy.stats import chi2_contingency, ttest_ind, shapiro, kruskal, mannwhitneyu, f as scipy_f\n"
        "pd.set_option('display.max_colwidth', 120)"
    ))

    cells.append(_code(
        f"_b64 = \"\"\"\n{b64}\n\"\"\"\n"
        "df = pd.read_csv(io.StringIO(base64.b64decode(_b64.strip()).decode()))\n"
        f"THEME_ORDER = {theme_order_repr}\n"
        "palette = px.colors.qualitative.Pastel\n"
        "NONE_THEME = 'None of the above'\n"
        "# Assign colors only to real themes (same order as the live app: rarest → palette[0]).\n"
        "_colored = [t for t in THEME_ORDER if t != NONE_THEME]\n"
        "THEME_COLORS = {t: palette[i % len(palette)] for i, t in enumerate(reversed(_colored))}\n"
        "df_analysis = df[df['primary_theme'] != NONE_THEME].copy()\n"
        "VALENCE_ORDER = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']\n"
        "VALENCE_COLORS = ['#d62728', '#ff7f0e', '#aec7e8', '#98df8a', '#2ca02c']\n"
        "EMOTION_ORDER = ['Angry','Frustrated','Disappointed','Worried','Confused','Neutral','Relieved','Satisfied']\n"
        "EMOTION_COLORS = ['#d62728','#ff7f0e','#e377c2','#9467bd','#8c564b','#aec7e8','#98df8a','#2ca02c']\n"
        f"{_STYLE_FN}"
        f"{_WRAP_FN}"
        f"{_ASSOCIATION_FN}"
        f"{_WELCH_FN}"
        f"print(f'{{len(df):,}} responses · {{df.shape[1]}} columns')\n"
        "df.head()"
    ))

    # ── Theme distribution ─────────────────────────────────────
    if is_multi:
        theme_intro = (
            "Chart shows **theme tags** — each response can contribute to more than one bar. "
            "Statistical analyses below use `primary_theme` (one value per response)."
        )
        count_code = (
            "# Explode pipe-separated themes; exclude 'None of the above' as the live app does.\n"
            "_exploded = df_analysis.copy()\n"
            "_exploded['_t'] = _exploded['theme_list'].str.split(' | ', regex=False)\n"
            "_exploded = _exploded.explode('_t')\n"
            "_exploded = _exploded[_exploded['_t'] != NONE_THEME]\n"
            "counts = _exploded['_t'].value_counts().reset_index()\n"
            "counts.columns = ['Theme', 'Count']\n"
            "_denom = counts['Count'].sum()  # % of total theme tags\n"
        )
    else:
        theme_intro = f"Most common theme: **{top_theme}** ({top_pct}% of responses)."
        count_code = (
            "counts = df_analysis['primary_theme'].value_counts().reset_index()\n"
            "counts.columns = ['Theme', 'Count']\n"
            "_denom = len(df)  # % of all responses (including 'None of the above')\n"
        )
    cells.append(_md(f"## Theme Distribution\n\n{theme_intro}"))
    cells.append(_code(
        f"{count_code}"
        "counts['Pct'] = (counts['Count'] / _denom * 100).round(1)\n"
        "counts['Label'] = counts.apply(lambda r: f\"{r['Count']} ({r['Pct']}%)\", axis=1)\n"
        "counts = counts.sort_values('Count')  # ascending — largest bar ends at top\n\n"
        "fig = go.Figure(go.Bar(\n"
        "    x=counts['Count'], y=counts['Theme'], orientation='h',\n"
        "    text=counts['Label'], textposition='outside', textfont=dict(size=15),\n"
        "    marker_color=[THEME_COLORS.get(t, '#c9c9c9') for t in counts['Theme']],\n"
        "))\n"
        "fig.update_layout(\n"
        "    title='Theme Distribution', showlegend=False, bargap=0.3,\n"
        "    xaxis={'title': 'Number of Responses', 'range': [0, counts['Count'].max() * 1.4]},\n"
        "    yaxis={'automargin': True},\n"
        "    margin={'l': 0, 'r': 20, 't': 60, 'b': 40},\n"
        ")\n"
        "_style(fig).show()"
    ))

    # ── Confidence ─────────────────────────────────────────────
    if has_conf:
        avg_conf = round(_themed_df["confidence"].mean(), 2)
        low_n = int((_themed_df["confidence"] < 0.5).sum())
        cells.append(_md(
            f"## Coding Confidence\n\n"
            f"Mean: **{avg_conf}** · Low-confidence responses (<0.5): **{low_n}** "
            f"({round(low_n / n_resp * 100, 1)}%).\n\n"
            "Low-confidence responses are good candidates for manual review.\n\n"
            "_Scoped to themed responses; 'None of the above' responses are excluded._"
        ))
        cells.append(_code(
            "fig = px.box(\n"
            "    df_analysis, x='primary_theme', y='confidence', color='primary_theme',\n"
            "    color_discrete_map=THEME_COLORS,\n"
            "    category_orders={'primary_theme': THEME_ORDER},\n"
            "    title='Confidence Distribution by Theme',\n"
            "    labels={'confidence': 'Confidence (0 = uncertain, 1 = certain)', 'primary_theme': ''},\n"
            "    points='outliers',\n"
            ")\n"
            "fig.update_layout(\n"
            "    showlegend=False,\n"
            "    xaxis={'automargin': True, 'tickangle': -30},\n"
            "    yaxis_title='Confidence', xaxis_title='',\n"
            "    margin={'l': 10, 'r': 10, 't': 60, 'b': 140},\n"
            ")\n"
            "_style(fig).show()\n\n"
            "print(f\"Mean confidence: {df_analysis['confidence'].mean():.2f}\")\n"
            "print(f\"Low confidence (<0.5): {(df_analysis['confidence'] < 0.5).sum()} responses\")"
        ))

    # ── Covariate analysis ─────────────────────────────────────
    if covariate_cols:
        n_cov = len(covariate_cols)
        bonf_thresh = round(0.05 / n_cov, 4) if n_cov > 1 else 0.05
        mc_note = (
            f"> **Multiple comparisons:** {n_cov} covariates tested. "
            + (f"Consider Bonferroni threshold **p < {bonf_thresh}**." if n_cov > 1 else "")
        )
        cells.append(_md(
            f"## Covariate Analysis\n\n"
            f"All tests use `primary_theme` (one row = one response = one group).\n\n{mc_note}"
        ))

        for cov in covariate_cols:
            cov_type = (covariate_types or {}).get(cov, _detect_type(coded_df[cov]))
            cells.append(_md(f"### `{cov}`"))

            if cov_type == "categorical":
                try:
                    stats = chi_square_summary(_themed_df, "primary_theme", cov)
                    sig = "significant" if stats["significant"] else "not significant"
                    p_text = "<0.001" if stats["p_value"] < 0.001 else f'{stats["p_value"]:.3f}'
                    stat_note = (
                        f'{stats["method"]}: statistic = {stats["chi2"]:.2f}, '
                        f'p = {p_text} ({sig}), bias-corrected Cramér\'s V = '
                        f'{stats["cramers_v"]:.3f}. Under the null hypothesis of no association, '
                        f'the smallest expected table-cell count was {stats["expected_min"]:.2f}, '
                        f'and {stats["expected_below_5_pct"]:.1f}% of cells had expected counts below 5. '
                        'These are model-expected cell counts, not observed category totals. '
                        + ('Because the usual Pearson chi-square assumptions were not met, the p-value uses permutations.'
                           if stats["method"].startswith("Permutation") else
                           'The expected counts were adequate for the Pearson chi-square approximation.')
                    )
                except Exception:
                    stat_note = ""
                cells.append(_md(stat_note))
                cells.append(_code(
                    # Limit to top 10 categories by frequency to keep charts readable.
                    # To show all categories, comment out the next two lines.
                    f"_top_cats = df_analysis['{cov}'].value_counts().head(10).index\n"
                    f"_plot_df = df_analysis[df_analysis['{cov}'].isin(_top_cats)].copy()\n\n"
                    # Stacked bar matching covariate_stacked_bar() in analysis.py
                    f"_ct = pd.crosstab(_plot_df['{cov}'], _plot_df['primary_theme'])\n"
                    f"_pct = (_ct.div(_ct.sum(axis=1), axis=0) * 100).round(1)\n"
                    f"_cat_labels = [_wrap_label(str(c), width=20) for c in _pct.index]\n"
                    f"_height = max(450, len(_pct) * 55 + 200)\n"
                    f"_themes_rev = [t for t in reversed(THEME_ORDER) if t in _pct.columns]\n"
                    f"fig = go.Figure()\n"
                    f"for _ri, theme in enumerate(_themes_rev):\n"
                    f"    fig.add_trace(go.Bar(\n"
                    f"        name=theme, x=_cat_labels, y=_pct[theme].tolist(),\n"
                    f"        marker_color=THEME_COLORS.get(theme, palette[_ri % len(palette)]),\n"
                    f"    ))\n"
                    f"fig.update_layout(\n"
                    f"    barmode='stack', title='Theme Mix by {cov}',\n"
                    f"    yaxis={{'title': '% of responses', 'range': [0, 101]}},\n"
                    f"    xaxis={{'title': '{cov}', 'automargin': True, 'tickangle': -30}},\n"
                    f"    legend={{'title': 'Theme', 'traceorder': 'reversed'}}, height=_height,\n"
                    f"    margin={{'l': 10, 'r': 10, 't': 60, 'b': 160}},\n"
                    f")\n"
                    f"_style(fig).show()\n\n"
                    # Heatmap matching covariate_heatmap()
                    f"_hct = pd.crosstab(_plot_df['{cov}'], _plot_df['primary_theme'], normalize='index') * 100\n"
                    f"_hct = _hct.round(1)\n"
                    f"_hct = _hct.reindex([t for t in THEME_ORDER if t in _hct.columns], axis=1)\n"
                    f"_wrapped_x = [_wrap_label(str(c)) for c in _hct.index]\n"
                    f"fig2 = px.imshow(\n"
                    f"    _hct.T, text_auto=True, aspect='auto',\n"
                    f"    color_continuous_scale='Blues',\n"
                    f"    title='Theme Distribution by {cov} (% of each category)',\n"
                    f"    labels={{'color': '% of category', 'x': '{cov}', 'y': 'Theme'}},\n"
                    f")\n"
                    f"fig2.update_xaxes(tickvals=list(range(len(_hct.index))), ticktext=_wrapped_x, tickangle=-30)\n"
                    f"fig2.update_layout(margin={{'l': 10, 'r': 10, 't': 60, 'b': 160}})\n"
                    f"_style(fig2).show()\n\n"
                    # Chi-square (run on full themed data, not just top 10)
                    f"_assoc = association_test(df_analysis, '{cov}', 'primary_theme')\n"
                    f"_p_text = '<0.001' if _assoc['p_value'] < 0.001 else f\"{{_assoc['p_value']:.3f}}\"\n"
                    f"print(f\"{{_assoc['method']}}: statistic={{_assoc['chi2']:.2f}}, "
                    f"p={{_p_text}}, df={{_assoc['dof']}}, bias-corrected Cramér's V={{_assoc['cramers_v']:.3f}}\")\n"
                    f"print(f\"Expected-count check: under no association, the smallest expected table-cell count "
                    f"was {{_assoc['expected_min']:.2f}}, and {{_assoc['expected_below_5_pct']:.1f}}% of cells "
                    f"had expected counts below 5. These are expected cell counts, not observed category totals.\")\n"
                    f"print('The p-value uses permutations because the Pearson expected-count assumptions were not met.' "
                    f"if _assoc['method'].startswith('Permutation') else "
                    f"'Expected counts were adequate for the Pearson chi-square approximation.')"
                ))

            elif cov_type == "numeric":
                cells.append(_code(
                    # Box plot matching anova_box_chart()
                    f"fig = px.box(\n"
                    f"    df_analysis, x='primary_theme', y='{cov}', color='primary_theme',\n"
                    f"    color_discrete_map=THEME_COLORS,\n"
                    f"    category_orders={{'primary_theme': THEME_ORDER}},\n"
                    f"    title='Distribution of {cov} by Theme', points='outliers',\n"
                    f")\n"
                    f"fig.update_layout(\n"
                    f"    showlegend=False,\n"
                    f"    xaxis={{'automargin': True, 'tickangle': -30}},\n"
                    f"    yaxis_title='{cov}', xaxis_title='',\n"
                    f"    margin={{'l': 10, 'r': 10, 't': 60, 'b': 140}},\n"
                    f")\n"
                    f"_style(fig).show()\n\n"
                    # Auto normality → ANOVA or Kruskal-Wallis
                    f"_groups = {{name: g['{cov}'].dropna().values\n"
                    f"           for name, g in df_analysis.groupby('primary_theme')\n"
                    f"           if g['{cov}'].dropna().shape[0] > 2}}\n"
                    f"_any_fail = any(shapiro(v).pvalue < 0.05 for v in _groups.values() if len(v) >= 3)\n"
                    f"if _any_fail:\n"
                    f"    _h, _p = kruskal(*_groups.values())\n"
                    f"    _n_total = sum(len(v) for v in _groups.values())\n"
                    f"    _k = len(_groups)\n"
                    f"    _eps2 = round(max(0.0, (_h - _k + 1) / (_n_total - _k)), 3)\n"
                    f"    _pt = '<0.001' if _p < 0.001 else f'{{_p:.3f}}'\n"
                    f"    print(f'Kruskal-Wallis: H={{_h:.2f}}, p={{_pt}}, ε²={{_eps2:.3f}}')\n"
                    f"    if _p < 0.05:\n"
                    f"        _pairs = list(itertools.combinations(sorted(_groups), 2))\n"
                    f"        _rows = []\n"
                    f"        for a, b in _pairs:\n"
                    f"            _, pv = mannwhitneyu(_groups[a], _groups[b], alternative='two-sided')\n"
                    f"            _rows.append({{'Theme A': a, 'Theme B': b,\n"
                    f"                'Median A': round(pd.Series(_groups[a]).median(), 2),\n"
                    f"                'Median B': round(pd.Series(_groups[b]).median(), 2),\n"
                    f"                'p-adj (Bonferroni)': round(min(1.0, pv * len(_pairs)), 4)}})\n"
                    f"        _ph = pd.DataFrame(_rows).sort_values('p-adj (Bonferroni)')\n"
                    f"        _ph['Sig.'] = _ph['p-adj (Bonferroni)'].apply(lambda p: '✓' if p < 0.05 else '')\n"
                    f"        display(_ph)\n"
                    f"else:\n"
                    f"    _all = pd.concat([pd.Series(v) for v in _groups.values()])\n"
                    f"    _gm = _all.mean()\n"
                    f"    _f, _p, _df1, _df2 = welch_anova(_groups)\n"
                    f"    _ssb = sum(len(v) * (v.mean() - _gm)**2 for v in _groups.values())\n"
                    f"    _sst = ((_all - _gm)**2).sum()\n"
                    f"    _eta2 = round(_ssb / _sst, 3) if _sst > 0 else 0\n"
                    f"    _pt = '<0.001' if _p < 0.001 else f'{{_p:.3f}}'\n"
                    f"    print(f\"Welch's ANOVA: F={{_f:.2f}}, df={{_df1}},{{_df2:.1f}}, p={{_pt}}, descriptive η²={{_eta2:.3f}}\")\n"
                    f"    if _p < 0.05:\n"
                    f"        _pairs = list(itertools.combinations(sorted(_groups), 2))\n"
                    f"        _rows = []\n"
                    f"        for a, b in _pairs:\n"
                    f"            _, pv = ttest_ind(_groups[a], _groups[b], equal_var=False)\n"
                    f"            _rows.append({{'Theme A': a, 'Theme B': b,\n"
                    f"                'Mean A': round(_groups[a].mean(), 2),\n"
                    f"                'Mean B': round(_groups[b].mean(), 2),\n"
                    f"                'Diff (A-B)': round(_groups[a].mean() - _groups[b].mean(), 2),\n"
                    f"                'p-adj (Bonferroni)': round(min(1.0, pv * len(_pairs)), 4)}})\n"
                    f"        _ph = pd.DataFrame(_rows).sort_values('p-adj (Bonferroni)')\n"
                    f"        _ph['Sig.'] = _ph['p-adj (Bonferroni)'].apply(lambda p: '✓' if p < 0.05 else '')\n"
                    f"        display(_ph)\n"
                ))

            else:  # date
                cells.append(_code(
                    f"df_analysis['_date'] = pd.to_datetime(df_analysis['{cov}'], errors='coerce')\n"
                    f"df_analysis['_period'] = df_analysis['_date'].dt.to_period('M').dt.start_time\n"
                    f"_grouped = df_analysis.groupby(['_period', 'primary_theme']).size().unstack(fill_value=0)\n"
                    f"_grouped = _grouped.stack().rename('count').reset_index()\n"
                    f"_grouped.columns = ['_period', 'primary_theme', 'count']\n"
                    f"_grouped['pct'] = (_grouped['count'] /\n"
                    f"    _grouped.groupby('_period')['count'].transform('sum') * 100).round(1)\n"
                    f"_rev_order = list(reversed(THEME_ORDER))\n"
                    f"fig = px.area(\n"
                    f"    _grouped, x='_period', y='pct', color='primary_theme',\n"
                    f"    color_discrete_map=THEME_COLORS,\n"
                    f"    category_orders={{'primary_theme': _rev_order}},\n"
                    f"    title='Theme Proportion Over Time',\n"
                    f"    labels={{'_period': 'Month', 'pct': '% of responses', 'primary_theme': 'Theme'}},\n"
                    f")\n"
                    f"_nt = len(fig.data)\n"
                    f"for _ti, _tr in enumerate(fig.data): _tr.legendrank = _nt - _ti\n"
                    f"fig.update_layout(\n"
                    f"    yaxis_title='% of responses', xaxis_title='Month',\n"
                    f"    legend_title='Theme', margin={{'l': 10, 'r': 10, 't': 60, 'b': 40}},\n"
                    f")\n"
                    f"_style(fig).show()"
                ))

    # ── Valence ────────────────────────────────────────────────
    if include_valence and "valence_score" in coded_df.columns:
        avg_val = round(_themed_df["valence_score"].mean(), 2)
        cells.append(_md(
            f"## Valence Analysis\n\n"
            f"Mean valence: **{avg_val} / 5** (1 = Very Negative → 5 = Very Positive)\n\n"
            f"Median: **{_themed_df['valence_score'].median()} / 5**"
        ))
        cells.append(_code(
            # Overall distribution — matches sentiment_distribution_chart()
            "_val_order = [l for l in VALENCE_ORDER if l in df_analysis['valence_label'].values]\n"
            "_val_colors = dict(zip(VALENCE_ORDER, VALENCE_COLORS))\n"
            "_val_counts = df_analysis['valence_label'].value_counts().reindex(_val_order).fillna(0).reset_index()\n"
            "_val_counts.columns = ['Valence', 'Count']\n"
            "fig = go.Figure(go.Bar(\n"
            "    x=_val_counts['Valence'], y=_val_counts['Count'],\n"
            "    marker_color=[_val_colors.get(v, '#aaa') for v in _val_counts['Valence']],\n"
            "))\n"
            "fig.update_layout(\n"
            "    title='Valence Distribution', showlegend=False, bargap=0.3,\n"
            "    xaxis={'categoryorder': 'array', 'categoryarray': _val_order,\n"
            "           'automargin': True, 'tickangle': 0, 'title': 'Valence'},\n"
            "    yaxis_title='Count',\n"
            "    margin={'l': 10, 'r': 10, 't': 60, 'b': 80},\n"
            ")\n"
            "_style(fig).show()\n\n"
            # By theme box — matches sentiment_by_theme_chart()
            "fig2 = px.box(\n"
            "    df_analysis, x='primary_theme', y='valence_score', color='primary_theme',\n"
            "    color_discrete_map=THEME_COLORS,\n"
            "    category_orders={'primary_theme': THEME_ORDER},\n"
            "    title='Valence Score by Theme  (1 = Very Negative → 5 = Very Positive)',\n"
            "    points='outliers',\n"
            ")\n"
            "fig2.update_layout(\n"
            "    showlegend=False,\n"
            "    yaxis={'tickvals': [1,2,3,4,5],\n"
            "           'ticktext': ['1 Very Neg','2 Neg','3 Neutral','4 Pos','5 Very Pos'],\n"
            "           'range': [0.5, 5.5]},\n"
            "    xaxis={'automargin': True, 'tickangle': -30},\n"
            "    yaxis_title='Valence Score', xaxis_title='',\n"
            "    margin={'l': 10, 'r': 10, 't': 60, 'b': 140},\n"
            ")\n"
            "_style(fig2).show()"
        ))
        cells.append(_md(
            "### Statistical Test: Valence Score × Primary Theme\n\n"
            "Auto-selects ANOVA or Kruskal-Wallis based on per-group normality (Shapiro-Wilk)."
        ))
        cells.append(_code(
            "_vg = {name: g['valence_score'].dropna().values\n"
            "       for name, g in df_analysis.groupby('primary_theme')\n"
            "       if g['valence_score'].dropna().shape[0] > 2}\n"
            "_vfail = any(shapiro(v).pvalue < 0.05 for v in _vg.values() if len(v) >= 3)\n"
            "if _vfail:\n"
            "    _h, _p = kruskal(*_vg.values())\n"
            "    _nt = sum(len(v) for v in _vg.values()); _k = len(_vg)\n"
            "    _eps2 = round(max(0.0, (_h - _k + 1) / (_nt - _k)), 3)\n"
            "    _pt = '<0.001' if _p < 0.001 else f'{_p:.3f}'\n"
            "    print(f'Kruskal-Wallis: H={_h:.2f}, p={_pt}, ε²={_eps2:.3f}')\n"
            "    if _p < 0.05:\n"
            "        _pairs = list(itertools.combinations(sorted(_vg), 2))\n"
            "        _rows = []\n"
            "        for a, b in _pairs:\n"
            "            _, pv = mannwhitneyu(_vg[a], _vg[b], alternative='two-sided')\n"
            "            _rows.append({'Theme A': a, 'Theme B': b,\n"
            "                'Median A': round(pd.Series(_vg[a]).median(), 2),\n"
            "                'Median B': round(pd.Series(_vg[b]).median(), 2),\n"
            "                'p-adj (Bonferroni)': round(min(1.0, pv * len(_pairs)), 4)})\n"
            "        _ph = pd.DataFrame(_rows).sort_values('p-adj (Bonferroni)')\n"
            "        _ph['Sig.'] = _ph['p-adj (Bonferroni)'].apply(lambda p: '✓' if p < 0.05 else '')\n"
            "        display(_ph)\n"
            "else:\n"
            "    _all = pd.concat([pd.Series(v) for v in _vg.values()])\n"
            "    _gm = _all.mean()\n"
            "    _f, _p, _df1, _df2 = welch_anova(_vg)\n"
            "    _ssb = sum(len(v) * (v.mean() - _gm)**2 for v in _vg.values())\n"
            "    _sst = ((_all - _gm)**2).sum()\n"
            "    _eta2 = round(_ssb / _sst, 3) if _sst > 0 else 0\n"
            "    _pt = '<0.001' if _p < 0.001 else f'{_p:.3f}'\n"
            "    print(f\"Welch's ANOVA: F={_f:.2f}, df={_df1},{_df2:.1f}, p={_pt}, descriptive η²={_eta2:.3f}\")\n"
            "    if _p < 0.05:\n"
            "        _pairs = list(itertools.combinations(sorted(_vg), 2))\n"
            "        _rows = []\n"
            "        for a, b in _pairs:\n"
            "            _, pv = ttest_ind(_vg[a], _vg[b], equal_var=False)\n"
            "            _rows.append({'Theme A': a, 'Theme B': b,\n"
            "                'Mean A': round(_vg[a].mean(), 2),\n"
            "                'Mean B': round(_vg[b].mean(), 2),\n"
            "                'Diff (A-B)': round(_vg[a].mean() - _vg[b].mean(), 2),\n"
            "                'p-adj (Bonferroni)': round(min(1.0, pv * len(_pairs)), 4)})\n"
            "        _ph = pd.DataFrame(_rows).sort_values('p-adj (Bonferroni)')\n"
            "        _ph['Sig.'] = _ph['p-adj (Bonferroni)'].apply(lambda p: '✓' if p < 0.05 else '')\n"
            "        display(_ph)\n"
        ))

    # ── Emotion ────────────────────────────────────────────────
    if include_emotion and "emotion" in coded_df.columns:
        cells.append(_md("## Emotion Analysis"))
        cells.append(_code(
            # Overall distribution — matches emotion_distribution_chart()
            "_emo_df = df_analysis[df_analysis['emotion'].notna() & (df_analysis['emotion'].str.strip() != '')].copy()\n"
            "_emo_order = [e for e in EMOTION_ORDER if e in _emo_df['emotion'].values]\n"
            "_emo_colors = dict(zip(EMOTION_ORDER, EMOTION_COLORS))\n"
            "_emo_counts = _emo_df['emotion'].value_counts().reindex(_emo_order).fillna(0).reset_index()\n"
            "_emo_counts.columns = ['Emotion', 'Count']\n"
            "_emo_counts = _emo_counts.iloc[::-1].reset_index(drop=True)  # reverse for horizontal chart\n"
            "fig = go.Figure(go.Bar(\n"
            "    x=_emo_counts['Count'], y=_emo_counts['Emotion'], orientation='h',\n"
            "    marker_color=[_emo_colors.get(e, '#aaa') for e in _emo_counts['Emotion']],\n"
            "))\n"
            "fig.update_layout(\n"
            "    title='Emotion Distribution', showlegend=False, bargap=0.3,\n"
            "    xaxis_title='Count', yaxis={'automargin': True},\n"
            "    margin={'l': 0, 'r': 10, 't': 60, 'b': 40},\n"
            ")\n"
            "_style(fig).show()\n\n"
            # By theme — matches emotion_by_theme_chart() (denominator = all responses per theme)
            "_theme_totals = df_analysis.groupby('primary_theme').size()\n"
            "_labeled = _emo_df\n"
            "_emo_ct = pd.crosstab(_labeled['primary_theme'], _labeled['emotion'])\n"
            "_emo_pct = (_emo_ct.div(_theme_totals, axis=0) * 100).round(1)\n"
            "_themes_y = [t for t in reversed(THEME_ORDER) if t in _emo_pct.index]\n"
            "_emo_pct = _emo_pct.reindex(_themes_y)\n"
            "_emotions_present = [e for e in EMOTION_ORDER if e in _emo_pct.columns]\n"
            "fig2 = go.Figure()\n"
            "for emo in _emotions_present:\n"
            "    fig2.add_trace(go.Bar(\n"
            "        name=emo, y=_themes_y, x=_emo_pct[emo].tolist(), orientation='h',\n"
            "        marker_color=_emo_colors.get(emo),\n"
            "    ))\n"
            "fig2.update_layout(\n"
            "    barmode='stack', title='Emotion Mix by Theme',\n"
            "    xaxis_title='% of all responses in theme',\n"
            "    yaxis={'automargin': True}, legend_title='Emotion',\n"
            "    margin={'l': 10, 'r': 10, 't': 60, 'b': 40},\n"
            ")\n"
            "_style(fig2).show()"
        ))
        cells.append(_md(
            "### Statistical Test: Emotion Label × Primary Theme\n\n"
            "Chi-square test of independence — do emotion distributions differ across themes?"
        ))
        cells.append(_code(
            "_emo_assoc = association_test(_emo_df, 'emotion', 'primary_theme')\n"
            "_emo_p = '<0.001' if _emo_assoc['p_value'] < 0.001 else f\"{_emo_assoc['p_value']:.3f}\"\n"
            "print(f\"{_emo_assoc['method']}: statistic={_emo_assoc['chi2']:.2f}, "
            "df={_emo_assoc['dof']}, p={_emo_p}, bias-corrected Cramér's V={_emo_assoc['cramers_v']:.3f}\")\n"
            "print(f\"Expected-count check: under no association, the smallest expected table-cell count "
            "was {_emo_assoc['expected_min']:.2f}, and {_emo_assoc['expected_below_5_pct']:.1f}% of cells "
            "had expected counts below 5. These are expected cell counts, not observed category totals.\")\n"
            "print('The p-value uses permutations because the Pearson expected-count assumptions were not met.' "
            "if _emo_assoc['method'].startswith('Permutation') else "
            "'Expected counts were adequate for the Pearson chi-square approximation.')\n"
            "display(_emo_assoc['table'])"
        ))

    # ── Statistical notes ──────────────────────────────────────
    cells.append(_md(
        "## Statistical Notes\n\n"
        "### Categorical association test\n"
        "Uses Pearson's chi-square p-value only when expected counts are adequate. When they are sparse, "
        "the notebook reports a deterministic permutation p-value instead. An expected count is the number "
        "of responses a table cell would contain if theme and group were unrelated; it is not the observed "
        "number of responses in a category. The usual approximation is treated as unreliable when the "
        "smallest expected cell is below 1 or more than 20% of cells are below 5.\n\n"
        "**Cramér's V effect size:** < 0.1 negligible · 0.1–0.3 moderate · > 0.3 strong\n\n"
        "### Welch's ANOVA / Kruskal-Wallis\n"
        "Welch's ANOVA tests whether mean values differ across theme groups without assuming equal variances. "
        "Kruskal-Wallis is used automatically when any group fails Shapiro-Wilk (p < 0.05). "
        "Post-hoc uses **Welch t-test** (ANOVA) or **Mann-Whitney U** (Kruskal-Wallis) "
        "with **Bonferroni correction**.\n\n"
        "**η²:** < 0.06 small · 0.06–0.14 medium · > 0.14 large  \n"
        "**ε²:** same benchmarks\n\n"
        "### Multiple comparisons\n"
        "Effect size (Cramér's V, η², ε²) is generally more informative than p-values alone."
    ))

    # ── License / credit footer ────────────────────────────────
    cells.append(_md(
        "---\n\n"
        "### About\n\n"
        "This report was generated with the "
        "[Survey Response Coder](https://survey-response-coder.streamlit.app), "
        "an AI-assisted qualitative coding tool created by "
        "[Jolie Martin](https://github.com/joliem/survey-response-coder).\n\n"
        "Released under the **MIT License** — free to use, modify, and distribute "
        "with attribution. © 2026 Jolie Martin."
    ))

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }
    return json.dumps(nb, indent=2).encode()
