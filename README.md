# Survey Response Coder

An AI-assisted qualitative coding tool for open-ended survey responses, built with Streamlit. Upload a dataset, generate a thematic taxonomy from your data, refine it, and code every response — all in a browser-based UI with no programming required.

**[Live demo →](https://survey-response-coder.streamlit.app)**

![Refine taxonomy](screenshots/01%20refine%20taxonomy.png)
![Theme distribution](screenshots/02%20theme%20distribution.png)
![Covariate analysis](screenshots/03%20covariate%20analysis.png)

---

## What it does

Qualitative coding of open-ended survey responses is time-consuming and hard to scale. This tool uses large language models to:

1. **Generate a taxonomy** — the model reads a sample of your responses and proposes a set of mutually exclusive, collectively exhaustive themes
2. **Let you refine it** — add, rename, merge, split, or rewrite themes before committing
3. **Code every response** — each response is assigned one or more themes, with confidence scores
4. **Analyze results** — built-in charts for theme distribution, sentiment, cross-tabulations by covariates, statistical tests (chi-square, ANOVA, Kruskal-Wallis), and inter-rater reliability
5. **Export** — download coded data as CSV/Excel, or generate a Jupyter notebook with the full analysis

---

## Features

- **Multi-provider**: works with Anthropic (Claude), OpenAI (GPT), and Google Gemini — including Gemini's free tier
- **Demo mode**: explore a pre-run analysis of real CFPB consumer complaint data without any API key
- **Flexible coding**: single-theme or multi-theme assignment, optional sentiment valence (1–5 scale) and emotion labels
- **Taxonomy refinement**: edit theme names/descriptions, split themes in two, add custom themes
- **Statistical analysis**: chi-square tests for theme × covariate relationships, ANOVA/Kruskal-Wallis for continuous outcomes, trend tests for ordinal variables
- **Inter-rater reliability**: built-in IRR workflow with Cohen's Kappa and Krippendorff's Alpha
- **Cost estimates**: per-run cost shown before you commit, with links to each provider's pricing page

---

## Getting started

### Prerequisites

- Python 3.10+
- An API key from at least one provider:
  - [Anthropic](https://console.anthropic.com/settings/keys)
  - [OpenAI](https://platform.openai.com/api-keys)
  - [Google AI Studio](https://aistudio.google.com/app/apikey) (free tier available)

### Installation

```bash
git clone https://github.com/joliem/survey-response-coder.git
cd survey-response-coder
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Input format

Upload a CSV or Excel file with at least one column of open-ended text responses. Any additional columns (demographics, Likert ratings, timestamps, etc.) can be used as covariates in the analysis.

---

## Providers and models

| Provider | Models | Free tier |
|---|---|---|
| Anthropic | Claude Opus 4.8, Sonnet 4.6, Haiku 4.5 | No |
| OpenAI | GPT-4.1, GPT-4.1 Mini, GPT-4.1 Nano, GPT-4o | No |
| Google Gemini | Gemini 2.5 Flash, 2.5 Flash Lite, 2.5 Pro, 2.0 Flash | Yes (2.5 Flash / Lite) |

All models share the same workflow. Use the "Other (enter model ID)…" option in the dropdown to try models not listed.

---

## Cost

Costs depend on dataset size, model, and number of iterations. As a rough guide for a 2,000-response dataset:

- **GPT-4o Mini**: ~$0.02 per coding run
- **Gemini 2.5 Flash**: within free tier for most datasets
- **Claude Haiku 4.5**: ~$0.03 per coding run

The app shows a per-run cost estimate before you start.

---

## Tech stack

- [Streamlit](https://streamlit.io) — UI framework
- [Anthropic Python SDK](https://github.com/anthropic/anthropic-sdk-python) — Claude API
- [OpenAI Python SDK](https://github.com/openai/openai-python) — OpenAI and Gemini (via OpenAI-compatible endpoint)
- [Pandas](https://pandas.pydata.org) — data handling
- [Plotly](https://plotly.com/python/) — charts
- [SciPy](https://scipy.org) — statistical tests

---

## Privacy

This tool does not store or log any uploaded data. However, response text is transmitted to the AI provider you select (Anthropic, OpenAI, or Google) to generate the taxonomy and coding. Review your provider's API data usage policy before uploading personally identifiable, sensitive, or regulated data.

---

## License

MIT License — free to use, modify, and distribute with attribution.

Copyright (c) 2026 Jolie Martin

Permission is hereby granted, free of charge, to any person obtaining a copy of this software to use, copy, modify, merge, publish, distribute, and/or sublicense it, subject to the following conditions:

**Attribution required**: any use, distribution, or derivative work must include the above copyright notice and a reference to the original project.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND. THE AUTHORS ARE NOT LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY ARISING FROM USE OF THIS SOFTWARE.

---

## Contributing

Issues and pull requests welcome. If you extend this for a specific domain (healthcare surveys, UX research, academic qualitative coding, etc.) feel free to open a PR.
