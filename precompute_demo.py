"""Pre-compute the demo's coded results by running the REAL coding pipeline over the
bundled CFPB data — so Demo Mode shows genuine, model-quality coding (not heuristics)
without anyone needing an API key at view time.

Run ONCE with an API key, then commit the updated cfpb_sample.csv:

    OPENAI_API_KEY=sk-...  python precompute_demo.py                  # default: GPT-4.1 mini
    MODEL=gpt-4.1          OPENAI_API_KEY=sk-...  python precompute_demo.py
    PROVIDER=Anthropic  MODEL=claude-sonnet-4-6  ANTHROPIC_API_KEY=sk-ant-...  python precompute_demo.py

It writes _true_theme / _true_theme_2 / _true_confidence / _true_valence_score /
_true_valence_label / _true_emotion into cfpb_sample.csv. code_responses_demo serves
these directly; demo-only polish (e.g. excluding redacted text from quotes) is applied
at view time in select_quotes_demo, so it doesn't need to be baked here.
"""

import os
import pathlib

import pandas as pd

from providers import code_responses, NONE_THEME
from demo_data import DEMO_TAXONOMY

DATA_PATH = pathlib.Path(__file__).parent / "cfpb_sample.csv"
TEXT_COL = "consumer_narrative"

PROVIDER = os.environ.get("PROVIDER", "OpenAI")
MODEL = os.environ.get("MODEL", "gpt-4.1-mini")
API_KEY = (
    os.environ.get("OPENAI_API_KEY")
    or os.environ.get("ANTHROPIC_API_KEY")
    or os.environ.get("GEMINI_API_KEY")
)

_VALID = {t["name"] for t in DEMO_TAXONOMY}


def _clean_themes(themes):
    kept = [th for th in (themes or []) if isinstance(th, str) and th in _VALID]
    return kept or [NONE_THEME]


def main():
    if not API_KEY:
        raise SystemExit(
            "No API key found. Set OPENAI_API_KEY (or ANTHROPIC_API_KEY / GEMINI_API_KEY) "
            "in the environment, e.g.:\n    OPENAI_API_KEY=sk-... python precompute_demo.py"
        )

    df = pd.read_csv(DATA_PATH)
    responses = df[TEXT_COL].fillna("").str.strip().tolist()
    print(f"Coding {len(responses):,} CFPB narratives with {PROVIDER} / {MODEL} "
          f"(multi-theme + valence + emotion)...")

    def _progress(p):
        print(f"  {p * 100:5.1f}%", end="\r", flush=True)

    results = code_responses(
        PROVIDER, MODEL, API_KEY, responses, DEMO_TAXONOMY,
        multi_theme=True, include_valence=True, include_emotion=True,
        progress_callback=_progress,
    )
    print()

    if len(results) != len(df):  # safety: align to dataframe length
        results = (results + [{}] * len(df))[:len(df)]

    cleaned = [_clean_themes(r.get("themes") if isinstance(r, dict) else None) for r in results]
    df["_true_theme"] = [c[0] for c in cleaned]
    df["_true_theme_2"] = [c[1] if len(c) > 1 else None for c in cleaned]
    df["_true_confidence"] = [round(float(r.get("confidence", 0.5)), 2) if isinstance(r, dict) else 0.5
                              for r in results]
    df["_true_valence_score"] = [r.get("score") if isinstance(r, dict) else None for r in results]
    df["_true_valence_label"] = [r.get("label") if isinstance(r, dict) else None for r in results]
    df["_true_emotion"] = [r.get("emotion") if isinstance(r, dict) else None for r in results]

    df.to_csv(DATA_PATH, index=False)

    # Quick summary so you can sanity-check the bake before committing
    print(f"\nWrote codings for {len(df):,} rows to {DATA_PATH.name}.\n")
    print("Theme distribution:")
    print(df["_true_theme"].value_counts().to_string())
    print(f"\nMean confidence: {df['_true_confidence'].mean():.2f} | "
          f"secondary themes: {df['_true_theme_2'].notna().sum():,} | "
          f"None of the above: {(df['_true_theme'] == NONE_THEME).sum():,}")


if __name__ == "__main__":
    main()
