"""Representative-quote selection helpers.

Domain-agnostic, dependency-light (only re + pandas). Used by both the live
(LLM-driven) and demo (keyword) quote pipelines.

Flow:
    build_candidates(theme_subset)  →  small candidate pool (code, free)
    [LLM or heuristic picks ids + verbatim excerpts]
    finalize_picks(raw_picks, ...)  →  validate each excerpt is genuinely
                                       verbatim; if not, substitute a real
                                       sentence from the source. Never paraphrase.
"""

import re
import itertools

import pandas as pd


def _normalize(s: str) -> str:
    """Lowercase + collapse whitespace, for tolerant substring matching."""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [p.strip() for p in parts if len(p.strip()) >= 15]


def validate_quote(quote: str, source_text: str) -> bool:
    """True only if `quote` is genuinely verbatim from `source_text`.

    Allows the model to join two contiguous spans with an ellipsis; each span
    must independently appear in the source. Guards against paraphrasing.
    """
    src = _normalize(source_text)
    parts = re.split(r"\s*(?:\.\.\.|…)\s*", str(quote))
    parts = [p for p in (_normalize(p) for p in parts) if len(p) >= 12]
    if not parts:
        return False
    return all(p in src for p in parts)


def extract_fallback_sentence(text: str, max_words: int = 40) -> str:
    """Pull a real, substantive sentence from `text` (used when an LLM excerpt
    fails verbatim validation — we never fall back to paraphrase)."""
    sentences = split_sentences(text)
    candidates = [s for s in sentences if len(s.split()) >= 6]
    if not candidates:
        snippet = " ".join(str(text).split())
    else:
        under = [s for s in candidates if len(s.split()) <= max_words]
        pool = under if under else candidates
        snippet = max(pool, key=lambda s: len(s.split()))
    words = snippet.split()
    if len(words) > max_words:
        snippet = " ".join(words[:max_words]) + "…"
    return snippet


def build_candidates(sub_df: pd.DataFrame, text_col: str,
                     max_candidates: int = 18, min_words: int = 5) -> list[dict]:
    """Build a diverse, quote-worthy candidate pool for ONE theme.

    `sub_df` is the subset of coded rows for a single theme (None excluded).
    Ranks by confidence, then interleaves across sentiment buckets (if present)
    so the pool isn't all one tone. Returns 1-based-id'd dicts.
    """
    df = sub_df.copy()
    df = df[df[text_col].notna()]
    if df.empty:
        return []
    df["_wc"] = df[text_col].astype(str).str.split().str.len()
    df = df[df["_wc"] >= min_words]
    if df.empty:
        return []

    if "confidence" in df.columns:
        df = df.sort_values("confidence", ascending=False)

    has_val = "valence_label" in df.columns
    if has_val:
        buckets: dict = {}
        for idx, row in df.iterrows():
            buckets.setdefault(row["valence_label"], []).append(idx)
        ordered_idx = []
        for tup in itertools.zip_longest(*buckets.values()):
            ordered_idx.extend(i for i in tup if i is not None)
        df = df.loc[ordered_idx]

    selected = df.head(max_candidates)
    cands = []
    for i, (idx, row) in enumerate(selected.iterrows()):
        full = " ".join(str(row[text_col]).split())
        words = full.split()
        truncated = full if len(words) <= 90 else " ".join(words[:90]) + " …"
        conf = None
        if "confidence" in row and pd.notna(row["confidence"]):
            conf = float(row["confidence"])
        cands.append({
            "id": i + 1,
            "row": idx,
            "text_full": full,
            "text": truncated,
            "confidence": conf,
            "valence": row["valence_label"] if has_val else None,
        })
    return cands


def finalize_picks(raw_picks, candidates: list[dict],
                   max_representative: int = 3, allow_nuance: bool = True) -> list[dict]:
    """Map model/heuristic picks to the final quote list.

    - Maps each pick back to a candidate by id.
    - Shows the full response text (pre-truncated in build_candidates).
    - De-dupes by source row, enforces representative/nuance caps.
    Returns: [{"row", "role", "quote", "reason", "confidence"}], reps then nuance.
    """
    by_id = {c["id"]: c for c in candidates}
    reps, nuances = [], []
    seen_rows = set()

    for pick in (raw_picks or []):
        if not isinstance(pick, dict):
            continue
        try:
            cid = int(pick.get("id"))
        except (TypeError, ValueError):
            continue
        cand = by_id.get(cid)
        if not cand or cand["row"] in seen_rows:
            continue

        role = "nuance" if str(pick.get("role", "")).lower().startswith("nuance") else "representative"
        entry = {
            "row": cand["row"],
            "role": role,
            "quote": cand["text"],  # full response text (truncated at build_candidates limit)
            "reason": str(pick.get("reason", "")).strip(),
            "confidence": cand["confidence"],
        }
        seen_rows.add(cand["row"])
        (nuances if role == "nuance" else reps).append(entry)

    reps = reps[:max_representative]
    nuances = nuances[:(1 if allow_nuance else 0)]
    return reps + nuances
