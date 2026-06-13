"""Pre-baked themes and keyword-based coding for Demo Mode (no API key required)."""

import random
import re

import pandas as pd

# Taxonomy derived from a fresh read of the bundled CFPB consumer complaint dataset —
# no prior framework, 8 themes identified from patterns in the raw responses.
DEMO_TAXONOMY = [
    {
        "name": "Identity Theft and Fraudulent Accounts",
        "description": "Complaints related to identity theft, including unauthorized accounts, "
                       "transactions, and inquiries appearing on credit reports without the "
                       "consumer's knowledge or consent.",
        "examples": [
            "I never gave anyone permission to pull my consumer report or to open any accounts "
            "in my name. This is identity theft. Please remove all attached items.",
            "I discovered that my personal information was used without my consent to open accounts "
            "or make unauthorized purchases. I did not initiate or authorize any of the transactions "
            "or accounts associated with my name.",
        ],
    },
    {
        "name": "Inaccurate or Erroneous Credit Reporting",
        "description": "Concerns regarding incorrect, outdated, or unverifiable information on "
                       "credit reports, such as wrong payment histories, balances, or account statuses.",
        "examples": [
            "I have always made my payments on time, yet a late payment appeared on my credit "
            "report. The credit bureau verified it as accurate without any genuine investigation.",
            "I submitted documentation to Experian confirming my account was in deferment while "
            "I was still in school. Experian refused to update the account status to reflect this.",
        ],
    },
    {
        "name": "Dispute and Investigation Process Issues",
        "description": "Issues involving failures or delays by credit bureaus or furnishers to "
                       "properly investigate, verify, or respond to consumer disputes within "
                       "mandated time frames.",
        "examples": [
            "I submitted a formal dispute with documentation confirming my payments were made on "
            "time. In response, the bureau simply stated the account was 'verified' without "
            "explaining how any investigation was conducted.",
            "This account continued to report on my credit report regardless of my previous "
            "written requests. I sent a dispute letter but never received a reply within the "
            "required 30 days.",
        ],
    },
    {
        "name": "Unauthorized Credit Inquiries",
        "description": "Complaints about hard credit inquiries made without consumer permission "
                       "or knowledge, often linked to identity theft or improper practices.",
        "examples": [
            "I received a hard credit inquiry without any consent, form, request, or verbal or "
            "written authorization. I found out when my credit monitoring alert notified me.",
            "While checking my most recent credit report, I noticed unauthorized credit inquiries "
            "by companies I never applied to and did not give permission to access my credit file.",
        ],
    },
    {
        "name": "Debt Collection and Validation Issues",
        "description": "Problems related to debt collectors reporting debts without proper "
                       "validation, requests for documentation, undue harassment, or inaccurate "
                       "debt information.",
        "examples": [
            "A collection agency is reporting an account that has already been paid. They agreed "
            "to delete it upon payment, but they are still reporting this account in collections.",
            "I sent a dispute letter to the collection agency and they chose not only to ignore "
            "it but to add it to my credit report — a violation of the FDCPA.",
        ],
    },
    {
        "name": "Banking and Account Access Problems",
        "description": "Complaints about bank account issues including unauthorized transactions, "
                       "frozen or closed accounts, delayed fund releases, or unexplained denials "
                       "of services.",
        "examples": [
            "My bank denied my claim for two unauthorized transactions totaling $800 without "
            "providing a specific explanation or the documentation used to make that determination.",
            "My bank closed my account without notice, leaving me unable to access my funds. "
            "Despite repeated follow-ups over several weeks, I have not received my remaining balance.",
        ],
    },
    {
        "name": "Loan Servicing and Payment Disputes",
        "description": "Concerns related to mortgage, student, or auto loan servicing errors "
                       "including misapplication of payments, inaccurate late payments, or "
                       "faulty loan modifications.",
        "examples": [
            "I called my student loan servicer to request that my overpayments be applied to the "
            "principal and was told that is not possible and I cannot choose where the money goes.",
            "I became ill and called my lender to arrange a payment plan. Despite being a "
            "long-term customer, they reported me late to the credit bureaus without notice.",
        ],
    },
    {
        "name": "Customer Service and Communication Failures",
        "description": "Frustrations with poor customer support, lack of timely responses, unclear "
                       "or contradictory information from financial institutions or credit bureaus.",
        "examples": [
            "A representative told me I could make a payment by a certain date to avoid a late "
            "report but did not specify a cut-off time. I followed their instructions and was "
            "still reported late.",
            "I have repeatedly attempted to contact Equifax by phone and through their website "
            "to place a fraud alert on my credit file, but they have refused to assist me.",
        ],
    },
]

# Keyword fallback rules — used when _true_theme is absent (user's own data in demo mode).
# Ordered from most-specific to broadest; first match wins.
_KEYWORD_RULES = [
    ("Identity Theft and Fraudulent Accounts", [
        "identity theft", "data breach", "stolen my identity",
        "victim of identity", "victim of a data breach",
        "opened in my name", "accounts in my name",
        "never opened this account", "did not open this account",
        "i did not open", "i never opened", "fraudulent account",
        "fraudulent accounts", "unauthorized accounts",
        "my information was compromised", "my personal information was compromised",
        "impersonating me", "impersonated me",
        "blocking and deletion", "block and delete",
        "1681c-2", "605b",
    ]),
    ("Unauthorized Credit Inquiries", [
        "unauthorized inquir", "hard inquir", "hard pull",
        "permissible purpose", "pulled my credit without",
        "inquiries on my credit report",
    ]),
    ("Banking and Account Access Problems", [
        "account was closed without", "closed without notice", "closed without warning",
        "closed without notification", "account was shut down", "account frozen",
        "account was frozen", "frozen my account", "locked my account",
        "account was locked", "card was deactivated", "card was defunded",
        "account was deactivated", "account was suspended",
        "access to my funds", "unable to withdraw", "funds were locked",
        "overdraft fee", "annual fee", "maintenance fee", "cash advance",
        "unauthorized charge", "unauthorized transaction",
        "fraudulent charge", "fraudulent transaction",
        "double charge", "duplicate charge", "billing error",
        "interest rate was increased", "credit limit was lowered",
        "credit limit was reduced", "cut in half", "gift card",
    ]),
    ("Loan Servicing and Payment Disputes", [
        "mortgage", "escrow", "loan servicer", "loan servic",
        "student loan", "auto loan", "vehicle loan", "vehicle lease",
        "refinanc", "foreclos", "loan modification", "forbear",
        "navient", "mohela", "aidvantage", "sallie mae", "nelnet",
        "great lakes", "fedloan", "edfinancial", "home equity", "heloc",
        "pslf", "public service loan", "income-driven", "income based repayment",
        "income driven repayment",
    ]),
    ("Debt Collection and Validation Issues", [
        "debt collector", "debt collection", "collection agency",
        "collection company", "validate the debt", "debt validation",
        "alleged debt", "cease and desist", "not my debt",
        "does not belong to me", "i do not owe this debt",
        "reinserted", "reinsertion",
        "calling my employer", "called my employer",
        "calling me every day", "multiple times a day",
    ]),
    ("Customer Service and Communication Failures", [
        "no response", "never responded", "couldn't reach", "could not reach",
        "hold for hours", "hours on hold", "transferred multiple times",
        "contradictory information", "different answer every time",
        "poor customer service", "unhelpful", "hung up on me",
        "refused to help", "no one could help",
    ]),
    ("Dispute and Investigation Process Issues", [
        "multiple disputes", "several disputes", "numerous disputes",
        "failed to investigate", "refused to investigate",
        "without investigation", "no response to my dispute",
        "never responded to my dispute", "still reporting after",
        "continues to report", "still has not been removed",
        "section 611", "1681i", "reasonable investigation",
        "over 30 days", "30 days",
    ]),
    ("Inaccurate or Erroneous Credit Reporting", [
        "inaccurate", "incorrect", "late payment", "late mark",
        "paid in full", "paid off", "credit report", "credit bureau",
        "credit bureaus", "fcra", "1681", "consumer report",
        "equifax", "transunion", "experian", "dispute", "reporting",
    ]),
]

# Sentiment valence scoring keywords (checked in order; first match sets the score)
_VALENCE_RULES = [
    (1, ["stolen", "identity theft", "police report", "illegal", "criminal", "fraud"]),
    (2, ["unauthorized", "rude", "refused", "ignored", "disconnected", "wrong", "incorrect", "never resolved"]),
    (4, ["partially resolved", "some help", "explained", "apologized"]),
    (5, ["resolved", "refunded", "thank you", "excellent", "great service", "finally fixed"]),
]

_VALENCE_LABELS = {1: "Very Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Very Positive"}

# Emotion keyword rules
_EMOTION_RULES = [
    ("Angry",        ["outrageous", "unacceptable", "furious", "angry", "livid"]),
    ("Frustrated",   ["frustrated", "repeatedly", "still not", "keep calling", "months", "weeks", "multiple times", "no one"]),
    ("Worried",      ["worried", "concerned", "afraid", "fear", "nervous"]),
    ("Disappointed", ["disappointed", "expected", "promised", "supposed to", "let down"]),
    ("Confused",     ["unclear", "different information", "confusing", "don't understand", "nobody could explain"]),
    ("Satisfied",    ["resolved", "thank", "appreciate", "helpful", "great"]),
    ("Relieved",     ["finally", "relief", "resolved at last", "issue was fixed"]),
]


_NONE_THEME = "None of the above"  # must match providers.NONE_THEME
_NONE_PHRASES = {"good", "ok", "okay", "fine", "great", "n/a", "na", "none", "yes", "no", "no comment"}


def _assign_theme(text: str, taxonomy: list[dict]) -> tuple[str, float]:
    stripped = text.strip()
    if len(stripped) < 20 or stripped.lower() in _NONE_PHRASES:
        return _NONE_THEME, 1.0

    text_lower = stripped.lower()
    rule_lookup = dict(_KEYWORD_RULES)
    scores: dict[str, float] = {}

    for t in taxonomy:
        name = t["name"]
        user_kws = t.get("_keywords", [])
        user_hits = sum(1 for kw in user_kws if kw in text_lower)
        if user_hits > 0:
            score = user_hits * 3
        else:
            rule_kws = rule_lookup.get(name, [])
            score = sum(1 for kw in rule_kws if kw in text_lower)
            if score == 0:
                name_words = [w.lower() for w in name.split() if len(w) > 3]
                score = sum(1 for w in name_words if w in text_lower) * 0.5
        scores[name] = score

    if not scores:
        return taxonomy[0]["name"], 0.20

    best_name = max(scores, key=scores.get)
    best_score = scores[best_name]
    sorted_vals = sorted(scores.values(), reverse=True)
    second_score = sorted_vals[1] if len(sorted_vals) > 1 else 0.0

    if best_score == 0:
        return _NONE_THEME, 1.0
    else:
        gap_ratio = (best_score - second_score) / max(best_score, 1.0)
        score_factor = min(1.0, best_score / 5.0)
        confidence = round(min(0.95, 0.30 + score_factor * 0.50 + gap_ratio * 0.15), 2)

    return best_name, confidence


def _assign_valence(text: str) -> tuple[int, str]:
    text_lower = text.lower()
    for score, keywords in _VALENCE_RULES:
        if any(kw in text_lower for kw in keywords):
            return score, _VALENCE_LABELS[score]
    return 3, _VALENCE_LABELS[3]


def _assign_emotion(text: str) -> str:
    text_lower = text.lower()
    for emotion, keywords in _EMOTION_RULES:
        if any(kw in text_lower for kw in keywords):
            return emotion
    return ""


def _parse_seeds(user_seeds: str) -> list[dict]:
    parsed = []
    for line in user_seeds.splitlines():
        line = line.strip()
        if not line:
            continue
        for sep in (" — ", " – ", " - "):
            if sep in line:
                name, kw_part = line.split(sep, 1)
                keywords = [k.strip().lower() for k in kw_part.split(",") if k.strip()]
                parsed.append({"name": name.strip(), "keywords": keywords})
                break
        else:
            parsed.append({"name": line.strip(), "keywords": []})
    return parsed


def _taxonomy_from_seeds(user_seeds: str) -> list[dict]:
    demo_lookup = {t["name"].lower(): t for t in DEMO_TAXONOMY}
    parsed = _parse_seeds(user_seeds)
    themes = []
    for item in parsed:
        name = item["name"]
        matched = next(
            (demo_lookup[k] for k in demo_lookup if k in name.lower() or name.lower() in k),
            None,
        )
        themes.append({
            "name": name,
            "description": matched["description"] if matched else f"Responses related to {name.lower()}.",
            "examples": matched["examples"] if matched else [],
            "_keywords": item["keywords"],
        })
    return themes if themes else DEMO_TAXONOMY


def suggest_themes_demo(responses: list[str], user_seeds: str = "", max_themes: int = 8) -> list[dict]:
    if user_seeds.strip():
        return _taxonomy_from_seeds(user_seeds)
    return DEMO_TAXONOMY[:max_themes]


def code_responses_demo(
    responses: list[str],
    taxonomy: list[dict],
    df=None,
    progress_callback=None,
    multi_theme: bool = False,
    include_valence: bool = False,
    include_emotion: bool = False,
) -> list[dict]:
    """
    Serve pre-computed demo results without calling the API.
    When df carries _true_theme / _true_confidence / _true_theme_2 (produced by
    precompute_demo.py), those values are used directly — no keyword heuristics.
    Falls back to keyword matching only for user-uploaded data in demo mode.
    Returns list[dict] matching the format of providers.code_responses.
    """
    use_precomputed = (
        df is not None
        and "_true_theme" in df.columns
        and "_true_confidence" in df.columns
    )
    if use_precomputed:
        taxonomy_names = {t["name"] for t in taxonomy}
        stored_themes = set(df["_true_theme"].dropna().unique())
        use_precomputed = stored_themes.issubset(taxonomy_names)

    has_secondary = use_precomputed and "_true_theme_2" in df.columns
    taxonomy_name_set = {t["name"] for t in taxonomy}

    rng = random.Random(42)
    all_theme_names = [t["name"] for t in taxonomy]
    results = []
    total = len(responses)

    for i, text in enumerate(responses):
        if use_precomputed:
            primary = df["_true_theme"].iloc[i]
            confidence = float(df["_true_confidence"].iloc[i])
        else:
            primary, confidence = _assign_theme(text, taxonomy)

        if multi_theme:
            themes = [primary]
            if use_precomputed and has_secondary:
                secondary = df["_true_theme_2"].iloc[i]
                if pd.notna(secondary) and secondary in taxonomy_name_set:
                    themes.append(secondary)
            elif not use_precomputed:
                others = [n for n in all_theme_names if n != primary]
                if others and rng.random() < 0.25:
                    themes.append(rng.choice(others))
        else:
            themes = [primary]

        score = label = emotion = None
        if include_valence:
            if use_precomputed and "_true_valence_label" in df.columns and pd.notna(df["_true_valence_label"].iloc[i]):
                label = df["_true_valence_label"].iloc[i]
                _s = df["_true_valence_score"].iloc[i] if "_true_valence_score" in df.columns else None
                score = int(_s) if pd.notna(_s) else 3
            else:
                score, label = _assign_valence(text)
        if include_emotion:
            if use_precomputed and "_true_emotion" in df.columns:
                _e = df["_true_emotion"].iloc[i]
                emotion = _e if (pd.notna(_e) and str(_e).strip()) else None
            else:
                emotion = _assign_emotion(text)

        results.append({
            "themes": themes,
            "score": score,
            "label": label,
            "emotion": emotion,
            "confidence": confidence,
        })

        if progress_callback and i % 10 == 0:
            progress_callback((i + 1) / total)

    if progress_callback:
        progress_callback(1.0)

    return results


# Flat list of sentiment-laden keywords, for scoring quote-worthy sentences in demo mode
_VALENCE_KW = [kw for _, kws in _VALENCE_RULES for kw in kws]

# The demo dataset masks PII as XXXX and dates as XX/XX/XXXX — these read poorly as
# pull-quotes, so we skip them in demo mode only (real uploaded data is unaffected).
_REDACTION_RE = re.compile(r"x{3,}|xx/xx", re.IGNORECASE)


def _is_redacted(text: str) -> bool:
    return bool(_REDACTION_RE.search(text))


def _too_redacted(text: str) -> bool:
    """True if more than ~15% of words are redaction placeholders — reads poorly as a quote."""
    words = text.split()
    if not words:
        return True
    masked = sum(1 for w in words if _REDACTION_RE.search(w))
    return masked / len(words) > 0.15


def select_quotes_demo(theme, candidates, max_representative=3, allow_nuance=True):
    """Heuristic counterpart to providers.select_quotes (no API key).

    Picks highest-confidence, non-redacted-heavy responses as representatives
    and one differing-sentiment response as nuance. Full response text is shown.
    """
    from quotes import finalize_picks
    if not candidates:
        return []

    # Filter responses that are too redaction-heavy to read well
    clean = [c for c in candidates if not _too_redacted(c["text_full"])]
    ranked = sorted(
        clean or candidates,  # fall back to all if all are heavily redacted
        key=lambda c: (c.get("confidence") or 0),
        reverse=True,
    )

    raw_picks = []
    used = set()
    for c in ranked:
        if len(raw_picks) >= max_representative:
            break
        raw_picks.append({"id": c["id"], "role": "representative", "reason": ""})
        used.add(c["id"])

    if allow_nuance:
        vals = [c.get("valence") for c in ranked if c.get("valence")]
        dominant = max(set(vals), key=vals.count) if vals else None
        for c in ranked:
            if c["id"] in used or not c.get("valence") or c.get("valence") == dominant:
                continue
            raw_picks.append({"id": c["id"], "role": "nuance", "reason": ""})
            break

    return finalize_picks(raw_picks, candidates, max_representative, allow_nuance)
