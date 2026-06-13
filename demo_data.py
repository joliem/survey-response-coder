"""Pre-baked themes and keyword-based coding for Demo Mode (no API key required)."""

import random
import re

import pandas as pd

# Taxonomy derived from a fresh read of the bundled CFPB consumer complaint dataset —
# no prior framework, 8 themes identified from patterns in the raw responses.
DEMO_TAXONOMY = [
    {
        "name": "Identity Theft & Fraudulent Accounts",
        "description": "Someone opened accounts, generated charges, or created credit entries in "
                       "the consumer's name without authorization — including data-breach victims "
                       "invoking FCRA 605B to request blocking and deletion of fraudulent tradelines.",
        "examples": [
            "My personal information was stolen in XXXX, since then I've been seeing a lot of "
            "unauthorized inquiries and accounts. I've attached a police report pertaining to the "
            "incident. In accordance with the Fair Credit Reporting Act, multiple unauthorized "
            "inquiries have been found on my credit report — please investigate and remove them immediately.",
            "I am writing to formally request the immediate blocking and deletion of the fraudulent "
            "accounts listed below from my credit file. I am a victim of a data breach, and these "
            "items were not authorized by me.",
        ],
    },
    {
        "name": "Dispute Process Failures",
        "description": "Consumer submitted formal disputes that were ignored, rubber-stamped as "
                       "'verified' without genuine investigation, or never responded to within the "
                       "required 30-day window — the grievance is the broken process itself, not "
                       "just the underlying data error.",
        "examples": [
            "I have disputed errors and inaccuracies on my credit report via Equifax's online "
            "disputing service on several occasions over the past few months and they have yet to "
            "be removed. Equifax refused to further investigate the inaccuracies I reported and "
            "refused to transfer the call to a manager.",
            "It has been over 30 days since my dispute was received, yet the agency failed to "
            "provide the required reinvestigation results or any written notice explaining a delay. "
            "This constitutes a direct violation of the Fair Credit Reporting Act.",
        ],
    },
    {
        "name": "Inaccurate Payment History & Account Data",
        "description": "Wrong data on the consumer's own existing account: late marks the consumer "
                       "disputes, paid accounts still showing delinquent, incorrect balances, "
                       "accounts not updated after payoff, or personal-information errors. "
                       "No claim that someone else opened the account.",
        "examples": [
            "I have always made my payments on time. For some reason, I realized that there was a "
            "late payment on my credit report. I called and was told their system automatically "
            "put me on paperless billing, which I did not request. I tried contacting the bureaus "
            "but they refuse to correct this error.",
            "Standard Mortgage Corporation reported my payment as late. It was paid on time and "
            "cleared my checking account on the due date. Attempts to dispute with the bureaus "
            "were unsuccessful because both are stating the information was verified as 'accurate'.",
        ],
    },
    {
        "name": "Debt Collection Practices",
        "description": "A debt collector is pursuing, reporting, or harassing over a debt the "
                       "consumer disputes, hasn't validated, considers paid, or doesn't recognize — "
                       "including FDCPA violations, calls to employers or neighbors, and debts "
                       "reinserted after prior deletion.",
        "examples": [
            "Performant Recovery is harassing me with the amount of phone calls I receive. They "
            "call me every day multiple times throughout the day and have called me repeatedly "
            "before 8am. They are also calling my place of employment every day more than once "
            "and have even faxed my employer several times.",
            "I received a document from a law office notifying I owed a balance of $2,300 from a "
            "previous apartment complex. I tried contacting the company on multiple occasions and "
            "have not been able to talk to anyone, yet they continue to report this debt.",
        ],
    },
    {
        "name": "Loan & Mortgage Servicing",
        "description": "Problems with how a servicer manages a loan: escrow mismanagement, payment "
                       "misapplication, servicer-transfer errors, unauthorized forbearance, "
                       "foreclosure concerns, or student-loan servicing failures including "
                       "PSLF and income-driven repayment issues.",
        "examples": [
            "MOHELA has been repeatedly placing my loans into forbearance without my consent. "
            "I have contacted MOHELA on multiple occasions to request that they stop, but they "
            "have repeatedly ignored my requests. My auto-debit payments are not being processed, "
            "causing my loans to remain unpaid and interest to improperly accrue.",
            "We had an FHA loan through Flagstar Bank. During COVID we took a deferment. Instead "
            "of adding the deferred amount to the current mortgage balance, they filed a second "
            "lien against the property — which we only discovered years later when trying to sell.",
        ],
    },
    {
        "name": "Billing, Fees & Card Disputes",
        "description": "Unexpected or undisclosed fees, unauthorized card transactions, billing "
                       "errors, credit-limit reductions the consumer did not request, or "
                       "gift-card and transfer scams where the consumer disputes liability.",
        "examples": [
            "I am writing to file a complaint against Navy Federal Credit Union regarding multiple "
            "overdraft fees I have incurred despite having sufficient funds. I was charged fees "
            "due to lagged posting of charges and credits, which aligns with recent CFPB findings "
            "against Navy Federal for similar practices.",
            "I went to make a purchase and found that my credit limit had been cut in half without "
            "any notice, which greatly impacted my credit score. Customer service said they sent "
            "a letter — they did not — and refused to reverse the change.",
        ],
    },
    {
        "name": "Account Restriction & Fund Access",
        "description": "A bank, payment app, or digital-wallet account was unexpectedly closed, "
                       "frozen, or suspended — leaving the consumer unable to access their own "
                       "money. Covers banks, fintech apps (Chime, Cash App), and credit unions.",
        "examples": [
            "Cash App locked my account and emailed me stating I had 5 days to respond about a "
            "transaction I never made. I replied immediately explaining I was at work overseas, "
            "and they went ahead and closed my account without any further response.",
            "US Bank closed my account without notice. I still had funds in the account. The "
            "branch manager said they sent a cashiers check but refused to give a tracking number. "
            "After weeks of follow-up I still have not received my money.",
        ],
    },
    {
        "name": "Unauthorized Credit Inquiries",
        "description": "Hard pulls placed on the consumer's credit file without their knowledge "
                       "or consent — where no full account was opened. The complaint centers on "
                       "the inquiry itself and permissible purpose under FCRA Section 604, "
                       "distinct from broader identity theft cases.",
        "examples": [
            "I applied for a personal loan and got denied, but when they pulled my credit report "
            "3 hard inquiries were placed. I also applied for an auto loan which was also denied "
            "and had another hard inquiry placed — none of these were disclosed upfront.",
            "I noticed unauthorized inquiries on my credit file. I reached out to the bureau to "
            "get them removed and was told I needed to get permission from the 'data furnisher' "
            "before they would remove any inquiry — but they were placed without my permission.",
        ],
    },
]

# Keyword fallback rules — used when _true_theme is absent (user's own data in demo mode).
# Ordered from most-specific to broadest; first match wins.
_KEYWORD_RULES = [
    ("Identity Theft & Fraudulent Accounts", [
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
    ("Account Restriction & Fund Access", [
        "account was closed without", "closed without notice", "closed without warning",
        "closed without notification", "account was shut down", "account frozen",
        "account was frozen", "frozen my account", "locked my account",
        "account was locked", "card was deactivated", "card was defunded",
        "account was deactivated", "account was suspended",
        "access to my funds", "unable to withdraw", "funds were locked",
    ]),
    ("Loan & Mortgage Servicing", [
        "mortgage", "escrow", "loan servicer", "loan servic",
        "student loan", "auto loan", "vehicle loan", "vehicle lease",
        "refinanc", "foreclos", "loan modification", "forbear",
        "navient", "mohela", "aidvantage", "sallie mae", "nelnet",
        "great lakes", "fedloan", "edfinancial", "home equity", "heloc",
        "pslf", "public service loan", "income-driven", "income based repayment",
        "income driven repayment",
    ]),
    ("Debt Collection Practices", [
        "debt collector", "debt collection", "collection agency",
        "collection company", "validate the debt", "debt validation",
        "alleged debt", "cease and desist", "not my debt",
        "does not belong to me", "i do not owe this debt",
        "reinserted", "reinsertion",
        "calling my employer", "called my employer",
        "calling me every day", "multiple times a day",
    ]),
    ("Billing, Fees & Card Disputes", [
        "overdraft fee", "annual fee", "maintenance fee", "cash advance",
        "unauthorized charge", "unauthorized transaction",
        "fraudulent charge", "fraudulent transaction",
        "double charge", "duplicate charge", "billing error",
        "interest rate was increased", "credit limit was lowered",
        "credit limit was reduced", "cut in half", "gift card",
    ]),
    ("Dispute Process Failures", [
        "multiple disputes", "several disputes", "numerous disputes",
        "failed to investigate", "refused to investigate",
        "without investigation", "no response to my dispute",
        "never responded to my dispute", "still reporting after",
        "continues to report", "still has not been removed",
        "section 611", "1681i", "reasonable investigation",
        "over 30 days", "30 days",
    ]),
    ("Inaccurate Payment History & Account Data", [
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
