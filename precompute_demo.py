"""
Pre-compute _true_theme, _true_confidence, and _true_theme_2 for every row
in cfpb_sample.csv. Heuristics mirror what a live model run produces:
priority-ordered theme assignment, confidence from signal strength and ambiguity,
secondary theme when a response genuinely crosses two issue areas.
Run once: python precompute_demo.py
"""

import pathlib
import pandas as pd

DATA_PATH = pathlib.Path(__file__).parent / "cfpb_sample.csv"

THEMES = [
    "Identity Theft & Fraudulent Accounts",
    "Unauthorized Credit Inquiries",
    "Account Restriction & Fund Access",
    "Loan & Mortgage Servicing",
    "Debt Collection Practices",
    "Billing, Fees & Card Disputes",
    "Dispute Process Failures",
    "Inaccurate Payment History & Account Data",
]

# Base confidence reflects how distinctively a theme's signals identify it.
# Higher tiers have more specific vocabulary; the default bucket is genuinely ambiguous.
_BASE_CONF = {
    "Identity Theft & Fraudulent Accounts": 0.86,
    "Unauthorized Credit Inquiries": 0.83,
    "Account Restriction & Fund Access": 0.80,
    "Loan & Mortgage Servicing": 0.85,
    "Debt Collection Practices": 0.82,
    "Billing, Fees & Card Disputes": 0.79,
    "Dispute Process Failures": 0.73,
    "Inaccurate Payment History & Account Data": 0.52,
}


def _score(text: str) -> dict:
    """Return keyword hit counts per theme."""
    t = text.lower()
    s = {}

    # ── Identity Theft & Fraudulent Accounts ─────────────────────────────────
    kws = [
        "identity theft", "data breach", "stolen my identity",
        "victim of identity", "victim of a data breach",
        "someone opened", "opened in my name", "open in my name",
        "accounts in my name", "never opened this account", "did not open this account",
        "i did not open", "i never opened",
        "my personal information was compromised",
        "my information has been exposed", "my information was compromised",
        "impersonating me", "impersonated me",
        "fraudulent account", "fraudulent accounts", "unauthorized accounts",
        "blocking and deletion", "block and delete", "1681c-2", "605b", "605 b",
    ]
    hits = sum(1 for kw in kws if kw in t)
    if "data breach" in t or ("victim" in t and "fraud" in t):
        hits = max(hits, 1)
    if any(kw in t for kw in ["police report", "ftc report", "ftc identity"]) and \
       any(kw in t for kw in ["fraud", "identity", "unauthorized", "fraudulent"]):
        hits = max(hits, 1)
    s["Identity Theft & Fraudulent Accounts"] = hits

    # ── Unauthorized Credit Inquiries ─────────────────────────────────────────
    inq_kws = [
        "unauthorized inquir", "hard inquir", "hard pull", "permissible purpose",
        "pulled my credit without", "ran my credit without",
    ]
    inq_hits = sum(1 for kw in inq_kws if kw in t)
    has_acct = any(kw in t for kw in [
        "fraudulent account", "opened in my name", "identity theft", "unauthorized accounts",
    ])
    s["Unauthorized Credit Inquiries"] = inq_hits if not has_acct else 0

    # ── Account Restriction & Fund Access ─────────────────────────────────────
    acct_kws = [
        "account was closed", "account closed without", "closed my account",
        "closed without notice", "closed without warning", "closed without notification",
        "account was shut", "account shut down", "account was terminated",
        "account frozen", "frozen my account", "account was frozen",
        "my account was deactivated", "card was deactivated", "card was defunded",
        "card was shut", "account was deactivated", "was defunded",
        "account was suspended", "account suspended",
        "locked my account", "account was locked",
        "access to my funds", "access my funds",
        "unable to withdraw", "cannot withdraw",
        "funds were locked", "funds locked",
    ]
    acct_hits = sum(1 for kw in acct_kws if kw in t)
    if acct_hits > 0 and any(kw in t for kw in ["credit report", "credit bureau", "credit bureaus"]):
        acct_hits = 0
    s["Account Restriction & Fund Access"] = acct_hits

    # ── Loan & Mortgage Servicing ─────────────────────────────────────────────
    loan_kws = [
        "mortgage", "escrow", "loan servicer", "loan servic",
        "student loan", "auto loan", "vehicle loan", "vehicle lease",
        "refinanc", "foreclos", "loan modification", "forbear",
        "navient", "mohela", "aidvantage", "sallie mae", "nelnet",
        "great lakes", "fedloan", "edfinancial", "home equity", "heloc", "pmi",
        "pslf", "public service loan", "income-driven", "income based repayment",
        "income driven repayment",
    ]
    s["Loan & Mortgage Servicing"] = sum(1 for kw in loan_kws if kw in t)

    # ── Debt Collection Practices ─────────────────────────────────────────────
    debt_kws = [
        "debt collector", "debt collection", "collection agency",
        "collection company", "collection firm",
        "validate the debt", "debt validation", "alleged debt",
        "cease and desist", "not my debt", "does not belong to me",
        "i do not owe this debt", "i do not owe",
        "third-party collect", "third party collect",
        "reinsertion", "reinserted",
        "calling my employer", "called my employer",
        "calling my workplace", "calls to my employer",
        "faxed my employer", "faxed my employment",
        "calling me every day", "calling me daily",
        "multiple times a day", "multiple times throughout the day",
    ]
    debt_hits = sum(1 for kw in debt_kws if kw in t)
    if "collection" in t and any(kw in t for kw in [
        "validate", "validation", "alleged", "not mine",
        "i do not owe", "paid this debt", "harass",
    ]):
        debt_hits = max(debt_hits, 1)
    s["Debt Collection Practices"] = debt_hits

    # ── Billing, Fees & Card Disputes ─────────────────────────────────────────
    billing_kws = [
        "overdraft fee", "annual fee", "maintenance fee",
        "cash advance", "reward point", "cash back",
        "unauthorized charge", "unauthorized transaction",
        "fraudulent charge", "fraudulent transaction",
        "double charge", "duplicate charge", "billing error", "billing dispute",
        "interest rate was increased", "rate increase",
        "credit limit was lowered", "credit limit decreased",
        "credit limit was reduced", "cut in half",
        "gift card", "wire transfer scam",
    ]
    s["Billing, Fees & Card Disputes"] = sum(1 for kw in billing_kws if kw in t)

    # ── Dispute Process Failures ──────────────────────────────────────────────
    dispute_hits = 0
    if "dispute" in t or "investigation" in t or "investigate" in t:
        multi = any(kw in t for kw in [
            "multiple", "several", "numerous", "many times", "again and again",
            "over and over", "submitted again", "filed again", "re-submitted",
            "repeatedly", "second time", "third time", "fourth time",
            "multiple disputes", "multiple attempts", "keep disputing", "still disputing",
        ])
        no_resp = any(kw in t for kw in [
            "no response", "never respond", "failed to respond", "failed to reply",
            "ignored my", "refused to investigate", "failed to investigate",
            "without investigation", "adequate investigation", "proper investigation",
            "not properly investigated", "not adequately investigated",
            "still reporting", "still showing", "still on my report",
            "continue to report", "continues to report", "still has not",
            "has not been removed", "have not removed", "over 30 days", "30 days",
        ])
        if multi or no_resp:
            dispute_hits = (1 if multi else 0) + (1 if no_resp else 0)
    if any(kw in t for kw in [
        "section 611", "1681i", "reasonable investigation",
        "failed to investigate", "failed to maintain",
    ]):
        dispute_hits = max(dispute_hits, 1)
    s["Dispute Process Failures"] = dispute_hits

    # ── Inaccurate Payment History & Account Data ─────────────────────────────
    inaccurate_kws = [
        "late payment", "late mark", "inaccurate", "incorrect balance",
        "paid in full", "paid off", "paid this account",
        "shows as delinquent", "reporting as delinquent",
        "never late", "always paid on time", "on time payment",
    ]
    s["Inaccurate Payment History & Account Data"] = sum(1 for kw in inaccurate_kws if kw in t)

    return s


def _assign(text: str) -> tuple[str, float, str]:
    """Return (primary_theme, confidence, secondary_theme_or_empty)."""
    scores = _score(text)

    # Primary: first theme in priority order with any hits; default if none
    primary = "Inaccurate Payment History & Account Data"
    for theme in THEMES[:-1]:
        if scores.get(theme, 0) > 0:
            primary = theme
            break

    primary_hits = scores.get(primary, 0)

    # Confidence: base for tier + signal bonus − competition penalty
    base = _BASE_CONF[primary]

    if primary == "Inaccurate Payment History & Account Data":
        # Default bucket: differentiate by whether specific inaccuracy signals are present
        if primary_hits == 0:
            base = 0.46   # generic fallthrough — response mentions credit reports but nothing more specific
        elif primary_hits == 1:
            base = 0.58
        else:
            base = 0.65

    signal_bonus = min(0.09, max(0.0, (primary_hits - 1) * 0.025))
    competing = sum(1 for tn, ts in scores.items() if tn != primary and ts > 0)
    competition_penalty = min(0.07, competing * 0.018)

    confidence = round(min(0.95, max(0.38, base + signal_bonus - competition_penalty)), 2)

    # Secondary: highest-priority non-primary theme with any hits
    secondary = ""
    for theme in THEMES:
        if theme != primary and scores.get(theme, 0) > 0:
            secondary = theme
            break

    return primary, confidence, secondary


def main():
    df = pd.read_csv(DATA_PATH)

    results = df["consumer_narrative"].fillna("").apply(_assign)
    df["_true_theme"] = [r[0] for r in results]
    df["_true_confidence"] = [r[1] for r in results]
    df["_true_theme_2"] = [r[2] for r in results]

    print("Theme distribution:")
    dist = df["_true_theme"].value_counts()
    for theme, n in dist.items():
        print(f"  {theme}: {n} ({n / len(df) * 100:.1f}%)")

    with_secondary = (df["_true_theme_2"] != "").sum()
    print(f"\nWith secondary theme: {with_secondary} ({with_secondary / len(df) * 100:.1f}%)")

    print("\nConfidence distribution:")
    print(f"  Mean:    {df['_true_confidence'].mean():.3f}")
    print(f"  <0.50:   {(df['_true_confidence'] < 0.50).sum()}")
    print(f"  0.50–0.70: {((df['_true_confidence'] >= 0.50) & (df['_true_confidence'] < 0.70)).sum()}")
    print(f"  0.70–0.85: {((df['_true_confidence'] >= 0.70) & (df['_true_confidence'] < 0.85)).sum()}")
    print(f"  ≥0.85:   {(df['_true_confidence'] >= 0.85).sum()}")

    df.to_csv(DATA_PATH, index=False)
    print(f"\nSaved {len(df)} rows to {DATA_PATH}")


if __name__ == "__main__":
    main()
