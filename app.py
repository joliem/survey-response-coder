"""Survey Response Coder — AI-assisted qualitative coding for open-ended survey data."""

import html as _html
import os
import streamlit as st
import streamlit.components.v1 as _components
import pandas as pd

_TRACKING_URL = "https://script.google.com/macros/s/AKfycbx0UXfGtAjn30NYk8gY80gsV5vZewuERtBdk3mvHPGeSnHBMXTl54Q8mJtDNWYun28S/exec"
_GA_MEASUREMENT_ID = "G-JKFCS1EWQE"
_GA_MP_URL = "https://www.google-analytics.com/mp/collect"

def _get_session_id():
    """Return a stable UUID for this browser session."""
    import uuid
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def _get_ga_secret():
    """Read GA4 Measurement Protocol API secret from Streamlit secrets."""
    try:
        return st.secrets["GA_API_SECRET"]
    except Exception:
        return None

def _track_ga(event_name: str, params: dict = None):
    """Send a GA4 event via Measurement Protocol (server-side, no iframe issues)."""
    import threading, requests
    secret = _get_ga_secret()
    if not secret:
        return
    payload = {
        "client_id": _get_session_id(),
        "events": [{"name": event_name, "params": params or {}}],
    }
    def _send():
        try:
            requests.post(
                _GA_MP_URL,
                params={"measurement_id": _GA_MEASUREMENT_ID, "api_secret": secret},
                json=payload,
                timeout=5,
            )
        except Exception:
            pass
    threading.Thread(target=_send, daemon=True).start()

def _track(event: str, **kwargs):
    """Fire-and-forget event to Google Sheets + GA4. Never raises."""
    import threading, requests
    payload = {
        "event": event,
        "session_id": _get_session_id(),
        **kwargs,
    }
    def _send():
        try:
            requests.post(_TRACKING_URL, json=payload, timeout=5)
        except Exception:
            pass
    threading.Thread(target=_send, daemon=True).start()
    # Mirror to GA4
    _track_ga(event, params=kwargs)

def _feedback_form(key: str):
    """Render a feedback form. `key` must be unique per call site."""
    with st.form(f"feedback_form_{key}", clear_on_submit=True):
        fb_text = st.text_area(
            "Your feedback",
            placeholder="What worked, what didn't, or what you'd love to see added…",
            label_visibility="collapsed",
            key=f"fb_text_{key}",
        )
        fb_contact = st.text_input(
            "Email (optional)",
            placeholder="Email (optional, if you'd like a reply)",
            label_visibility="collapsed",
            key=f"fb_contact_{key}",
        )
        if st.form_submit_button("Send", use_container_width=True):
            if fb_text.strip():
                _step_names = {
                    1: "1. Load Data",
                    2: "2. Explore Themes",
                    3: "3. Refine Taxonomy",
                    4: "4. Code Responses",
                    5: "5. Analyze",
                }
                _cur_step = st.session_state.get("step", 0)
                _track(
                    "feedback",
                    feedback_text=fb_text.strip(),
                    contact=fb_contact.strip(),
                    page=_step_names.get(_cur_step, str(_cur_step)),
                    source=key,
                )
                st.success("Thanks — your feedback was sent! 🙏")
            else:
                st.warning("Please enter a comment before sending.")

_SCROLL_TOP = """<script>
(function() {
    function go() {
        var doc = window.parent.document;
        var el = doc.querySelector('[data-testid="stMain"]')
              || doc.querySelector('section.main')
              || doc.querySelector('.main');
        if (el) el.scrollTop = 0;
        window.parent.scrollTo(0, 0);
    }
    go();
    setTimeout(go, 80);
    setTimeout(go, 200);
})();
</script>"""

from sample_data import load_sample
from providers import suggest_themes, code_responses, split_theme, select_quotes, preflight_check, PROVIDERS, DEFAULT_MODEL, NONE_THEME
from demo_data import suggest_themes_demo, code_responses_demo, select_quotes_demo
from quotes import build_candidates
from analysis import (
    theme_bar_chart,
    covariate_heatmap,
    covariate_stacked_bar,
    chi_square_summary,
    detect_covariate_type,
    theme_over_time_chart,
    theme_over_time_line,
    confidence_box_chart,
    anova_box_chart,
    anova_summary,
    anova_posthoc,
    normality_summary,
    kruskal_summary,
    kruskal_posthoc,
    trend_test_summary,
    explode_themes,
    sentiment_distribution_chart,
    sentiment_by_theme_chart,
    emotion_distribution_chart,
    emotion_by_theme_chart,
    irr_cohen_kappa,
    irr_krippendorff_alpha,
)
from report import generate_notebook

def _quote(text: str) -> None:
    """Display response text as a blockquote without markdown interpretation."""
    st.markdown(
        f'<div style="border-left:3px solid #d0d0d0;padding:0.35em 0.75em;'
        f'margin:0.2em 0;color:inherit;font-size:0.95em">'
        f'{_html.escape(str(text))}</div>',
        unsafe_allow_html=True,
    )


def _friendly_api_error(e, provider: str = "") -> str:
    """Turn a provider API exception into a clear, actionable message (no traceback)."""
    name = type(e).__name__
    status = getattr(e, "status_code", None)
    prov = provider or "Your provider"
    model = st.session_state.get("model", "")
    blob = (str(getattr(e, "body", "")) + " " + str(e)).lower()
    # Out of credits (OpenAI/Anthropic) — a 429, but billing not throttling
    if provider != "Google Gemini" and "insufficient_quota" in blob:
        return (
            f"**Out of credits** — {prov} reports no remaining balance for this key. "
            "Add credit/billing in your provider dashboard, or switch to a provider/model whose "
            "key has credit, then try again."
        )
    if name == "RateLimitError" or status == 429:
        msg = f"**Rate limit or quota reached** — {prov} stopped accepting requests.\n\n"
        if provider == "Google Gemini":
            _is_day = ("perday" in blob) or ("per day" in blob)
            _is_min = ("perminute" in blob) or ("per minute" in blob)
            if _is_day:
                msg += (
                    "This is Gemini's **daily** request cap (free tier), which resets **once a day, "
                    "around midnight Pacific** — not throughout the day. "
                )
            elif _is_min:
                msg += (
                    "This is Gemini's **per-minute** rate limit (free tier), which resets within a minute. "
                    "The app normally waits and retries automatically, so seeing it here means the free "
                    "tier is heavily throttled right now — wait a minute and try again. "
                )
            else:
                msg += (
                    "This is a Gemini free-tier limit — either the **per-minute** throttle (resets within "
                    "a minute) or the **daily** cap (resets ~midnight Pacific). "
                )
            if model == "gemini-2.5-flash-lite":
                msg += (
                    "Flash Lite has the smallest free allowance — switch to **Gemini 2.5 Flash**, or if "
                    "that's also spent, a paid model (OpenAI **GPT-4.1 nano** is fast and a few cents)."
                )
            else:
                msg += (
                    "For a full dataset, a paid model — OpenAI **GPT-4.1 nano**, fast and a few cents — is "
                    "much more reliable than waiting on free quota."
                )
        else:
            msg += (
                "This can mean a per-minute rate limit, an exhausted daily quota, or no remaining "
                "credits. Check your account's limits and billing, wait a moment, or switch to a "
                "different model or provider in the sidebar, then try again."
            )
        return msg
    if name == "AuthenticationError" or status == 401:
        return f"**Authentication failed** — check that your API key is valid for {prov}."
    if name == "NotFoundError" or status == 404:
        return ("**Model not found** — the selected model ID may be wrong or unavailable on your key. "
                "Double-check the model in the sidebar.")
    if name == "InternalServerError" or status in (500, 503):
        return "**The model is temporarily unavailable** (server error). Wait a moment and run coding again."
    return f"**Coding failed** ({name}). Try again, or switch model/provider in the sidebar."


def _coding_signature(n_responses, taxonomy, multi_theme, include_valence, include_emotion) -> str:
    """Identify a coding run so partial progress can only resume a compatible one.

    Deliberately excludes provider/model — so you can hit a free-tier wall on one
    model and resume the rest on another.
    """
    import hashlib
    theme_part = tuple((t.get("name", ""), t.get("description", "")) for t in taxonomy)
    key = repr((theme_part, n_responses, multi_theme, include_valence, include_emotion))
    return hashlib.md5(key.encode()).hexdigest()


def _responses_fingerprint(responses) -> str:
    """Hash of the exact response texts — verifies a resume file matches the dataset."""
    import hashlib
    h = hashlib.md5()
    h.update("\n".join(responses).encode("utf-8", "ignore"))
    return h.hexdigest()


def _build_resume_blob(*, text_col, covariate_cols, taxonomy, options, total, results, responses) -> bytes:
    """Serialize everything needed to continue a coding run on a fresh session."""
    import json
    payload = {
        "app": "survey-response-coder",
        "kind": "coding-resume",
        "version": 1,
        "text_col": text_col,
        "covariate_cols": list(covariate_cols),
        "taxonomy": taxonomy,
        "options": options,
        "total": total,
        "responses_hash": _responses_fingerprint(responses),
        "results": results,
    }
    return json.dumps(payload).encode("utf-8")


def _parse_resume_blob(data) -> dict:
    """Parse + validate an uploaded resume file. Raises ValueError if malformed."""
    import json
    blob = json.loads(bytes(data).decode("utf-8"))
    if not isinstance(blob, dict) or blob.get("kind") != "coding-resume":
        raise ValueError("Not a Survey Response Coder resume file.")
    for k in ("text_col", "taxonomy", "options", "total", "responses_hash", "results"):
        if k not in blob:
            raise ValueError("Resume file is missing required fields.")
    return blob


# ── Browser auto-save (localStorage) ───────────────────────────────────────────
# Persists taxonomy + coding progress to the user's browser so a session reset
# (Streamlit Cloud recycling, refresh, sleep) doesn't lose work. Dependency-free:
# uses window.parent.localStorage, same as the scroll-to-top helper.
_LS_KEY = "survey_coder_session_v1"


def _autosave_to_browser():
    """Save current recoverable state to localStorage — only when it changes."""
    import base64
    try:
        df = st.session_state.df
        text_col = st.session_state.text_col
        taxonomy = st.session_state.taxonomy
        if df is None or not text_col or not taxonomy:
            return
        prog = st.session_state.coding_progress
        results = prog["results"] if prog else []
        # Cheap change-signature so we don't re-emit a large blob on every rerun
        sig = (len(results), len(taxonomy), text_col,
               bool(st.session_state.multi_theme),
               bool(st.session_state.include_valence),
               bool(st.session_state.include_emotion))
        if st.session_state.get("_ls_sig") == sig:
            return
        responses = df[text_col].fillna("").str.strip().tolist()
        blob = _build_resume_blob(
            text_col=text_col, covariate_cols=st.session_state.covariate_cols,
            taxonomy=taxonomy,
            options={"multi_theme": st.session_state.multi_theme,
                     "include_valence": st.session_state.include_valence,
                     "include_emotion": st.session_state.include_emotion},
            total=len(responses), results=results, responses=responses,
        )
        b64 = base64.b64encode(blob).decode()
        _components.html(
            f"<script>try{{window.parent.localStorage.setItem('{_LS_KEY}','{b64}');}}"
            f"catch(e){{}}</script>",
            height=0,
        )
        st.session_state._ls_sig = sig
    except Exception:
        pass


def _clear_browser_save():
    """Remove the auto-saved session from the browser (on Start Over)."""
    try:
        _components.html(
            f"<script>try{{window.parent.localStorage.removeItem('{_LS_KEY}');}}"
            f"catch(e){{}}</script>",
            height=0,
        )
        st.session_state._ls_sig = None
    except Exception:
        pass


st.set_page_config(
    page_title="Survey Response Coder",
    page_icon="📊",
    layout="wide",
)

# Inject Google Analytics via component (works on Streamlit Cloud)
_components.html("""
<script async src="https://www.googletagmanager.com/gtag/js?id=G-JKFCS1EWQE"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-JKFCS1EWQE');
</script>
""", height=0)

# --- Session state defaults ---
DEFAULTS = {
    "step": 1,
    "df": None,
    "text_col": None,
    "covariate_cols": [],
    "taxonomy": [],
    "preview_sample": None,
    "coded_df": None,
    "user_seeds": "",
    "provider": "Anthropic",
    "model": DEFAULT_MODEL["Anthropic"],
    "api_key": "",
    "demo_mode": False,       # derived from provider == "Demo Mode"; kept for compatibility
    "multi_theme": False,
    "sentiment_enabled": False,
    "include_valence": False,
    "include_emotion": False,
    "irr_sample": None,
    "irr_labels": {},
    "irr_pos": 0,
    "rerun_requested": False,
    "theme_quotes": {},       # cached representative quotes per theme (lazy-computed)
    "coding_progress": None,  # {"sig", "results", "total"} for checkpoint/resume of coding
    "_ls_sig": None,          # dedup signature for browser auto-save
    "_rendered_step": -1,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Fire a page_view once per session
if "ga_page_view_sent" not in st.session_state:
    st.session_state.ga_page_view_sent = True
    _track_ga("page_view", {"page_title": "Survey Response Coder", "page_location": "https://survey-response-coder.streamlit.app/"})


def go_to(step: int):
    st.session_state.step = step
    st.session_state._rendered_step = -1  # trigger scroll on next render


# --- Sidebar ---
STEPS = [
    "1. Load Data",
    "2. Explore Themes",
    "3. Refine Taxonomy",
    "4. Code Responses",
    "5. Visualize & Analyze",
]

_ENV_KEY_NAMES = {
    "Anthropic":    "ANTHROPIC_API_KEY",
    "OpenAI":       "OPENAI_API_KEY",
    "Google Gemini": "GEMINI_API_KEY",
}

with st.sidebar:
    st.title("Survey Coder")
    st.caption("AI-assisted qualitative coding for open-ended survey responses")
    st.divider()
    for i, label in enumerate(STEPS, 1):
        if i < st.session_state.step:
            st.markdown(f"✅ {label}")
        elif i == st.session_state.step:
            st.markdown(f"**▶ {label}**")
        else:
            st.markdown(f"⬜ {label}")
    st.divider()

    # ── Demo Mode toggle ───────────────────────────────────────
    demo_toggle = st.toggle(
        "Demo Mode",
        value=st.session_state.demo_mode,
        help="No API key needed — uses a bundled CFPB dataset with pre-run results.",
        key="_sidebar_demo_toggle",
    )
    if demo_toggle != st.session_state.demo_mode:
        st.session_state.demo_mode = demo_toggle
        if demo_toggle:
            _track("demo_mode_enabled")
        st.session_state.provider = "Demo Mode" if demo_toggle else "Anthropic"
        if not demo_toggle:
            st.session_state.model = DEFAULT_MODEL["Anthropic"]
        st.session_state.api_key = ""
        st.session_state.taxonomy = []
        st.session_state.coded_df = None
        st.session_state.preview_sample = None
        st.rerun()

    if demo_toggle:
        st.info(
            "**Demo Mode** — shows a pre-run analysis of real CFPB consumer complaints "
            "using Claude rather than live API calls. Results reflect what live mode would "
            "produce on this dataset, but won't adapt to your own data or instructions.",
            icon="🧪",
        )
    else:
        # ── Provider selector ─────────────────────────────────
        _provider_list = list(PROVIDERS.keys())
        _provider_labels = {
            "Anthropic":     "Anthropic (Claude)",
            "OpenAI":        "OpenAI (GPT)",
            "Google Gemini": "Google Gemini",
        }
        _cur_provider = st.session_state.provider if st.session_state.provider in _provider_list else "Anthropic"
        selected_provider = st.selectbox(
            "Provider",
            options=_provider_list,
            format_func=lambda p: _provider_labels.get(p, p),
            index=_provider_list.index(_cur_provider),
            key="_sidebar_provider",
        )
        if selected_provider != st.session_state.provider:
            st.session_state.provider = selected_provider
            st.session_state.model = DEFAULT_MODEL[selected_provider]
            st.session_state.api_key = ""
            st.session_state.taxonomy = []
            st.session_state.coded_df = None
            st.session_state.preview_sample = None
            st.rerun()

        pinfo = PROVIDERS[selected_provider]

        # ── Model selector ─────────────────────────────────────
        _model_ids    = [m["id"]    for m in pinfo["models"]]
        _model_labels = {m["id"]: m["label"] for m in pinfo["models"]}
        # If the stored model isn't in the list it's a custom ID — map to __custom__
        _stored_model = st.session_state.model
        _is_custom = _stored_model not in _model_ids or _stored_model == "__custom__"
        _cur_model = "__custom__" if _is_custom else _stored_model
        _cur_custom_text = _stored_model if (_is_custom and _stored_model != "__custom__") else ""
        selected_model = st.selectbox(
            "Model",
            options=_model_ids,
            format_func=lambda m: _model_labels.get(m, m),
            index=_model_ids.index(_cur_model),
        )
        if selected_model == "__custom__":
            _custom_id = st.text_input(
                "Model ID",
                value=_cur_custom_text,
                placeholder="e.g. claude-opus-4-8",
                key="_sidebar_custom_model_id",
            )
            _effective_model = _custom_id.strip() if _custom_id.strip() else "__custom__"
        else:
            _effective_model = selected_model
        if _effective_model != st.session_state.model:
            st.session_state.model = _effective_model

        # Free-tier reality check, right at model choice
        if selected_provider == "Google Gemini" and _effective_model in (
            "gemini-2.5-flash", "gemini-2.5-flash-lite"
        ):
            st.caption(
                "🆓 Great for the **taxonomy step** and **small samples**. Free daily request caps "
                "are low (and change often), so a **full coding run will likely hit the cap** — if "
                "your dataset is large, set up paid credits (OpenAI / Anthropic / paid Gemini) "
                "*before* you start so you're not stuck mid-run."
            )

        # ── API key input ──────────────────────────────────────
        _env_key = os.environ.get(_ENV_KEY_NAMES.get(selected_provider, ""), "")
        _displayed = st.session_state.api_key or _env_key
        _entered = st.text_input(
            "API Key",
            value=_displayed,
            type="default",
            placeholder=f"Paste your {selected_provider} API key here",
            key="_sidebar_api_key",
        )
        if _entered != st.session_state.api_key:
            st.session_state.api_key = _entered

        _key_ok = bool(_entered)
        if _key_ok:
            _display_model = _effective_model if _effective_model != "__custom__" else "your custom model"
            st.success(f"API key set — ready to use {_display_model}.", icon="⚡")
            if st.button("✓ Test key", use_container_width=True,
                         help="Sends one tiny request to confirm the key works and isn't out of "
                              "credits or already capped right now."):
                try:
                    with st.spinner("Testing…"):
                        preflight_check(selected_provider, _effective_model, _entered)
                    st.success("Key works — a test request just went through. ✓", icon="✅")
                    if selected_provider == "Google Gemini":
                        st.caption(
                            "This only confirms one call succeeds now. On the free tier it can't tell "
                            "whether your remaining **daily** quota covers a full coding run."
                        )
                except Exception as _pf_err:
                    st.error(_friendly_api_error(_pf_err, selected_provider), icon="⚠️")
        else:
            if pinfo["free_tier"]:
                st.info(
                    f"**Free tier available.** {pinfo['free_tier_note']}  \n"
                    f"[Get a free API key →]({pinfo['key_url']})",
                    icon="🎁",
                )
            else:
                st.warning(
                    f"No API key. [Get one here]({pinfo['key_url']})  \n"
                    "Or enable **Demo Mode** above — no key needed.",
                    icon="🔑",
                )

    st.divider()
    if st.session_state.step > 1:
        if st.button("↩ Start Over", use_container_width=True):
            _clear_browser_save()
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()

    with st.expander("💬 Feedback / feature requests"):
        _feedback_form("sidebar")

    st.caption("Powered by AI · Built with Streamlit")
    st.caption("© 2026 [Jolie Martin](https://github.com/joliem/survey-response-coder)")


# Guard against an inconsistent step (e.g. after a session reset that left `step`
# ahead of the data it needs) — clamp back to the earliest step with prerequisites met.
if st.session_state.step >= 2 and st.session_state.df is None:
    st.session_state.step = 1
if st.session_state.step >= 4 and not st.session_state.taxonomy:
    st.session_state.step = 2
if st.session_state.step == 5 and st.session_state.coded_df is None:
    st.session_state.step = 4

# Continuously back up recoverable state to the browser (survives a session reset)
_autosave_to_browser()

# Scroll to top only when navigating to a new step, not on widget re-renders
if st.session_state._rendered_step != st.session_state.step:
    _components.html(_SCROLL_TOP, height=1)
    st.session_state._rendered_step = st.session_state.step


# ============================================================
# STEP 1 — Load Data
# ============================================================
if st.session_state.step == 1:
    st.title("📊 Survey Response Coder")

    st.markdown(
        """
**Survey Response Coder** is an AI-assisted tool for qualitative coding of open-ended survey data,
designed for quantitative UX and market researchers who need to make sense of free-text at scale.

**How it works — 5 steps:**

| Step | What happens |
|---|---|
| **1. Load Data** | Upload a CSV/Excel file or use the built-in sample dataset |
| **2. Explore Themes** | The AI reads your responses and proposes a coding taxonomy |
| **3. Refine Taxonomy** | Merge, split, or rename themes; preview coding on a sample before committing |
| **4. Code Responses** | The AI assigns themes — and optionally valence/emotion — to every response |
| **5. Analyze** | Visualize distributions, explore covariate relationships, run statistical tests |
        """
    )

    if not st.session_state.demo_mode and not st.session_state.api_key:
        st.info(
            "No API key yet? Toggle **Demo Mode** in the sidebar to walk through the full workflow "
            "using a bundled sample of real CFPB consumer complaints — no setup required.",
            icon="🧪",
        )

    st.divider()

    if st.session_state.demo_mode:
        st.subheader("Load the demo dataset")
        st.markdown(
            "Real consumer complaints from the "
            "[CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) "
            "— bundled with the app, no download needed."
        )
        if st.button("Load Demo Dataset", use_container_width=True, type="secondary"):
            st.session_state.df = load_sample(500)
            st.success("Loaded 500 real CFPB complaints.")
    else:
        col_upload, col_sample = st.columns([1, 1], gap="large")

        with col_upload:
            st.subheader("Upload your own file")
            uploaded = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])
            if uploaded:
                try:
                    if uploaded.name.endswith((".xlsx", ".xls")):
                        df = pd.read_excel(uploaded)
                    else:
                        df = pd.read_csv(uploaded)
                    st.success(f"Loaded {len(df):,} rows · {len(df.columns)} columns")
                    st.session_state.df = df
                except Exception as e:
                    st.error(f"Could not read file: {e}")

        with col_sample:
            st.subheader("Use the demo dataset")
            st.markdown(
                "Real consumer complaints from the "
                "[CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) "
                "— bundled with the app, no download needed."
            )
            if st.button("Load Demo Dataset", use_container_width=True, type="secondary"):
                st.session_state.df = load_sample(500)
                st.success("Loaded 500 real CFPB complaints.")

    if st.session_state.df is not None:
        df = st.session_state.df
        st.divider()
        st.subheader("Preview")
        st.dataframe(df.head(5), use_container_width=True)

        st.subheader("Configure Columns")
        text_options = [
            c for c in df.columns
            if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])
        ]
        if not text_options:
            text_options = list(df.columns)

        default_text = (
            "consumer_narrative" if "consumer_narrative" in text_options else text_options[0]
        )
        text_col = st.selectbox(
            "Which column contains the open-ended responses?",
            options=text_options,
            index=text_options.index(default_text),
        )

        default_covariates = [
            c for c in ["product", "company", "state", "date_submitted", "outcome"]
            if c in df.columns and c != text_col
        ]
        covariate_cols = st.multiselect(
            "Select covariate columns for cross-analysis (optional)",
            options=[c for c in df.columns if c != text_col],
            default=default_covariates,
        )

        n_valid = df[text_col].dropna().str.strip().ne("").sum()
        st.caption(f"{n_valid:,} non-empty responses in '{text_col}'")

        if st.button("Continue →", type="primary", use_container_width=True):
            st.session_state.text_col = text_col
            st.session_state.covariate_cols = covariate_cols
            go_to(2)
            st.rerun()

        with st.expander("↪️ Resume an interrupted coding run"):
            st.caption(
                "If a coding run was interrupted, pick up where it stopped. Load the **same dataset** "
                "above first, then either recover the **auto-saved** copy from this browser or upload "
                "a resume file you saved."
            )
            # Auto-recover: read the browser's auto-saved session and offer it as a download
            # that can be fed straight into the uploader below.
            _components.html(
                f"""<div id="rec" style="font-family:sans-serif;font-size:14px"></div>
                <script>
                  try {{
                    var d = window.parent.localStorage.getItem('{_LS_KEY}');
                    if (d) {{
                      var blob = new Blob([atob(d)], {{type:'application/json'}});
                      var url = URL.createObjectURL(blob);
                      document.getElementById('rec').innerHTML =
                        '✅ Auto-saved progress found in this browser. '+
                        '<a href="'+url+'" download="coding_resume.json">⬇ Download it</a>, '+
                        'then upload it just below.';
                    }} else {{
                      document.getElementById('rec').innerHTML =
                        '<span style="color:#888">No auto-saved progress found in this browser.</span>';
                    }}
                  }} catch(e) {{
                    document.getElementById('rec').innerHTML =
                      '<span style="color:#888">Browser storage unavailable here.</span>';
                  }}
                </script>""",
                height=44,
            )
            _ru = st.file_uploader("Resume file (.json)", type=["json"], key="resume_uploader")
            if _ru is not None:
                try:
                    _blob = _parse_resume_blob(_ru.getvalue())
                except Exception:
                    st.error("That doesn't look like a valid resume file.")
                else:
                    _tcol = _blob["text_col"]
                    if _tcol not in df.columns:
                        st.error(
                            f"This resume file needs a `{_tcol}` column, which isn't in the loaded "
                            "dataset — make sure you loaded the original file."
                        )
                    elif _responses_fingerprint(
                        df[_tcol].fillna("").str.strip().tolist()
                    ) != _blob["responses_hash"]:
                        st.error(
                            "This resume file doesn't match the loaded dataset. "
                            "Load the exact same file you were coding before."
                        )
                    else:
                        _done_n = len(_blob["results"])
                        st.success(
                            f"Resume file matches — **{_done_n:,} of {_blob['total']:,}** responses "
                            "already coded."
                        )
                        if st.button("↪️ Continue this run", type="primary", key="resume_continue"):
                            _opts = _blob["options"]
                            st.session_state.text_col = _tcol
                            st.session_state.covariate_cols = _blob.get("covariate_cols", [])
                            st.session_state.taxonomy = _blob["taxonomy"]
                            st.session_state.multi_theme = _opts.get("multi_theme", False)
                            st.session_state.include_valence = _opts.get("include_valence", False)
                            st.session_state.include_emotion = _opts.get("include_emotion", False)
                            st.session_state.sentiment_enabled = (
                                st.session_state.include_valence or st.session_state.include_emotion
                            )
                            _sig = _coding_signature(
                                _blob["total"], _blob["taxonomy"],
                                st.session_state.multi_theme,
                                st.session_state.include_valence,
                                st.session_state.include_emotion,
                            )
                            st.session_state.coding_progress = {
                                "sig": _sig, "results": _blob["results"], "total": _blob["total"],
                            }
                            st.session_state.coded_df = None
                            go_to(4)
                            st.rerun()


# ============================================================
# STEP 2 — Explore Themes
# ============================================================
elif st.session_state.step == 2:

    df = st.session_state.df
    text_col = st.session_state.text_col

    st.title("🔍 Explore Themes")
    st.markdown(
        "The model reads a sample of your responses and builds a coding taxonomy from what it finds. "
        "You can optionally share your hypotheses below — they'll be treated as research framing, "
        "not a fixed list."
    )

    st.subheader("Sample Responses")
    sample = df[text_col].dropna().sample(min(10, len(df)), random_state=1).tolist()
    for r in sample:
        _quote(r)

    st.divider()
    st.subheader("Your Theme Hypotheses (optional)")
    st.markdown(
        "If you have hunches about what themes exist in the data, enter them below — "
        "**one suspected theme per line**, followed by any keywords or phrases that signal it.\n\n"
        "**Example format:**\n"
        "```\n"
        "Billing errors — duplicate charge, wrong amount, unauthorized fee, refund\n"
        "Poor customer service — hold time, hung up, rude, no callback, transferred\n"
        "Fraud — identity theft, account hacked, unauthorized access, police report\n"
        "```\n"
        "**In live mode**, the model combines your hunches with what it finds in the data. "
        "It won't force-fit your suggestions — themes may be merged if they overlap, "
        "split if one covers too much ground, renamed to better reflect respondents' language, "
        "or skipped if the data doesn't support them. It may also surface themes you didn't anticipate.\n\n"
        "_Leave blank to let the model explore the data with no prior framing._"
    )
    user_seeds = st.text_area(
        "Theme hypotheses",
        value=st.session_state.user_seeds,
        placeholder="Billing errors — duplicate charge, wrong amount, refund\nPoor customer service — hold time, rude, no callback",
        height=150,
        label_visibility="collapsed",
    )
    st.session_state.user_seeds = user_seeds

    st.divider()
    st.subheader("How many themes?")
    theme_range = st.slider(
        "Target number of themes (the model picks the optimal count within this range)",
        min_value=2, max_value=15, value=(5, 8), step=1,
    )
    min_themes, max_themes = theme_range

    total_responses = int(df[text_col].dropna().str.strip().ne("").sum())
    max_cap = min(total_responses, 1200)

    if st.session_state.demo_mode:
        max_for_taxonomy = max_cap
    else:
        st.subheader("Responses to use for taxonomy development")
        max_for_taxonomy = st.slider(
            "How many responses should the model read to build the taxonomy?",
            min_value=min(50, max_cap),
            max_value=max_cap,
            value=min(500, max_cap),
            step=50,
            help="More responses → better coverage of edge-case themes, but higher cost and longer wait. "
                 "Responses are randomly sampled so order in your file doesn't bias the result. "
                 "Hard cap of 1,200 to stay within the model's context window.",
        )
        _provider = st.session_state.provider
        _model = st.session_state.model
        _pricing = PROVIDERS.get(_provider, {}).get("pricing", {}).get(_model, {})
        _pricing_url = PROVIDERS.get(_provider, {}).get("pricing_url", "")
        _pricing_link = f" [Current rates ↗]({_pricing_url})" if _pricing_url else ""
        if PROVIDERS.get(_provider, {}).get("free_tier"):
            _cost_note = (
                f"{_provider} free tier is ideal for this **taxonomy step** and **small / test runs**. "
                "For coding a larger dataset, the free daily cap (and free-hosting time limits) make a "
                "paid model more reliable — see the note in the sidebar."
            )
        elif not _pricing or _model == "__custom__":
            _cost_note = (
                f"Cost estimate unavailable for this model. "
                f"Check{_pricing_link if _pricing_link else ' the provider pricing page'} for current rates."
            )
        else:
            _in_price  = _pricing["input"]
            _out_price = _pricing["output"]
            _avg_tok = 100
            _est = (max_for_taxonomy * _avg_tok / 1_000_000 * _in_price
                    + 1_500 / 1_000_000 * _out_price)
            _coding_est_lo = total_responses * _avg_tok / 1_000_000 * _in_price
            _coding_est_hi = _coding_est_lo * 1.3
            _cost_note = (
                f"Each taxonomy run samples {max_for_taxonomy:,} responses: ~\${_est:.3f} per run. "
                f"Each coding run covers all {total_responses:,} responses: ~\${_coding_est_lo:.2f}–\${_coding_est_hi:.2f} per run. "
                f"Prices may vary.{_pricing_link}"
            )
        st.caption(_cost_note)

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("← Back"):
            go_to(1)
            st.rerun()
    with col2:
        _btn_label = "Suggest Themes (Demo) →" if st.session_state.demo_mode else "Suggest Themes →"
        if st.button(_btn_label, type="primary", use_container_width=True):
            if not st.session_state.demo_mode and not st.session_state.api_key:
                st.error("No API key — enter one in the sidebar or switch to Demo Mode.", icon="🔑")
            else:
                responses = df[text_col].dropna().str.strip().tolist()
                try:
                    if st.session_state.demo_mode:
                        taxonomy = suggest_themes_demo(responses, user_seeds, max_themes=max_themes)
                    else:
                        with st.spinner("Reading your responses and building a taxonomy..."):
                            taxonomy = suggest_themes(
                                st.session_state.provider, st.session_state.model,
                                st.session_state.api_key or None,
                                responses, user_seeds,
                                max_responses=max_for_taxonomy,
                                min_themes=min_themes, max_themes=max_themes,
                            )
                except Exception as _tax_err:
                    st.error(_friendly_api_error(_tax_err, st.session_state.provider), icon="⚠️")
                    _track("taxonomy_failed",
                           provider=st.session_state.provider,
                           model=st.session_state.model,
                           error=type(_tax_err).__name__)
                    st.stop()
                st.session_state.taxonomy = taxonomy
                st.session_state.preview_sample = None
                _track("taxonomy_generated",
                       provider=st.session_state.provider,
                       model=st.session_state.model,
                       num_responses=len(responses),
                       demo_mode=st.session_state.demo_mode)
                go_to(3)
                st.rerun()


# ============================================================
# STEP 3 — Refine Taxonomy
# ============================================================
elif st.session_state.step == 3:

    df = st.session_state.df
    text_col = st.session_state.text_col

    st.title("✏️ Refine Your Taxonomy")
    st.markdown(
        "Review the taxonomy below and shape it until it fits your research framing. "
        "You can **edit** names and descriptions directly in the table, **merge** overlapping themes, "
        "or **split** a theme that covers too much ground. "
        "Use **Preview** to run a quick test-code on a sample of responses before committing."
    )

    taxonomy = st.session_state.taxonomy

    # ── Editable table ─────────────────────────────────────────
    taxonomy_df = pd.DataFrame([
        {"Theme Name": t["name"], "Description": t["description"]}
        for t in taxonomy
    ])
    edited = st.data_editor(
        taxonomy_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Theme Name": st.column_config.TextColumn("Theme Name", width="medium"),
            "Description": st.column_config.TextColumn("Description", width="large"),
        },
    )

    def _read_taxonomy(edited_df):
        """Convert data_editor output back to taxonomy list, preserving examples/keywords."""
        orig_lookup = {t["name"]: t for t in st.session_state.taxonomy}
        rows = edited_df.dropna(subset=["Theme Name"])
        rows = rows[rows["Theme Name"].str.strip() != ""]
        return [
            {
                "name": row["Theme Name"].strip(),
                "description": str(row.get("Description") or ""),
                "examples": orig_lookup.get(row["Theme Name"].strip(), {}).get("examples", []),
                "_keywords": orig_lookup.get(row["Theme Name"].strip(), {}).get("_keywords", []),
            }
            for _, row in rows.iterrows()
        ]

    # ── Merge / Split tools ────────────────────────────────────
    col_merge, col_split = st.columns(2, gap="large")

    with col_merge:
        with st.expander("🔀 Merge Themes"):
            st.caption("Combine two or more overlapping themes into one.")
            current_names = [t["name"] for t in taxonomy]
            to_merge = st.multiselect("Select themes to merge", current_names, key="merge_select")
            merged_name = st.text_input("Name for merged theme", key="merge_name")
            if st.button("Merge", key="do_merge"):
                if len(to_merge) < 2:
                    st.warning("Select at least 2 themes to merge.")
                elif not merged_name.strip():
                    st.warning("Enter a name for the merged theme.")
                else:
                    current = _read_taxonomy(edited)
                    merged_examples = []
                    merged_keywords = []
                    for t in current:
                        if t["name"] in to_merge:
                            merged_examples.extend(t.get("examples", []))
                            merged_keywords.extend(t.get("_keywords", []))
                    new_taxonomy = [t for t in current if t["name"] not in to_merge]
                    new_taxonomy.append({
                        "name": merged_name.strip(),
                        "description": f"Combined theme covering: {', '.join(to_merge)}.",
                        "examples": merged_examples[:3],
                        "_keywords": merged_keywords,
                    })
                    st.session_state.taxonomy = new_taxonomy
                    st.session_state.preview_sample = None
                    _track("themes_merged", num_merged=len(to_merge))
                    st.rerun()

    with col_split:
        with st.expander("✂️ Split a Theme"):
            st.caption("Divide one theme into two more specific ones.")
            current_names = [t["name"] for t in taxonomy]
            to_split = st.selectbox("Theme to split", current_names, key="split_select")
            new_name_1 = st.text_input("Name for first new theme", key="split_name1")
            new_name_2 = st.text_input("Name for second new theme", key="split_name2")
            if st.session_state.demo_mode:
                st.caption(
                    "**Demo Mode:** provide keywords so responses can be correctly assigned "
                    "to each new theme. In live mode, the model handles this automatically."
                )
                kw_1 = st.text_input(
                    "Keywords for first theme (comma-separated)",
                    key="split_kw1",
                    placeholder="e.g. wait, hold, slow, hours",
                )
                kw_2 = st.text_input(
                    "Keywords for second theme (comma-separated)",
                    key="split_kw2",
                    placeholder="e.g. rude, supervisor, unhelpful, dismissed",
                )
            split_label = "Split (Demo)" if st.session_state.demo_mode else "Split with AI"
            if st.button(split_label, key="do_split"):
                if not new_name_1.strip() or not new_name_2.strip():
                    st.warning("Enter names for both new themes.")
                else:
                    current = _read_taxonomy(edited)
                    old_theme = next((t for t in current if t["name"] == to_split), None)
                    new_taxonomy = [t for t in current if t["name"] != to_split]

                    if st.session_state.demo_mode or not st.session_state.api_key:
                        examples = old_theme.get("examples", []) if old_theme else []
                        mid = max(1, len(examples) // 2)
                        kws_1 = [k.strip().lower() for k in kw_1.split(",") if k.strip()] if st.session_state.demo_mode else []
                        kws_2 = [k.strip().lower() for k in kw_2.split(",") if k.strip()] if st.session_state.demo_mode else []
                        new_taxonomy.extend([
                            {
                                "name": new_name_1.strip(),
                                "description": f"Responses related to {new_name_1.strip().lower()}.",
                                "examples": examples[:mid],
                                "_keywords": kws_1,
                            },
                            {
                                "name": new_name_2.strip(),
                                "description": f"Responses related to {new_name_2.strip().lower()}.",
                                "examples": examples[mid:],
                                "_keywords": kws_2,
                            },
                        ])
                    else:
                        # Live: ask the model to generate descriptions + examples
                        sample_for_split = (
                            df[df[text_col].notna()][text_col]
                            .sample(min(30, len(df)), random_state=42)
                            .tolist()
                        )
                        try:
                            with st.spinner(f"Splitting '{to_split}' with AI..."):
                                new_themes = split_theme(
                                    st.session_state.provider,
                                    st.session_state.model,
                                    st.session_state.api_key or None,
                                    old_theme or {"name": to_split, "description": ""},
                                    new_name_1.strip(),
                                    new_name_2.strip(),
                                    sample_for_split,
                                )
                        except Exception as _split_err:
                            st.error(_friendly_api_error(_split_err, st.session_state.provider), icon="⚠️")
                            st.stop()
                        new_taxonomy.extend(new_themes)

                    st.session_state.taxonomy = new_taxonomy
                    st.session_state.preview_sample = None
                    _track("theme_split")
                    st.rerun()

    # ── Preview ────────────────────────────────────────────────
    st.divider()
    st.subheader("👁 Preview Coding on a Sample")
    st.markdown(
        "Run a quick test-code on a sample of responses using the current taxonomy. "
        "This helps you spot miscoded responses and decide if any themes need further refinement "
        "before committing to coding all responses. Preview always uses single-theme coding."
    )

    preview_n = st.slider("Sample size", min_value=10, max_value=60, value=30, step=5)
    preview_label = "Run Preview (Demo)" if st.session_state.demo_mode else "Run Preview with AI"

    if st.button(preview_label, use_container_width=True):
        if not st.session_state.demo_mode and not st.session_state.api_key:
            st.error("No API key — enter one in the sidebar or switch to Demo Mode.", icon="🔑")
        else:
            current_taxonomy = _read_taxonomy(edited)
            _valid_mask = df[text_col].notna() & df[text_col].str.strip().ne("")
            sample_df = (
                df[_valid_mask]
                .sample(min(preview_n, _valid_mask.sum()), random_state=99)
                .reset_index(drop=True)
            )
            sample_texts = sample_df[text_col].str.strip().tolist()

            try:
                with st.spinner("Coding sample..."):
                    if st.session_state.demo_mode:
                        results = code_responses_demo(
                            sample_texts, current_taxonomy,
                            df=sample_df, progress_callback=None,
                            multi_theme=False,
                        )
                    else:
                        results = code_responses(
                            st.session_state.provider,
                            st.session_state.model,
                            st.session_state.api_key or None,
                            sample_texts, current_taxonomy,
                            multi_theme=False,
                        )
            except Exception as _preview_err:
                st.error(_friendly_api_error(_preview_err, st.session_state.provider), icon="⚠️")
                st.stop()

            _valid_preview = {t["name"] for t in current_taxonomy}
            def _first_valid_theme(r):
                return next((th for th in (r.get("themes") or []) if th in _valid_preview), NONE_THEME)
            st.session_state.preview_sample = [
                {"text": text, "primary_theme": _first_valid_theme(r)}
                for text, r in zip(sample_texts, results)
            ]
            st.rerun()

    if st.session_state.preview_sample:
        preview_df = pd.DataFrame(st.session_state.preview_sample)
        theme_counts = preview_df["primary_theme"].value_counts()
        n_other = (preview_df["primary_theme"] == "Other").sum()

        st.markdown(f"**{len(preview_df)} responses coded** across {preview_df['primary_theme'].nunique()} themes.")
        if n_other > 0:
            st.warning(
                f"{n_other} response(s) coded as 'Other' — these didn't fit any theme. "
                "Consider adding or broadening a theme to capture them."
            )

        for theme in theme_counts.index:
            theme_rows = preview_df[preview_df["primary_theme"] == theme]
            with st.expander(f"**{theme}** — {len(theme_rows)} responses"):
                for row in theme_rows.head(5).itertuples():
                    _quote(row.text)
                if len(theme_rows) > 5:
                    st.caption(f"... and {len(theme_rows) - 5} more in this theme")

    # ── Claude's initial examples ──────────────────────────────
    with st.expander("💡 Example Responses per Theme (from initial suggestion)"):
        for t in taxonomy:
            st.markdown(f"**{t['name']}** — *{t['description']}*")
            for ex in t.get("examples", []):
                _quote(ex)
            st.markdown("")

    # ── Navigation ─────────────────────────────────────────────
    st.divider()
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("← Back"):
            go_to(2)
            st.rerun()
    with col2:
        if st.button("Finalize Taxonomy & Set Coding Options →", type="primary", use_container_width=True):
            st.session_state.taxonomy = _read_taxonomy(edited)
            st.session_state.coded_df = None
            _track("taxonomy_finalized", num_themes=len(st.session_state.taxonomy))
            go_to(4)
            st.rerun()


# ============================================================
# STEP 4 — Code Responses
# ============================================================
elif st.session_state.step == 4:

    df = st.session_state.df
    text_col = st.session_state.text_col
    taxonomy = st.session_state.taxonomy

    st.title("⚙️ Code All Responses")
    theme_names = [t["name"] for t in taxonomy]
    st.markdown(
        f"The model will assign one of **{len(theme_names)} themes** to each of your "
        f"**{len(df):,} responses**."
    )
    st.markdown("**Themes:** " + " · ".join(f"`{n}`" for n in theme_names))

    with st.expander("⚙️ Coding Options", expanded=True):
        multi_theme = st.toggle(
            "Allow multiple themes per response",
            value=st.session_state.multi_theme,
            help="The model assigns up to 3 themes per response ranked by relevance. "
                 "The first theme is always treated as the primary theme for covariate analysis.",
        )
        st.session_state.multi_theme = multi_theme

        sentiment_enabled = st.toggle(
            "Assign sentiment to each response",
            value=st.session_state.sentiment_enabled,
        )
        st.session_state.sentiment_enabled = sentiment_enabled

        if sentiment_enabled:
            scol1, scol2 = st.columns(2)
            with scol1:
                include_valence = st.checkbox(
                    "Valence score  (1 = Very Negative → 5 = Very Positive)",
                    value=st.session_state.include_valence,
                )
                st.session_state.include_valence = include_valence
            with scol2:
                include_emotion = st.checkbox(
                    "Emotion label  (Frustrated, Angry, Satisfied, etc.)",
                    value=st.session_state.include_emotion,
                )
                st.session_state.include_emotion = include_emotion
        else:
            include_valence = False
            include_emotion = False

    if st.session_state.coded_df is None:
        responses = df[text_col].fillna("").str.strip().tolist()
        _n_total = len(responses)
        _sig = _coding_signature(_n_total, taxonomy, multi_theme, include_valence, include_emotion)

        # Drop stale checkpoint if taxonomy/options changed since it was saved
        _prog = st.session_state.coding_progress
        if _prog and _prog.get("sig") != _sig:
            _prog = None
            st.session_state.coding_progress = None
        _done = len(_prog["results"]) if _prog else 0
        _resume = bool(_prog) and 0 < _done < _n_total

        # Pre-run advisory: set expectations on time + the refresh-loses-progress caveat.
        _fast_provider = st.session_state.provider in ("OpenAI", "Anthropic")  # paid, high-throughput
        if not st.session_state.demo_mode and _n_total > 100:
            _batches = (_n_total + 19) // 20  # coding runs in batches of 20
            _est_min = max(1, round(_batches * (3 if _fast_provider else 6) / 60))
            st.warning(
                f"Coding {_n_total:,} responses is expected to take roughly **{_est_min} min**. "
                "Avoid refreshing the page while it runs — that loses progress. If you hit your "
                "model's request limit, click **Resume** to continue with a different (paid) model.",
                icon="⏱️",
            )

        _col1, _col2 = st.columns([1, 4])
        with _col1:
            if st.button("← Back", key="step4_back"):
                go_to(3)
                st.rerun()
        with _col2:
            if _resume:
                _label = f"▶ Resume coding ({_n_total - _done:,} of {_n_total:,} left)"
            else:
                _label = "Start Coding (Demo) →" if st.session_state.demo_mode else "Start Coding →"
            _start = st.button(_label, type="primary", use_container_width=True)

        if _resume:
            st.info(
                f"A previous run coded **{_done:,} of {_n_total:,}** responses before it stopped. "
                "Resume to finish the rest — you can switch to a different model or provider in the "
                "sidebar first (e.g. from a free tier that hit its daily cap).",
                icon="↪️",
            )
            _rcol1, _rcol2 = st.columns(2)
            with _rcol1:
                if st.button("↻ Start over instead", key="step4_restart", use_container_width=True):
                    st.session_state.coding_progress = None
                    st.rerun()
            with _rcol2:
                st.download_button(
                    "⬇ Save resume file",
                    data=_build_resume_blob(
                        text_col=text_col, covariate_cols=st.session_state.covariate_cols,
                        taxonomy=taxonomy,
                        options={"multi_theme": multi_theme, "include_valence": include_valence,
                                 "include_emotion": include_emotion},
                        total=_n_total, results=st.session_state.coding_progress["results"],
                        responses=responses,
                    ),
                    file_name="coding_resume.json", mime="application/json",
                    key="dl_resume_persist", use_container_width=True,
                    help="Download progress so you can continue even if the page/session resets — "
                         "re-upload it on the Load Data step.",
                )

        if _start:
            if not st.session_state.demo_mode and not st.session_state.api_key:
                st.error("No API key — enter one in the sidebar or switch to Demo Mode.", icon="🔑")
                st.stop()

            # (No pre-run preflight: a 1-token test can't predict a multi-request run and
            #  would waste one of a free tier's scarce requests. If the run can't start,
            #  the first batch fails immediately with the same clean error + checkpoint.)

            # Reuse the checkpoint when resuming; otherwise start a fresh one
            if not _resume:
                st.session_state.coding_progress = {"sig": _sig, "results": [], "total": _n_total}
            _start_index = len(st.session_state.coding_progress["results"])
            _remaining = responses[_start_index:]
            progress_bar = st.progress(_start_index / _n_total, text="Starting...")

            def _on_batch(batch_results):
                st.session_state.coding_progress["results"].extend(batch_results)
                _d = len(st.session_state.coding_progress["results"])
                progress_bar.progress(min(_d / _n_total, 1.0),
                                      text=f"Coding responses... {_d:,}/{_n_total:,}")

            try:
                if st.session_state.demo_mode:
                    _demo_results = code_responses_demo(
                        _remaining, taxonomy,
                        df=df.iloc[_start_index:].reset_index(drop=True),
                        multi_theme=multi_theme,
                        include_valence=include_valence, include_emotion=include_emotion,
                    )
                    _on_batch(_demo_results)
                else:
                    code_responses(
                        st.session_state.provider,
                        st.session_state.model,
                        st.session_state.api_key or None,
                        _remaining, taxonomy,
                        multi_theme=multi_theme,
                        include_valence=include_valence, include_emotion=include_emotion,
                        on_batch=_on_batch,
                    )
            except Exception as _coding_err:
                progress_bar.empty()
                _autosave_to_browser()  # persist the partial to the browser before anything resets
                _saved = len(st.session_state.coding_progress["results"])
                st.error(
                    _friendly_api_error(_coding_err, st.session_state.provider)
                    + f"\n\n**{_saved:,} of {_n_total:,} responses are saved** (auto-backed-up to this "
                      "browser). Click **Resume coding** to finish now, switch to a different model "
                      "first, or save the resume file below.",
                    icon="⚠️",
                )
                st.download_button(
                    "⬇ Save resume file",
                    data=_build_resume_blob(
                        text_col=text_col, covariate_cols=st.session_state.covariate_cols,
                        taxonomy=taxonomy,
                        options={"multi_theme": multi_theme, "include_valence": include_valence,
                                 "include_emotion": include_emotion},
                        total=_n_total, results=st.session_state.coding_progress["results"],
                        responses=responses,
                    ),
                    file_name="coding_resume.json", mime="application/json", key="dl_resume_fail",
                )
                _track("coding_failed",
                       provider=st.session_state.provider,
                       model=st.session_state.model,
                       error=type(_coding_err).__name__,
                       coded_so_far=_saved)
                st.stop()

            results = st.session_state.coding_progress["results"]
            # Safety: keep results aligned to the dataframe length
            if len(results) > _n_total:
                results = results[:_n_total]
            elif len(results) < _n_total:
                results += [{"themes": [NONE_THEME], "score": None, "label": None,
                             "emotion": None, "confidence": 0.0}] * (_n_total - len(results))
            # Validate theme names against the taxonomy — drop anything that isn't a real
            # theme (JSON-parsing artifacts like "score" / "confidence" can leak in as fake
            # themes). Then enforce single-theme if multi wasn't requested. All-invalid → None.
            _valid_themes = {t["name"] for t in taxonomy}
            for _r in results:
                _kept = [th for th in (_r.get("themes") or []) if th in _valid_themes]
                if not multi_theme:
                    _kept = _kept[:1]
                _r["themes"] = _kept or [NONE_THEME]

            progress_bar.progress(1.0, text="Done!")
            _track("coding_completed",
                   provider=st.session_state.provider,
                   model=st.session_state.model,
                   num_responses=_n_total,
                   demo_mode=st.session_state.demo_mode)
            coded_df = df.copy()
            coded_df["theme"] = [" | ".join(r["themes"]) for r in results]
            coded_df["primary_theme"] = [r["themes"][0] for r in results]
            coded_df["confidence"] = [r.get("confidence", 0.5) for r in results]
            if include_valence:
                coded_df["valence_score"] = [r["score"] for r in results]
                coded_df["valence_label"] = [r["label"] for r in results]
            if include_emotion:
                coded_df["emotion"] = [r["emotion"] for r in results]
            st.session_state.coded_df = coded_df
            st.session_state.coding_progress = None  # checkpoint consumed
            st.session_state.theme_quotes = {}  # invalidate cached quotes
            st.rerun()
    else:
        st.success(f"Coding complete — {len(st.session_state.coded_df):,} responses coded.")
        if st.button("↺ Re-run Coding with New Options", key="step4_rerun"):
            st.session_state.coded_df = None
            st.session_state.coding_progress = None
            st.session_state.theme_quotes = {}
            _track("coding_rerun")
            st.rerun()

    if st.session_state.coded_df is not None:
        coded_df = st.session_state.coded_df
        is_multi_theme = bool(st.session_state.get("multi_theme", False)) and \
            coded_df["theme"].str.contains(" | ", regex=False).any()
        show_cols = (
            [text_col, "primary_theme"]
            + (["theme"] if is_multi_theme else [])
            + (["confidence"] if "confidence" in coded_df.columns else [])
            + (["valence_score", "valence_label"] if "valence_score" in coded_df.columns else [])
            + (["emotion"] if "emotion" in coded_df.columns else [])
        )
        st.subheader("Spot Check (one per theme)")
        n_themes = coded_df["primary_theme"].nunique()
        per_theme = max(1, 10 // n_themes)
        # Explicit loop avoids pandas groupby/apply dropping the groupby column as index
        spot = pd.concat([
            grp[show_cols].sample(min(per_theme, len(grp)), random_state=7)
            for _, grp in coded_df.groupby("primary_theme")
        ]).sample(frac=1, random_state=7).reset_index(drop=True)
        if is_multi_theme:
            spot = spot.rename(columns={"theme": "theme_list"})
        st.dataframe(
            spot,
            use_container_width=True,
            hide_index=True,
            column_config={text_col: st.column_config.TextColumn(width="medium")},
        )
        st.caption(
            "One randomly selected response per theme. "
            + ("**theme_list** shows all assigned themes; **primary_theme** is used for statistical analysis. " if is_multi_theme else "")
            + ("**confidence** reflects certainty in the **primary theme** assignment — "
               "pre-computed signal strength in demo mode, the model's self-reported certainty in live mode: "
               "1.0 = unambiguous fit, 0.5 = another theme could apply, 0.0 = forced. "
               "Low-confidence responses are good candidates for manual review."
               if "confidence" in coded_df.columns else "")
        )

        # ── Inter-Rater Reliability ────────────────────────────────
        st.divider()
        with st.expander("🎯 Inter-Rater Reliability Check"):
            st.markdown(
                "Rate a random sample of responses yourself — **without seeing the model's labels** — "
                "then measure agreement using Cohen's Kappa and Krippendorff's Alpha."
            )
            if is_multi_theme:
                st.info(
                    "Multi-theme mode is on. IRR is computed on the **primary theme** only — "
                    "multi-label kappa extensions exist but are non-standard, and primary theme is "
                    "consistent with how all statistical analysis in this app is done.",
                    icon="ℹ️",
                )

            if st.session_state.irr_sample is None:
                irr_n = st.slider("Responses to rate", 5, 50, 20, step=5, key="irr_n_slider")
                if irr_n < 10:
                    st.caption("Fewer than 10 responses gives very noisy estimates — results are illustrative only.")
                if st.button("Begin Rating", key="irr_start"):
                    _track("irr_started", n=irr_n)
                    sample_rows = coded_df.sample(min(irr_n, len(coded_df)), random_state=42).reset_index(drop=True)
                    st.session_state.irr_sample = [
                        {"text": row[text_col], "model_theme": row["primary_theme"]}
                        for _, row in sample_rows.iterrows()
                    ]
                    st.session_state.irr_labels = {}
                    st.session_state.irr_pos = 0
                    st.rerun()

            elif st.session_state.irr_pos < len(st.session_state.irr_sample):
                irr_sample = st.session_state.irr_sample
                pos = st.session_state.irr_pos
                n_total = len(irr_sample)

                st.progress(pos / n_total, text=f"Response {pos + 1} of {n_total}")
                st.markdown("**Read the full response, then assign the best-fitting theme:**")
                # Full text — no truncation
                st.markdown(
                    f'<div style="border-left:3px solid #d0d0d0;padding:0.5em 0.75em;'
                    f'margin:0.2em 0 0.8em 0;color:inherit;font-size:0.95em;'
                    f'white-space:pre-wrap;word-wrap:break-word">'
                    f'{_html.escape(str(irr_sample[pos]["text"]))}</div>',
                    unsafe_allow_html=True,
                )

                theme_names = [t["name"] for t in st.session_state.taxonomy] + [NONE_THEME]
                prev = st.session_state.irr_labels.get(pos)
                prev_idx = theme_names.index(prev) if prev in theme_names else None

                choice = st.radio(
                    "Your theme:",
                    theme_names,
                    index=prev_idx,
                    key=f"irr_radio_{pos}",
                )

                col_back, col_next = st.columns([1, 3])
                with col_back:
                    if pos > 0 and st.button("← Back", key="irr_back"):
                        st.session_state.irr_pos -= 1
                        st.rerun()
                with col_next:
                    is_last = pos == n_total - 1
                    if st.button(
                        "Finish & See Results" if is_last else "Next →",
                        type="primary",
                        key="irr_next",
                    ):
                        st.session_state.irr_labels[pos] = choice
                        st.session_state.irr_pos = n_total if is_last else pos + 1
                        st.rerun()

                if st.button("↩ Start Over", key="irr_reset"):
                    st.session_state.irr_sample = None
                    st.session_state.irr_labels = {}
                    st.session_state.irr_pos = 0
                    st.rerun()

            else:
                irr_sample = st.session_state.irr_sample
                n_total = len(irr_sample)
                human = [st.session_state.irr_labels.get(i, "Other") for i in range(n_total)]
                model = [item["model_theme"] for item in irr_sample]

                kappa_r = irr_cohen_kappa(human, model)
                alpha_r = irr_krippendorff_alpha(human, model)

                c1, c2, c3 = st.columns(3)
                c1.metric("% Agreement", f"{kappa_r['pct_agree']}%")
                c2.metric("Cohen's Kappa", kappa_r["kappa"])
                c3.metric("Krippendorff's α", alpha_r["alpha"])

                st.markdown(f"**Interpretation:** {kappa_r['interpretation']} agreement.")
                st.markdown(
                    "**Cohen's Kappa** corrects for chance using each rater's individual label frequencies — "
                    "the standard metric for two-rater categorical agreement. "
                    "Landis & Koch benchmarks: < 0.20 Slight · 0.20–0.40 Fair · "
                    "0.40–0.60 Moderate · 0.60–0.80 Substantial · > 0.80 Almost perfect. "
                    "\n\n"
                    "**Krippendorff's Alpha** uses pooled label frequencies instead, making it slightly more "
                    "conservative and generalizable to more than two raters or ordinal/interval scales. "
                    "For two raters on nominal data the two metrics are usually close; they diverge when one "
                    "rater's label distribution is very different from the other's. "
                    "Krippendorff recommends α ≥ 0.80 for reliable conclusions and 0.67–0.80 for tentative ones."
                )

                mismatches = [(i, item) for i, item in enumerate(irr_sample) if human[i] != model[i]]
                n_agree = n_total - len(mismatches)

                comparison = pd.DataFrame({
                    "Response": [item["text"] for item in irr_sample],
                    "Your Label": human,
                    "Model Label": model,
                    "Match": ["✓" if h == m else "✗" for h, m in zip(human, model)],
                })
                st.dataframe(
                    comparison,
                    use_container_width=True,
                    hide_index=True,
                    column_config={"Response": st.column_config.TextColumn(width="large")},
                )

                # ── Actionable mismatches ──────────────────────────
                if mismatches:
                    st.markdown(f"#### Disagreements ({len(mismatches)} of {n_total})")
                    # Confusion summary: which theme pairs disagree most
                    from collections import Counter
                    pair_counts = Counter(
                        (human[i], item["model_theme"]) for i, item in enumerate(irr_sample)
                        if human[i] != item["model_theme"]
                    )
                    if pair_counts:
                        st.markdown("**Most common disagreements** (your label → model label):")
                        for (h_lbl, m_lbl), cnt in pair_counts.most_common(5):
                            st.markdown(f"- `{h_lbl}` ↔ `{m_lbl}`: {cnt} response(s)")

                    with st.expander("Review each mismatch"):
                        for i, item in mismatches:
                            st.markdown(
                                f"**Response {i+1}** — You: `{human[i]}` · Model: `{item['model_theme']}`"
                            )
                            st.markdown(
                                f'<div style="border-left:3px solid #f0a0a0;padding:0.4em 0.75em;'
                                f'margin:0.2em 0 0.6em 0;font-size:0.9em;white-space:pre-wrap">'
                                f'{_html.escape(str(item["text"]))}</div>',
                                unsafe_allow_html=True,
                            )

                    if kappa_r["kappa"] < 0.6:
                        if st.session_state.demo_mode:
                            st.info(
                                "**Low agreement detected.** In live mode you can refine the taxonomy "
                                "descriptions to clarify boundaries between confused themes, then re-run "
                                "coding — the updated descriptions are passed directly to the model.",
                                icon="💡",
                            )
                        else:
                            st.info(
                                "**Low agreement detected.** Go back to **Step 3 → Refine Taxonomy** and "
                                "tighten the descriptions for the themes that are being confused — "
                                "especially the pairs listed above. Clearer boundary descriptions directly "
                                "improve coding accuracy. Then re-run Step 4.",
                                icon="💡",
                            )
                            if st.button("← Go to Refine Taxonomy", key="irr_goto_step3"):
                                st.session_state.irr_sample = None
                                st.session_state.irr_labels = {}
                                st.session_state.irr_pos = 0
                                go_to(3)
                                st.rerun()

                if st.button("↩ Start Over", key="irr_reset_done"):
                    st.session_state.irr_sample = None
                    st.session_state.irr_labels = {}
                    st.session_state.irr_pos = 0
                    st.rerun()

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("← Back"):
                go_to(3)
                st.rerun()
        with col2:
            if st.button("View Analysis →", type="primary", use_container_width=True):
                go_to(5)
                st.rerun()


# ============================================================
# STEP 5 — Visualize & Analyze
# ============================================================
elif st.session_state.step == 5:
    coded_df = st.session_state.coded_df
    text_col = st.session_state.text_col
    covariate_cols = st.session_state.covariate_cols



    st.title("📊 Visualize & Analyze")

    # Blank (no-text) responses — excluded from charts and analysis entirely.
    _blank_mask = coded_df[text_col].isna() | (coded_df[text_col].astype(str).str.strip() == "")
    _n_blank = int(_blank_mask.sum())

    # Treat as multi-theme only if the user CHOSE it and the data has multiple tags
    # (guards against a model returning >1 theme on a single-theme run).
    is_multi = bool(st.session_state.get("multi_theme", False)) and \
        coded_df["theme"].str.contains(" | ", regex=False).any()
    _chart_col = "theme" if is_multi else "primary_theme"

    # themed_df: real themes only — blanks and "None of the above" removed.
    themed_df = coded_df[(~_blank_mask) & (coded_df["primary_theme"] != NONE_THEME)].copy()
    analysis_df = explode_themes(themed_df) if is_multi else themed_df.copy()
    covariate_df = themed_df  # one row per response, blanks + None excluded — all stats use this

    _none_count = int(((coded_df["primary_theme"] == NONE_THEME) & ~_blank_mask).sum())
    _total = len(coded_df)

    # Consistent theme order + color map (used by every chart on this page)
    import plotly.express as _px
    _palette = _px.colors.qualitative.Pastel
    _by_count = analysis_df[_chart_col].value_counts().sort_values()
    _theme_color_map = {t: _palette[i % len(_palette)] for i, t in enumerate(_by_count.index)}
    _theme_order = list(_by_count.index[::-1])  # most common → rarest

    # ── Theme Distribution ──────────────────────────────────────
    st.subheader("Theme Distribution")
    if is_multi:
        st.info(
            f"Multi-theme coding: chart shows **theme tags** ({len(analysis_df):,}) across "
            f"{len(themed_df):,} themed responses. Covariate analysis uses the **primary theme** "
            f"(most relevant) per response to preserve statistical independence.",
            icon="ℹ️",
        )
    st.plotly_chart(
        theme_bar_chart(analysis_df, _chart_col, color_map=_theme_color_map, total=_total),
        use_container_width=True,
    )
    st.caption(
        f"Each bar shows the count and its share of all {_total:,} responses. "
        "Blank and unmatched ('None of the above') responses are excluded — see below."
    )

    # Notes for the excluded buckets
    _excluded_notes = []
    if _n_blank > 0:
        _excluded_notes.append(
            f"**{_n_blank:,} ({_n_blank / _total * 100:.1f}%)** had no response (blank) — excluded."
        )
    if _none_count > 0:
        _excluded_notes.append(
            f"**{_none_count:,} ({_none_count / _total * 100:.1f}%)** had a response but didn't fit "
            f"any theme (*{NONE_THEME}*) — excluded from charts and analysis."
        )
    if _excluded_notes:
        st.info("  \n".join(_excluded_notes), icon="ℹ️")

    # ── Coding Confidence ───────────────────────────────────────
    if "confidence" in covariate_df.columns:
        with st.expander("Coding Confidence"):
            avg_conf = covariate_df["confidence"].mean()
            low_n = (covariate_df["confidence"] < 0.5).sum()
            st.plotly_chart(
                confidence_box_chart(covariate_df, "primary_theme", color_map=_theme_color_map, theme_order=_theme_order),
                use_container_width=True,
            )
            c1, c2 = st.columns(2)
            c1.metric("Mean confidence", f"{avg_conf:.2f}")
            c2.metric("Low-confidence responses (<0.5)", f"{low_n} ({low_n / len(covariate_df) * 100:.1f}%)")
            st.caption(
                "Confidence reflects the model's self-assessed certainty that the primary theme is the "
                "best fit (1.0 = unambiguous, 0.5 = plausible but another theme could apply, "
                "0.0 = forced fit). "
                "Low-confidence responses are good candidates to review manually or to re-examine "
                "whether the taxonomy needs a new theme."
            )

    # ── Representative Quotes ───────────────────────────────────
    with st.expander("Representative Quotes"):
        st.caption(
            "Verbatim excerpts that best capture each theme — the prevailing view, plus the occasional "
            "less-common but meaningful angle. "
            "Selection favors substantive comments that name specific pain points or requests."
        )

        _SMALL_THEME = 6  # themes with fewer than this get 1 quote max, no nuance
        _tax_lookup = {t["name"]: t for t in st.session_state.taxonomy}

        def _generate_quotes():
            tq = {}
            for theme in _theme_order:
                sub = covariate_df[covariate_df["primary_theme"] == theme]
                cands = build_candidates(sub, text_col)
                is_small = len(sub) < _SMALL_THEME
                max_rep = 1 if is_small else 3
                allow_nuance = not is_small
                theme_dict = _tax_lookup.get(theme, {"name": theme, "description": ""})
                if st.session_state.demo_mode:
                    picks = select_quotes_demo(theme_dict, cands, max_rep, allow_nuance)
                else:
                    picks = select_quotes(
                        st.session_state.provider, st.session_state.model,
                        st.session_state.api_key or None,
                        theme_dict, cands, max_rep, allow_nuance,
                    )
                tq[theme] = picks
            st.session_state.theme_quotes = tq
            _track("quotes_generated", demo_mode=st.session_state.demo_mode)

        def _render_quotes():
            for theme in _theme_order:
                picks = st.session_state.theme_quotes.get(theme, [])
                _n = int((covariate_df["primary_theme"] == theme).sum())
                st.markdown(f"**{theme}**  ·  {_n} response{'s' if _n != 1 else ''}")
                if not picks:
                    st.caption("_No sufficiently representative quote found — this may be a thin theme._")
                    continue
                for p in picks:
                    _quote(p["quote"])
                    _label = "Less common angle" if p["role"] == "nuance" else "Representative"
                    st.caption(_label + (f" · {p['reason']}" if p.get("reason") else ""))
                st.write("")

        if st.session_state.demo_mode:
            # Free and instant in demo mode — generate automatically, no button.
            if not st.session_state.theme_quotes:
                with st.spinner("Selecting representative quotes…"):
                    _generate_quotes()
            _render_quotes()
        elif st.session_state.theme_quotes:
            _render_quotes()
        else:
            # Live mode: optional (costs an API call per theme) → button + cost estimate.
            _qp = PROVIDERS.get(st.session_state.provider, {})
            _qprice = _qp.get("pricing", {}).get(st.session_state.model, {})
            _n_themes = len(_theme_order)
            if _qp.get("free_tier"):
                _q_cost_note = f"Uses your selected model — typically within {st.session_state.provider}'s free tier."
            elif _qprice:
                _q_total = (_n_themes * 2500 / 1_000_000 * _qprice["input"]
                            + _n_themes * 350 / 1_000_000 * _qprice["output"])
                _q_cost_note = (
                    f"Makes one small API call per theme ({_n_themes} themes) — "
                    f"about \${_q_total:.3f} total with your selected model. Cached afterward."
                )
            else:
                _q_cost_note = (
                    f"Makes one small API call per theme ({_n_themes} themes) using your selected model. "
                    "Cached afterward."
                )
            if st.button("✨ Generate representative quotes", key="gen_quotes"):
                try:
                    with st.spinner("Selecting representative quotes…"):
                        _generate_quotes()
                    st.rerun()
                except Exception as _quotes_err:
                    st.session_state.theme_quotes = {}
                    st.error(_friendly_api_error(_quotes_err, st.session_state.provider), icon="⚠️")
            st.caption(_q_cost_note)

    # ── Covariate Analysis ──────────────────────────────────────
    if covariate_cols:
        st.divider()
        st.subheader("Covariate Analysis")
        st.caption(
            "All covariate analyses use the **primary theme** per response (the single most relevant theme "
            "assigned by the model), ensuring each response contributes to exactly one group. "
            "This preserves the independence assumption required for chi-square, ANOVA, and Kruskal-Wallis."
        )

        selected_cov = st.selectbox("Select a covariate", covariate_cols)
        detected = detect_covariate_type(covariate_df[selected_cov])
        TYPE_LABELS = {
            "categorical": "Categorical (groups, labels)",
            "numeric": "Numeric / Continuous",
            "date": "Date / Time",
        }
        cov_type = st.radio(
            f"Variable type — auto-detected as **{TYPE_LABELS[detected]}**. Override if needed:",
            options=["categorical", "numeric", "date"],
            format_func=lambda x: TYPE_LABELS[x],
            index=["categorical", "numeric", "date"].index(detected),
            horizontal=True,
        )

        # ── Categorical ─────────────────────────────────────────
        if cov_type == "categorical":
            n_cats = covariate_df[selected_cov].nunique()
            if n_cats > 15:
                st.info(
                    f"'{selected_cov}' has {n_cats} categories — stacked bar shows top 10 by frequency. "
                    "Use the heatmap to page through all categories."
                )
                top_cats = covariate_df[selected_cov].value_counts().head(10).index
                plot_df = covariate_df[covariate_df[selected_cov].isin(top_cats)].copy()
            else:
                plot_df = covariate_df.copy()

            tab1, tab2 = st.tabs(["Stacked Bar", "Heatmap"])
            with tab1:
                st.plotly_chart(
                    covariate_stacked_bar(plot_df, "primary_theme", selected_cov, color_map=_theme_color_map, theme_order=_theme_order),
                    use_container_width=True,
                )
            with tab2:
                _HMAP_PAGE_SIZE = 10
                # Order by frequency so page 1 matches the stacked bar's top-10
                _all_hmap_cats = covariate_df[selected_cov].value_counts().index.tolist()
                _n_hmap_pages = max(1, (len(_all_hmap_cats) + _HMAP_PAGE_SIZE - 1) // _HMAP_PAGE_SIZE)
                _hmap_pg_key = f"_hmap_pg_{selected_cov}"
                if _hmap_pg_key not in st.session_state:
                    st.session_state[_hmap_pg_key] = 0
                _hmap_cur = min(st.session_state[_hmap_pg_key], _n_hmap_pages - 1)

                _hmap_cats = _all_hmap_cats[_hmap_cur * _HMAP_PAGE_SIZE : (_hmap_cur + 1) * _HMAP_PAGE_SIZE]
                _hmap_df = covariate_df[covariate_df[selected_cov].isin(_hmap_cats)]

                st.plotly_chart(
                    covariate_heatmap(_hmap_df, "primary_theme", selected_cov, theme_order=_theme_order),
                    use_container_width=True,
                )

                if _n_hmap_pages > 1:
                    _hc1, _hc2, _hc3 = st.columns([1, 3, 1])
                    with _hc1:
                        if st.button("← Prev", disabled=_hmap_cur == 0, key=f"_hmap_prev_{selected_cov}"):
                            st.session_state[_hmap_pg_key] = _hmap_cur - 1
                            st.rerun()
                    with _hc2:
                        _hmap_start = _hmap_cur * _HMAP_PAGE_SIZE + 1
                        _hmap_end = min((_hmap_cur + 1) * _HMAP_PAGE_SIZE, len(_all_hmap_cats))
                        st.caption(f"Categories {_hmap_start}–{_hmap_end} of {len(_all_hmap_cats)}")
                    with _hc3:
                        if st.button("Next →", disabled=_hmap_cur >= _n_hmap_pages - 1, key=f"_hmap_next_{selected_cov}"):
                            st.session_state[_hmap_pg_key] = _hmap_cur + 1
                            st.rerun()

            st.subheader(f"Chi-Square Test: Theme × {selected_cov}")
            stats = chi_square_summary(plot_df, "primary_theme", selected_cov)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Chi² Statistic", stats["chi2"])
            m2.metric("p-value", stats["p_value"])
            m3.metric("Degrees of Freedom", stats["dof"])
            m4.metric("Cramér's V", stats["cramers_v"])
            if stats["significant"]:
                strength = (
                    "weak" if stats["cramers_v"] < 0.1
                    else "moderate" if stats["cramers_v"] < 0.3
                    else "strong"
                )
                st.success(
                    f"Statistically significant association between themes and {selected_cov} "
                    f"(p = {stats['p_value']}). Cramér's V = {stats['cramers_v']} — {strength} effect."
                )
            else:
                st.info(f"No statistically significant association found (p = {stats['p_value']}).")
            n_cov = len(covariate_cols)
            if n_cov > 1:
                bonf_thresh = round(0.05 / n_cov, 4)
                st.caption(
                    "**Chi-square test of independence** asks whether the distribution of themes differs "
                    "significantly across groups of the covariate. A significant result (p < 0.05) means "
                    "the pattern is unlikely due to chance. **Cramér's V** measures effect size on a 0–1 scale: "
                    "< 0.1 = negligible, 0.1–0.3 = moderate, > 0.3 = strong. "
                    f"N = {stats['n']:,} responses.  \n"
                    f"⚠️ **Multiple comparisons:** you have {n_cov} covariates selected. Testing all at α = 0.05 "
                    f"gives up to {round(n_cov * 0.05, 2)} expected false positives. "
                    f"Consider a Bonferroni threshold of **p < {bonf_thresh}** when interpreting all tests together."
                )
            else:
                st.caption(
                    "**Chi-square test of independence** asks whether the distribution of themes differs "
                    "significantly across groups of the covariate. A significant result (p < 0.05) means "
                    "the pattern is unlikely due to chance. **Cramér's V** measures effect size on a 0–1 scale: "
                    "< 0.1 = negligible, 0.1–0.3 = moderate, > 0.3 = strong. "
                    f"N = {stats['n']:,} responses."
                )

        # ── Date / Time ─────────────────────────────────────────
        elif cov_type == "date":
            granularity = st.radio(
                "Time granularity", ["Month", "Week", "Day"], horizontal=True,
                help="Use coarser granularity when individual periods have fewer than ~20 responses.",
            )
            n_periods = covariate_df[selected_cov].pipe(
                lambda s: pd.to_datetime(s, errors="coerce")
                .dt.to_period({"Month": "M", "Week": "W", "Day": "D"}[granularity])
                .nunique()
            )
            if n_periods < 3:
                st.warning(
                    f"Only {n_periods} {granularity.lower()}(s) of data — try a coarser granularity."
                )

            tab1, tab2 = st.tabs(["Stacked Area", "Line"])
            with tab1:
                st.plotly_chart(
                    theme_over_time_chart(covariate_df, "primary_theme", selected_cov, granularity,
                                          color_map=_theme_color_map, theme_order=_theme_order),
                    use_container_width=True,
                )
            with tab2:
                st.plotly_chart(
                    theme_over_time_line(covariate_df, "primary_theme", selected_cov, granularity,
                                         color_map=_theme_color_map, theme_order=_theme_order),
                    use_container_width=True,
                )
            st.caption(
                "Each point shows what **percentage of responses in that time period** belonged to each theme. "
                "A rising line means that theme is becoming more prevalent over time; a falling line means less prevalent. "
                f"Tip: use coarser granularity if any {granularity.lower()} has fewer than ~20 responses, "
                "as sparse periods produce noisy estimates."
            )

            st.subheader("Linear Trend Test by Theme")
            trend_df = trend_test_summary(covariate_df, "primary_theme", selected_cov, granularity)
            if trend_df.empty:
                st.warning(f"Need at least 3 {granularity.lower()}s of data to run trend tests.")
            else:
                _theme_pos = {t: i for i, t in enumerate(_theme_order)}
                trend_df = trend_df.sort_values(
                    "Theme", key=lambda s: s.map(lambda t: _theme_pos.get(t, 999))
                ).reset_index(drop=True)

                def _trend_flag(row):
                    if row["Sig."] == "✓" and row["Slope (pp/period)"] > 0:
                        return "🟢"
                    if row["Sig."] == "✓" and row["Slope (pp/period)"] < 0:
                        return "🔴"
                    return ""

                trend_display = trend_df.copy()
                trend_display.insert(0, "", trend_display.apply(_trend_flag, axis=1))
                st.dataframe(trend_display, use_container_width=True, hide_index=True)
                sig_rising = trend_df[(trend_df["Sig."] == "✓") & (trend_df["Slope (pp/period)"] > 0)]
                sig_falling = trend_df[(trend_df["Sig."] == "✓") & (trend_df["Slope (pp/period)"] < 0)]
                if not sig_rising.empty:
                    rising_names = ", ".join(f"**{t}**" for t in sig_rising["Theme"])
                    st.success(f"Significantly rising over time: {rising_names}")
                if not sig_falling.empty:
                    falling_names = ", ".join(f"**{t}**" for t in sig_falling["Theme"])
                    st.info(f"Significantly falling over time: {falling_names}")
                if sig_rising.empty and sig_falling.empty:
                    st.info("No themes show a statistically significant linear trend at p < 0.05.")
                st.caption(
                    "**Slope** is the change in percentage points per time period (positive = rising, "
                    "negative = falling). **R²** measures how well a straight line fits the trend. "
                    "**p-value** tests whether the slope is significantly different from zero. "
                    "Rows are ordered by theme frequency (most common first); "
                    "🟢 = significant rising trend, 🔴 = significant falling trend. "
                    "⚠️ Treat as exploratory: time series data can have autocorrelation that inflates "
                    "significance, and short windows may reflect noise rather than real trends."
                )

        # ── Numeric / Continuous ─────────────────────────────────
        elif cov_type == "numeric":
            st.plotly_chart(
                anova_box_chart(covariate_df, "primary_theme", selected_cov, color_map=_theme_color_map, theme_order=_theme_order),
                use_container_width=True,
            )

            norm = normality_summary(covariate_df, "primary_theme", selected_cov)
            use_kruskal = norm["recommendation"] == "kruskal"

            with st.expander("Normality check (Shapiro-Wilk per group)"):
                norm_df = pd.DataFrame(norm["groups"])
                st.dataframe(norm_df, use_container_width=True, hide_index=True)
                if norm["any_fail"]:
                    st.warning(
                        "One or more groups failed the normality test (p < 0.05). "
                        "**Kruskal-Wallis** is used instead of ANOVA."
                    )
                else:
                    st.success(
                        "All testable groups appear normally distributed. "
                        "**ANOVA** is used."
                    )
                st.caption(
                    "**Shapiro-Wilk** tests whether each group's values are plausibly drawn from a normal "
                    "distribution. p < 0.05 means normality is rejected for that group. ANOVA assumes "
                    "normality within groups; Kruskal-Wallis is the nonparametric alternative when that "
                    "assumption doesn't hold."
                )

            if use_kruskal:
                st.subheader(f"Kruskal-Wallis Test: {selected_cov} × Theme")
                stats = kruskal_summary(covariate_df, "primary_theme", selected_cov)
                if stats is None:
                    st.warning("Not enough data per group to run Kruskal-Wallis.")
                else:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("H Statistic", stats["h_stat"])
                    m2.metric("p-value", stats["p_value"])
                    m3.metric("ε² (Epsilon-squared)", stats["epsilon_squared"])
                    if stats["significant"]:
                        strength = (
                            "small" if stats["epsilon_squared"] < 0.06
                            else "medium" if stats["epsilon_squared"] < 0.14
                            else "large"
                        )
                        st.success(
                            f"Significant difference in **{selected_cov}** across themes "
                            f"(H = {stats['h_stat']}, p = {stats['p_value']}). "
                            f"ε² = {stats['epsilon_squared']} — {strength} effect size."
                        )
                    else:
                        st.info(
                            f"No significant difference in {selected_cov} across themes "
                            f"(p = {stats['p_value']})."
                        )
                    st.caption(
                        "**Kruskal-Wallis** is a nonparametric test of whether the distributions of a "
                        "numeric variable differ across theme groups — used here because at least one group "
                        "failed the normality check. It compares median ranks rather than means. "
                        "**ε² (epsilon-squared)** is the effect size: < 0.06 = small, 0.06–0.14 = medium, "
                        "> 0.14 = large."
                    )
                    if stats["significant"]:
                        with st.expander("Post-hoc pairwise comparisons"):
                            st.markdown(
                                "Bonferroni-corrected pairwise Mann-Whitney U tests — which specific theme "
                                "pairs drive the significant Kruskal-Wallis result."
                            )
                            ph = kruskal_posthoc(covariate_df, "primary_theme", selected_cov)
                            if not ph.empty:
                                st.dataframe(ph, use_container_width=True, hide_index=True)
                                st.caption(
                                    "Medians (not means) are shown since Kruskal-Wallis is rank-based. "
                                    "p-values are Bonferroni-adjusted across all "
                                    f"{len(ph)} pairwise comparisons. **Sig. ✓** means adjusted p < 0.05."
                                )
            else:
                st.subheader(f"One-Way ANOVA: {selected_cov} × Theme")
                stats = anova_summary(covariate_df, "primary_theme", selected_cov)
                if stats is None:
                    st.warning("Not enough data per group to run ANOVA.")
                else:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("F Statistic", stats["f_stat"])
                    m2.metric("p-value", stats["p_value"])
                    m3.metric("η² (Eta-squared)", stats["eta_squared"])
                    if stats["significant"]:
                        strength = (
                            "small" if stats["eta_squared"] < 0.06
                            else "medium" if stats["eta_squared"] < 0.14
                            else "large"
                        )
                        st.success(
                            f"Significant difference in **{selected_cov}** across themes "
                            f"(F = {stats['f_stat']}, p = {stats['p_value']}). "
                            f"η² = {stats['eta_squared']} — {strength} effect size."
                        )
                    else:
                        st.info(
                            f"No significant difference in {selected_cov} across themes "
                            f"(p = {stats['p_value']})."
                        )
                    st.caption(
                        "**One-way ANOVA** tests whether the mean value of the numeric variable differs "
                        "significantly across theme groups. **η² (eta-squared)** measures the proportion of "
                        "variance explained by theme membership: < 0.06 = small, 0.06–0.14 = medium, > 0.14 = large."
                    )
                    if stats["significant"]:
                        with st.expander("Post-hoc pairwise comparisons"):
                            st.markdown(
                                "Bonferroni-corrected pairwise Welch t-tests — which specific theme pairs "
                                "drive the significant ANOVA result."
                            )
                            ph = anova_posthoc(covariate_df, "primary_theme", selected_cov)
                            if not ph.empty:
                                st.dataframe(ph, use_container_width=True, hide_index=True)
                                st.caption(
                                    "p-values are adjusted using Bonferroni correction across all "
                                    f"{len(ph)} pairwise comparisons. "
                                    "Welch's t-test (unequal variances) is used for each pair. "
                                    "**Sig. ✓** means the adjusted p < 0.05."
                                )

    # ── Valence ─────────────────────────────────────────────────
    if "valence_score" in covariate_df.columns:
        st.divider()
        st.subheader("Valence Analysis")
        tab1, tab2 = st.tabs(["Overall Distribution", "By Primary Theme"])
        with tab1:
            st.plotly_chart(
                sentiment_distribution_chart(covariate_df, "valence_label"), use_container_width=True
            )
        with tab2:
            st.plotly_chart(
                sentiment_by_theme_chart(covariate_df, "primary_theme", "valence_score", color_map=_theme_color_map, theme_order=_theme_order),
                use_container_width=True,
            )
        avg = covariate_df["valence_score"].mean()
        median = covariate_df["valence_score"].median()
        c1, c2 = st.columns(2)
        c1.metric("Mean Valence Score", f"{avg:.2f} / 5")
        c2.metric("Median Valence Score", f"{median:.1f} / 5")
        st.caption(
            "Valence is scored on a 1–5 scale: 1 = Very Negative, 2 = Negative, 3 = Neutral, "
            "4 = Positive, 5 = Very Positive. The 'By Primary Theme' box plot shows the median score "
            "and spread within each theme group. In complaint datasets, scores are typically skewed negative — "
            "look for which themes have the most negative or most variable valence rather than absolute values."
        )

        with st.expander("Statistical Test: Valence Score × Primary Theme"):
            _val_norm = normality_summary(covariate_df, "primary_theme", "valence_score")
            _val_use_kruskal = _val_norm["recommendation"] == "kruskal"
            _val_test_name = "Kruskal-Wallis" if _val_use_kruskal else "One-Way ANOVA"
            st.markdown(
                f"Testing whether **valence scores differ significantly across themes** using "
                f"**{_val_test_name}** (chosen based on normality check below)."
            )

            with st.expander("Normality check (Shapiro-Wilk per theme group)"):
                _val_norm_df = pd.DataFrame(_val_norm["groups"])
                st.dataframe(_val_norm_df, use_container_width=True, hide_index=True)
                if _val_norm["any_fail"]:
                    st.warning("One or more groups failed the normality test — using Kruskal-Wallis.")
                else:
                    st.success("All groups appear normally distributed — using ANOVA.")

            if _val_use_kruskal:
                _val_stats = kruskal_summary(covariate_df, "primary_theme", "valence_score")
                if _val_stats is None:
                    st.warning("Not enough data per theme group to run Kruskal-Wallis.")
                else:
                    _vm1, _vm2, _vm3 = st.columns(3)
                    _vm1.metric("H Statistic", _val_stats["h_stat"])
                    _vm2.metric("p-value", _val_stats["p_value"])
                    _vm3.metric("ε² (Epsilon-squared)", _val_stats["epsilon_squared"])
                    if _val_stats["significant"]:
                        _val_strength = (
                            "small" if _val_stats["epsilon_squared"] < 0.06
                            else "medium" if _val_stats["epsilon_squared"] < 0.14
                            else "large"
                        )
                        st.success(
                            f"Significant difference in valence scores across themes "
                            f"(H = {_val_stats['h_stat']}, p = {_val_stats['p_value']}). "
                            f"ε² = {_val_stats['epsilon_squared']} — {_val_strength} effect."
                        )
                        with st.expander("Post-hoc pairwise comparisons (Bonferroni-corrected Mann-Whitney U)"):
                            _val_ph = kruskal_posthoc(covariate_df, "primary_theme", "valence_score")
                            if not _val_ph.empty:
                                st.dataframe(_val_ph, use_container_width=True, hide_index=True)
                                st.caption(
                                    "Medians shown; p-values Bonferroni-adjusted across all pairwise comparisons. "
                                    "**Sig. ✓** means adjusted p < 0.05."
                                )
                    else:
                        st.info(
                            f"No significant difference in valence scores across themes "
                            f"(p = {_val_stats['p_value']})."
                        )
                    st.caption(
                        "**Kruskal-Wallis** tests whether valence score distributions differ across theme groups. "
                        "**ε²** (epsilon-squared) is the effect size: < 0.06 = small, 0.06–0.14 = medium, > 0.14 = large."
                    )
            else:
                _val_stats = anova_summary(covariate_df, "primary_theme", "valence_score")
                if _val_stats is None:
                    st.warning("Not enough data per theme group to run ANOVA.")
                else:
                    _vm1, _vm2, _vm3 = st.columns(3)
                    _vm1.metric("F Statistic", _val_stats["f_stat"])
                    _vm2.metric("p-value", _val_stats["p_value"])
                    _vm3.metric("η² (Eta-squared)", _val_stats["eta_squared"])
                    if _val_stats["significant"]:
                        _val_strength = (
                            "small" if _val_stats["eta_squared"] < 0.06
                            else "medium" if _val_stats["eta_squared"] < 0.14
                            else "large"
                        )
                        st.success(
                            f"Significant difference in valence scores across themes "
                            f"(F = {_val_stats['f_stat']}, p = {_val_stats['p_value']}). "
                            f"η² = {_val_stats['eta_squared']} — {_val_strength} effect."
                        )
                        with st.expander("Post-hoc pairwise comparisons (Bonferroni-corrected Welch t-tests)"):
                            _val_ph = anova_posthoc(covariate_df, "primary_theme", "valence_score")
                            if not _val_ph.empty:
                                st.dataframe(_val_ph, use_container_width=True, hide_index=True)
                                st.caption(
                                    "p-values Bonferroni-adjusted across all pairwise comparisons. "
                                    "**Sig. ✓** means adjusted p < 0.05."
                                )
                    else:
                        st.info(
                            f"No significant difference in valence scores across themes "
                            f"(p = {_val_stats['p_value']})."
                        )
                    st.caption(
                        "**One-way ANOVA** tests whether mean valence score differs across theme groups. "
                        "**η²** (eta-squared) is the proportion of variance explained by theme: "
                        "< 0.06 = small, 0.06–0.14 = medium, > 0.14 = large."
                    )

    if "emotion" in covariate_df.columns:
        st.divider()
        st.subheader("Emotion Analysis")
        tab1, tab2 = st.tabs(["Overall Distribution", "By Primary Theme"])
        with tab1:
            st.plotly_chart(
                emotion_distribution_chart(covariate_df, "emotion"), use_container_width=True
            )
        with tab2:
            st.plotly_chart(
                emotion_by_theme_chart(covariate_df, "primary_theme", "emotion", color_map=_theme_color_map, theme_order=_theme_order),
                use_container_width=True,
            )
        st.caption(
            "Emotion labels reflect the predominant emotional tone of each response as interpreted by the model. "
            "Emotions are ordered most-negative to most-positive: "
            "Angry → Frustrated → Disappointed → Worried → Confused → Neutral → Relieved → Satisfied. "
            "The 'By Primary Theme' bars show each emotion as a **percentage of all responses in that theme** "
            "(including unlabeled ones), so bars do not sum to 100% — the gap represents responses the model "
            "did not assign a clear emotion. This makes coverage differences across themes visible."
        )

        with st.expander("Statistical Test: Emotion Label × Primary Theme"):
            _emo_clean = covariate_df.dropna(subset=["emotion"])
            if len(_emo_clean) < 10 or _emo_clean["emotion"].nunique() < 2:
                st.warning("Not enough labeled responses to run a chi-square test.")
            else:
                _emo_stats = chi_square_summary(_emo_clean, "primary_theme", "emotion")
                st.markdown(
                    "Testing whether **emotion label distributions differ significantly across themes** "
                    "using a **chi-square test of independence**."
                )
                _em1, _em2, _em3, _em4 = st.columns(4)
                _em1.metric("Chi² Statistic", _emo_stats["chi2"])
                _em2.metric("p-value", _emo_stats["p_value"])
                _em3.metric("Degrees of Freedom", _emo_stats["dof"])
                _em4.metric("Cramér's V", _emo_stats["cramers_v"])
                if _emo_stats["significant"]:
                    _emo_strength = (
                        "weak" if _emo_stats["cramers_v"] < 0.1
                        else "moderate" if _emo_stats["cramers_v"] < 0.3
                        else "strong"
                    )
                    st.success(
                        f"Significant association between emotion labels and primary theme "
                        f"(p = {_emo_stats['p_value']}). "
                        f"Cramér's V = {_emo_stats['cramers_v']} — {_emo_strength} effect."
                    )
                else:
                    st.info(
                        f"No statistically significant association between emotion and theme "
                        f"(p = {_emo_stats['p_value']})."
                    )
                st.caption(
                    "**Chi-square test of independence** asks whether emotion label frequencies differ "
                    "across theme groups beyond what chance would produce. "
                    "**Cramér's V** measures effect size on a 0–1 scale: "
                    "< 0.1 = negligible, 0.1–0.3 = moderate, > 0.3 = strong. "
                    f"N = {_emo_stats['n']:,} responses with emotion labels."
                )

    # ── Download ────────────────────────────────────────────────
    st.divider()
    st.subheader("Download")
    _is_multi = bool(st.session_state.get("multi_theme", False)) and \
        coded_df["theme"].str.contains(" | ", regex=False).any()
    _drop_cols = ([] if _is_multi else ["theme"]) + [c for c in coded_df.columns if c.startswith("_")]
    _rename_cols = {"theme": "theme_list"} if _is_multi else {}
    _download_df = coded_df.drop(columns=_drop_cols).rename(columns=_rename_cols)

    # Flag responses surfaced as representative quotes (if generated)
    _quotes_ready = bool(st.session_state.theme_quotes)
    if _quotes_ready:
        _role_by_row, _excerpt_by_row = {}, {}
        for _picks in st.session_state.theme_quotes.values():
            for _p in _picks:
                _role_by_row[_p["row"]] = _p["role"]
                _excerpt_by_row[_p["row"]] = _p["quote"]
        _download_df["highlighted_quote"] = _download_df.index.map(lambda i: _role_by_row.get(i, ""))
        _download_df["quote_excerpt"] = _download_df.index.map(lambda i: _excerpt_by_row.get(i, ""))

    _dl_col1, _dl_col2 = st.columns(2)
    with _dl_col1:
        if st.download_button(
            "⬇ Coded Dataset (CSV)",
            data=_download_df.to_csv(index=False).encode(),
            file_name="coded_responses.csv",
            mime="text/csv",
            use_container_width=True,
        ):
            _track("download_csv", demo_mode=st.session_state.demo_mode)
        st.caption(
            "Includes the original data plus: "
            "`primary_theme`, `confidence`, "
            + ("`theme_list` (all assigned themes), " if _is_multi else "")
            + "and `valence_score` / `valence_label` / `emotion` if enabled."
            + (" Plus `highlighted_quote` / `quote_excerpt` marking responses chosen as representative quotes."
               if _quotes_ready else
               " *Generate representative quotes above to also flag them in the download.*")
        )
    with _dl_col2:
        _nb_include_valence = "valence_score" in coded_df.columns
        _nb_include_emotion = "emotion" in coded_df.columns
        _nb_bytes = generate_notebook(
            coded_df=_download_df,
            text_col=st.session_state.text_col,
            covariate_cols=st.session_state.covariate_cols,
            taxonomy=st.session_state.taxonomy,
            include_valence=_nb_include_valence,
            include_emotion=_nb_include_emotion,
            theme_order=_theme_order,
            quotes=st.session_state.theme_quotes or None,
        )
        if st.download_button(
            "⬇ Analysis Notebook (Jupyter)",
            data=_nb_bytes,
            file_name="survey_analysis.ipynb",
            mime="application/x-ipynb+json",
            use_container_width=True,
        ):
            _track("download_notebook", demo_mode=st.session_state.demo_mode)
        st.caption(
            "Self-contained Jupyter notebook with all charts and statistical tests. "
            "The coded dataset is embedded — no external files needed. "
            "Open in JupyterLab, VS Code, or Google Colab and run top-to-bottom."
        )

    if st.button("← Back to Coding"):
        go_to(4)
        st.rerun()

    # ── Feedback ───────────────────────────────────────────────
    st.divider()
    st.subheader("💬 Feedback")
    st.caption(
        "Made it to the end? I'd love to hear what worked, what didn't, "
        "or any features you'd like to see."
    )
    _feedback_form("end")
