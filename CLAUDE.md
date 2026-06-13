# Survey Response Coder — project notes

AI-assisted qualitative coding tool for open-ended survey responses. Streamlit app.
- Live: https://survey-response-coder.streamlit.app/
- Repo: joliem/survey-response-coder (push to `main` → Streamlit Cloud auto-redeploys, which **resets the live session** — never push while the user is mid-run on the live app).

## The owner & the lens that matters most

Owner: **Jolie Martin — a UX researcher.** This is a **portfolio piece**, so the *craft of the user
flow IS the deliverable*, judged by a UX/research-savvy audience.

**Apply a UX lens to everything.** Prioritize:
- **Visible thoughtfulness over invisible robustness.** Graceful failure states (never raw
  tracebacks), honest copy that doesn't overpromise, smooth recovery, sensible empty/first-run
  states. A janky edge case reads as "didn't design for real users" to this audience.
- **First-impression polish** (the first 30s a cold visitor sees) is high-leverage and cheaper than
  edge-case hardening — weight it accordingly.
- **Don't overstate / don't mislead.** The user repeatedly (rightly) pushes back on confident-but-
  wrong claims and copy that recommends the thing they're already using. When unsure, hedge honestly.
- **Avoid robustness rabbit holes.** "Doesn't embarrass in a realistic demo" is the bar, not
  "survives a meteor strike." Call the resilience layer done once verified.

## Conventions worth keeping

- **Model labels describe relative position only** ("most capable / balanced / fastest · cheapest").
  Absolute prices live ONLY in the `PROVIDERS` `pricing` dict in `providers.py` — so a price change
  touches one number, not labels/warnings. No "paid" label (anything without a "free" tag is paid).
- **Provider-aware error messages** (`_friendly_api_error`): Gemini = OpenAI-compatible client, so its
  errors are `openai.*` — key messages off the selected provider, not the exception class.
- A monthly scheduled task (`monthly-llm-model-pricing-audit`) checks the 3 providers for new
  models/pricing and opens a PR to `providers.py`.
- Resilience already built: graceful failures, in-session checkpoint/resume, durable resume file,
  browser localStorage auto-save, chunked coding (survives mid-run refresh). Don't re-litigate these.

## Open / next ideas
- First-impression heuristic pass on the cold new-user flow (Demo Mode discoverability, empty states,
  legibility of the 5-step model).
