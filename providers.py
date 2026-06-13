"""Multi-provider LLM interface: Anthropic, OpenAI, Google Gemini.

Includes representative-quote selection (select_quotes).
"""

import json

# ── Registry ───────────────────────────────────────────────────────────────────

_CUSTOM = {"id": "__custom__", "label": "Other (enter model ID)…"}

PROVIDERS = {
    "Anthropic": {
        "label": "Anthropic (Claude)",
        "models": [
            {"id": "claude-opus-4-8",    "label": "Claude Opus 4.8   (most capable)"},
            {"id": "claude-sonnet-4-6",  "label": "Claude Sonnet 4.6 (balanced)"},
            {"id": "claude-haiku-4-5",   "label": "Claude Haiku 4.5  (fastest · cheapest)"},
            _CUSTOM,
        ],
        "free_tier": False,
        "free_tier_note": "",
        "key_env": "ANTHROPIC_API_KEY",
        "key_url": "https://console.anthropic.com/settings/keys",
        "pricing_url": "https://www.anthropic.com/pricing",
        "pricing": {
            "claude-opus-4-8":           {"input":  5.00, "output": 25.00},
            "claude-opus-4-7":           {"input":  5.00, "output": 25.00},
            "claude-opus-4-6":           {"input":  5.00, "output": 25.00},
            "claude-sonnet-4-6":         {"input":  3.00, "output": 15.00},
            "claude-haiku-4-5":          {"input":  1.00, "output":  5.00},
            # Legacy ID kept for backward compatibility
            "claude-haiku-4-5-20251001": {"input":  1.00, "output":  5.00},
        },
    },
    "OpenAI": {
        "label": "OpenAI (GPT)",
        "models": [
            {"id": "gpt-4.1",      "label": "GPT-4.1      (most capable)"},
            {"id": "gpt-4.1-mini", "label": "GPT-4.1 mini (balanced)"},
            {"id": "gpt-4.1-nano", "label": "GPT-4.1 nano (fastest · cheapest)"},
            _CUSTOM,
        ],
        "free_tier": False,
        "free_tier_note": "",
        "key_env": "OPENAI_API_KEY",
        "key_url": "https://platform.openai.com/api-keys",
        "pricing_url": "https://openai.com/api/pricing",
        "pricing": {
            "gpt-4.1":      {"input": 2.00, "output":  8.00},
            "gpt-4.1-mini": {"input": 0.40, "output":  1.60},
            "gpt-4.1-nano": {"input": 0.10, "output":  0.40},
            "gpt-4o":       {"input": 2.50, "output": 10.00},
            "gpt-4o-mini":  {"input": 0.15, "output":  0.60},
        },
    },
    "Google Gemini": {
        "label": "Google Gemini",
        "models": [
            {"id": "gemini-2.5-pro",        "label": "Gemini 2.5 Pro       (most capable)"},
            {"id": "gemini-2.5-flash",      "label": "Gemini 2.5 Flash     (balanced · free, larger daily cap)"},
            {"id": "gemini-2.5-flash-lite", "label": "Gemini 2.5 Flash Lite (fastest · free, smallest daily cap)"},
            _CUSTOM,
        ],
        "free_tier": True,
        "free_tier_note": (
            "Great for the **taxonomy step** and **small / test runs** (a few hundred responses). "
            "**2.5 Flash** has the larger free daily allowance, **Flash Lite** the smaller. For "
            "**full coding of larger datasets**, free daily caps and free-hosting time limits make a "
            "paid model (e.g. OpenAI GPT-4.1 nano/mini — fast, a few cents) far more reliable. "
            "2.5 Pro requires billing."
        ),
        "key_env": "GEMINI_API_KEY",
        "key_url": "https://aistudio.google.com/app/apikey",
        "pricing_url": "https://ai.google.dev/pricing",
        "pricing": {
            "gemini-2.5-pro":        {"input": 1.25, "output": 10.00},
            "gemini-2.5-flash":      {"input": 0.15, "output":  0.60},
            "gemini-2.5-flash-lite": {"input": 0.10, "output":  0.40},
            "gemini-2.0-flash":      {"input": 0.10, "output":  0.40},
        },
    },
}

DEFAULT_MODEL = {
    p: next(m["id"] for m in info["models"] if m["id"] != "__custom__")
    for p, info in PROVIDERS.items()
}

NONE_THEME = "None of the above"

# ── Shared prompts ─────────────────────────────────────────────────────────────

_SUGGEST_SYSTEM_TEMPLATE = """\
You are an expert qualitative researcher helping build a coding taxonomy for open-ended survey responses.

Given a sample of responses, suggest {min_themes}–{max_themes} mutually exclusive and collectively exhaustive themes.
Choose the number that best fits the actual diversity in the data — don't pad with thin themes or merge distinct ones just to hit a target.
For each theme provide:
  - name: short label (2-5 words)
  - description: one clear sentence defining what belongs in this theme
  - examples: 2-3 verbatim short excerpts from the responses that fit this theme

Return ONLY valid JSON in this exact structure — no prose, no markdown fences:
{{"themes": [{{"name": "...", "description": "...", "examples": ["...", "..."]}}]}}
"""

EMOTIONS = [
    "Frustrated", "Confused", "Angry", "Disappointed",
    "Worried", "Satisfied", "Relieved", "Neutral",
]


def _build_system(taxonomy, multi_theme, include_valence, include_emotion):
    taxonomy_lines = "\n".join(f'- "{t["name"]}": {t["description"]}' for t in taxonomy)
    theme_rule = (
        "Assign 1–3 themes in order of relevance (most relevant first)."
        if multi_theme else
        "Assign EXACTLY ONE theme — the single best fit."
    )
    sentiment_lines = []
    output_keys = ['"themes"']
    example_obj: dict = {"themes": ["Theme A"]}

    if include_valence:
        sentiment_lines.append(
            "- Rate sentiment on a 1–5 scale: 1 = Very Negative, 2 = Negative, 3 = Neutral, "
            '4 = Positive, 5 = Very Positive. Include "score" (integer 1–5) and "label" (the text). '
            "Default to 3 / Neutral when tone is unclear."
        )
        output_keys += ['"score"', '"label"']
        example_obj.update({"score": 2, "label": "Negative"})

    if include_emotion:
        emotions_str = ", ".join(EMOTIONS)
        sentiment_lines.append(
            f'- Identify the primary emotion. Choose from: {emotions_str}. '
            'Include "emotion". Use "" when no clear emotion is expressed.'
        )
        output_keys.append('"emotion"')
        example_obj["emotion"] = "Frustrated"

    output_keys.append('"confidence"')
    example_obj["confidence"] = 0.85

    sentiment_block = "\n".join(sentiment_lines) + "\n" if sentiment_lines else ""
    confidence_line = (
        '- Rate your confidence in the primary theme: "confidence" (float 0.0–1.0; '
        '1.0 = unambiguous fit, 0.5 = plausible but another theme could apply, 0.0 = forced fit).\n'
    )
    keys_str = ", ".join(output_keys)
    example = f"[{json.dumps(example_obj)}]"

    return (
        "You are an expert qualitative researcher coding open-ended survey responses.\n\n"
        f"CODING TAXONOMY:\n{taxonomy_lines}\n"
        f'- "{NONE_THEME}": use for responses that are too vague, short, or off-topic to '
        'meaningfully fit any theme — e.g. "good", "ok", "N/A", blank text, or unrelated content.\n\n'
        "Instructions:\n"
        f"- {theme_rule}\n"
        "- Use theme names verbatim as listed above.\n"
        f"{sentiment_block}"
        f"{confidence_line}"
        "- Return ONLY a JSON array of objects, one per response, in input order.\n"
        f"- Each object must have exactly these keys: {keys_str}\n"
        "- No explanation, no markdown fences — just the JSON array.\n"
        f"- Example: {example}\n"
    )


def _strip_fences(text: str) -> str:
    """Extract the first complete JSON object or array from model output.

    Handles: markdown fences, leading prose, trailing prose/comments.
    """
    text = text.strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if "```" in text:
            text = text.rsplit("```", 1)[0]
        text = text.strip()

    # Find the first JSON start character
    start = -1
    start_char = ""
    for i, ch in enumerate(text):
        if ch in ("{", "["):
            start = i
            start_char = ch
            break
    if start == -1:
        return text  # nothing to strip — let JSON parser surface the error

    end_char = "}" if start_char == "{" else "]"
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == start_char:
            depth += 1
        elif ch == end_char:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    # Unbalanced — return from start and let JSON parser surface the error
    return text[start:]


# ── Anthropic path ─────────────────────────────────────────────────────────────

def _anthropic_client(api_key: str):
    import anthropic
    return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()


def _suggest_anthropic(model, api_key, responses, user_seeds, max_responses, min_themes, max_themes):
    import random
    client = _anthropic_client(api_key)
    pool = list(responses)
    random.shuffle(pool)
    sample = pool[:min(max_responses, 1200)]
    numbered = "\n\n".join(f"{i+1}. {r}" for i, r in enumerate(sample))
    user_msg = f"Here are {len(sample)} survey responses:\n\n{numbered}"
    if user_seeds.strip():
        user_msg += f"\n\nThe researcher suggests these possible themes: {user_seeds}"
    user_msg += "\n\nSuggest a coding taxonomy."
    system = _SUGGEST_SYSTEM_TEMPLATE.format(min_themes=min_themes, max_themes=max_themes)
    msg = client.messages.create(
        model=model, max_tokens=2048, system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    return [_normalize_theme(t) for t in json.loads(msg.content[0].text)["themes"]]


# Blank / non-answer responses carry no thematic content — they're auto-coded as
# "None of the above" and NEVER sent to the model (no tokens spent on empty cells).
_UNINFORMATIVE = {
    "", "nothing", "nothing.", "nothing really", "none", "no", "nope", "na", "n/a", "nan",
    "nil", "nada", "good", "great", "ok", "okay", "fine", "nice", "cool", "yes", "yeah",
    "all good", "not really", "no comment", "not applicable", "thanks", "thank you", "n a",
}


def is_uninformative(text) -> bool:
    """True for blank / non-answer responses that carry no thematic content."""
    t = str(text).strip().lower().strip(".!?,-_ ")
    return len(t) <= 1 or t in _UNINFORMATIVE


def _none_result(include_valence):
    return {"themes": [NONE_THEME],
            "score": 3 if include_valence else None,
            "label": "Neutral" if include_valence else None,
            "emotion": None, "confidence": 1.0}


def _code_one_batch(batch, call_model, results, include_valence):
    """Code one batch, skipping blank/non-answer responses (no API tokens spent on them).
    Appends exactly len(batch) result dicts to `results`, in original order; skipped
    responses get a 'None of the above' result. Retries parse failures up to 3x. If a batch
    is entirely non-answers, the model isn't called at all."""
    inf_idx = [i for i, r in enumerate(batch) if not is_uninformative(r)]
    parsed = []
    if inf_idx:
        inf = [batch[i] for i in inf_idx]
        numbered = "\n\n".join(f"{j+1}. {r}" for j, r in enumerate(inf))
        for _attempt in range(3):
            try:
                parsed = []
                _parse_batch(call_model(numbered, len(inf)), parsed)
                break
            except (ValueError, SyntaxError, KeyError, TypeError):
                parsed = []
                if _attempt == 2:
                    raise
    inf_set = set(inf_idx)
    _it = iter(parsed)
    for i in range(len(batch)):
        if i in inf_set:
            r = next(_it, None)
            results.append(r if isinstance(r, dict) else _none_result(include_valence))
        else:
            results.append(_none_result(include_valence))


def _code_anthropic(model, api_key, responses, taxonomy, batch_size,
                    progress_callback, multi_theme, include_valence, include_emotion, on_batch=None):
    client = _anthropic_client(api_key)
    system_text = _build_system(taxonomy, multi_theme, include_valence, include_emotion)
    results = []
    total_batches = (len(responses) + batch_size - 1) // batch_size
    for batch_num, start in enumerate(range(0, len(responses), batch_size)):
        batch = responses[start:start + batch_size]
        before = len(results)

        def _call(numbered, n):
            msg = client.messages.create(
                model=model, max_tokens=8192,
                system=[{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": f"Code these {n} responses:\n\n{numbered}"}],
            )
            return msg.content[0].text

        _code_one_batch(batch, _call, results, include_valence)
        if on_batch:
            on_batch(results[before:])
        if progress_callback:
            progress_callback((batch_num + 1) / total_batches)
    return results


def _split_anthropic(model, api_key, theme, new_name_1, new_name_2, sample_responses):
    client = _anthropic_client(api_key)
    prompt = _split_prompt(theme, new_name_1, new_name_2, sample_responses)
    msg = client.messages.create(model=model, max_tokens=4096,
                                  messages=[{"role": "user", "content": prompt}])
    return [_normalize_theme(t) for t in json.loads(msg.content[0].text)["themes"]]


# ── OpenAI / Gemini path (OpenAI-compatible API) ───────────────────────────────

def _openai_client(provider, api_key):
    from openai import OpenAI
    if provider == "Google Gemini":
        return OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    return OpenAI(api_key=api_key)


def _is_per_day_quota(e) -> bool:
    """True if a 429 is a per-DAY quota violation (retrying won't help until reset).

    Scans both the structured error body and the message text, since Gemini's
    OpenAI-compatible endpoint sometimes only embeds the quota id in the message.
    """
    blob = ""
    try:
        body = getattr(e, "body", None)
        if body:
            blob += str(body)
    except Exception:
        pass
    try:
        blob += " " + str(e)
    except Exception:
        pass
    blob = blob.lower()
    return ("perday" in blob) or ("per day" in blob) or ("requests per day" in blob)


def _chat(client, model, system, user, max_tokens=2048, _retries=6, response_format=None):
    import time
    from openai import RateLimitError, InternalServerError
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if response_format is not None:
        kwargs["response_format"] = response_format
    for attempt in range(_retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except (RateLimitError, InternalServerError) as e:
            # Daily quota exhausted — retrying wastes minutes for nothing; fail now.
            if isinstance(e, RateLimitError) and _is_per_day_quota(e):
                raise
            if attempt == _retries - 1:
                raise
            # Try to extract retry-after delay from error details
            wait = 60  # default fallback
            try:
                details = e.body.get("error", {}).get("details", [])
                for d in details:
                    if d.get("@type", "").endswith("RetryInfo"):
                        delay_str = d.get("retryDelay", "60s")
                        wait = int("".join(filter(str.isdigit, delay_str))) + 2
                        break
            except Exception:
                pass
            time.sleep(wait)


def _suggest_openai_compat(provider, model, api_key, responses, user_seeds,
                            max_responses, min_themes, max_themes):
    import random
    client = _openai_client(provider, api_key)
    pool = list(responses)
    random.shuffle(pool)
    sample = pool[:min(max_responses, 1200)]
    numbered = "\n\n".join(f"{i+1}. {r}" for i, r in enumerate(sample))
    user_msg = f"Here are {len(sample)} survey responses:\n\n{numbered}"
    if user_seeds.strip():
        user_msg += f"\n\nThe researcher suggests these possible themes: {user_seeds}"
    user_msg += "\n\nSuggest a coding taxonomy."
    system = _SUGGEST_SYSTEM_TEMPLATE.format(min_themes=min_themes, max_themes=max_themes)
    raw = _chat(client, model, system, user_msg, max_tokens=8192)
    return [_normalize_theme(t) for t in _robust_json_loads(raw)["themes"]]


def _code_openai_compat(provider, model, api_key, responses, taxonomy, batch_size,
                        progress_callback, multi_theme, include_valence, include_emotion, on_batch=None):
    client = _openai_client(provider, api_key)
    system_text = _build_system(taxonomy, multi_theme, include_valence, include_emotion)
    # OpenAI supports JSON mode — forces syntactically valid JSON, eliminating parse errors.
    use_json_mode = provider == "OpenAI"
    rf = {"type": "json_object"} if use_json_mode else None
    results = []
    total_batches = (len(responses) + batch_size - 1) // batch_size
    for batch_num, start in enumerate(range(0, len(responses), batch_size)):
        batch = responses[start:start + batch_size]
        before = len(results)

        def _call(numbered, n):
            user_msg = f"Code these {n} responses:\n\n{numbered}"
            if use_json_mode:
                user_msg += '\n\nReturn a JSON object of the form {"results": [ one object per response, in order ]}.'
            return _strip_fences(_chat(client, model, system_text, user_msg,
                                       max_tokens=8192, response_format=rf))

        _code_one_batch(batch, _call, results, include_valence)
        if on_batch:
            on_batch(results[before:])
        if progress_callback:
            progress_callback((batch_num + 1) / total_batches)
    return results


def _split_openai_compat(provider, model, api_key, theme, new_name_1, new_name_2, sample_responses):
    client = _openai_client(provider, api_key)
    raw = _chat(client, model, "", _split_prompt(theme, new_name_1, new_name_2, sample_responses))
    return [_normalize_theme(t) for t in _robust_json_loads(raw)["themes"]]


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _split_prompt(theme, new_name_1, new_name_2, sample_responses):
    samples_text = "\n".join(f"- {r}" for r in sample_responses[:15])
    return (
        f'A researcher wants to split the theme "{theme["name"]}" into two more specific themes '
        f'named "{new_name_1}" and "{new_name_2}".\n\n'
        f'Original description: {theme["description"]}\n\n'
        f"Sample responses currently coded as this theme:\n{samples_text}\n\n"
        f'Create exactly these two themes with clear descriptions and 2–3 example excerpts each.\n'
        f'Return ONLY valid JSON — no prose, no markdown fences:\n'
        f'{{"themes": [{{"name": "...", "description": "...", "examples": ["...", "..."]}}, '
        f'{{"name": "...", "description": "...", "examples": ["...", "..."]}}]}}'
    )


def _robust_json_loads(text: str):
    """Parse JSON with fallbacks for common model output quirks."""
    import ast, re
    cleaned = _strip_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Remove trailing commas before ] or } (common model mistake)
    no_trailing = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        return json.loads(no_trailing)
    except json.JSONDecodeError:
        pass
    # Last resort: Python literal eval (handles single-quoted strings)
    return ast.literal_eval(cleaned)


# Maps variations Gemini sometimes uses back to the expected keys
_THEME_KEY_MAP = {
    "theme name": "name",
    "theme_name": "name",
    "title": "name",
    "theme": "name",
    "description": "description",
    "theme description": "description",
    "theme_description": "description",
    "examples": "examples",
    "example responses": "examples",
    "example_responses": "examples",
    "sample responses": "examples",
}

def _normalize_theme(t: dict) -> dict:
    """Remap any non-standard keys a model might use to the expected schema."""
    normalized = {}
    for k, v in t.items():
        normalized[_THEME_KEY_MAP.get(k.lower(), k.lower())] = v
    return normalized


def _parse_batch(raw: str, results: list) -> None:
    batch_raw = _robust_json_loads(raw)
    # JSON-mode responses wrap the array in an object, e.g. {"results": [...]}
    if isinstance(batch_raw, dict):
        _lists = [v for v in batch_raw.values() if isinstance(v, list)]
        batch_raw = _lists[0] if _lists else [batch_raw]
    for item in batch_raw:
        results.append({
            "themes":     item.get("themes", [item]) if isinstance(item, dict) else [item],
            "score":      item.get("score")           if isinstance(item, dict) else None,
            "label":      item.get("label")           if isinstance(item, dict) else None,
            "emotion":    item.get("emotion")         if isinstance(item, dict) else None,
            "confidence": item.get("confidence", 0.5) if isinstance(item, dict) else 0.5,
        })


# ── Public API ─────────────────────────────────────────────────────────────────

def preflight_check(provider, model, api_key):
    """Make one tiny request to verify the key/model work and aren't already capped.

    Single attempt (no retries) so it's fast. Raises the provider's exception on
    failure (bad key, no credits, rate/quota limit, unknown model); returns True on
    success. Costs ~1 token.
    """
    if provider == "Anthropic":
        client = _anthropic_client(api_key)
        client.messages.create(
            model=model, max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
    else:
        client = _openai_client(provider, api_key)
        client.chat.completions.create(
            model=model, max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
    return True


def suggest_themes(provider, model, api_key, responses, user_seeds="",
                   max_responses=500, min_themes=5, max_themes=8):
    if provider == "Anthropic":
        return _suggest_anthropic(model, api_key, responses, user_seeds,
                                   max_responses, min_themes, max_themes)
    return _suggest_openai_compat(provider, model, api_key, responses, user_seeds,
                                   max_responses, min_themes, max_themes)


def code_responses(provider, model, api_key, responses, taxonomy, batch_size=20,
                   progress_callback=None, multi_theme=False,
                   include_valence=False, include_emotion=False, on_batch=None):
    if provider == "Anthropic":
        return _code_anthropic(model, api_key, responses, taxonomy, batch_size,
                                progress_callback, multi_theme, include_valence, include_emotion,
                                on_batch=on_batch)
    return _code_openai_compat(provider, model, api_key, responses, taxonomy, batch_size,
                                progress_callback, multi_theme, include_valence, include_emotion,
                                on_batch=on_batch)


def split_theme(provider, model, api_key, theme, new_name_1, new_name_2, sample_responses):
    if provider == "Anthropic":
        return _split_anthropic(model, api_key, theme, new_name_1, new_name_2, sample_responses)
    return _split_openai_compat(provider, model, api_key, theme, new_name_1, new_name_2, sample_responses)


# ── Representative quote selection ─────────────────────────────────────────────

_QUOTE_SYSTEM = (
    "You help a researcher pull representative VERBATIM quotes from open-ended survey "
    "responses that were already coded to a theme. Your job is SELECTION and EXTRACTION ONLY. "
    "You must NEVER paraphrase, rewrite, fix grammar, translate, or invent text. Every quote "
    "you return must be copied EXACTLY, word-for-word, from one of the provided responses. You "
    "may shorten by quoting a contiguous span (a sentence or two), or join two contiguous spans "
    "with an ellipsis (…), but you may not alter any words within a span."
)


def _quote_user_prompt(theme, candidates, max_representative, allow_nuance):
    listing = "\n\n".join(f"[{c['id']}] {c['text']}" for c in candidates)
    instr = (
        f'Theme: "{theme["name"]}"\n'
        f'Definition: {theme.get("description", "")}\n\n'
        f"Responses coded to this theme:\n\n{listing}\n\n"
        f"Select up to {max_representative} response(s) that best REPRESENT this theme. "
        "Prioritise quotes with substance — ones that name a specific pain point or request a "
        "specific improvement, and that reflect the prevailing view within the theme. "
        "Avoid generic, boilerplate, or purely procedural/legalistic text. "
    )
    if allow_nuance:
        instr += (
            'Additionally, you may select up to 1 response that captures a meaningful but '
            'LESS COMMON angle within this theme — mark its role "nuance". '
        )
    instr += (
        "Only include a quote if it is genuinely substantive; it is better to return fewer "
        "(even zero) than to pad with weak quotes. For each selection, extract the single most "
        "representative verbatim excerpt (ideally 25 words or fewer, never more than ~40), "
        "copied exactly from that response.\n\n"
        "Return ONLY valid JSON — no prose, no markdown fences:\n"
        '{"quotes": [{"id": <number>, "role": "representative" | "nuance", '
        '"quote": "<verbatim excerpt>", "reason": "<why, 12 words max>"}]}'
    )
    return instr


def select_quotes(provider, model, api_key, theme, candidates,
                  max_representative=3, allow_nuance=True):
    """Pick representative + nuance verbatim quotes for a single theme.

    `candidates` comes from quotes.build_candidates(). Excerpts are validated as
    verbatim downstream; any that aren't are replaced with a real source sentence.
    """
    from quotes import finalize_picks
    if not candidates:
        return []
    user = _quote_user_prompt(theme, candidates, max_representative, allow_nuance)
    if provider == "Anthropic":
        client = _anthropic_client(api_key)
        msg = client.messages.create(
            model=model, max_tokens=1500, system=_QUOTE_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        raw = msg.content[0].text
    else:
        client = _openai_client(provider, api_key)
        raw = _chat(client, model, _QUOTE_SYSTEM, user, max_tokens=1500)
    try:
        parsed = _robust_json_loads(raw)
        picks = parsed.get("quotes", []) if isinstance(parsed, dict) else parsed
    except Exception:
        picks = []
    return finalize_picks(picks, candidates, max_representative, allow_nuance)
