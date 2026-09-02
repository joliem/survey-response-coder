"""Claude API calls for theme suggestion and response coding."""

import json
import anthropic

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"

EMOTIONS = [
    "Frustrated", "Confused", "Angry", "Disappointed",
    "Worried", "Satisfied", "Relieved", "Neutral",
]

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


def suggest_themes(responses: list[str], user_seeds: str = "", max_responses: int = 500,
                   min_themes: int = 5, max_themes: int = 8) -> list[dict]:
    """
    Return a list of theme dicts {name, description, examples} from a random sample.
    Uses up to max_responses responses; capped at 1200 to stay within context limits.
    Responses are randomly shuffled before sampling so order in the file doesn't bias themes.
    """
    import random
    pool = list(responses)
    random.shuffle(pool)
    sample = pool[:min(max_responses, 1200)]
    numbered = "\n\n".join(f"{i+1}. {r}" for i, r in enumerate(sample))

    user_content = f"Here are {len(sample)} survey responses:\n\n{numbered}"
    if user_seeds.strip():
        user_content += f"\n\nThe researcher suggests these possible themes to consider: {user_seeds}"
    user_content += "\n\nSuggest a coding taxonomy."

    system = _SUGGEST_SYSTEM_TEMPLATE.format(min_themes=min_themes, max_themes=max_themes)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    )
    result = json.loads(msg.content[0].text)
    return result["themes"]


def _build_system(
    taxonomy: list[dict],
    multi_theme: bool,
    include_valence: bool,
    include_emotion: bool,
) -> str:
    taxonomy_lines = "\n".join(
        f'- "{t["name"]}": {t["description"]}' for t in taxonomy
    )

    theme_rule = (
        "Assign 1–3 themes in order of relevance (most relevant first)."
        if multi_theme
        else "Assign EXACTLY ONE theme — the single best fit."
    )

    sentiment_lines = []
    output_keys = ['"themes"']
    example_obj: dict = {"themes": ["Theme A"]}

    if include_valence:
        sentiment_lines.append(
            "- Rate sentiment on a 1–5 scale: "
            "1 = Very Negative, 2 = Negative, 3 = Neutral, 4 = Positive, 5 = Very Positive. "
            'Include "score" (integer 1–5) and "label" (the text label). '
            "Default to 3 / Neutral when tone is unclear."
        )
        output_keys += ['"score"', '"label"']
        example_obj.update({"score": 2, "label": "Negative"})

    if include_emotion:
        emotions_str = ", ".join(EMOTIONS)
        sentiment_lines.append(
            f"- Identify the primary emotion expressed. Choose from: {emotions_str}. "
            'Include "emotion". Use an empty string "" when no clear emotion is expressed.'
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
    import json as _json
    example = f'[{_json.dumps(example_obj)}]'

    return f"""\
You are an expert qualitative researcher coding open-ended survey responses.

CODING TAXONOMY:
{taxonomy_lines}
- "Other": use only when absolutely no theme fits

Instructions:
- {theme_rule}
- Use theme names verbatim as listed above.
{sentiment_block}\
{confidence_line}\
- Return ONLY a JSON array of objects, one per response, in input order.
- Each object must have exactly these keys: {keys_str}
- No explanation, no markdown fences — just the JSON array.
- Example: {example}
"""


def split_theme(
    theme: dict,
    new_name_1: str,
    new_name_2: str,
    sample_responses: list[str],
) -> list[dict]:
    """
    Ask Claude to split one theme into two, generating descriptions and examples.
    Returns a list of exactly two theme dicts.
    """
    samples_text = "\n".join(f"- {r}" for r in sample_responses[:15])
    prompt = (
        f'A researcher wants to split the theme "{theme["name"]}" into two more specific themes '
        f'named "{new_name_1}" and "{new_name_2}".\n\n'
        f'Original description: {theme["description"]}\n\n'
        f"Sample responses currently coded as this theme:\n{samples_text}\n\n"
        f'Create exactly these two themes with clear descriptions and 2–3 example excerpts each.\n'
        f'Return ONLY valid JSON — no prose, no markdown fences:\n'
        f'{{"themes": [{{"name": "...", "description": "...", "examples": ["...", "..."]}}, '
        f'{{"name": "...", "description": "...", "examples": ["...", "..."]}}]}}'
    )
    msg = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    result = json.loads(msg.content[0].text)
    return result["themes"]


def code_responses(
    responses: list[str],
    taxonomy: list[dict],
    batch_size: int = 25,
    progress_callback=None,
    multi_theme: bool = False,
    include_valence: bool = False,
    include_emotion: bool = False,
) -> list[dict]:
    """
    Code responses using Claude.
    Always returns list[dict] with keys:
      themes (list[str]), score (int|None), label (str|None), emotion (str|None)
    """
    system_text = _build_system(taxonomy, multi_theme, include_valence, include_emotion)
    results: list[dict] = []
    total_batches = (len(responses) + batch_size - 1) // batch_size

    for batch_num, start in enumerate(range(0, len(responses), batch_size)):
        batch = responses[start : start + batch_size]
        numbered = "\n\n".join(f"{i+1}. {r}" for i, r in enumerate(batch))
        user_content = f"Code these {len(batch)} responses:\n\n{numbered}"

        msg = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=[{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_content}],
        )

        batch_raw = json.loads(msg.content[0].text)
        for item in batch_raw:
            results.append({
                "themes": item.get("themes", [item]) if isinstance(item, dict) else [item],
                "score": item.get("score") if isinstance(item, dict) else None,
                "label": item.get("label") if isinstance(item, dict) else None,
                "emotion": item.get("emotion") if isinstance(item, dict) else None,
                "confidence": item.get("confidence", 0.5) if isinstance(item, dict) else 0.5,
            })

        if progress_callback:
            progress_callback((batch_num + 1) / total_batches)

    return results
