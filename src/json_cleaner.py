"""
src/json_cleaner.py

Robust helpers to parse model outputs that are *intended* to be JSON but often
contain extra text, templates, or Python-style quotes.

Functions:
- clean_model_output(raw_text) -> dict : best-effort parse; always returns a dict.
- create_error_response(msg) -> dict : consistent error dict used by the pipeline.
"""

import json
import re
import ast
from typing import Any, Dict, Optional

# Find all {...} and [...] blocks non-greedily (Python-compatible)
_BRACE_ITER_RE = re.compile(r"\{[\s\S]*?\}", re.S)
_BRACKET_ITER_RE = re.compile(r"\[[\s\S]*?\]", re.S)


def _extract_json_candidate(text: str) -> str:
    """
    Extract the most-likely JSON substring from text by preferring the last {...}
    or [...] block. This helps when prompts include example JSON templates
    earlier in the text and the assistant's JSON reply appears at the end.
    """
    if not text:
        return text

    txt = text.strip()

    # If the entire thing already looks like JSON, return as-is
    if (txt.startswith("{") and txt.endswith("}")) or (txt.startswith("[") and txt.endswith("]")):
        return txt

    # 1) Find all {...} occurrences and prefer the last sensible one
    br_matches = list(_BRACE_ITER_RE.finditer(txt))
    if br_matches:
        # choose the last reasonably long candidate (avoid tiny braces)
        for m in reversed(br_matches):
            cand = m.group(0).strip()
            if len(cand) > 10:
                return cand
        # fallback to last match
        return br_matches[-1].group(0)

    # 2) If no braces, look for [...] and prefer the last
    sq_matches = list(_BRACKET_ITER_RE.finditer(txt))
    if sq_matches:
        for m in reversed(sq_matches):
            cand = m.group(0).strip()
            if len(cand) > 10:
                return cand
        return sq_matches[-1].group(0)

    # 3) Fallback: try slicing between first '{' and last '}' if present
    try:
        first = txt.index("{")
        last = txt.rindex("}")
        if last > first:
            return txt[first : last + 1]
    except ValueError:
        pass

    # 4) Nothing found — return original text
    return txt


def _normalize_quotes(text: str) -> str:
    """
    Heuristic conversion of Python-style single-quoted dicts/lists into JSON
    by converting certain single-quoted tokens into double-quoted ones.

    Conservative: if the text already appears JSON-like (contains '":'), return it.
    """
    if not text:
        return text

    # If it already looks like JSON with double-quoted keys, return unchanged
    if '"' in text and '":' in text:
        return text

    t = text

    # Replace single-quoted keys/values that are followed by separators with double quotes.
    # This is heuristic and intentionally conservative to avoid breaking valid content.
    t = re.sub(r"(?<=[:\s\[,]?)'([^']*?)'(?=[\s,\]\}])", r'"\1"', t)

    # If we still have no double quotes at all, fallback to replacing all single quotes
    if '"' not in t:
        t = t.replace("'", '"')

    return t


def create_error_response(message: str, raw: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a consistent error response dict for downstream code.
    """
    out: Dict[str, Any] = {
        "error": True,
        "error_message": message,
    }
    if raw is not None:
        out["raw_model_output"] = raw[:5000]  # store up to 5k chars for debugging
    return out


def clean_model_output(raw_text: Optional[str]) -> Dict[str, Any]:
    """
    Robustly parse the raw text returned by the model into a Python dict.

    Strategy (in order):
      1) Try json.loads on whole text
      2) Extract last {...} or [...] block and try json.loads
      3) Try ast.literal_eval on candidate (handles Python dicts using single quotes)
      4) Normalize quotes heuristically and json.loads
      5) If all fail, return an error dict containing a trimmed raw_model_output

    Always returns a dict. If parsing succeeds and yields a dict, that dict is
    returned with an explicit "error" key set to False if missing. If parsing
    succeeds but returns a list/other, it's returned as {"error": False, "parsed": <obj>}.
    """
    if raw_text is None:
        return create_error_response("Model returned no text", raw=None)

    text = raw_text.strip()

    # 1) Direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            parsed.setdefault("error", False)
            return parsed
        return {"error": False, "parsed": parsed}
    except Exception:
        pass

    # 2) Extract candidate JSON-like substring (prefer last occurrence)
    candidate = _extract_json_candidate(text)
    if candidate:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                parsed.setdefault("error", False)
                return parsed
            return {"error": False, "parsed": parsed}
        except Exception:
            pass

    # 3) Try ast.literal_eval (safe for Python-literal dicts using single quotes)
    try:
        parsed = ast.literal_eval(candidate if candidate else text)
        if isinstance(parsed, dict):
            parsed.setdefault("error", False)
            return parsed
        return {"error": False, "parsed": parsed}
    except Exception:
        pass

    # 4) Normalize quotes heuristically then try json.loads
    try:
        normalized = _normalize_quotes(candidate if candidate else text)
        parsed = json.loads(normalized)
        if isinstance(parsed, dict):
            parsed.setdefault("error", False)
            return parsed
        return {"error": False, "parsed": parsed}
    except Exception:
        pass

    # 5) Give up — return error with the raw text attached for debugging
    return create_error_response("Invalid JSON from model: could not parse output.", raw=text)
