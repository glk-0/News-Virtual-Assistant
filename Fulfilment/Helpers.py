#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
import requests
import re
import dateparser
import configparser

def load_Newsapi_keys():
    config = configparser.ConfigParser()
    config.read('.cfg')
    keys = config['newsapi']['api_keys']
    return [k.strip() for k in keys.split(',')]

def _load_gnews_keys():
    config = configparser.ConfigParser()
    config.read('.cfg')
    keys = config['gnews']['api_keys']
    return [k.strip() for k in keys.split(',')]


def _unwrap(slot_val, default=""):
    """Unwrap a list slot value to a scalar, return default if empty."""
    if isinstance(slot_val, list):
        return slot_val[0].strip() if slot_val else default
    if isinstance(slot_val, str):
        return slot_val.strip()
    return default if slot_val is None else slot_val


def _parse_season(raw) -> int | None:
    """
    Convert a season slot value to an integer year.

    Handles:
        "2024"        → 2024
        "last year"   → current_year - 1
        "this year"   → current_year
        "last season" → current_year - 1
        2024          → 2024
        None / ""     → None
    """
    if raw is None:
        return None

    val = _unwrap(raw).lower().strip()

    if not val:
        return None

    # Numeric string
    if val.isdigit():
        return int(val)

    # Natural language
    now = datetime.datetime.now().year
    if val in ("last year", "last season", "previous year", "previous season"):
        return now - 1
    if val in ("this year", "this season", "current year", "current season"):
        return now
    if val in ("next year", "next season"):
        return now + 1

    # e.g. "2024-2025" → take first year
    if "-" in val:
        part = val.split("-")[0].strip()
        if part.isdigit():
            return int(part)

    return None


def _parse_date(raw, format=0) -> str:
    """
    Convert a date slot to YYYYMMDD string for ESPN API.

    Handles:
        "20250101"          → "20250101"
        "2025-01-01"        → "20250101"
        "today"             → today as YYYYMMDD
        "yesterday"         → yesterday as YYYYMMDD
        None / ""           → ""
    """
    if raw is None:
        return ""

    val = _unwrap(raw).lower().strip()

    if not val:
        return ""

    date = dateparser.parse(val).strftime("%Y-%m-%d")

    # Strip hyphens: "2025-01-01" → "20250101"
    if format == 0:
      cleaned = date.replace("-", "")
    else:
      cleaned = date
    if cleaned.isdigit() and len(cleaned) == 8:
        return cleaned

    return ""

def _parse_bool_slot(val, default=True) -> bool:
    """Convert slot values like 'on', 'off', 'true', 'false' to bool."""
    if val is None:
        return default
    s = str(val).lower().strip()
    return s in ("on", "true", "yes", "1", "enable", "enabled")

def _parse_duration_to_seconds(raw) -> int:
    """Robust parser for natural language and numeric durations."""
    if not raw:
        return 0

    val = str(raw).lower().strip()
    total = 0

    # 1. Strict Colon Format (HH:MM:SS or MM:SS)
    colon_match = re.fullmatch(r'(?:(\d+):)?(\d+):(\d{2})', val)
    if colon_match:
        h, m, s = colon_match.groups()
        total += int(h) * 3600 if h else 0
        total += int(m) * 60
        total += int(s)
        return total

    # 2. Strict Word Boundaries for Natural Language
    # \b ensures we match the exact word boundaries, avoiding cross-contamination
    h_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:hour|hr|h)s?\b', val)
    m_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:minute|min|m)s?\b', val)
    s_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:second|sec|s)s?\b', val)

    if h_match or m_match or s_match:
        total += int(float(h_match.group(1)) * 3600) if h_match else 0
        total += int(float(m_match.group(1)) * 60) if m_match else 0
        total += int(float(s_match.group(1))) if s_match else 0
        return total

    # 3. Bare Number Fallback (Defaulting to seconds for safety)
    digits = re.sub(r'\D', '', val)
    if digits:
        return int(digits)

    return 0


def _seconds_to_hms(seconds: int) -> str:
    """Format seconds as HH:MM:SS (omits hours if zero)."""
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def _gnews_get(endpoint: str, params: dict) -> dict:
    params["lang"] = params.get("lang", "en")
    api_keys = _load_gnews_keys()

    for key in api_keys:
        params["apikey"] = key
        r = requests.get(f"https://gnews.io/api/v4/{endpoint}", params=params)
        if r.status_code != 429:
            r.raise_for_status()
            return r.json()

    raise Exception(f"All {len(api_keys)} GNews API keys are rate limited (429)")