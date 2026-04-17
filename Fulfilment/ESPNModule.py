#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""#Sports intent Fulfilment

JSONS
"""
import json
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
from typing import Optional
import re
from Fulfilment.Helpers import _unwrap, _parse_date, _parse_season

with open("Fulfilment/JSONS/TEAM_ALIASES.JSON") as f:
    TEAM_ALIASES = json.load(f)

with open("Fulfilment/JSONS/TEAM_LEAGUE.JSON") as f:
    TEAM_LEAGUE = json.load(f)

"""Functions"""

LEAGUE_CONFIG = {
    # --- American sports ---
    "nfl":  {"sport": "football",   "league": "nfl",  "type": "american"},
    "nba":  {"sport": "basketball", "league": "nba",  "type": "american"},
    "nhl":  {"sport": "hockey",     "league": "nhl",  "type": "american"},
    # --- Soccer (Big 5 + extras) ---
    "epl":          {"sport": "soccer", "league": "eng.1", "type": "soccer"},
    "eng.1":        {"sport": "soccer", "league": "eng.1", "type": "soccer"},
    "laliga":       {"sport": "soccer", "league": "esp.1", "type": "soccer"},
    "esp.1":        {"sport": "soccer", "league": "esp.1", "type": "soccer"},
    "bundesliga":   {"sport": "soccer", "league": "ger.1", "type": "soccer"},
    "ger.1":        {"sport": "soccer", "league": "ger.1", "type": "soccer"},
    "seriea":       {"sport": "soccer", "league": "ita.1", "type": "soccer"},
    "ita.1":        {"sport": "soccer", "league": "ita.1", "type": "soccer"},
    "ligue1":       {"sport": "soccer", "league": "fra.1", "type": "soccer"},
    "fra.1":        {"sport": "soccer", "league": "fra.1", "type": "soccer"},
}

BASE_SITE   = "https://site.api.espn.com/apis/site/v2/sports"
BASE_CORE   = "https://sports.core.api.espn.com/v2/sports"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_league(league_key: str) -> dict:
    key = league_key.lower().replace(" ", "").replace("-", "")
    cfg = LEAGUE_CONFIG.get(key)
    if not cfg:
        try:
            print(key)
            key = TEAM_LEAGUE.get(key)
            cfg = LEAGUE_CONFIG.get(key.lower())
        except Exception:
            raise ValueError(
                f"Unknown league '{league_key}'. "
                f"Valid options: {', '.join(LEAGUE_CONFIG.keys())}"
            )
    return cfg


def _get(url: str, params: Optional[dict] = None) -> dict:
    """
    HTTP GET via urllib directly with ProxyHandler({}) to bypass any
    environment proxy that injects urllib3.Timeout objects into requests.
    """
    try:
        if params:
            url = url + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers=HEADERS)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.reason}"}
    except urllib.error.URLError as e:
        return {"error": f"Connection failed: {e.reason}"}
    except TimeoutError:
        return {"error": "Request timed out"}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON response: {e}"}
    except Exception as e:
        return {"error": str(e)}


def _find_team_in_event(event: dict, team_name: str) -> bool:
    """Return True if team_name (case-insensitive substring) appears in event."""
    name_lower = team_name.lower()
    for comp in event.get("competitions", []):
        for c in comp.get("competitors", []):
            t = c.get("team", {})
            if (name_lower in t.get("displayName", "").lower()
                    or name_lower in t.get("shortDisplayName", "").lower()
                    or name_lower in t.get("abbreviation", "").lower()
                    or name_lower in t.get("name", "").lower()):
                return True
    return False


def _parse_score_event(event: dict) -> dict:
    """Normalise a scoreboard event into a clean score dict."""
    comp = event["competitions"][0]
    status = comp["status"]
    competitors = comp["competitors"]

    home = next((c for c in competitors if c["homeAway"] == "home"), competitors[0])
    away = next((c for c in competitors if c["homeAway"] == "away"), competitors[1])

    result = {
        "event_id":   event.get("id"),
        "name":       event.get("name"),
        "date":       event.get("date"),
        "status":     status["type"].get("description", "Unknown"),
        "status_detail": status.get("displayClock", status["type"].get("shortDetail", "")),
        "home_team":  home["team"].get("displayName"),
        "home_abbrev": home["team"].get("abbreviation"),
        "home_score": home.get("score", "0"),
        "away_team":  away["team"].get("displayName"),
        "away_abbrev": away["team"].get("abbreviation"),
        "away_score": away.get("score", "0"),
        "completed":  status["type"].get("completed", False),
    }

    # Period / half detail for soccer
    if "period" in status:
        result["period"] = status["period"]

    # Line scores (quarters / halves)
    home_line = home.get("linescores", [])
    away_line = away.get("linescores", [])
    if home_line:
        result["home_linescores"] = [ls.get("value", 0) for ls in home_line]
        result["away_linescores"] = [ls.get("value", 0) for ls in away_line]

    # Winner
    home_winner = home.get("winner")
    away_winner = away.get("winner")
    if home_winner:
        result["winner"] = home["team"].get("displayName")
    elif away_winner:
        result["winner"] = away["team"].get("displayName")

    return result

def _resolve_team_league(team_raw: str) -> tuple[str, str]:
    """
    Given a raw team string (alias or full name), return
    (canonical_name, league_key_for_resolve_league).

    Returns ("", "") if not found.
    """
    if not team_raw:
        return "", ""

    canonical = TEAM_ALIASES.get(team_raw.lower(), team_raw)
    league_str = TEAM_LEAGUE.get(canonical, "")

    LEAGUE_KEY_MAP = {
        "NFL":        "nfl",
        "NBA":        "nba",
        "NHL":        "nhl",
        "EPL":        "epl",
        "LA_LIGA":    "laliga",
        "BUNDESLIGA": "bundesliga",
        "SERIE_A":    "seriea",
        "LIGUE_1":    "ligue1",
    }
    league_key = LEAGUE_KEY_MAP.get(league_str.upper(), league_str.lower())
    return canonical, league_key


def _normalize_league_key(raw: str) -> str:
    """Map TEAM_LEAGUE values or user strings to _resolve_league keys."""
    LEAGUE_KEY_MAP = {
        "NFL":        "nfl",
        "NBA":        "nba",
        "NHL":        "nhl",
        "EPL":        "epl",
        "LA_LIGA":    "laliga",
        "BUNDESLIGA": "bundesliga",
        "SERIE_A":    "seriea",
        "LIGUE_1":    "ligue1",
    }
    return LEAGUE_KEY_MAP.get(raw.upper(), raw.lower())


# ---------------------------------------------------------------------------
# 1. GetGameScore
# ---------------------------------------------------------------------------

def GetGameScore(slots: dict) -> dict:
    """
    Get live or recent game score(s).

    Slot keys:
        TEAM     (str, optional) : filter to any game involving this team
        TEAM2    (str, optional) : head-to-head — both teams must appear
        LEAGUE   (str, optional) : override auto-detected league
        DATE     (str, optional) : YYYYMMDD — defaults to today/current

    Behaviour:
        - league is auto-detected from TEAM via TEAM_ALIASES + TEAM_LEAGUE
        - No team slots  → LEAGUE is required explicitly
        - TEAM only      → all games involving that team
        - TEAM + TEAM2   → the specific matchup between the two teams
    """
    league_key   = _unwrap(slots.get("LEAGUE", ""))
    team_raw     = _unwrap(slots.get("TEAM", ""))
    team2_raw    = _unwrap(slots.get("TEAM2", ""))
    date_str     = _parse_date(slots.get("DATE", datetime.datetime.utcnow().strftime("%Y-%m-%d")))

    # Auto-detect league from team name
    if not league_key and team_raw:
        canonical, league_key = _resolve_team_league(team_raw)
        team_raw = canonical  # use canonical name for filtering
        if not league_key:
            return {"error": f"Could not determine league for team '{team_raw}'. "
                             "Please provide a LEAGUE slot explicitly."}
    elif team_raw:
        # League was explicit — still resolve alias for cleaner filtering
        canonical, _ = _resolve_team_league(team_raw)
        if canonical:
            team_raw = canonical

    if team2_raw:
        canonical2, _ = _resolve_team_league(team2_raw)
        if canonical2:
            team2_raw = canonical2

    if not league_key:
        return {"error": "Please provide at least a TEAM or LEAGUE slot."}

    league_key = _normalize_league_key(league_key)

    try:
        cfg = _resolve_league(league_key)
    except ValueError as e:
        return {"error": str(e)}

    sport  = cfg["sport"]
    league = cfg["league"]
    url    = f"{BASE_SITE}/{sport}/{league}/scoreboard"

    params = {}
    if date_str != "":
        params["dates"] = date_str

    data = _get(url, params)
    if "error" in data:
        return data

    events = data.get("events", [])
    games  = []
    for ev in events:
        if team_raw and team2_raw:
            if not (_find_team_in_event(ev, team_raw) and
                    _find_team_in_event(ev, team2_raw)):
                continue
        elif team_raw:
            if not _find_team_in_event(ev, team_raw):
                continue
        games.append(_parse_score_event(ev))

    return {
        "league": league,
        "date":   date_str,
        "team":   team_raw or None,
        "team2":  team2_raw or None,
        "games":  games,
        "count":  len(games),
    }


# ---------------------------------------------------------------------------
# 2. GetTeamStanding
# ---------------------------------------------------------------------------
def GetTeamStanding(slots: dict) -> dict:
    """
    Get league standings table.

    Slot keys:
        TEAM     (str, optional) : auto-detects league; filters to that team row
        LEAGUE   (str, optional) : explicit league override
    """
    league_key  = _unwrap(slots.get("LEAGUE", ""))
    team_raw    = _unwrap(slots.get("TEAM", ""))

    # Auto-detect league from team
    if not league_key and team_raw:
        canonical, league_key = _resolve_team_league(team_raw)
        team_raw = canonical
        if not league_key:
            return {"error": f"Could not determine league for team '{team_raw}'. "
                             "Please provide a LEAGUE slot explicitly."}
    elif team_raw:
        canonical, _ = _resolve_team_league(team_raw)
        if canonical:
            team_raw = canonical

    if not league_key:
        return {"error": "Please provide at least a TEAM or LEAGUE slot."}

    league_key = _normalize_league_key(league_key)

    try:
        cfg = _resolve_league(league_key)
    except ValueError as e:
        return {"error": str(e)}
    season = datetime.datetime.now().year
    sport  = cfg["sport"]
    league = cfg["league"]

    if team_raw:
        teams_url  = f"{BASE_SITE}/{sport}/{league}/teams"
        teams_data = _get(teams_url)
        if "error" in teams_data:
            return teams_data

        team_id    = None
        name_lower = team_raw.lower()
        for t_entry in teams_data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
            t = t_entry.get("team", {})
            if (name_lower in t.get("displayName", "").lower()
                    or name_lower in t.get("abbreviation", "").lower()
                    or name_lower in t.get("shortDisplayName", "").lower()):
                team_id = t["id"]
                break

    #hockey/leagues/nhl/seasons/2026/types/2/groups/9/standings/0?lang=en&region=us example
    # basketball url types/2/groups/7/standings/0?lang=en&region=us

    urls = {
        "nba": f"http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons/{season}/types/2/groups/7/standings/0?lang=en&region=us",
        "nhl": f"http://sports.core.api.espn.com/v2/sports/hockey/leagues/nhl/seasons/{season}/types/2/groups/9/standings/0?lang=en&region=us",
        "nfl": f"http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}/types/2/groups/9/standings/0?lang=en&region=us",
        "epl": f"http://sports.core.api.espn.com/v2/sports/soccer/leagues/eng.1/seasons/{season}/types/1/groups/1/standings/0?lang=en&region=us",
        "laliga": f"http://sports.core.api.espn.com/v2/sports/soccer/leagues/esp.1/seasons/{season}/types/1/groups/1/standings/0?lang=en&region=us",
        "bundesliga": f"http://sports.core.api.espn.com/v2/sports/soccer/leagues/ger.1/seasons/{season}/types/1/groups/1/standings/0?lang=en&region=us",
        "seriea": f"http://sports.core.api.espn.com/v2/sports/soccer/leagues/ita.1/seasons/{season}/types/1/groups/1/standings/0?lang=en&region=us",
        "ligue1": f"http://sports.core.api.espn.com/v2/sports/soccer/leagues/fra.1/seasons/{season}/types/1/groups/1/standings/0?lang=en&region=us",
    }
    #add season format to code
    url = urls.get(league)

    params = {}
    data = _get(url, params)
    if "error" in data:
        return data

    season_label = "current"
    team_position = "Unknown"
    standings = data.get("standings")
    for position, rank in enumerate(standings, start=1):
      link = rank["team"]["$ref"]
      ref_teamId = re.search(r'(?:teams/)(\d*)(?:\?)', link).group(1)
      if ref_teamId == team_id:
        team_position = position
        Overall_record = rank["records"][0]["displayValue"]

    # Step 2 — season stats via core API
    season_label = str(season) if season else "current"
    season_year  = season if season else datetime.datetime.now().year
    stats_out    = {}

    stats_url  = (f"{BASE_CORE}/{sport}/leagues/{league}"
                  f"/seasons/{season_year}/types/2/teams/{team_id}/statistics")
    stats_data = _get(stats_url)

    if "error" not in stats_data:
        for category in stats_data.get("splits", {}).get("categories", []):
            cat_name = category.get("displayName", category.get("name", ""))
            for stat in category.get("stats", []):
                key = f"{cat_name}.{stat['name']}"
                stats_out[key] = stat.get("displayValue", stat.get("value"))

    return {
        "league": league,
        "season": season_label,
        "position": team_position,
        "Overall_Record": Overall_record,
        "League_size": len(standings),
        "Stats": stats_out
    }


# ---------------------------------------------------------------------------
# 3. GetLeagueSchedule
# ---------------------------------------------------------------------------

def GetLeagueSchedule(slots: dict) -> dict:
    """
    Get upcoming and/or recent schedule for a league or specific team.

    Slot keys:
        (str, required)  : e.g. "nfl", "nba", "nhl", "epl"
        TEAM     (str, optional)  : triggers team-schedule endpoint
        DATE     (str, optional)  : YYYYMMDD or date range "YYYYMMDD-YYYYMMDD"
        WEEK     (int, optional)  : NFL week
        SEASON   (int, optional)  : season year or natural language
        LIMIT    (int, optional)  : max results (default 20)
    """
    league_key  = _unwrap(slots.get("LEAGUE", ""))
    team_raw    = _unwrap(slots.get("TEAM", ""))
    date_str    = _parse_date(slots.get("DATE"))
    week        = _unwrap(slots.get("WEEK")) or None
    season      = _parse_season(slots.get("SEASON"))
    limit_raw   = slots.get("LIMIT", 20)
    limit       = int(_unwrap(limit_raw) if isinstance(limit_raw, list) else limit_raw)

    # Auto-detect league from team
    if not league_key and team_raw:
        canonical, league_key = _resolve_team_league(team_raw)
        team_raw = canonical
        if not league_key:
            return {"error": f"Could not determine league for team '{team_raw}'. "
                             "Please provide a LEAGUE slot explicitly."}
    elif team_raw:
        canonical, _ = _resolve_team_league(team_raw)
        if canonical:
            team_raw = canonical

    if not league_key:
        return {"error": "Please provide at least a TEAM or LEAGUE slot."}

    league_key = _normalize_league_key(league_key)

    try:
        cfg = _resolve_league(league_key)
    except ValueError as e:
        return {"error": str(e)}

    sport  = cfg["sport"]
    league = cfg["league"]

    if team_raw:
        teams_url  = f"{BASE_SITE}/{sport}/{league}/teams"
        teams_data = _get(teams_url)
        if "error" in teams_data:
            return teams_data

        team_id    = None
        name_lower = team_raw.lower()
        for t_entry in teams_data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
            t = t_entry.get("team", {})
            if (name_lower in t.get("displayName", "").lower()
                    or name_lower in t.get("abbreviation", "").lower()
                    or name_lower in t.get("shortDisplayName", "").lower()):
                team_id = t["id"]
                break

        if not team_id:
            return {"error": f"Team '{team_raw}' not found in {league}"}

        url    = f"{BASE_SITE}/{sport}/{league}/teams/{team_id}/schedule"
        params = {}
        if season:
            params["season"] = season
        data   = _get(url, params)
        events = data.get("events", [])
    else:
        url    = f"{BASE_SITE}/{sport}/{league}/scoreboard"
        params = {"limit": limit}
        if date_str:
            params["dates"] = date_str
        if week:
            params["week"] = week
        data   = _get(url, params)
        events = data.get("events", [])

    if "error" in data:
        return data

    schedule = [_parse_score_event(ev) for ev in events[:limit]]

    return {
        "league":   league,
        "team":     team_raw or None,
        "schedule": schedule,
        "count":    len(schedule),
    }
