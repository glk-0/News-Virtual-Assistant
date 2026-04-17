#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import requests
from rapidfuzz import process, fuzz
import pycountry
import string
from Fulfilment.Helpers import _gnews_get, _unwrap, load_Newsapi_keys

def GetTopHeadlines( slots : dict) -> dict:
    """
    Get current top headlines.

    Slots: none required. Optional: REGION, COUNT
    """
    data = _gnews_get("top-headlines", {
        "max": 10,
    })

    return {
        "intent":   "GetTopHeadlines",
        "articles": data.get("articles", [0]),
    }

def GetTopicNews(slots: dict) -> dict:
    """
    Get news for a specific topic category.

    Slots: TOPIC (required), COUNT (optional)
    """
    topic = "".join(slots.get("TOPIC", ["general"])).lower()

    data = _gnews_get("search", {
        "q": topic
    })

    return {
        "intent":   "GetTopicNews",
        "topic":    topic,
        "articles": data.get("articles", [0]),
    }

def GetRegionNews(slots: dict) -> dict:
    """
    Get top news from a specific country or larger region.

    Slots: REGION (required), TOPIC (optional), COUNT (optional)
    """

    LARGE_REGIONS = {
        "north america": ["US", "CA", "MX"],
        "south america": ["BR", "AR", "CL", "CO", "PE", "UY", "PY", "BO", "EC", "VE"],
        "europe": ["FR", "DE", "IT", "ES", "GB", "NL", "BE", "SE", "NO", "DK", "FI", "PL", "AT", "CH"],
        "eastern europe": ["PL", "CZ", "SK", "HU", "RO", "BG", "UA", "BY", "MD", "RU"],
        "western europe": ["FR", "DE", "BE", "NL", "LU", "AT", "CH", "IE", "GB"],
        "northern europe": ["SE", "NO", "DK", "FI", "IS", "EE", "LV", "LT"],
        "southern europe": ["IT", "ES", "PT", "GR", "HR", "SI", "MT", "CY"],
        "scandinavia": ["SE", "NO", "DK", "FI", "IS"],
        "eastern asia": ["CN", "JP", "KR", "MN", "TW"],
        "southeast asia": ["ID", "MY", "SG", "TH", "PH", "VN", "KH", "LA", "MM", "BN", "TL"],
        "middle east": ["SA", "AE", "IL", "IR", "IQ", "JO", "KW", "QA", "OM", "TR"],
        "latin america": ["MX", "BR", "AR", "CL", "CO", "PE", "UY", "PY", "BO", "EC", "VE"],
    }

    # Unwrap list slots safely
    region_raw = slots.get("REGION", ["ca"])
    region = (region_raw[0] if isinstance(region_raw, list) else region_raw).lower()

    count_raw = slots.get("COUNT", [10])
    max_results = int(count_raw[0] if isinstance(count_raw, list) else count_raw)

    from_date = slots.get("DATE", datetime.datetime.now())
    to_date   = slots.get("DATE", datetime.datetime.now())

    # --- 1. Try to match a large region using rapidfuzz ---
    best_match, score, _ = process.extractOne(
        region,
        LARGE_REGIONS.keys(),
        scorer=fuzz.ratio
    )

    if score >= 90:
        articles = []
        for country_code in LARGE_REGIONS[best_match]:
            data = _gnews_get("top-headlines", {
                "country": country_code,
                "max":     min(max_results, 3),
                "to":      to_date,
                "from":    from_date,
            })
            articles.extend(data.get("articles", []))  # fixed missing quote in key
    else:
        try:
            country_code = pycountry.countries.search_fuzzy(region)[0].alpha_2
        except LookupError:
            country_code = None

        if country_code:
            data = _gnews_get("top-headlines", {
                "country": country_code,
                "max":     min(max_results, 5),
                "to":      to_date,
                "from":    from_date,
            })
        else:
            data = _gnews_get("search", {
                "q":   region,
                "max": min(max_results, 5),
            })
        articles = data.get("articles", [])  # fixed missing quote in key

    return {
        "intent":   "GetRegionNews",
        "region":   region,
        "articles": articles,
    }

def GetPublisherHeadlines(slots: dict) -> dict:
    """
    Get headlines from a specific publisher/source domain.

    Slots: PUBLISHER (required, e.g. "BBC", "CNN", "bbc.com"), COUNT (optional)
    """
    publisher_raw = _unwrap(slots.get("SOURCE", []))
    count_raw     = slots.get("COUNT", [10])
    max_results   = int(count_raw[0] if isinstance(count_raw, list) else count_raw)

    publisher_map = {
        "bbc": "bbc.com", "bbc news": "bbc.com",
        "cnn": "cnn.com",
        "fox": "foxnews.com", "fox news": "foxnews.com",
        "nyt": "nytimes.com", "new york times": "nytimes.com",
        "washington post": "washingtonpost.com", "wapo": "washingtonpost.com",
        "reuters": "reuters.com",
        "ap": "apnews.com", "associated press": "apnews.com",
        "the guardian": "theguardian.com", "guardian": "theguardian.com",
        "espn": "espn.com",
        "sky sports": "skysports.com",
        "nbc": "nbcnews.com", "nbc news": "nbcnews.com",
        "abc": "abcnews.go.com", "abc news": "abcnews.go.com",
        "techcrunch": "techcrunch.com",
        "the verge": "theverge.com",
        "wired": "wired.com",
    }

    publisher_key = publisher_raw.lower().strip(string.punctuation)
    domain = process.extractOne(publisher_key, publisher_map)[0]
    if not domain.endswith((".com", ".co.uk", ".org", ".net", ".go.com")):
        domain = domain + ".com"

    url = "https://newsapi.org/v2/everything"
    
    api_keys = load_Newsapi_keys()
    
    for key in api_keys:
        params = {
            "apiKey": key,
            "domains": domain,
            "pageSize": 15
        }
        response = requests.get(url, params=params)
        if response.status_code != 429:
            return response

    response.raise_for_status()  # raises if request failed
    data = response.json()

    articles = data.get("articles", [])

    # Keep only articles whose URL contains the target domain
    filtered = [
        a for a in articles
        if domain in a.get("url", "") or domain in a.get("source", {}).get("url", "")
    ]

    return {
        "intent":    "GetPublisherHeadlines",
        "publisher": publisher_raw,
        "domain":    domain,
        "articles":  filtered[:max_results],
    }