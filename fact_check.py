"""
Google Fact Check Tools API integration.

Aggregates real published verdicts from independent fact-checking
organizations (PolitiFact, Snopes, FactCheck.org, AFP Fact Check,
Reuters Fact Check, Full Fact, and others) via Google's ClaimReview index.

Setup (free, no billing required for this API's quota):
  1. Go to console.cloud.google.com
  2. Create or select a project
  3. APIs & Services -> Library -> search "Fact Check Tools API" -> Enable
  4. APIs & Services -> Credentials -> Create Credentials -> API key
  5. Set it as an environment variable: GOOGLE_FACTCHECK_API_KEY
"""
import os
import requests

API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

FALSE_KEYWORDS = ("false", "fake", "fabricat", "pants on fire", "incorrect", "hoax", "misleading")
TRUE_KEYWORDS = ("true", "correct", "accurate", "verified")


def search_fact_checks(query, max_results=5):
    """Queries the Fact Check Tools API for claims matching `query`.

    Returns a dict: {"query": str, "claims": [...]} on success,
    or {"error": str} on failure (missing key, network issue, bad response).
    """
    api_key = os.environ.get("GOOGLE_FACTCHECK_API_KEY")
    if not api_key:
        return {"error": "Missing GOOGLE_FACTCHECK_API_KEY environment variable. "
                          "See fact_check.py docstring for setup steps."}

    query = (query or "").strip()
    if not query:
        return {"error": "Empty query."}

    # Fact-check search works best on a short claim, not a full paragraph.
    short_query = query.split(".")[0][:180] if "." in query else query[:180]

    try:
        resp = requests.get(API_URL, params={
            "query": short_query,
            "languageCode": "en",
            "pageSize": max_results,
            "key": api_key
        }, timeout=8)
    except requests.RequestException as e:
        return {"error": f"Network error contacting Fact Check API: {e}"}

    if resp.status_code != 200:
        return {"error": f"Fact Check API error {resp.status_code}: {resp.text[:200]}"}

    data = resp.json()
    claims = []
    for c in data.get("claims", []):
        reviews = []
        for r in c.get("claimReview", []):
            reviews.append({
                "publisher": (r.get("publisher") or {}).get("name", "Unknown publisher"),
                "url": r.get("url"),
                "title": r.get("title"),
                "rating": r.get("textualRating", "Unrated"),
                "review_date": r.get("reviewDate"),
            })
        claims.append({
            "text": c.get("text", short_query),
            "claimant": c.get("claimant"),
            "reviews": reviews,
        })
    return {"query": short_query, "claims": claims}


def strongest_rating(claims):
    """Scans matched claims for the most decisive published rating.

    Returns ("false", review_dict), ("true", review_dict), or None if
    nothing decisive was found (unreviewed, or only ambiguous ratings).
    """
    worst = None
    best = None
    for c in claims or []:
        for r in c.get("reviews", []):
            rating = (r.get("rating") or "").lower()
            if any(k in rating for k in FALSE_KEYWORDS):
                worst = r
            elif any(k in rating for k in TRUE_KEYWORDS) and not best:
                best = r
    if worst:
        return ("false", worst)
    if best:
        return ("true", best)
    return None
