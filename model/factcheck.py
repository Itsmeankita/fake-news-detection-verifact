"""
Optional integration with Google's Fact Check Tools API — searches for
existing professional fact-checks related to the article's key terms.

This is a genuinely free, public API (no billing required for the Fact
Check Tools endpoint), but it does need a Google Cloud API key with the
"Fact Check Tools API" enabled. If no key is configured, this module
degrades gracefully — the app just won't show the fact-check section,
rather than crashing.

To enable:
1. Go to https://console.cloud.google.com/, create a project (free).
2. Enable the "Fact Check Tools API" for that project.
3. Create an API key under APIs & Services > Credentials.
4. Set it as an environment variable before running the app:
     Windows:  set FACTCHECK_API_KEY=your_key_here
     Mac/Linux: export FACTCHECK_API_KEY=your_key_here
"""
import os
import re

import requests

API_KEY = os.environ.get("FACTCHECK_API_KEY", "")
ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


def extract_query(text, max_words=12):
    """Pulls a short, search-friendly query out of a longer article by
    taking the first sentence (or first N words), stripped of noise."""
    first_sentence = re.split(r"(?<=[.!?])\s", text.strip())[0]
    words = first_sentence.split()
    return " ".join(words[:max_words])


def get_related_factchecks(text, max_results=3):
    """Returns a list of {text, rating, publisher, url} dicts, or an empty
    list if the API key isn't configured or nothing relevant is found."""
    if not API_KEY:
        return []

    query = extract_query(text)
    if len(query) < 10:
        return []

    try:
        resp = requests.get(ENDPOINT, params={
            "key": API_KEY,
            "query": query,
            "languageCode": "en",
        }, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    results = []
    for claim in data.get("claims", [])[:max_results]:
        reviews = claim.get("claimReview", [])
        if not reviews:
            continue
        review = reviews[0]
        results.append({
            "text": claim.get("text", "")[:200],
            "rating": review.get("textualRating", "Unrated"),
            "publisher": review.get("publisher", {}).get("name", "Unknown source"),
            "url": review.get("url", ""),
        })
    return results
