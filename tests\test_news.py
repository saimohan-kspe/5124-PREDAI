"""Test 11: Deterministic offline news entity tagging without network dependencies."""
import json
import pytest


def tag_news_headline(headline: str, entity_keywords: dict):
    """Deterministic keyword matching for financial headlines."""
    headline_lower = headline.lower()
    matched_symbols = []
    for symbol, keywords in entity_keywords.items():
        if any(kw in headline_lower for kw in keywords):
            matched_symbols.append(symbol)
    return matched_symbols


def test_deterministic_offline_news_tagging():
    """Verify headlines are deterministically tagged to correct symbols."""
    entity_keywords = {
        "RELIANCE": ["oil", "refining", "telecom", "jio"],
        "TCS": ["software", "tcs", "it exports"],
        "HDFCBANK": ["bank", "deposits", "credit", "npa"],
    }

    h1 = "Global crude oil surge pressures Indian refining margins"
    h2 = "US tech slowdown raises caution on IT exports and software spending"

    tags1 = tag_news_headline(h1, entity_keywords)
    tags2 = tag_news_headline(h2, entity_keywords)

    assert "RELIANCE" in tags1
    assert "TCS" in tags2
    assert "HDFCBANK" not in tags1
