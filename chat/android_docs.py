"""Fetch and format official Android developer documentation."""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urljoin

import httpx
from lxml import html as lxml_html

BASE = "https://developer.android.com"
SEARCH_URL = "https://developer.android.com/s/results?q={query}"
USER_AGENT = "codescope/0.1 (+https://developer.android.com)"
CACHE_TTL_SEC = 7 * 24 * 3600  # 7 days
MAX_BODY_CHARS = 12_000

# Curated topic → canonical doc URL (developer.android.com)
TOPIC_URLS: dict[str, str] = {
    "room": "https://developer.android.com/training/data-storage/room",
    "room database": "https://developer.android.com/training/data-storage/room",
    "stateflow": "https://developer.android.com/kotlin/flow/stateflow-and-sharedflow",
    "sharedflow": "https://developer.android.com/kotlin/flow/stateflow-and-sharedflow",
    "flow": "https://developer.android.com/kotlin/flow",
    "viewmodel": "https://developer.android.com/topic/libraries/architecture/viewmodel",
    "viewmodel compose": "https://developer.android.com/topic/libraries/architecture/viewmodel/viewmodel-compose",
    "viewmodel stateflow": "https://developer.android.com/topic/libraries/architecture/viewmodel/viewmodel-compose",
    "livedata": "https://developer.android.com/topic/libraries/architecture/livedata",
    "compose": "https://developer.android.com/jetpack/compose",
    "jetpack compose": "https://developer.android.com/jetpack/compose",
    "navigation": "https://developer.android.com/guide/navigation",
    "navgraph": "https://developer.android.com/guide/navigation/design",
    "navigation compose": "https://developer.android.com/develop/ui/compose/navigation",
    "material 3": "https://developer.android.com/develop/ui/compose/designsystems/material3",
    "material3": "https://developer.android.com/develop/ui/compose/designsystems/material3",
    "hilt": "https://developer.android.com/training/dependency-injection/hilt-android",
    "dagger": "https://developer.android.com/training/dependency-injection/hilt-android",
    "coroutines": "https://developer.android.com/kotlin/coroutines",
    "repository": "https://developer.android.com/topic/architecture/data-layer",
    "datastore": "https://developer.android.com/topic/libraries/architecture/datastore",
    "workmanager": "https://developer.android.com/topic/libraries/architecture/workmanager",
    "activity": "https://developer.android.com/guide/components/activities/intro-activities",
    "fragment": "https://developer.android.com/guide/fragments",
    "recyclerview": "https://developer.android.com/develop/ui/views/layout/recyclerview",
    "lazy column": "https://developer.android.com/develop/ui/compose/lists",
    "lazycolumn": "https://developer.android.com/develop/ui/compose/lists",
}


def _normalize_topic(topic: str) -> str:
    return re.sub(r"\s+", " ", topic.strip().lower())


def resolve_topic_url(topic: str) -> str | None:
    key = _normalize_topic(topic)
    if key in TOPIC_URLS:
        return TOPIC_URLS[key]
    # partial match
    for k, url in TOPIC_URLS.items():
        if key in k or k in key:
            return url
    return None


def _cache_path(cache_dir: Path | None) -> Path:
    if cache_dir:
        return cache_dir / "android_docs_cache.jsonl"
    return Path.home() / ".codescope" / "android_docs_cache.jsonl"


def _cache_get(cache_path: Path, key: str) -> dict | None:
    if not cache_path.exists():
        return None
    now = time.time()
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("key") == key and now - rec.get("ts", 0) < CACHE_TTL_SEC:
                return rec.get("doc")
    return None


def _cache_put(cache_path: Path, key: str, doc: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "ts": time.time(), "doc": doc}, ensure_ascii=False) + "\n")


def _fetch_html(url: str) -> tuple[str, str]:
    r = httpx.get(
        _canonical_url(url),
        headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
        timeout=25,
        follow_redirects=True,
    )
    r.raise_for_status()
    return r.text, _canonical_url(str(r.url))


def _canonical_url(url: str) -> str:
    url = url.split("#")[0].strip()
    if url.startswith("//"):
        url = "https:" + url
    if url.startswith("/"):
        url = BASE + url
    return url


def _search_first_url(query: str) -> str | None:
    """Pick the first developer.android.com guide/reference link from search."""
    url = SEARCH_URL.format(query=quote_plus(query))
    try:
        body, _ = _fetch_html(url)
    except Exception:
        return None
    tree = lxml_html.fromstring(body)
    for a in tree.xpath("//a[@href]"):
        href = a.get("href", "")
        if not href:
            continue
        full = urljoin(BASE, href)
        if "developer.android.com" not in full:
            continue
        if any(x in full for x in ("/reference/", "/training/", "/guide/", "/develop/", "/jetpack/", "/kotlin/")):
            if "/s/results" not in full:
                return _canonical_url(full)
    return None


def pick_primary_topic(user_query: str, extra_context: str = "") -> str:
    """Choose one topic for a single android_docs fetch."""
    text = f"{user_query} {extra_context}".lower()
    priority = [
        ("room", "room"),
        ("stateflow", "stateflow"),
        ("compose", "jetpack compose"),
        ("viewmodel", "viewmodel compose"),
        ("navigation", "navigation compose"),
        ("material", "material 3"),
        ("coroutines", "coroutines"),
        ("repository", "repository"),
    ]
    for needle, topic in priority:
        if needle in text:
            return topic
    return "android jetpack architecture"


def _relevance_score(text: str, query: str, topic: str) -> float:
    """Simple keyword overlap score 0..1."""
    hay = text.lower()
    terms = set(re.findall(r"[a-z][a-z0-9]{2,}", f"{query} {topic}".lower()))
    terms -= {"the", "and", "for", "with", "from", "this", "that", "android", "please"}
    if not terms:
        return 0.5
    hits = sum(1 for t in terms if t in hay)
    return hits / len(terms)


def validate_android_doc(
    url: str,
    user_query: str,
    topic: str = "",
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Fetch *url* and verify it is relevant to the user's question.
    Returns validation result with excerpt; may suggest a better URL.
    """
    url = _canonical_url(url)
    try:
        html_body, final_url = _fetch_html(url)
        title, body, sections = _html_to_text(html_body)
    except Exception as e:
        return {
            "valid": False,
            "url": url,
            "error": f"Could not fetch URL: {e}",
            "recommendation": "Call android_docs with a different topic query.",
        }

    score = _relevance_score(f"{title}\n{body}", user_query, topic)
    valid = score >= 0.25 and len(body) > 200

    result: dict[str, Any] = {
        "valid":           valid,
        "url":             final_url,
        "title":           title,
        "relevance_score": round(score, 2),
        "excerpt":         body[:2500],
        "section_headings": [s.get("heading", "") for s in sections[:8] if s.get("heading")],
    }

    if not valid:
        alt_topic = pick_primary_topic(user_query, topic)
        alt_url = resolve_topic_url(alt_topic) or _search_first_url(alt_topic)
        if alt_url and alt_url != url:
            result["alternate_url"] = _canonical_url(alt_url)
            result["alternate_topic"] = alt_topic
            result["recommendation"] = (
                f"Page may not match the query (score={score:.2f}). "
                f"Retry android_docs with topic={alt_topic!r} or use alternate_url."
            )
        else:
            result["recommendation"] = "Try android_docs with a more specific topic."

    result["_hint"] = (
        "Validation complete. Use ONLY the url field above in your answer — do not invent links."
    )
    return result


def _html_to_text(html_body: str) -> tuple[str, str, list[dict[str, str]]]:
    tree = lxml_html.fromstring(html_body)
    title_el = tree.xpath("//title")
    title = (title_el[0].text_content().strip() if title_el else "Android Documentation")

    # Devsite article body
    nodes = tree.xpath(
        "//article//div[contains(@class,'devsite-article-body')]"
        " | //div[contains(@class,'devsite-article-body')]"
    )
    if not nodes:
        nodes = tree.xpath("//article | //main")

    sections: list[dict[str, str]] = []
    parts: list[str] = []

    for node in nodes[:1]:
        for el in node.iter():
            if el.tag in ("h1", "h2", "h3", "h4"):
                heading = el.text_content().strip()
                if heading:
                    sections.append({"heading": heading, "content": ""})
                    parts.append(f"\n## {heading}\n")
            elif el.tag in ("p", "li", "pre", "code"):
                text = el.text_content().strip()
                if not text or len(text) < 4:
                    continue
                if sections and sections[-1]["content"] == "":
                    sections[-1]["content"] = text[:2000]
                parts.append(text)

    body = re.sub(r"\n{3,}", "\n\n", "\n".join(parts)).strip()
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + "\n\n… [truncated for context]"
    return title, body, sections


def fetch_android_doc(
    topic: str,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Fetch one Android doc topic. Returns structured dict for agent + terminal display.
    """
    topic = topic.strip()
    if not topic:
        raise ValueError("topic is required")

    cache_path = _cache_path(cache_dir)
    cache_key = hashlib.sha256(_normalize_topic(topic).encode()).hexdigest()[:16]
    cached = _cache_get(cache_path, cache_key)
    if cached:
        return {**cached, "cached": True}

    url = resolve_topic_url(topic)
    if not url:
        url = _search_first_url(topic)
    if not url:
        return {
            "topic": topic,
            "error": f"No Android documentation URL found for {topic!r}.",
            "suggestions": sorted(set(TOPIC_URLS.keys()))[:20],
        }

    try:
        html_body, final_url = _fetch_html(url)
        title, body, sections = _html_to_text(html_body)
    except Exception as e:
        return {
            "topic": topic,
            "url": url,
            "error": f"Failed to fetch documentation: {e}",
        }

    doc = {
        "topic":    topic,
        "title":    title,
        "url":      final_url,
        "summary":  body[:500] + ("…" if len(body) > 500 else ""),
        "content":  body,
        "sections": sections[:12],
        "cached":   False,
        "_hint":    "Official Android docs loaded. Do not call android_docs again for this topic this turn.",
    }
    _cache_put(cache_path, cache_key, doc)
    return doc


def fetch_android_docs(
    topics: list[str],
    cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    return [fetch_android_doc(t, cache_dir=cache_dir) for t in topics if t.strip()]


def format_doc_terminal(doc: dict[str, Any]) -> str:
    """Plain-text block for Rich Markdown / console."""
    if doc.get("error"):
        lines = [f"## {doc.get('topic', '?')}", f"**Error:** {doc['error']}"]
        if doc.get("url"):
            lines.append(f"URL: {doc['url']}")
        if doc.get("suggestions"):
            lines.append("\n**Known topics:** " + ", ".join(doc["suggestions"][:15]))
        return "\n".join(lines)

    topic = doc.get("topic", "Android")
    url = doc.get("url", "")
    lines = [
        f"## {doc.get('title', topic)}",
        f"{topic}: {url}",
        "",
        doc.get("content", doc.get("summary", "")),
    ]
    if doc.get("cached"):
        lines.insert(2, "*(cached)*")
    return "\n".join(lines)
