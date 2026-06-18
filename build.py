#!/usr/bin/env python3
"""
Build a personal news dashboard from RSS/Reddit sources, optionally enriched
with a Claude pass that summarizes, tags, and relevance-ranks each item.

Output: index.html (committed by the GitHub Action and served by GitHub Pages).

Env:
  ANTHROPIC_API_KEY   If set, the LLM enrich pass runs. If absent, the build
                      still works — it just shows raw titles, ranked by source
                      weight + recency, with no summaries or relevance filter.
  FEED_MODEL          Claude model id (default: claude-opus-4-8). For a cheaper
                      every-few-hours cron, set claude-haiku-4-5 or
                      claude-sonnet-4-6 as a repo variable.
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import feedparser
import requests
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("FEED_MODEL", "claude-opus-4-8")
# A browser-like UA matters: Reddit (and some CDNs) 403 generic/bot agents.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
# Batch size for the LLM enrich call — keeps each request well within limits.
LLM_BATCH = 40


# ── Fetching ──────────────────────────────────────────────────────────────
def _ts(entry) -> float:
    """Best-effort published timestamp as a UNIX float (0.0 if unknown)."""
    for key in ("published_parsed", "updated_parsed"):
        val = entry.get(key)
        if val:
            return time.mktime(val)
    return 0.0


def fetch_rss(src: dict) -> list[dict]:
    feed = feedparser.parse(src["url"], agent=USER_AGENT)
    items = []
    for e in feed.entries:
        title = (e.get("title") or "").strip()
        link = (e.get("link") or "").strip()
        if not title or not link:
            continue
        items.append(
            {
                "title": title,
                "url": link,
                "source": src["name"],
                "weight": float(src.get("weight", 1.0)),
                "tag_hint": src.get("tag_hint", ""),
                "ts": _ts(e),
                "blurb": html.unescape((e.get("summary") or "")[:500]),
            }
        )
    return items


def fetch_reddit(src: dict) -> list[dict]:
    r = requests.get(src["url"], headers={"User-Agent": USER_AGENT}, timeout=20)
    r.raise_for_status()
    items = []
    for child in r.json().get("data", {}).get("children", []):
        d = child.get("data", {})
        if d.get("stickied"):
            continue
        title = (d.get("title") or "").strip()
        if not title:
            continue
        # Prefer the linked article; fall back to the reddit comments page.
        permalink = "https://www.reddit.com" + d.get("permalink", "")
        url = d.get("url_overridden_by_dest") or permalink
        items.append(
            {
                "title": title,
                "url": url,
                "comments_url": permalink,
                "source": src["name"],
                "weight": float(src.get("weight", 1.0)),
                "tag_hint": src.get("tag_hint", ""),
                "ts": float(d.get("created_utc") or 0.0),
                "blurb": (d.get("selftext") or "")[:500],
                "score": d.get("score", 0),
            }
        )
    return items


def add_twitter_sources(cfg: dict) -> None:
    """Expand the `twitter:` block into one RSS source per account.

    Needs a working RSSHub base URL (X has no free native RSS). With no
    base configured, the accounts are listed but skipped — no error.
    """
    tw = cfg.get("twitter") or {}
    base = (tw.get("rsshub_base") or "").rstrip("/")
    accounts = [a.lstrip("@") for a in (tw.get("accounts") or [])]
    if not accounts:
        return
    if not base:
        print(
            f"  (twitter: {len(accounts)} accounts configured but "
            "rsshub_base is empty — skipping)",
            file=sys.stderr,
        )
        return
    cfg.setdefault("sources", [])
    for h in accounts:
        cfg["sources"].append(
            {
                "name": f"@{h}",
                "type": "rss",
                "url": f"{base}/twitter/user/{h}",
                "weight": float(tw.get("weight", 1.1)),
                "tag_hint": tw.get("tag_hint", "ai"),
            }
        )


def fetch_all(cfg: dict) -> list[dict]:
    items: list[dict] = []
    for src in cfg["sources"]:
        try:
            kind = src.get("type", "rss")
            got = fetch_reddit(src) if kind == "reddit" else fetch_rss(src)
            print(f"  {src['name']}: {len(got)} items", file=sys.stderr)
            items.extend(got)
        except Exception as ex:  # one bad source shouldn't sink the build
            print(f"  ! {src['name']} failed: {ex}", file=sys.stderr)
    return items


def strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s or "")).strip()


def ensure_detail(items: list[dict]) -> list[dict]:
    """Guarantee every card has something to show when expanded."""
    for it in items:
        if not it.get("detail"):
            it["detail"] = strip_html(it.get("blurb", ""))[:400]
    return items


def dedup(items: list[dict]) -> list[dict]:
    seen, out = set(), []
    for it in items:
        # Normalize URL for dedup (strip query/tracking and trailing slash).
        key = it["url"].split("?")[0].rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


# ── LLM enrich pass ─────────────────────────────────────────────────────────
ENRICH_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "i": {"type": "integer"},
                    "summary": {"type": "string"},
                    "detail": {"type": "string"},
                    "tag": {
                        "type": "string",
                        "enum": ["ai", "tech", "world", "science", "business", "fun", "other"],
                    },
                    "score": {"type": "integer"},
                },
                "required": ["i", "summary", "detail", "tag", "score"],
            },
        }
    },
    "required": ["items"],
}


def enrich_batch(client, interests: str, batch: list[dict]) -> dict[int, dict]:
    listing = "\n".join(
        f'{idx}. [{it["source"]}] {it["title"]}'
        + (f' — {it["blurb"][:200]}' if it.get("blurb") else "")
        for idx, it in enumerate(batch)
    )
    prompt = (
        "You are curating a personal news dashboard for someone with these "
        f"interests:\n{interests}\n\n"
        "For each numbered item below, return:\n"
        "- summary: one tight sentence, no preamble, max ~22 words (the card headline).\n"
        "- detail: a genuinely useful 2-4 sentence brief (~50-80 words) for someone "
        "who clicks in — what happened, the key facts/numbers, and why it matters. "
        "Plain English, no fluff, no 'this article discusses'.\n"
        "- tag: a single category.\n"
        "- score: relevance 0-10 for how much THIS person would care "
        "(10 = must-read, 0 = irrelevant/spam).\n"
        "Use the item's index as `i`.\n\n"
        f"{listing}"
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=12000,
        output_config={"format": {"type": "json_schema", "schema": ENRICH_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    text = next((b.text for b in resp.content if b.type == "text"), "{}")
    data = json.loads(text)
    return {row["i"]: row for row in data.get("items", [])}


def enrich(items: list[dict], cfg: dict) -> list[dict]:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  (no ANTHROPIC_API_KEY — skipping LLM enrich)", file=sys.stderr)
        return items
    import anthropic

    client = anthropic.Anthropic()
    interests = cfg.get("interests", "general technology and world news")
    for start in range(0, len(items), LLM_BATCH):
        batch = items[start : start + LLM_BATCH]
        try:
            rows = enrich_batch(client, interests, batch)
        except Exception as ex:
            print(f"  ! enrich batch @{start} failed: {ex}", file=sys.stderr)
            continue
        for idx, it in enumerate(batch):
            row = rows.get(idx)
            if row:
                it["summary"] = row["summary"]
                it["detail"] = row["detail"]
                it["tag"] = row["tag"]
                it["score"] = int(row["score"])
    return items


# ── Ranking ──────────────────────────────────────────────────────────────
def rank(items: list[dict], cfg: dict) -> list[dict]:
    now = time.time()
    have_scores = any("score" in it for it in items)
    min_score = int(cfg.get("min_score", 0))
    ranked = []
    for it in items:
        age_h = max(0.0, (now - it["ts"]) / 3600.0) if it["ts"] else 48.0
        recency = 1.0 / (1.0 + age_h / 12.0)  # ~half-life of half a day
        rel = it.get("score", 5) / 10.0
        it["rank"] = it["weight"] * (0.6 * rel + 0.4 * recency)
        if have_scores and it.get("score", 10) < min_score:
            continue  # LLM judged it below the relevance bar
        ranked.append(it)
    ranked.sort(key=lambda x: x["rank"], reverse=True)

    # Curate: cap how many any single source can contribute, then cap total.
    max_per = int(cfg.get("max_per_source", 0))
    if max_per:
        per: dict[str, int] = {}
        kept = []
        for it in ranked:
            n = per.get(it["source"], 0)
            if n >= max_per:
                continue
            per[it["source"]] = n + 1
            kept.append(it)
        ranked = kept

    max_items = int(cfg.get("max_items", 0))
    if max_items:
        ranked = ranked[:max_items]
    return ranked


# ── Render ───────────────────────────────────────────────────────────────
def render(items: list[dict], dropped: int) -> str:
    env = Environment(
        loader=FileSystemLoader(HERE),
        autoescape=select_autoescape(["html"]),
    )
    tmpl = env.get_template("template.html")
    tags = sorted({it.get("tag", "other") for it in items})
    return tmpl.render(
        items=items,
        tags=tags,
        dropped=dropped,
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        enriched=bool(os.environ.get("ANTHROPIC_API_KEY")),
    )


def main() -> int:
    with open(os.path.join(HERE, "sources.yml")) as f:
        cfg = yaml.safe_load(f)

    add_twitter_sources(cfg)
    print("Fetching sources…", file=sys.stderr)
    items = dedup(fetch_all(cfg))
    print(f"{len(items)} unique items", file=sys.stderr)

    before = len(items)
    items = enrich(items, cfg)
    items = ensure_detail(items)
    items = rank(items, cfg)
    dropped = before - len(items)

    out = render(items, dropped)
    with open(os.path.join(HERE, "index.html"), "w") as f:
        f.write(out)
    print(f"Wrote index.html — {len(items)} items ({dropped} filtered out)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
