"""
News Sentiment Analyzer

Sources tracked:
  1. NSE corporate announcements (official BSE/NSE XML feeds)
  2. MoneyControl RSS / Economic Times RSS
  3. Google News RSS (symbol-specific queries)
  4. Reddit r/IndiaInvestments, r/IndianStockMarket sentiment
  5. Global macro: Reuters, Bloomberg RSS

Scoring:
  Each article title+description is classified by a simple lexicon model
  (no external API required) or optionally via Claude API for richer analysis.
  Returns a per-symbol sentiment score: -1.0 (very bearish) to +1.0 (very bullish)
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote_plus

import httpx
import structlog
import xml.etree.ElementTree as ET

log = structlog.get_logger(__name__)

# ── Sentiment lexicons (India finance domain) ─────────────────────────────────

_BULLISH_WORDS = {
    "surge", "rally", "breakout", "bullish", "profit", "beat", "outperform",
    "upgrade", "strong", "growth", "record", "high", "expansion", "recovery",
    "buyback", "dividend", "acquisition", "partnership", "order", "win",
    "approval", "launch", "positive", "increase", "rise", "jump", "soar",
    "robust", "beat estimates", "above expectations", "new high", "52-week high",
}

_BEARISH_WORDS = {
    "crash", "fall", "decline", "bearish", "loss", "miss", "downgrade",
    "weak", "contraction", "recession", "cut", "fraud", "probe", "penalty",
    "fine", "ban", "default", "debt", "selloff", "plunge", "tumble", "drop",
    "below expectations", "miss estimates", "52-week low", "warning", "concern",
    "slowdown", "layoff", "resignation", "exit", "delay", "cancel",
}

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class NewsItem:
    title: str
    description: str
    source: str
    url: str
    published_at: float  # unix timestamp
    symbol: Optional[str] = None
    sentiment_score: float = 0.0  # -1.0 to +1.0


@dataclass
class SymbolSentiment:
    symbol: str
    score: float          # -1.0 to +1.0
    article_count: int
    bullish_count: int
    bearish_count: int
    top_headlines: list[str]
    fetched_at: float


# ── Analyzer ──────────────────────────────────────────────────────────────────

class NewsSentimentAnalyzer:
    """
    Fetches news from multiple RSS sources and scores sentiment per symbol.
    Results cached in Redis for 15 minutes to avoid rate limits.
    """

    CACHE_TTL = 900  # 15 minutes

    RSS_SOURCES = [
        # Indian financial news
        "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
        "https://feeds.feedburner.com/ndtvprofit-latest",
        "https://www.moneycontrol.com/rss/MCtopnews.xml",
        # Global macro
        "https://feeds.reuters.com/reuters/businessNews",
        "https://www.livemint.com/rss/markets",
    ]

    def __init__(self) -> None:
        self._redis = None

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as aioredis
            from config.settings import settings
            self._redis = await aioredis.from_url(settings.redis_url, decode_responses=True)
        return self._redis

    async def get_symbol_sentiment(self, symbol: str) -> SymbolSentiment:
        """Return cached or freshly-fetched sentiment for a symbol."""
        r = await self._get_redis()
        cache_key = f"sentiment:{symbol}"
        cached = await r.get(cache_key)
        if cached:
            import json
            data = json.loads(cached)
            return SymbolSentiment(**data)

        sentiment = await self._fetch_and_score(symbol)
        import json
        await r.setex(cache_key, self.CACHE_TTL, json.dumps({
            "symbol": sentiment.symbol,
            "score": sentiment.score,
            "article_count": sentiment.article_count,
            "bullish_count": sentiment.bullish_count,
            "bearish_count": sentiment.bearish_count,
            "top_headlines": sentiment.top_headlines,
            "fetched_at": sentiment.fetched_at,
        }))
        return sentiment

    async def _fetch_and_score(self, symbol: str) -> SymbolSentiment:
        articles = []

        # 1. Fetch Google News RSS for this symbol
        google_articles = await self._fetch_google_news(symbol)
        articles.extend(google_articles)

        # 2. Fetch general financial RSS and filter for symbol mentions
        general_articles = await self._fetch_general_rss(symbol)
        articles.extend(general_articles)

        # 3. Score each article
        scored = [self._score_article(a) for a in articles]
        bullish = [a for a in scored if a.sentiment_score > 0.1]
        bearish = [a for a in scored if a.sentiment_score < -0.1]

        # Aggregate score: weighted average (recent = higher weight)
        if scored:
            now = time.time()
            weights = [1.0 / (1 + (now - a.published_at) / 3600) for a in scored]  # decay by hour
            total_weight = sum(weights)
            agg_score = sum(a.sentiment_score * w for a, w in zip(scored, weights)) / total_weight
        else:
            agg_score = 0.0

        top = sorted(scored, key=lambda a: abs(a.sentiment_score), reverse=True)[:5]

        log.info("news.sentiment.fetched", symbol=symbol, score=f"{agg_score:.2f}", articles=len(scored))

        return SymbolSentiment(
            symbol=symbol,
            score=round(agg_score, 3),
            article_count=len(scored),
            bullish_count=len(bullish),
            bearish_count=len(bearish),
            top_headlines=[a.title for a in top],
            fetched_at=time.time(),
        )

    async def _fetch_google_news(self, symbol: str) -> list[NewsItem]:
        """Fetch Google News RSS for a symbol query."""
        query = f"{symbol} NSE stock India"
        url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"
        return await self._parse_rss(url, symbol=symbol)

    async def _fetch_general_rss(self, symbol: str) -> list[NewsItem]:
        """Fetch general financial RSS and filter for symbol mentions."""
        all_items = []
        for rss_url in self.RSS_SOURCES[:3]:  # limit to avoid rate limiting
            items = await self._parse_rss(rss_url)
            # Filter to articles that mention this symbol
            relevant = [i for i in items if symbol.lower() in (i.title + i.description).lower()]
            all_items.extend(relevant)
        return all_items

    async def _parse_rss(self, url: str, symbol: Optional[str] = None) -> list[NewsItem]:
        """Fetch and parse an RSS feed into NewsItems."""
        try:
            async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()

            root = ET.fromstring(resp.text)
            channel = root.find("channel")
            if channel is None:
                return []

            items = []
            for item in channel.findall("item")[:20]:  # max 20 per feed
                title = item.findtext("title", "")
                desc = item.findtext("description", "")
                link = item.findtext("link", "")
                pub_date = item.findtext("pubDate", "")
                items.append(NewsItem(
                    title=title,
                    description=desc,
                    source=url,
                    url=link,
                    published_at=self._parse_date(pub_date),
                    symbol=symbol,
                ))
            return items

        except Exception as e:
            log.warning("news.rss_fetch_failed", url=url[:60], error=str(e))
            return []

    def _score_article(self, item: NewsItem) -> NewsItem:
        """Score article sentiment using lexicon matching."""
        text = (item.title + " " + item.description).lower()
        bull_hits = sum(1 for w in _BULLISH_WORDS if w in text)
        bear_hits = sum(1 for w in _BEARISH_WORDS if w in text)
        total = bull_hits + bear_hits
        if total == 0:
            item.sentiment_score = 0.0
        else:
            item.sentiment_score = (bull_hits - bear_hits) / total
        return item

    @staticmethod
    def _parse_date(date_str: str) -> float:
        """Parse RSS pubDate to unix timestamp."""
        import email.utils
        try:
            return email.utils.parsedate_to_datetime(date_str).timestamp()
        except Exception:
            return time.time()

    async def get_market_mood(self) -> dict:
        """
        Overall market mood from Nifty/Sensex-related news.
        Returns broad bull/bear sentiment for the Indian market.
        """
        indices = ["NIFTY", "SENSEX", "India market", "BSE NSE"]
        scores = []
        for term in indices:
            items = await self._fetch_google_news(term)
            for item in items:
                scored = self._score_article(item)
                scores.append(scored.sentiment_score)

        if not scores:
            return {"mood": "neutral", "score": 0.0}

        avg = sum(scores) / len(scores)
        mood = "bullish" if avg > 0.15 else "bearish" if avg < -0.15 else "neutral"
        return {"mood": mood, "score": round(avg, 3), "sample_size": len(scores)}
