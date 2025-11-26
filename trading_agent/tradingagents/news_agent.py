from __future__ import annotations

import contextlib
import functools
import html
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import URLError
from urllib.parse import quote_plus
from urllib.request import urlopen

import xml.etree.ElementTree as ET

import pathway as pw
import requests
import yfinance as yf

from tradingagents import llm_client

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover - handled at runtime
    SentimentIntensityAnalyzer = None  # type: ignore

_SENTIMENT_ANALYSER = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None

@dataclass
class NewsArticle:
    headline: str
    published_at: Optional[str]
    summary: Optional[str]
    source: Optional[str]
    url: Optional[str]
    sentiment: str
    sentiment_score: int


@dataclass
class NewsWeightReport:
    ticker: str
    weight: float
    as_of: str
    lookback_days: int
    judgement: str
    points: List[str]
    articles: List[NewsArticle]
    generated_via_llm: bool = False

    def to_markdown(self, include_articles: bool = True) -> str:
        header = (
            f"# News-Based Weight Review: {self.ticker}\n\n"
            f"- **As of:** {self.as_of}\n"
            f"- **Assigned Weight:** {self.weight:.2%}\n"
            f"- **News Lookback:** {self.lookback_days} day(s)\n\n"
        )

        bullet_lines = "\n".join(f"- {point}" for point in self.points)
        sections = [header, "## Coverage Assessment\n", bullet_lines, "\n"]

        if include_articles and self.articles:
            sections.extend(["## Notable Headlines\n", _format_articles_table(self.articles), "\n"])

        return "".join(sections)


def _parse_as_of(as_of: str) -> date:
    try:
        return datetime.strptime(as_of, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("as_of must be in YYYY-MM-DD format") from exc


def _article_to_dict(article: NewsArticle) -> Dict[str, Any]:
    return {
        "headline": article.headline,
        "published_at": article.published_at,
        "summary": article.summary,
        "source": article.source,
        "url": article.url,
        "sentiment": article.sentiment,
        "sentiment_score": int(article.sentiment_score),
    }


def _article_from_dict(payload: Dict[str, Any]) -> NewsArticle:
    sentiment_score = payload.get("sentiment_score", 0)
    try:
        sentiment_int = int(sentiment_score)
    except (TypeError, ValueError):
        sentiment_int = 0
    return NewsArticle(
        headline=str(payload.get("headline", "")),
        published_at=payload.get("published_at"),
        summary=payload.get("summary"),
        source=payload.get("source"),
        url=payload.get("url"),
        sentiment=str(payload.get("sentiment", "neutral")),
        sentiment_score=sentiment_int,
    )


def _compute_news_payload(
    ticker: str,
    weight: float,
    as_of: str,
    lookback_days: int,
    max_articles: int,
    use_llm: bool,
    llm_model: Optional[str],
) -> Dict[str, Any]:
    as_of_date = _parse_as_of(as_of)
    lookback = int(lookback_days)
    if lookback <= 0:
        raise ValueError("lookback_days must be positive")
    max_count = int(max_articles)
    if max_count <= 0:
        raise ValueError("max_articles must be positive")

    start_date = as_of_date - timedelta(days=lookback)

    articles = _fetch_news(ticker, start_date, as_of_date)
    # Debug: log fetched article counts before and after scoring
    try:
        print(f"[DEBUG news_agent] {ticker}: fetched_raw_articles={len(articles)}")
    except Exception:
        pass
    articles = _score_articles(articles)
    try:
        print(f"[DEBUG news_agent] {ticker}: scored_articles={len(articles)}, use_llm={use_llm}, llm_model={llm_model}")
    except Exception:
        pass
    articles = articles[:max_count]

    judgement, supporting_points = _build_opinion(weight, articles)
    points = [judgement] + supporting_points
    points = [str(point) for point in points[:4]]
    generated_via_llm = False

    if use_llm:
        article_summaries = _articles_prompt_digest(articles)
        net_sentiment = sum(article.sentiment_score for article in articles)
        llm_points = llm_client.summarise_news(
            ticker=ticker,
            weight=weight,
            as_of=as_of_date.isoformat(),
            lookback_days=lookback,
            article_summaries=article_summaries,
            net_sentiment=net_sentiment,
            max_points=4,
            model=llm_model,
        )
        if llm_points:
            cleaned_points = [str(point) for point in llm_points if point]
            points = cleaned_points[:4] or points
            if points:
                judgement = points[0]
            generated_via_llm = True

    return {
        "ticker": ticker,
        "weight": float(weight),
        "as_of": as_of_date.isoformat(),
        "lookback_days": lookback,
        "judgement": judgement,
        "points": points,
        "articles": [_article_to_dict(article) for article in articles],
        "generated_via_llm": generated_via_llm,
    }


def _run_news_pipeline(
    *,
    ticker: str,
    weight: float,
    as_of: str,
    lookback_days: int,
    max_articles: int,
    use_llm: bool,
    llm_model: Optional[str],
) -> Dict[str, Any]:
    payload = _compute_news_payload(
        ticker,
        weight,
        as_of,
        lookback_days,
        max_articles,
        use_llm,
        llm_model,
    )
    payload_table = pw.debug.table_from_rows(
        pw.schema_from_types(payload=pw.Json),
        [(pw.Json(payload),)],
    )
    keys, columns = pw.debug.table_to_dicts(payload_table)
    if not keys:
        raise RuntimeError("News analysis produced no output.")
    payload = columns["payload"][keys[0]]
    if isinstance(payload, pw.Json):
        return payload.value
    return payload


def _payload_to_news_report(payload: Dict[str, Any]) -> NewsWeightReport:
    articles_raw = payload.get("articles", [])
    articles: List[NewsArticle] = []
    for entry in articles_raw:
        if isinstance(entry, NewsArticle):
            articles.append(entry)
            continue
        if not isinstance(entry, dict):
            continue
        with contextlib.suppress(Exception):
            articles.append(_article_from_dict(entry))

    points_raw = payload.get("points", [])
    points = [str(point) for point in points_raw if point is not None]

    return NewsWeightReport(
        ticker=str(payload.get("ticker", "")),
        weight=float(payload.get("weight", 0.0)),
        as_of=str(payload.get("as_of", "")),
        lookback_days=int(payload.get("lookback_days", 0)),
        judgement=str(payload.get("judgement", "")),
        points=points,
        articles=articles,
        generated_via_llm=bool(payload.get("generated_via_llm", False)),
    )


class NewsWeightReviewAgent:
    """Reviews an assigned portfolio weight against recent news flow using Pathway."""

    def __init__(self, *, default_as_of: Optional[date] = None):
        self._default_as_of = default_as_of or date.today()

    def generate_report(
        self,
        ticker: str,
        weight: float,
        *,
        as_of: Optional[str] = None,
        lookback_days: int = 7,
        max_articles: int = 8,
        use_llm: bool = False,
        llm_model: Optional[str] = None,
    ) -> NewsWeightReport:
        clean_ticker = ticker.strip().upper()
        if not clean_ticker:
            raise ValueError("Ticker symbol cannot be empty")
        if not (0.0 <= weight <= 1.0):
            raise ValueError("Weight must be between 0.0 and 1.0 inclusive")
        if lookback_days <= 0:
            raise ValueError("Lookback window must be positive")
        if max_articles <= 0:
            raise ValueError("max_articles must be positive")

        resolved_as_of = self._default_as_of if not as_of else _parse_as_of(as_of)
        as_of_str = resolved_as_of.isoformat()

        payload = _run_news_pipeline(
            ticker=clean_ticker,
            weight=float(weight),
            as_of=as_of_str,
            lookback_days=int(lookback_days),
            max_articles=int(max_articles),
            use_llm=bool(use_llm),
            llm_model=llm_model,
        )
        return _payload_to_news_report(payload)


def _fetch_news(ticker: str, start_date: date, end_date: date) -> List[NewsArticle]:
    articles: List[NewsArticle] = []
    queries = _news_queries_for_ticker(ticker)

    for query in queries:
        articles.extend(_fetch_google_news(query, start_date, end_date))
        articles.extend(_fetch_yahoo_news(query, start_date, end_date))

    articles.extend(_fetch_yfinance_news(ticker, start_date, end_date))

    deduped = _deduplicate_articles(articles)
    relevant = _filter_relevant_articles(deduped, ticker)
    try:
        print(
            f"[DEBUG news_agent] _fetch_news {ticker}: queries={len(queries)} raw={len(articles)} deduped={len(deduped)} relevant={len(relevant)}"
        )
    except Exception:
        pass
    return relevant


def _fetch_google_news(query: str, start_date: date, end_date: date) -> List[NewsArticle]:
    encoded = quote_plus(query)
    url = (
        "https://news.google.com/rss/search?q="
        f"{encoded}&hl=en-US&gl=US&ceid=US:en"
    )

    try:
        with contextlib.closing(urlopen(url, timeout=10)) as response:
            payload = response.read()
    except URLError as e:
        print(f"[DEBUG news_agent] _fetch_google_news URLError for query='{query}': {e}")
        return []
    except TimeoutError as e:
        print(f"[DEBUG news_agent] _fetch_google_news TimeoutError for query='{query}': {e}")
        return []

    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        return []

    articles: List[NewsArticle] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue

        pub_date_raw = item.findtext("pubDate")
        publish_dt: Optional[datetime] = None
        if pub_date_raw:
            try:
                publish_dt = parsedate_to_datetime(pub_date_raw)
                if publish_dt.tzinfo is None:
                    publish_dt = publish_dt.replace(tzinfo=timezone.utc)
                else:
                    publish_dt = publish_dt.astimezone(timezone.utc)
            except (TypeError, ValueError):
                publish_dt = None
        if publish_dt is None:
            continue
        if publish_dt.date() < start_date or publish_dt.date() > end_date:
            continue

        summary_raw = item.findtext("description") or ""
        summary = _strip_html(summary_raw).strip() or None
        source_elem = item.find("{http://news.google.com/newssources}news-source")
        source = source_elem.text.strip() if source_elem is not None and source_elem.text else None
        link = (item.findtext("link") or "").strip() or None

        articles.append(
            NewsArticle(
                headline=html.unescape(title),
                published_at=publish_dt.isoformat(),
                summary=html.unescape(summary) if summary else None,
                source=source,
                url=link,
                sentiment="neutral",
                sentiment_score=0,
            )
        )

    articles.sort(key=lambda article: article.published_at or "", reverse=True)
    deduped = _deduplicate_articles(articles)
    try:
        print(
            f"[DEBUG news_agent] _fetch_google_news query='{query}': parsed={len(articles)}, deduped={len(deduped)}"
        )
    except Exception:
        pass
    return deduped


def _fetch_yahoo_news(query: str, start_date: date, end_date: date) -> List[NewsArticle]:
    params = {
        "q": query,
        "quotesCount": 0,
        "newsCount": 20,
        "listsCount": 0,
    }
    try:
        resp = requests.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params=params,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        payload = resp.json()
    except Exception as exc:
        try:
            print(f"[DEBUG news_agent] _fetch_yahoo_news query='{query}' failed: {exc}")
        except Exception:
            pass
        return []

    news_items = payload.get("news", [])
    articles: List[NewsArticle] = []
    for item in news_items:
        if not isinstance(item, dict):
            continue
        published = _extract_publish_datetime(item)
        if published is None:
            continue
        if published.date() < start_date or published.date() > end_date:
            continue
        headline = (item.get("title") or "").strip()
        if not headline:
            continue
        source = (item.get("publisher") or "").strip() or None
        summary = (item.get("summary") or item.get("description") or "").strip() or None
        url = (item.get("link") or "").strip() or None
        articles.append(
            NewsArticle(
                headline=headline,
                published_at=published.isoformat(),
                summary=summary,
                source=source,
                url=url,
                sentiment="neutral",
                sentiment_score=0,
            )
        )

    if articles:
        articles.sort(key=lambda article: article.published_at or "", reverse=True)
        try:
            print(
                f"[DEBUG news_agent] _fetch_yahoo_news query='{query}': parsed={len(articles)}"
            )
        except Exception:
            pass
    return articles


def _fetch_yfinance_news(ticker: str, start_date: date, end_date: date) -> List[NewsArticle]:
    try:
        payload = yf.Ticker(ticker).news or []
    except Exception:
        payload = []

    articles: List[NewsArticle] = []
    for raw_item in payload:
        if not isinstance(raw_item, dict):
            continue

        item = _normalise_yfinance_item(raw_item)
        if not item:
            continue

        published = _extract_publish_datetime(item)
        if published is None:
            continue
        if published.date() < start_date or published.date() > end_date:
            continue

        headline = (item.get("title") or item.get("headline") or "").strip()
        if not headline:
            continue

        summary = (
            item.get("summary")
            or item.get("description")
            or item.get("content")
            or ""
        ).strip() or None
        source = (
            item.get("publisher")
            or item.get("source")
            or (item.get("provider") or {}).get("displayName")  # type: ignore[arg-type]
            or ""
        )
        source = source.strip() or None
        url = (
            item.get("url")
            or item.get("link")
            or _first_nested_url(item.get("canonicalUrl"))
            or _first_nested_url(item.get("clickThroughUrl"))
            or ""
        )
        url = url.strip() or None

        articles.append(
            NewsArticle(
                headline=headline,
                published_at=published.isoformat(),
                summary=summary,
                source=source,
                url=url,
                sentiment="neutral",
                sentiment_score=0,
            )
        )

    articles.sort(key=lambda article: article.published_at or "", reverse=True)
    deduped = _deduplicate_articles(articles)
    try:
        print(f"[DEBUG news_agent] _fetch_yfinance_news {ticker}: parsed={len(articles)}, deduped={len(deduped)}")
    except Exception:
        pass
    return deduped


@functools.lru_cache(maxsize=256)
def _news_queries_for_ticker(ticker: str) -> List[str]:
    base_queries = {ticker, f"{ticker} stock"}
    names = _ticker_name_candidates(ticker)
    for name in names:
        base_queries.add(name)
        base_queries.add(f"{name} stock")
        base_queries.add(f"{name} company")
    filtered = []
    for query in base_queries:
        if not query:
            continue
        stripped = query.strip()
        if len(stripped) < 4:
            continue
        if stripped.upper() == stripped and len(stripped) <= 4:
            continue
        filtered.append(stripped)
    # Keep queries deterministic for caching; sort by length descending to prioritise descriptive names
    filtered.sort(key=lambda val: (-len(val), val.lower()))
    return filtered[:10]


@functools.lru_cache(maxsize=256)
def _ticker_name_candidates(ticker: str) -> List[str]:
    try:
        info = yf.Ticker(ticker).get_info()
    except Exception:
        info = {}
    names: List[str] = []
    for key in ("longName", "shortName", "displayName", "name"):
        value = info.get(key)
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                names.append(cleaned)
    return names


def _normalise_yfinance_item(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten recent yfinance news responses into a single dictionary."""

    base: Dict[str, Any] = {}

    content = raw.get("content")
    if isinstance(content, dict):
        base.update(content)
        # Preserve providerPublishTime when nested payload omits it.
        if "providerPublishTime" in raw and "providerPublishTime" not in base:
            base["providerPublishTime"] = raw["providerPublishTime"]
    else:
        base.update(raw)

    provider = base.get("provider")
    if isinstance(provider, dict):
        base.setdefault("publisher", provider.get("displayName"))

    # Some responses wrap URLs in dictionaries (e.g. canonicalUrl: {"url": "..."}).
    for key in ("canonicalUrl", "clickThroughUrl"):
        value = base.get(key)
        if isinstance(value, dict) and "url" in value:
            base.setdefault("url", value.get("url"))

    return base


def _first_nested_url(entry: Any) -> Optional[str]:
    if isinstance(entry, dict):
        url = entry.get("url")
        if isinstance(url, str):
            return url
    return None


def _filter_relevant_articles(articles: List[NewsArticle], ticker: str) -> List[NewsArticle]:
    if not articles:
        return articles

    names = [name.lower() for name in _ticker_name_candidates(ticker)]
    names = [name for name in names if len(name) > 4]

    ticker_lower = ticker.lower()
    ticker_aliases = {ticker_lower}
    if "." in ticker_lower:
        ticker_aliases.add(ticker_lower.split(".")[0])
    ticker_aliases.add(ticker_lower.replace(".", " "))

    filtered: List[NewsArticle] = []
    for article in articles:
        haystack = " ".join(
            filter(None, [article.headline, article.summary, article.source])
        ).lower()
        if any(alias and alias in haystack for alias in ticker_aliases):
            filtered.append(article)
            continue
        if any(name in haystack for name in names):
            filtered.append(article)

    return filtered or articles


def _score_articles(articles: List[NewsArticle]) -> List[NewsArticle]:
    scored: List[NewsArticle] = []
    for article in articles:
        text = " ".join(filter(None, [article.headline, article.summary]))
        label, score = _score_text(text)
        scored.append(
            NewsArticle(
                headline=article.headline,
                published_at=article.published_at,
                summary=article.summary,
                source=article.source,
                url=article.url,
                sentiment=label,
                sentiment_score=score,
            )
        )
    scored.sort(key=lambda a: (a.sentiment_score, a.headline.lower()), reverse=True)
    return scored


def _build_opinion(weight: float, articles: List[NewsArticle]) -> Tuple[str, List[str]]:
    if not articles:
        judgement = (
            "No recent vendor news was available; maintain the current allocation until coverage improves."
        )
        return judgement, ["Absence of fresh headlines keeps the allocation decision data-light."]

    positives = [a for a in articles if a.sentiment_score > 0]
    negatives = [a for a in articles if a.sentiment_score < 0]
    net_score = sum(a.sentiment_score for a in articles)

    stance = _news_stance(net_score, len(positives), len(negatives), len(articles))
    weight_pct = weight * 100.0

    key_positive = positives[0] if positives else None
    key_negative = negatives[0] if negatives else None

    if stance == "support":
        anchor = _headline_snippet(key_positive or positives[:1] or articles[:1])
        judgement = (
            f"Maintain the {weight_pct:.1f}% allocation; upbeat coverage such as {anchor} reinforces the weight."
        )
    elif stance == "caution":
        anchor = _headline_snippet(key_negative or negatives[:1] or articles[:1])
        judgement = (
            f"Trim or closely monitor the {weight_pct:.1f}% allocation because bearish news like {anchor} challenges the position."
        )
    else:
        anchor = _headline_snippet(key_positive or key_negative or articles[:1])
        judgement = (
            f"Keep the {weight_pct:.1f}% allocation but note mixed coverage; {anchor} is the most material headline in the window."
        )

    supporting: List[str] = []
    coverage_summary = _coverage_summary(len(positives), len(negatives), len(articles))
    supporting.append(coverage_summary)

    for idx, article in enumerate(_top_articles(positives, negatives)):
        tone = "positive" if article.sentiment_score > 0 else "negative"
        snippet = _headline_snippet(article)
        if tone == "positive":
            supporting.append(
                f"Positive driver: {snippet} — supports holding {weight_pct:.1f}% exposure."
            )
        else:
            supporting.append(
                f"Risk flag: {snippet} — consider hedging part of the {weight_pct:.1f}% weight."
            )
        if len(supporting) >= 3:
            break

    if len(supporting) < 3 and not negatives and positives:
        supporting.append(
            "Coverage skewed positive during the window with no headline-level risks identified."
        )
    if len(supporting) < 3 and not positives and negatives:
        supporting.append(
            "Coverage skewed negative during the window with no offsetting positive headlines."
        )

    return judgement, supporting


def _top_articles(positives: List[NewsArticle], negatives: List[NewsArticle]) -> Iterable[NewsArticle]:
    ordered = sorted(positives, key=lambda a: -a.sentiment_score) + sorted(
        negatives, key=lambda a: a.sentiment_score
    )
    return ordered


def _coverage_summary(pos_count: int, neg_count: int, total: int) -> str:
    if total == 0:
        return "Coverage volume was negligible over the review window."
    neutral_count = total - pos_count - neg_count
    return (
        f"News tone snapshot: {pos_count} positive, {neg_count} negative, {neutral_count} neutral items in the sample."
    )


def _headline_snippet(article_or_list: Any) -> str:
    if isinstance(article_or_list, list):
        target = article_or_list[0] if article_or_list else None
    else:
        target = article_or_list
    if not isinstance(target, NewsArticle):
        return "recent coverage"
    date_str = target.published_at.split("T")[0] if target.published_at else "recent"
    source = target.source or "press"
    return f"{source} on {date_str}: '{target.headline}'"


def _news_stance(net_score: int, pos_count: int, neg_count: int, total: int) -> str:
    if total == 0:
        return "mixed"
    pos_ratio = pos_count / total
    neg_ratio = neg_count / total
    if net_score >= 1 and pos_ratio >= 0.5:
        return "support"
    if net_score <= -1 and neg_ratio >= 0.4:
        return "caution"
    return "mixed"


def _articles_prompt_digest(articles: List[NewsArticle]) -> str:
    lines: List[str] = []
    for article in articles:
        tone = article.sentiment
        date_str = article.published_at or "recent"
        source = article.source or "vendor"
        summary = article.summary or "(no summary provided)"
        lines.append(
            f"- {tone.title()} | {source} | {date_str}: {article.headline} — {summary}"
        )
    return "\n".join(lines)


def _compose_weight_statement(
    weight: float, net_score: int, pos_count: int, neg_count: int
) -> str:
    weight_pct = weight * 100.0
    return (
        f"Headline sentiment score: {net_score} ({pos_count} positive / {neg_count} negative) alongside a {weight_pct:.1f}% allocation."
    )


def _score_text(text: str) -> Tuple[str, int]:
    cleaned = text.strip()
    if not cleaned:
        return "neutral", 0

    if _SENTIMENT_ANALYSER is None:
        return "neutral", 0

    compound = _SENTIMENT_ANALYSER.polarity_scores(cleaned)["compound"]
    if compound >= 0.1:
        return "positive", 1
    if compound <= -0.1:
        return "negative", -1
    return "neutral", 0


def _extract_publish_datetime(item: dict) -> Optional[datetime]:
    raw = item.get("providerPublishTime")
    if raw is not None:
        try:
            return datetime.fromtimestamp(int(raw), tz=timezone.utc)
        except (OSError, OverflowError, TypeError, ValueError):
            pass

    for key in ("pubDate", "publishedAt", "date", "time"):
        raw_value = item.get(key)
        if raw_value is None:
            continue
        if isinstance(raw_value, (int, float)):
            try:
                return datetime.fromtimestamp(float(raw_value), tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                continue
        if isinstance(raw_value, str):
            candidate = raw_value.strip()
            if not candidate:
                continue
            if candidate.isdigit():
                try:
                    return datetime.fromtimestamp(int(candidate), tz=timezone.utc)
                except (OSError, OverflowError, ValueError):
                    continue
            try:
                return datetime.fromisoformat(candidate.replace("Z", "+00:00"))
            except ValueError:
                continue
    return None


def _strip_html(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value)


def _deduplicate_articles(articles: List[NewsArticle]) -> List[NewsArticle]:
    seen: set[str] = set()
    deduped: List[NewsArticle] = []
    for article in articles:
        key = article.headline.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(article)
    return deduped


def _format_articles_table(articles: List[NewsArticle]) -> str:
    header = "| Date | Source | Tone | Headline |\n| --- | --- | --- | --- |"
    rows = []
    for article in articles:
        date_str = article.published_at.split("T")[0] if article.published_at else "--"
        source = article.source or "--"
        tone = article.sentiment
        headline = article.headline.replace("|", "/")
        rows.append(f"| {date_str} | {source} | {tone} | {headline} |")
    return "\n".join([header] + rows)
