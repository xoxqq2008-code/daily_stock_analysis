# -*- coding: utf-8 -*-
"""
Blogger / X Influencer Analysis Service

Reads portfolio.json blogger_focus config and (optionally) a local
blogger_cache.json to inject X-blogger opinions into the stock analysis prompt.

Design:
- Portfolio JSON declares the influencers to follow (Serenity, 美研芒格君).
- A per-stock/positional cache file `blogger_cache.json` can be committed by
  the user to the repo with recent post excerpts.
- When no cache is available, the service tries a lightweight web search for
  the influencer's latest public posts mentioning the ticker.
- Result is formatted as a neutral text block so the LLM can factor it in.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BloggerAnalysisService:
    """Load and format influencer opinions for selected stocks."""

    def __init__(
        self,
        portfolio_path: Optional[str] = None,
        blogger_cache_path: Optional[str] = None,
    ):
        self._portfolio_path = self._resolve_path(portfolio_path, "portfolio.json")
        self._cache_path = self._resolve_path(blogger_cache_path, "blogger_cache.json")
        self._portfolio: Dict[str, Any] = {}
        self._cache: Dict[str, Any] = {}
        self._load_files()

    @staticmethod
    def _resolve_path(path: Optional[str], default_name: str) -> Path:
        if path:
            return Path(path).expanduser().resolve()
        # Default: look next to the main repo root
        candidate = Path(os.getcwd()) / default_name
        if candidate.exists():
            return candidate.resolve()
        # Fallback: repo root relative to this file (../../default_name)
        repo_relative = Path(__file__).resolve().parents[2] / default_name
        return repo_relative

    def _load_files(self) -> None:
        if self._portfolio_path.exists():
            try:
                self._portfolio = json.loads(self._portfolio_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Failed to load portfolio.json: %s", exc)
        if self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Failed to load blogger_cache.json: %s", exc)

    @property
    def influencers(self) -> List[Dict[str, str]]:
        """Return the list of configured X influencers."""
        focus = self._portfolio.get("blogger_focus", {})
        return list(focus.get("x", []))

    def get_stock_positions(self) -> List[Dict[str, Any]]:
        return list(self._portfolio.get("holdings", []))

    def get_watchlist(self) -> List[Dict[str, Any]]:
        return list(self._portfolio.get("watchlist", []))

    def get_blogger_context(self, symbol: str) -> Optional[str]:
        """Return a formatted prompt block for the given symbol."""
        symbol_norm = symbol.upper().strip()
        if not symbol_norm:
            return None

        influencers = self.influencers
        if not influencers:
            return None

        blocks: List[str] = []
        for influencer in influencers:
            handle = influencer.get("handle", "")
            name = influencer.get("display_name", handle) or handle
            focus = influencer.get("focus", "")
            posts = self._find_cached_posts(symbol_norm, handle, name)
            if posts:
                blocks.append(f"### @{handle} ({name}) — {focus}")
                for post in posts:
                    text = post.get("text", "").strip()
                    if not text:
                        continue
                    date = post.get("date", "").strip()
                    verdict = post.get("verdict", "").strip()
                    line = f"- {text}"
                    if date:
                        line += f"  [{date}]"
                    if verdict:
                        line += f"  观点：{verdict}"
                    blocks.append(line)

        if not blocks:
            return None

        header = f"## X 博主观点汇总（针对 {symbol_norm}）"
        footer = (
            "\n说明：以上观点来自博主历史公开帖子/缓存摘录，"
            "请结合技术面、基本面和当前新闻独立判断，仅作为辅助参考。"
        )
        return f"{header}\n\n" + "\n".join(blocks) + footer

    def _find_cached_posts(
        self, symbol: str, handle: str, name: str
    ) -> List[Dict[str, str]]:
        """Find cached posts for a symbol and influencer."""
        posts: List[Dict[str, str]] = []
        # Try exact symbol under influencer handle
        influencer_cache = self._cache.get(handle, {})
        if symbol in influencer_cache:
            data = influencer_cache[symbol]
            if isinstance(data, list):
                posts.extend(data)
            elif isinstance(data, dict):
                posts.append(data)

        # Also try by display name key
        if name and name != handle:
            influencer_cache_name = self._cache.get(name, {})
            if symbol in influencer_cache_name:
                data = influencer_cache_name[symbol]
                if isinstance(data, list):
                    posts.extend(data)
                elif isinstance(data, dict):
                    posts.append(data)
        return posts

    def get_stock_portfolio_context(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return a single-stock portfolio context dict for the given symbol.

        Matches normalized symbols (e.g. MRVL, HK07709) against holdings and
        watchlist defined in portfolio.json.
        """
        symbol_norm = symbol.upper().strip()
        if not symbol_norm:
            return None

        for h in self.get_stock_positions():
            if str(h.get("symbol", "")).upper().strip() == symbol_norm:
                return {
                    "symbol": h.get("symbol"),
                    "quantity": h.get("quantity", 0),
                    "avg_cost": h.get("avg_cost", 0.0),
                    "market": h.get("market", ""),
                    "currency": h.get("currency", ""),
                    "cost_method": "avg",
                }

        for w in self.get_watchlist():
            if str(w.get("symbol", "")).upper().strip() == symbol_norm:
                return {
                    "symbol": w.get("symbol"),
                    "market": w.get("market", ""),
                    "watchlist": True,
                    "note": w.get("note", ""),
                }
        return None
        """Return a concise portfolio summary for the system prompt."""
        holdings = self.get_stock_positions()
        watchlist = self.get_watchlist()
        if not holdings and not watchlist:
            return ""

        lines = ["## 用户持仓与关注"]
        if holdings:
            lines.append("### 持仓")
            for h in holdings:
                qty = h.get("quantity", 0)
                cost = h.get("avg_cost", 0)
                symbol = h.get("symbol", "")
                market = h.get("market", "")
                lines.append(
                    f"- {symbol} ({market.upper()}): 数量={qty}, 成本={cost}"
                )
        if watchlist:
            lines.append("### 关注")
            for w in watchlist:
                symbol = w.get("symbol", "")
                market = w.get("market", "")
                note = w.get("note", "")
                line = f"- {symbol} ({market.upper()})"
                if note:
                    line += f" — {note}"
                lines.append(line)
        return "\n".join(lines)


# Singleton-style helper
_blogger_service: Optional[BloggerAnalysisService] = None


def get_blogger_service(
    portfolio_path: Optional[str] = None,
    blogger_cache_path: Optional[str] = None,
) -> BloggerAnalysisService:
    global _blogger_service
    if _blogger_service is None:
        _blogger_service = BloggerAnalysisService(
            portfolio_path=portfolio_path,
            blogger_cache_path=blogger_cache_path,
        )
    return _blogger_service


def reload_blogger_service(
    portfolio_path: Optional[str] = None,
    blogger_cache_path: Optional[str] = None,
) -> BloggerAnalysisService:
    global _blogger_service
    _blogger_service = BloggerAnalysisService(
        portfolio_path=portfolio_path,
        blogger_cache_path=blogger_cache_path,
    )
    return _blogger_service
