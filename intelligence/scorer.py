"""
Signal Strength Scorer — Evidence-Based Confluence Engine

Produces a 0–100 confidence score for every trade signal.
The system requires EVIDENCE, not guesses.

Architecture:

  Layer 0 — Manipulation Gate    (BLOCK if triggered)
    • Stop hunts, fake breakouts, bull/bear traps
    • Pump & dump, churn (hidden distribution)
    • Volume-price divergence
    → If ≥2 manipulation signals detected → trade is BLOCKED entirely

  Layer 1 — Smart Money Evidence (max 25 pts)
    • Wyckoff phase (accumulation Spring/SOS or distribution UTAD/SOW)
    • SMC order block test (price at unmitigated OB)
    • Liquidity sweep + confirmation (stop hunt before real move)
    • Fair Value Gap alignment

  Layer 2 — Technical Confirmation (max 30 pts)
    • EMA stack alignment
    • RSI position
    • MACD histogram
    • Supertrend direction
    • Volume confirmation

  Layer 3 — News Sentiment        (max 20 pts)
    • FinBERT symbol-specific headline sentiment

  Layer 4 — Global Market Context (max 15 pts)
    • Gift NIFTY, US markets, India VIX, FII/DII flow

  Layer 5 — Options & Momentum   (max 10 pts)
    • NSE live PCR (contrarian interpretation)
    • 5-day price momentum

Signal Strength Labels:
  90–100 → STRONG BUY / STRONG SELL  (strong evidence, full size)
  70–89  → BUY / SELL                (confirmed, full size)
  50–69  → WEAK BUY / WEAK SELL      (marginal evidence, half size)
  < 50   → SKIP                      (insufficient evidence)
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd
import structlog

log = structlog.get_logger(__name__)


class SignalStrength(str, Enum):
    STRONG_BUY   = "STRONG_BUY"    # 90–100
    BUY          = "BUY"           # 70–89
    WEAK_BUY     = "WEAK_BUY"      # 50–69
    VERY_WEAK    = "VERY_WEAK"     # 30–49
    COUNTER      = "COUNTER"       # 0–29 (signal contradicted)
    STRONG_SELL  = "STRONG_SELL"   # 90–100 (sell side)
    SELL         = "SELL"          # 70–89
    WEAK_SELL    = "WEAK_SELL"     # 50–69


@dataclass
class ScoreBreakdown:
    # Layer scores (raw points)
    smart_money_pts: float = 0.0    # max 25 (Wyckoff + SMC + manipulation)
    technical_pts: float = 0.0      # max 30
    news_pts: float = 0.0           # max 20
    context_pts: float = 0.0        # max 15
    momentum_pts: float = 0.0       # max 10

    # Smart Money evidence
    wyckoff_phase: str = "Unknown"
    wyckoff_bias: Optional[str] = None
    wyckoff_confidence: float = 0.0
    smc_bullish_signals: int = 0
    smc_bearish_signals: int = 0
    liquidity_swept: Optional[str] = None
    at_order_block: bool = False
    manipulation_score: int = 0
    manipulation_blocked: bool = False
    operator_direction: Optional[str] = None

    # Technical components
    ema_aligned: bool = False
    rsi_value: float = 50.0
    rsi_favorable: bool = False
    macd_aligned: bool = False
    supertrend_aligned: bool = False
    volume_above_avg: bool = False
    price_above_vwap: bool = False

    news_score: float = 0.0         # -1 to +1
    market_mood: str = "neutral"
    context_score: float = 0.0      # -1 to +1
    vix_level: float = 15.0

    pcr: float = 1.0                # Put-Call Ratio
    momentum_5d: float = 0.0        # 5-day price change %

    # Warnings / blockers
    warnings: list[str] = field(default_factory=list)


@dataclass
class ScoredSignal:
    symbol: str
    action: str                    # BUY or SELL
    score: int                     # 0–100
    strength: SignalStrength
    size_multiplier: float         # 0.0–1.0 — scale order size by this
    breakdown: ScoreBreakdown
    rationale: str                 # Human-readable explanation
    skip: bool = False             # True if score too low to trade


class SignalScorer:
    """
    Scores any trade signal by gathering technical, news, and global context.
    All async — designed to complete in < 2 seconds during market hours.
    """

    # Minimum score to allow a trade
    MIN_SCORE_TO_TRADE = 50

    def __init__(self) -> None:
        pass

    async def score(
        self,
        symbol: str,
        action: str,
        strategy: str = "default",
        df: Optional[pd.DataFrame] = None,
    ) -> ScoredSignal:
        """
        Score a signal for the given symbol and action.

        Args:
            symbol: NSE symbol e.g. RELIANCE
            action: BUY or SELL
            strategy: strategy name (for strategy-specific config)
            df: pre-loaded OHLCV DataFrame (fetched if None)
        """
        breakdown = ScoreBreakdown()
        is_buy = action.upper() == "BUY"

        # Fetch all data in parallel
        df_task = self._get_ohlcv(symbol, df)
        sentiment_task = self._get_sentiment(symbol)
        context_task = self._get_context()
        pcr_task = self._get_pcr_async(symbol)

        ohlcv, sentiment, context, pcr = await asyncio.gather(
            df_task, sentiment_task, context_task, pcr_task,
            return_exceptions=True,
        )

        breakdown.pcr = float(pcr) if isinstance(pcr, (int, float)) else 1.0

        # ── Layer 0: Manipulation Gate ────────────────────────────────────
        # Run BEFORE scoring. If manipulation confirmed → block entirely.
        if isinstance(ohlcv, pd.DataFrame) and not ohlcv.empty:
            breakdown.smart_money_pts = self._score_smart_money(ohlcv, is_buy, breakdown)
        else:
            log.warning("scorer.no_data", symbol=symbol)
            breakdown.warnings.append("Price data unavailable — no smart money analysis")

        if breakdown.manipulation_blocked:
            score = 0
            strength = SignalStrength.COUNTER
            return ScoredSignal(
                symbol=symbol,
                action=action.upper(),
                score=0,
                strength=SignalStrength.COUNTER,
                size_multiplier=0.0,
                breakdown=breakdown,
                rationale=self._build_rationale(0, SignalStrength.COUNTER, breakdown, is_buy),
                skip=True,
            )

        # ── Layer 1: Technical (max 30 pts) ───────────────────────────────
        if isinstance(ohlcv, pd.DataFrame) and not ohlcv.empty:
            breakdown.technical_pts = self._score_technical(ohlcv, is_buy, breakdown)
        else:
            breakdown.technical_pts = 15.0  # neutral default
            breakdown.warnings.append("Technical data unavailable — using neutral score")

        # ── Layer 2: News Sentiment (max 20 pts) ──────────────────────────
        if not isinstance(sentiment, Exception):
            breakdown.news_pts = self._score_news(sentiment, is_buy, breakdown)
        else:
            breakdown.news_pts = 10.0  # neutral
            breakdown.warnings.append(f"News fetch failed: {sentiment}")

        # ── Layer 3: Global Context (max 15 pts) ──────────────────────────
        if not isinstance(context, Exception):
            breakdown.context_pts = self._score_context(context, is_buy, breakdown)
        else:
            breakdown.context_pts = 7.5  # neutral
            breakdown.warnings.append(f"Context fetch failed: {context}")

        # ── Layer 4: Momentum + PCR (max 10 pts) ──────────────────────────
        if isinstance(ohlcv, pd.DataFrame) and not ohlcv.empty:
            breakdown.momentum_pts = self._score_momentum(ohlcv, is_buy, breakdown)
        else:
            breakdown.momentum_pts = 5.0  # neutral

        # Aggregate
        total = (
            breakdown.smart_money_pts
            + breakdown.technical_pts
            + breakdown.news_pts
            + breakdown.context_pts
            + breakdown.momentum_pts
        )
        score = int(round(min(100, max(0, total))))
        strength = self._classify_strength(score, is_buy)
        size_mult = self._size_multiplier(score)

        result = ScoredSignal(
            symbol=symbol,
            action=action.upper(),
            score=score,
            strength=strength,
            size_multiplier=size_mult,
            breakdown=breakdown,
            rationale=self._build_rationale(score, strength, breakdown, is_buy),
            skip=score < self.MIN_SCORE_TO_TRADE,
        )

        log.info(
            "scorer.result",
            symbol=symbol,
            action=action,
            score=score,
            strength=strength.value,
            skip=result.skip,
        )
        return result

    # ── Layer 0: Smart Money + Manipulation Gate ──────────────────────────

    def _score_smart_money(self, df: pd.DataFrame, is_buy: bool, bd: ScoreBreakdown) -> float:
        """
        Score smart money evidence (max 25 pts) and run manipulation gate.
        If manipulation_blocked=True, the caller returns score=0 immediately.

        Points breakdown:
          Wyckoff phase aligned (Phase C or D):  10 pts
          SMC confluence (order block, FVG, sweep): up to 8 pts (3 each, cap 8)
          Manipulation penalty:  -10 pts per confirmed signal (blocks at 2+)
        """
        from intelligence.wyckoff import WyckoffAnalyzer
        from intelligence.smart_money import SMCAnalyzer
        from intelligence.manipulation import ManipulationDetector

        pts = 0.0

        # ── Manipulation check first (gate) ──
        manip = ManipulationDetector().detect(df)
        bd.manipulation_score = manip.manipulation_score
        bd.operator_direction = manip.operator_direction
        bd.warnings.extend(manip.warnings)

        if manip.block_trade:
            bd.manipulation_blocked = True
            bd.warnings.insert(0,
                f"MANIPULATION DETECTED ({manip.manipulation_score} signals) — trade blocked: "
                + "; ".join(w.split("]")[0][1:] for w in manip.warnings[:3])
            )
            return 0.0

        # Small penalty per warning (not enough to block alone)
        pts -= manip.manipulation_score * 3.0

        # ── Wyckoff phase ──
        wyckoff = WyckoffAnalyzer().analyze(df)
        bd.wyckoff_phase = wyckoff.phase.value
        bd.wyckoff_bias = wyckoff.trade_bias
        bd.wyckoff_confidence = wyckoff.confidence
        bd.warnings.extend(wyckoff.reasoning[:2])  # Top 2 reasons in breakdown

        if wyckoff.trade_bias == ("bullish" if is_buy else "bearish"):
            pts += 10.0 * wyckoff.confidence  # Max 10 pts, scaled by phase confidence
        elif wyckoff.trade_bias == "wait":
            pts += 2.0   # Neutral — in range but no clear trigger yet
        elif wyckoff.trade_bias == "avoid":
            pts += 0.0
            bd.warnings.append(f"Wyckoff phase unidentifiable — no structural edge")
        elif wyckoff.trade_bias not in (None, "bullish", "bearish"):
            pts += 0.0

        # Bearish divergence warning for buys
        if is_buy and wyckoff.pv_divergence == "bearish":
            bd.warnings.append("P/V bearish divergence — distribution in progress, avoid longs")
            pts -= 4.0

        # ── SMC (Smart Money Concepts) ──
        smc = SMCAnalyzer().analyze(df)
        bd.smc_bullish_signals = smc.bullish_confluence
        bd.smc_bearish_signals = smc.bearish_confluence
        bd.liquidity_swept = smc.liquidity_swept
        bd.at_order_block = smc.price_at_bullish_ob if is_buy else smc.price_at_bearish_ob

        if is_buy:
            smc_pts = 0.0
            if smc.price_at_bullish_ob:
                smc_pts += 4.0   # Price at unmitigated demand zone
            if smc.price_in_bullish_fvg:
                smc_pts += 3.0   # Inside unfilled imbalance (magnet)
            if smc.liquidity_swept == "sell_side" and smc.sweep_confirmed:
                smc_pts += 4.0   # Stop hunt below confirmed — operators bought
            if smc.bos_direction == "bullish":
                smc_pts += 2.0   # Structure confirmed bullish
            if smc.in_discount_zone:
                smc_pts += 2.0   # Price in wholesale zone
            # Counter-signals penalty
            smc_pts -= smc.bearish_confluence * 1.5
            pts += max(0.0, min(8.0, smc_pts))

            # Summary
            if smc.evidence_summary:
                bd.warnings.append("SMC: " + " | ".join(smc.evidence_summary[:3]))

        else:  # SELL
            smc_pts = 0.0
            if smc.price_at_bearish_ob:
                smc_pts += 4.0
            if smc.price_in_bearish_fvg:
                smc_pts += 3.0
            if smc.liquidity_swept == "buy_side" and smc.sweep_confirmed:
                smc_pts += 4.0
            if smc.bos_direction == "bearish":
                smc_pts += 2.0
            if smc.in_premium_zone:
                smc_pts += 2.0
            smc_pts -= smc.bullish_confluence * 1.5
            pts += max(0.0, min(8.0, smc_pts))

        # CHoCH detected (early reversal signal)
        if smc.choch_detected:
            if smc.choch_direction == ("bullish" if is_buy else "bearish"):
                pts += 2.0
                bd.warnings.append(f"CHoCH {smc.choch_direction} detected — first structure reversal")
            else:
                pts -= 2.0  # CHoCH against our direction

        return min(25.0, max(0.0, pts))

    # ── Layer 1: Technical Analysis ───────────────────────────────────────

    def _score_technical(self, df: pd.DataFrame, is_buy: bool, bd: ScoreBreakdown) -> float:
        try:
            import pandas_ta as ta
        except ImportError:
            from utils import ta_compat as ta
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        pts = 0.0

        # EMA stack alignment (12 pts)
        ema9 = ta.ema(close, 9)
        ema21 = ta.ema(close, 21)
        ema50 = ta.ema(close, 50)
        ema200 = ta.ema(close, 200)
        price = close.iloc[-1]

        if all(x is not None for x in [ema9.iloc[-1], ema21.iloc[-1], ema50.iloc[-1]]):
            e9, e21, e50 = ema9.iloc[-1], ema21.iloc[-1], ema50.iloc[-1]
            if is_buy:
                bd.ema_aligned = bool(e9 > e21 > e50 and price > e50)
            else:
                bd.ema_aligned = bool(e9 < e21 < e50 and price < e50)
            if bd.ema_aligned:
                pts += 12.0
            elif (is_buy and e9 > e21) or (not is_buy and e9 < e21):
                pts += 6.0  # partial alignment

        # Add long-term trend bonus (3 pts)
        if ema200.iloc[-1] is not None:
            e200 = ema200.iloc[-1]
            if (is_buy and price > e200) or (not is_buy and price < e200):
                pts += 3.0

        # RSI (8 pts)
        rsi = ta.rsi(close, 14)
        if rsi.iloc[-1] is not None:
            bd.rsi_value = float(rsi.iloc[-1])
            if is_buy:
                # Favorable: oversold recovery (30–55) or strong momentum (55–70)
                if 30 <= bd.rsi_value <= 55:
                    bd.rsi_favorable = True
                    pts += 8.0
                elif 55 < bd.rsi_value <= 70:
                    bd.rsi_favorable = True
                    pts += 5.0
                elif bd.rsi_value > 70:
                    bd.warnings.append(f"RSI overbought ({bd.rsi_value:.0f}) — risky buy")
                    pts += 1.0
            else:  # SELL
                if 45 <= bd.rsi_value <= 70:
                    bd.rsi_favorable = True
                    pts += 8.0
                elif 70 < bd.rsi_value:
                    bd.rsi_favorable = True
                    pts += 10.0  # overbought = strong sell signal
                elif bd.rsi_value < 30:
                    bd.warnings.append(f"RSI oversold ({bd.rsi_value:.0f}) — risky sell")

        # MACD (7 pts)
        macd = ta.macd(close)
        if macd is not None and not macd.empty:
            macd_cols = [c for c in macd.columns if c.startswith("MACD_")]
            sig_cols = [c for c in macd.columns if c.startswith("MACDs_")]
            hist_cols = [c for c in macd.columns if c.startswith("MACDh_")]
            if macd_cols and sig_cols and hist_cols:
                m_val = macd[macd_cols[0]].iloc[-1]
                s_val = macd[sig_cols[0]].iloc[-1]
                h_val = macd[hist_cols[0]].iloc[-1]
                h_prev = macd[hist_cols[0]].iloc[-2] if len(macd) > 1 else 0
                if is_buy:
                    bd.macd_aligned = bool(m_val > s_val and h_val > 0 and h_val > h_prev)
                else:
                    bd.macd_aligned = bool(m_val < s_val and h_val < 0 and h_val < h_prev)
                pts += 7.0 if bd.macd_aligned else (3.5 if (is_buy and m_val > s_val) or (not is_buy and m_val < s_val) else 0)

        # Supertrend (5 pts)
        try:
            st = ta.supertrend(high, low, close)
            if st is not None:
                dir_cols = [c for c in st.columns if c.startswith("SUPERTd")]
                if dir_cols:
                    direction = st[dir_cols[0]].iloc[-1]
                    bd.supertrend_aligned = bool((is_buy and direction == 1) or (not is_buy and direction == -1))
                    pts += 5.0 if bd.supertrend_aligned else 0.0
        except Exception:
            pass

        # Volume confirmation (5 pts)
        avg_vol = volume.rolling(20).mean().iloc[-1]
        curr_vol = volume.iloc[-1]
        if avg_vol and curr_vol > 0:
            bd.volume_above_avg = bool(curr_vol > avg_vol * 1.2)
            pts += 5.0 if bd.volume_above_avg else 0.0

        return min(30.0, pts)

    # ── Layer 2: News Sentiment ───────────────────────────────────────────

    def _score_news(self, sentiment, is_buy: bool, bd: ScoreBreakdown) -> float:
        from intelligence.news import SymbolSentiment
        if not isinstance(sentiment, SymbolSentiment):
            return 12.5

        bd.news_score = sentiment.score
        pts = 0.0

        # Symbol-specific news (max 15 pts)
        if is_buy:
            pts += max(0.0, (sentiment.score + 1) / 2) * 15
        else:
            pts += max(0.0, (1 - sentiment.score) / 2) * 15

        # Article volume bonus — more coverage = higher confidence (max 5 pts)
        if sentiment.article_count >= 10:
            pts += 5.0
        elif sentiment.article_count >= 5:
            pts += 3.0
        elif sentiment.article_count >= 2:
            pts += 1.5

        # Warning if sentiment strongly contradicts signal direction
        if is_buy and sentiment.score < -0.5:
            bd.warnings.append(f"News strongly bearish ({sentiment.score:.2f}) — contradicts BUY")
        if not is_buy and sentiment.score > 0.5:
            bd.warnings.append(f"News strongly bullish ({sentiment.score:.2f}) — contradicts SELL")

        return min(20.0, pts)

    # ── Layer 3: Global Market Context ───────────────────────────────────

    def _score_context(self, context, is_buy: bool, bd: ScoreBreakdown) -> float:
        from intelligence.market_context import GlobalContext
        if not isinstance(context, GlobalContext):
            return 10.0

        bd.context_score = context.context_score
        bd.vix_level = context.india_vix

        # Convert -1..+1 context score to 0..15 pts
        if is_buy:
            pts = (context.context_score + 1) / 2 * 15
        else:
            pts = (1 - context.context_score) / 2 * 15

        # ── India VIX calibration via IndiaVIXLevel thresholds ───────────
        from intelligence.knowledge_base import IndiaVIXLevel
        vix = context.india_vix
        if vix > IndiaVIXLevel.CRISIS:
            bd.warnings.append(f"VIX CRISIS ({vix:.1f}) — extreme risk, skip entries")
            pts *= 0.2
        elif vix > IndiaVIXLevel.HIGH:
            bd.warnings.append(f"VIX very high ({vix:.1f}) — 40% size only")
            pts *= 0.4
        elif vix > IndiaVIXLevel.ELEVATED:
            bd.warnings.append(f"VIX elevated ({vix:.1f}) — reduce size")
            pts *= 0.7
        elif vix < IndiaVIXLevel.COMPLACENCY:
            bd.warnings.append(f"VIX extremely low ({vix:.1f}) — complacency risk")
            pts *= 0.85

        return min(15.0, max(0.0, pts))

    # ── Layer 4: Momentum & Options ───────────────────────────────────────

    def _score_momentum(self, df: pd.DataFrame, is_buy: bool, bd: ScoreBreakdown) -> float:
        close = df["close"]
        pts = 0.0

        # 5-day momentum (10 pts)
        if len(close) >= 5:
            momentum = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100
            bd.momentum_5d = float(momentum)
            if is_buy and momentum > 0:
                pts += min(10.0, momentum * 1.5)
            elif not is_buy and momentum < 0:
                pts += min(10.0, abs(momentum) * 1.5)

        # PCR from live NSE option chain (5 pts)
        # Source: intelligence/nse_options.py — NSE official option chain API
        # PCR is CONTRARIAN: high PCR (>1.3) = bearish sentiment = bullish signal
        # bd.pcr is set externally by scorer.score() via _get_pcr_async()
        from intelligence.knowledge_base import PCRLevel
        pcr = bd.pcr if bd.pcr > 0 else 1.0

        if is_buy:
            if pcr > PCRLevel.STRONGLY_BULLISH:    # Extreme put buying = panic = contrarian BUY
                pts += 4.0
                bd.warnings.append(f"PCR {pcr:.2f} — extreme put buying, contrarian buy signal")
            elif pcr > PCRLevel.MILDLY_BULLISH:    # Mild put dominance = support
                pts += 2.5
            elif pcr < PCRLevel.STRONGLY_BEARISH:  # Too many calls = crowded long = caution
                pts += 0.0
                bd.warnings.append(f"PCR {pcr:.2f} — excessive call buying, caution on longs")
            else:
                pts += 1.5   # Neutral zone
        else:  # SELL
            if pcr < PCRLevel.STRONGLY_BEARISH:    # Excessive calls = crowded long = SELL signal
                pts += 4.0
            elif pcr < PCRLevel.MILDLY_BEARISH:
                pts += 2.0
            else:
                pts += 1.0

        return min(10.0, max(0.0, pts))

    async def _get_pcr_async(self, symbol: str) -> float:
        """Fetch real PCR from NSE option chain. Returns 1.0 (neutral) on failure."""
        try:
            from intelligence.nse_options import NSEOptionChainFetcher
            # Map individual stock to NIFTY for index PCR (most relevant for market bias)
            index_symbol = "NIFTY" if symbol not in {"BANKNIFTY", "FINNIFTY", "NIFTY"} else symbol
            fetcher = NSEOptionChainFetcher()
            metrics = await fetcher.get_metrics(index_symbol)
            return metrics.pcr_oi
        except Exception as e:
            log.warning("scorer.pcr_fetch_failed", symbol=symbol, error=str(e))
            return 1.0

    # ── Helpers ───────────────────────────────────────────────────────────

    async def _get_ohlcv(self, symbol: str, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if df is not None:
            return df
        from data.historical import HistoricalDataFetcher
        from datetime import datetime, timedelta
        fetcher = HistoricalDataFetcher()
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=250)).strftime("%Y-%m-%d")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, fetcher.get_with_indicators, symbol, from_date, to_date, None
        )

    async def _get_sentiment(self, symbol: str):
        # Use FinBERT if available, fall back to lexicon
        try:
            from intelligence.finbert_sentiment import AIEnhancedNewsSentiment
            analyzer = AIEnhancedNewsSentiment()
            return await analyzer.get_symbol_sentiment(symbol)
        except Exception:
            from intelligence.news import NewsSentimentAnalyzer
            analyzer = NewsSentimentAnalyzer()
            return await analyzer.get_symbol_sentiment(symbol)

    async def _get_context(self):
        from intelligence.market_context import MarketContextTracker
        tracker = MarketContextTracker()
        return await tracker.get_context()

    @staticmethod
    def _classify_strength(score: int, is_buy: bool) -> SignalStrength:
        if is_buy:
            if score >= 90: return SignalStrength.STRONG_BUY
            if score >= 70: return SignalStrength.BUY
            if score >= 50: return SignalStrength.WEAK_BUY
            return SignalStrength.VERY_WEAK
        else:
            if score >= 90: return SignalStrength.STRONG_SELL
            if score >= 70: return SignalStrength.SELL
            if score >= 50: return SignalStrength.WEAK_SELL
            return SignalStrength.VERY_WEAK

    @staticmethod
    def _size_multiplier(score: int) -> float:
        """Scale order size by confidence."""
        if score >= 90: return 1.0
        if score >= 70: return 0.75
        if score >= 50: return 0.5
        return 0.0  # skip

    @staticmethod
    def _build_rationale(score: int, strength: SignalStrength, bd: ScoreBreakdown, is_buy: bool) -> str:
        parts = [f"Score {score}/100 → {strength.value}"]

        if bd.manipulation_blocked:
            parts.append(f"BLOCKED: Manipulation detected ({bd.manipulation_score} signals)")
            parts.extend(bd.warnings[:3])
            return " | ".join(parts)

        # Smart money layer
        sm_parts = [f"Wyckoff:{bd.wyckoff_phase}({bd.wyckoff_confidence:.0%})"]
        if bd.at_order_block:
            sm_parts.append("OB✓")
        if bd.liquidity_swept:
            sm_parts.append(f"Sweep:{bd.liquidity_swept}")
        if bd.manipulation_score > 0:
            sm_parts.append(f"ManipWarn:{bd.manipulation_score}")
        parts.append(f"SmartMoney: {bd.smart_money_pts:.0f}/25 pts [{' '.join(sm_parts)}]")

        parts.append(f"Technical: {bd.technical_pts:.0f}/30 pts"
                     + (f" [EMA✓]" if bd.ema_aligned else "")
                     + (f" [RSI {bd.rsi_value:.0f}]" if bd.rsi_favorable else f" [RSI {bd.rsi_value:.0f}✗]")
                     + (f" [MACD✓]" if bd.macd_aligned else "")
                     + (f" [ST✓]" if bd.supertrend_aligned else "")
                     + (f" [Vol↑]" if bd.volume_above_avg else ""))
        news_dir = "bullish" if bd.news_score > 0 else "bearish"
        parts.append(f"News: {bd.news_pts:.0f}/20 pts [{news_dir} {bd.news_score:+.2f}]")
        ctx_dir = "supportive" if (is_buy and bd.context_score > 0) or (not is_buy and bd.context_score < 0) else "against"
        parts.append(f"Global: {bd.context_pts:.0f}/15 pts [context {ctx_dir}, VIX {bd.vix_level:.1f}]")
        parts.append(f"Momentum: {bd.momentum_pts:.0f}/10 pts [5d: {bd.momentum_5d:+.1f}%, PCR:{bd.pcr:.2f}]")
        if bd.warnings:
            # Show first 3 warnings only to keep rationale readable
            parts.append("Flags: " + "; ".join(bd.warnings[:3]))
        return " | ".join(parts)
