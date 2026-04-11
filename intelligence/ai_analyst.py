"""
AI Options Analyst — powered by OpenRouter (Google Gemma 4 with reasoning)

Every 30 min during market hours:
1. Collects market context + top scanner picks
2. Two-turn conversation: initial analysis → confirmation + JSON output
3. Parses structured OptionsSignal from response
4. Saves to Redis key "ai:options:latest" (TTL 45 min)
5. Sends Telegram notification
"""
from __future__ import annotations

import asyncio
import json
import re
import structlog
from dataclasses import dataclass, field
from typing import Optional

log = structlog.get_logger(__name__)

# Free models tried in order — first available wins (verified against OpenRouter /models)
MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",    # large, reliable
    "google/gemma-4-31b-it:free",                 # newest Gemma
    "nvidia/nemotron-3-super-120b-a12b:free",     # large reasoning model
    "openai/gpt-oss-120b:free",                   # GPT OSS
    "google/gemma-4-26b-a4b-it:free",             # original pick (often rate-limited)
    "nousresearch/hermes-3-llama-3.1-405b:free",  # huge fallback
]
BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class OptionsSignal:
    symbol: str
    action: str          # "BUY_CE" | "BUY_PE" | "SELL_CE" | "SELL_PE" | "HOLD"
    strike: Optional[float] = None
    expiry: str = "weekly"
    premium_est: Optional[float] = None
    target_pct: float = 30.0
    sl_pct: float = 15.0
    confidence: float = 0.0
    rationale: str = ""
    risk_factors: list[str] = field(default_factory=list)
    ai_reasoning_summary: str = ""


class AIOptionsAnalyst:
    """
    Uses OpenRouter + Gemma 4 with chain-of-thought reasoning to analyze F&O opportunities.

    Multi-turn: Turn 1 = think through market. Turn 2 = confirm + output JSON.
    """

    def __init__(self) -> None:
        from config.settings import settings
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            base_url=BASE_URL,
            api_key=settings.openrouter_api_key or "missing",
        )

    async def analyze(self, context_data: dict) -> list[OptionsSignal]:
        """Run 2-turn AI analysis. Tries MODELS in order until one succeeds."""
        prompt = self._build_prompt(context_data)

        for model in MODELS:
            try:
                return await self._analyze_with_model(model, prompt, context_data)
            except Exception as e:
                err = str(e)
                if "429" in err or "rate" in err.lower() or "temporarily" in err.lower():
                    log.warning("ai_analyst.model_rate_limited", model=model)
                    continue
                log.error("ai_analyst.model_failed", model=model, error=err)
                continue

        log.error("ai_analyst.all_models_failed")
        return []

    async def _analyze_with_model(self, model: str, prompt: str, context_data: dict) -> list[OptionsSignal]:
        """Run 2-turn conversation with a specific model."""
        try:
            # ── Turn 1: think through market ──────────────────────────────
            resp1 = await self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"reasoning": {"enabled": True}},
                timeout=90,
            )
            if not resp1.choices or resp1.choices[0].message is None:
                raise ValueError(f"Model {model} returned empty response")
            msg1 = resp1.choices[0].message
            if not msg1.content:
                raise ValueError(f"Model {model} returned None content (may be streaming-only)")
            reasoning_details = getattr(msg1, "reasoning_details", None)

            # ── Turn 2: confirm + output JSON ──────────────────────────────
            messages = [
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": msg1.content or "",
                    **({"reasoning_details": reasoning_details} if reasoning_details else {}),
                },
                {
                    "role": "user",
                    "content": (
                        "Review your analysis. Are you confident? Consider the risks.\n"
                        "Output ONLY valid JSON in exactly this format (no extra text before/after):\n"
                        '{"signals": [{"symbol": "NIFTY", "action": "BUY_CE", '
                        '"strike": 24000, "expiry": "weekly", "premium_est": 150, '
                        '"target_pct": 40, "sl_pct": 20, "confidence": 0.72, '
                        '"rationale": "brief reason", "risk_factors": ["VIX elevated", "global weak"]}]}\n'
                        "Use action=HOLD and empty signals array if no clear trade. Max 2 signals."
                    ),
                },
            ]

            resp2 = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                extra_body={"reasoning": {"enabled": True}},
                timeout=90,
            )
            final_content = resp2.choices[0].message.content or ""
            reasoning2 = getattr(resp2.choices[0].message, "reasoning_details", None)
            reasoning_summary = self._extract_reasoning_summary(reasoning_details or reasoning2)

            signals = self._parse_signals(final_content, reasoning_summary, context_data)
            log.info("ai_analyst.done", signals=len(signals), model=model)
            return signals

        except Exception as e:
            log.error("ai_analyst.model_error", model=model, error=str(e))
            raise  # let the model-fallback loop catch it

    def _build_prompt(self, ctx: dict) -> str:
        market = ctx.get("market_context", {})
        picks = ctx.get("scanner_picks", [])

        picks_text = "\n".join([
            f"  • {p.get('symbol', '?')}: score={p.get('score', 0)}, "
            f"action={p.get('action', '?')}, wyckoff={p.get('wyckoff_phase', '?')}, "
            f"breakout={p.get('breakout_type', 'none')}, confidence={p.get('confidence', 0):.2f}"
            for p in picks[:5]
        ]) or "  No scanner picks available yet."

        vix = market.get("india_vix", 15)
        vix_regime = (
            "HIGH (>20) — prefer SELLING options" if vix > 20
            else "MEDIUM (15-20) — balanced approach" if vix > 15
            else "LOW (<15) — aggressive option BUYING favored"
        )

        return f"""You are a professional NSE F&O (Futures & Options) trader with 15 years of experience.

\u2550\u2550\u2550 CURRENT MARKET CONDITIONS \u2550\u2550\u2550
Market Bias:      {market.get('bias', 'unknown').upper()} (score: {market.get('score', 0):+.2f})
India VIX:        {vix:.1f} \u2192 {vix_regime}
Gift Nifty:       {market.get('gift_nifty_change_pct', 0):+.2f}%
S&P 500:          {market.get('sp500_change_pct', 0):+.2f}%
Nasdaq:           {market.get('nasdaq_change_pct', 0):+.2f}%
FII Net:          \u20b9{market.get('fii_net_crore', 0):+.0f} Cr (positive = bullish)
DII Net:          \u20b9{market.get('dii_net_crore', 0):+.0f} Cr
USD/INR:          {market.get('usdinr', 84):.2f} (weaker INR = bearish for market)
Crude Oil:        ${market.get('crude_oil_usd', 80):.1f}/bbl

\u2550\u2550\u2550 TOP SCANNER PICKS \u2550\u2550\u2550
{picks_text}

\u2550\u2550\u2550 TASK \u2550\u2550\u2550
Analyze the complete market picture and recommend the best 1-2 F&O options trades.

DECISION FRAMEWORK:
1. Global cues (US markets + Gift Nifty) \u2192 morning gap direction
2. VIX level \u2192 option buying vs selling strategy
3. FII flow \u2192 institutional direction
4. Scanner picks \u2192 which stocks/indices have the clearest setups
5. Wyckoff phase + breakout confirmation \u2192 entry timing

OPTIONS STRATEGY GUIDE:
- BULLISH + VIX<15: BUY NIFTY CE or stock CE (ATM or 1-strike OTM)
- BEARISH + VIX<15: BUY NIFTY PE or stock PE
- HIGH VIX (>20): SELL CE/PE credit spreads, not naked buying
- Unclear/choppy: HOLD \u2014 no trade is a valid trade

For each signal provide: symbol (NIFTY/BANKNIFTY/NSE stock), action (BUY_CE/BUY_PE/SELL_CE/SELL_PE/HOLD), approximate strike, expiry (weekly/monthly), estimated premium range, target % gain on premium, SL % loss on premium, confidence 0-1.

Think through each factor systematically before concluding."""

    def _extract_reasoning_summary(self, reasoning_details) -> str:
        if not reasoning_details:
            return ""
        try:
            if isinstance(reasoning_details, list):
                parts = [r.get("thinking", "") if isinstance(r, dict) else str(r) for r in reasoning_details]
                full = " ".join(parts)
            else:
                full = str(reasoning_details)
            return full[:400]
        except Exception:
            return ""

    def _parse_signals(self, content: str, reasoning: str, context: dict) -> list[OptionsSignal]:
        signals = []

        # Try multiple JSON extraction strategies
        data = None
        # Strategy 1: find the outermost {...} block
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            candidate = content[start:end]
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # Strategy 2: extract json code block from markdown
        if data is None:
            md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if md_match:
                try:
                    data = json.loads(md_match.group(1))
                except json.JSONDecodeError:
                    pass

        if data and "signals" in data:
            for s in data["signals"]:
                action = s.get("action", "HOLD").upper()
                try:
                    signals.append(OptionsSignal(
                        symbol=s.get("symbol", "NIFTY").upper(),
                        action=action,
                        strike=float(s["strike"]) if s.get("strike") else None,
                        expiry=s.get("expiry", "weekly"),
                        premium_est=float(s["premium_est"]) if s.get("premium_est") else None,
                        target_pct=float(s.get("target_pct", 30)),
                        sl_pct=float(s.get("sl_pct", 15)),
                        confidence=float(s.get("confidence", 0.5)),
                        rationale=s.get("rationale", ""),
                        risk_factors=list(s.get("risk_factors", [])),
                        ai_reasoning_summary=reasoning,
                    ))
                except Exception as e:
                    log.warning("ai_analyst.signal_parse_error", error=str(e))
        elif data is None:
            log.warning("ai_analyst.json_parse_failed", content_snippet=content[:300])

        return [s for s in signals if s.action != "HOLD"]


async def run_ai_analysis_and_notify() -> None:
    """
    Full pipeline: collect data -> AI analysis -> save to Redis -> Telegram.
    Called by scheduler every 30 min.
    """
    from intelligence.market_context import MarketContextTracker
    from utils.notifications import Notifier

    try:
        # Collect market context
        tracker = MarketContextTracker()
        ctx = await tracker.get_context()
        market_data = {
            "bias": ctx.market_bias,
            "score": ctx.context_score,
            "india_vix": ctx.india_vix,
            "gift_nifty_change_pct": ctx.gift_nifty_change_pct,
            "sp500_change_pct": ctx.sp500_change_pct,
            "nasdaq_change_pct": ctx.nasdaq_change_pct,
            "fii_net_crore": ctx.fii_net_crore,
            "dii_net_crore": ctx.dii_net_crore,
            "usdinr": ctx.usdinr,
            "crude_oil_usd": ctx.crude_oil_usd,
        }

        # Collect scanner picks from Redis (cached)
        picks = []
        try:
            import redis.asyncio as aioredis
            from config.settings import settings
            import json as _json
            r = await aioredis.from_url(settings.redis_url, decode_responses=True)
            cached = await r.get("scanner:latest")
            if cached:
                scan_data = _json.loads(cached)
                picks = scan_data.get("top_picks", [])[:5]
        except Exception:
            pass

        # Run AI analysis
        analyst = AIOptionsAnalyst()
        signals = await analyst.analyze({
            "market_context": market_data,
            "scanner_picks": picks,
        })

        # Save to Redis
        try:
            import redis.asyncio as aioredis
            from config.settings import settings
            import time, json as _json
            r = await aioredis.from_url(settings.redis_url, decode_responses=True)
            payload = {
                "signals": [
                    {
                        "symbol": s.symbol,
                        "action": s.action,
                        "strike": s.strike,
                        "expiry": s.expiry,
                        "premium_est": s.premium_est,
                        "target_pct": s.target_pct,
                        "sl_pct": s.sl_pct,
                        "confidence": s.confidence,
                        "rationale": s.rationale,
                        "risk_factors": s.risk_factors,
                        "ai_reasoning_summary": s.ai_reasoning_summary,
                    }
                    for s in signals
                ],
                "market_context": market_data,
                "generated_at": time.time(),
                "model": MODELS[0],
            }
            await r.set("ai:options:latest", _json.dumps(payload), ex=2700)  # 45 min TTL
            log.info("ai_analyst.saved_to_redis", signals=len(signals))
        except Exception as e:
            log.warning("ai_analyst.redis_save_failed", error=str(e))

        # Send Telegram notification
        if signals:
            notifier = Notifier()
            await notifier.send_options_signals(signals, market_data)

    except Exception as e:
        log.error("ai_analyst.pipeline_failed", error=str(e), exc_info=True)
