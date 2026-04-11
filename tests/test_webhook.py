"""Tests for TradingView webhook server and payload parser."""
from __future__ import annotations

import pytest
from httpx import AsyncClient, ASGITransport

from webhook.parser import TradingViewAlert


# ── Parser tests ──────────────────────────────────────────────────────────────

class TestTradingViewAlert:
    def test_parse_full_symbol(self):
        alert = TradingViewAlert(symbol="NSE:RELIANCE", action="BUY", qty=10)
        assert alert.parsed_symbol == "RELIANCE"
        assert alert.parsed_exchange == "NSE"

    def test_parse_plain_symbol(self):
        alert = TradingViewAlert(symbol="HDFCBANK", action="SELL")
        assert alert.parsed_symbol == "HDFCBANK"
        assert alert.parsed_exchange == "NSE"  # default

    def test_action_normalization(self):
        assert TradingViewAlert(symbol="X", action="buy").action == "BUY"
        assert TradingViewAlert(symbol="X", action="BUY_LONG").action == "BUY"
        assert TradingViewAlert(symbol="X", action="CLOSE_LONG").action == "SELL"

    def test_invalid_action(self):
        with pytest.raises(ValueError):
            TradingViewAlert(symbol="X", action="HOLD")

    def test_market_order_when_price_zero(self):
        alert = TradingViewAlert(symbol="NSE:TCS", action="BUY", price=0)
        assert alert.price == 0.0  # 0 means MARKET

    def test_symbol_uppercase(self):
        alert = TradingViewAlert(symbol="nse:reliance", action="buy")
        assert alert.parsed_symbol == "RELIANCE"


# ── Webhook endpoint tests ─────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestWebhookEndpoints:
    @pytest.fixture
    def app(self, monkeypatch):
        monkeypatch.setenv("WEBHOOK_SECRET", "test_secret_key_1234")
        monkeypatch.setenv("API_SECRET_KEY", "test_api_key_5678")
        monkeypatch.setenv("PAPER_TRADING", "true")
        monkeypatch.setenv("ACTIVE_BROKER", "zerodha")
        # Reset cached settings so env vars are picked up
        from config.settings import get_settings
        get_settings.cache_clear()
        new_settings = get_settings()
        # Patch the module-level settings singleton in the server module
        import webhook.server as _server
        monkeypatch.setattr(_server, "settings", new_settings)
        from webhook.server import create_app
        return create_app()

    async def test_health_check(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["paper_trading"] is True

    async def test_webhook_valid_token(self, app):
        payload = {"symbol": "NSE:RELIANCE", "action": "BUY", "qty": 1, "strategy": "ema_cross"}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/webhook/test_secret_key_1234", json=payload)
        assert resp.status_code == 200
        assert resp.json()["status"] == "queued"

    async def test_webhook_invalid_token(self, app):
        payload = {"symbol": "NSE:RELIANCE", "action": "BUY"}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/webhook/wrong_token", json=payload)
        assert resp.status_code == 401

    async def test_dry_run_endpoint(self, app):
        payload = {"symbol": "BSE:TCS", "action": "SELL", "qty": 5, "price": 3500}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/webhook/test_secret_key_1234/test", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "dry_run"
        assert data["parsed"]["symbol"] == "TCS"
        assert data["parsed"]["exchange"] == "BSE"
        assert data["parsed"]["order_type"] == "LIMIT"
