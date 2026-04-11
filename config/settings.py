from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(str, Enum):
    development = "development"
    staging = "staging"
    production = "production"


class BrokerName(str, Enum):
    zerodha = "zerodha"
    upstox = "upstox"
    angelone = "angelone"
    finvasia = "finvasia"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────
    app_env: AppEnv = AppEnv.development
    log_level: str = "INFO"
    paper_trading: bool = True

    # ── Security ─────────────────────────────────────────────────────────
    webhook_secret: str = Field(..., min_length=16)
    api_secret_key: str = Field(..., min_length=16)

    # ── Redis ────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Broker selection ─────────────────────────────────────────────────
    active_broker: BrokerName = BrokerName.zerodha

    # ── Zerodha ──────────────────────────────────────────────────────────
    zerodha_api_key: Optional[str] = None
    zerodha_api_secret: Optional[str] = None
    zerodha_access_token: Optional[str] = None
    zerodha_user_id: Optional[str] = None
    zerodha_password: Optional[str] = None
    zerodha_totp_secret: Optional[str] = None

    # ── Upstox ───────────────────────────────────────────────────────────
    upstox_api_key: Optional[str] = None
    upstox_api_secret: Optional[str] = None
    upstox_redirect_uri: str = "http://localhost:8000/broker/upstox/callback"
    upstox_access_token: Optional[str] = None

    # ── Angel One ────────────────────────────────────────────────────────
    angelone_api_key: Optional[str] = None
    angelone_client_id: Optional[str] = None
    angelone_password: Optional[str] = None
    angelone_totp_secret: Optional[str] = None

    # ── Finvasia ─────────────────────────────────────────────────────────
    finvasia_user_id: Optional[str] = None
    finvasia_password: Optional[str] = None
    finvasia_api_key: Optional[str] = None
    finvasia_vendor_code: Optional[str] = None
    finvasia_imei: Optional[str] = None

    # ── Risk ─────────────────────────────────────────────────────────────
    max_capital: float = 100_000.0
    risk_per_trade_pct: float = 1.0
    max_open_positions: int = 5
    daily_loss_limit_pct: float = 3.0
    max_position_size_pct: float = 20.0

    # ── Telegram ─────────────────────────────────────────────────────────
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    # ── OpenRouter AI ─────────────────────────────────────────────────────
    openrouter_api_key: Optional[str] = None

    # ── Data ─────────────────────────────────────────────────────────────
    nse_rate_limit_delay: float = 1.0

    # ── TradingView ──────────────────────────────────────────────────────
    tradingview_ip_whitelist: str = ""

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return v.upper()

    @property
    def is_production(self) -> bool:
        return self.app_env == AppEnv.production

    @property
    def tv_allowed_ips(self) -> list[str]:
        if not self.tradingview_ip_whitelist:
            return []
        return [ip.strip() for ip in self.tradingview_ip_whitelist.split(",") if ip.strip()]

    @property
    def daily_loss_limit_inr(self) -> float:
        return self.max_capital * self.daily_loss_limit_pct / 100

    @property
    def max_position_size_inr(self) -> float:
        return self.max_capital * self.max_position_size_pct / 100


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
