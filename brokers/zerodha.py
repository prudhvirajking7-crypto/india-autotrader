from __future__ import annotations

import pyotp
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from brokers.base import BrokerBase, Order, OrderSide, OrderStatus, Position, ProductType
from config.settings import settings

log = structlog.get_logger(__name__)


class ZerodhaBroker(BrokerBase):
    """Zerodha Kite Connect broker integration via pykiteconnect."""

    def __init__(self, paper_trading: bool = True) -> None:
        super().__init__(paper_trading)
        self._kite = None

    def login(self) -> None:
        from kiteconnect import KiteConnect

        api_key = settings.zerodha_api_key
        access_token = settings.zerodha_access_token

        if not api_key:
            raise ValueError("ZERODHA_API_KEY is not set")

        self._kite = KiteConnect(api_key=api_key)

        if access_token:
            # Use pre-generated access token (refreshed externally / daily script)
            self._kite.set_access_token(access_token)
            log.info("zerodha.login", method="access_token")
        else:
            # Auto-login via TOTP (requires password + TOTP secret)
            self._auto_login()

        self._logged_in = True
        log.info("zerodha.login.success", paper=self.paper_trading)

    def _auto_login(self) -> None:
        """
        Automate Zerodha login using requests + TOTP.
        Requires: ZERODHA_USER_ID, ZERODHA_PASSWORD, ZERODHA_TOTP_SECRET
        """
        import requests

        user_id = settings.zerodha_user_id
        password = settings.zerodha_password
        totp_secret = settings.zerodha_totp_secret

        if not all([user_id, password, totp_secret, settings.zerodha_api_key, settings.zerodha_api_secret]):
            raise ValueError("Auto-login requires ZERODHA_USER_ID, PASSWORD, TOTP_SECRET, API_KEY, API_SECRET")

        session = requests.Session()
        totp = pyotp.TOTP(totp_secret).now()

        # Step 1: credentials
        r = session.post(
            "https://kite.zerodha.com/api/login",
            data={"user_id": user_id, "password": password},
        )
        r.raise_for_status()
        request_id = r.json()["data"]["request_id"]

        # Step 2: TOTP 2FA
        r = session.post(
            "https://kite.zerodha.com/api/twofa",
            data={"request_id": request_id, "twofa_value": totp, "user_id": user_id},
        )
        r.raise_for_status()

        # Step 3: extract request_token from redirect
        r = session.get(
            f"https://kite.trade/connect/login?api_key={settings.zerodha_api_key}&v=3",
            allow_redirects=False,
        )
        from urllib.parse import parse_qs, urlparse
        loc = r.headers.get("Location", "")
        request_token = parse_qs(urlparse(loc).query).get("request_token", [None])[0]

        if not request_token:
            raise RuntimeError("Could not extract request_token from Zerodha login redirect")

        # Step 4: generate access token
        data = self._kite.generate_session(request_token, api_secret=settings.zerodha_api_secret)
        self._kite.set_access_token(data["access_token"])
        log.info("zerodha.autologin.success")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
    def _live_place_order(self, order: Order) -> Order:
        side_map = {OrderSide.BUY: self._kite.TRANSACTION_TYPE_BUY, OrderSide.SELL: self._kite.TRANSACTION_TYPE_SELL}
        product_map = {
            ProductType.MIS: self._kite.PRODUCT_MIS,
            ProductType.CNC: self._kite.PRODUCT_CNC,
            ProductType.NRML: self._kite.PRODUCT_NRML,
        }
        order_type_map = {
            "MARKET": self._kite.ORDER_TYPE_MARKET,
            "LIMIT": self._kite.ORDER_TYPE_LIMIT,
            "SL": self._kite.ORDER_TYPE_SL,
            "SL-M": self._kite.ORDER_TYPE_SLM,
        }

        broker_id = self._kite.place_order(
            variety=self._kite.VARIETY_REGULAR,
            exchange=order.exchange,
            tradingsymbol=order.symbol,
            transaction_type=side_map[order.side],
            quantity=order.qty,
            product=product_map[order.product],
            order_type=order_type_map[order.order_type.value],
            price=order.price if order.price > 0 else None,
            trigger_price=order.trigger_price if order.trigger_price > 0 else None,
            tag=order.tag[:20] if order.tag else None,
        )
        order.broker_order_id = str(broker_id)
        order.status = OrderStatus.OPEN
        log.info("zerodha.order.placed", broker_id=broker_id, symbol=order.symbol, side=order.side)
        return order

    def _live_cancel_order(self, order_id: str) -> bool:
        self._kite.cancel_order(variety=self._kite.VARIETY_REGULAR, order_id=order_id)
        log.info("zerodha.order.cancelled", order_id=order_id)
        return True

    def get_positions(self) -> list[Position]:
        raw = self._kite.positions()
        positions = []
        for p in raw.get("day", []):
            if p["quantity"] == 0:
                continue
            positions.append(Position(
                symbol=p["tradingsymbol"],
                exchange=p["exchange"],
                product=ProductType(p["product"]),
                qty=p["quantity"],
                avg_price=p["average_price"],
                ltp=p["last_price"],
                pnl=p["pnl"],
            ))
        return positions

    def get_margins(self) -> dict:
        margins = self._kite.margins()
        equity = margins.get("equity", {})
        return {
            "available_cash": equity.get("available", {}).get("cash", 0),
            "utilised": equity.get("utilised", {}).get("debits", 0),
            "net": equity.get("net", 0),
        }

    def get_ltp(self, symbol: str, exchange: str) -> float:
        key = f"{exchange}:{symbol}"
        quote = self._kite.ltp([key])
        return quote[key]["last_price"]

    def get_instrument_token(self, symbol: str, exchange: str) -> str:
        instruments = self._kite.instruments(exchange)
        for inst in instruments:
            if inst["tradingsymbol"] == symbol:
                return str(inst["instrument_token"])
        raise ValueError(f"Instrument not found: {symbol} on {exchange}")
