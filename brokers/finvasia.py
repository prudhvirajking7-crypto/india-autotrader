from __future__ import annotations

import hashlib
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from brokers.base import BrokerBase, Order, OrderSide, OrderStatus, Position, ProductType
from config.settings import settings

log = structlog.get_logger(__name__)

_PRODUCT_MAP = {
    ProductType.MIS: "I",    # Intraday
    ProductType.CNC: "C",    # CNC delivery
    ProductType.NRML: "M",   # NRML carry forward
}
_ORDER_TYPE_MAP = {
    "MARKET": "MKT",
    "LIMIT": "LMT",
    "SL": "SL-LMT",
    "SL-M": "SL-MKT",
}


class FinvasiaBroker(BrokerBase):
    """Finvasia (Shoonya) zero-brokerage broker integration."""

    def __init__(self, paper_trading: bool = True) -> None:
        super().__init__(paper_trading)
        self._api = None

    def login(self) -> None:
        from NorenRestApiPy.NorenApi import NorenApi

        user_id = settings.finvasia_user_id
        password = settings.finvasia_password
        api_key = settings.finvasia_api_key
        vendor_code = settings.finvasia_vendor_code
        imei = settings.finvasia_imei

        if not all([user_id, password, api_key, vendor_code]):
            raise ValueError("Finvasia requires USER_ID, PASSWORD, API_KEY, VENDOR_CODE")

        # Hash password with SHA256
        pwd_hash = hashlib.sha256(password.encode()).hexdigest()
        app_key = hashlib.sha256(f"{user_id}|{api_key}".encode()).hexdigest()

        self._api = NorenApi(
            host="https://api.shoonya.com/NorenWClient/",
            websocket="wss://api.shoonya.com/NorenWSTP/",
        )

        ret = self._api.login(
            userid=user_id,
            password=pwd_hash,
            twoFA=api_key,
            vendor_code=vendor_code,
            api_secret=app_key,
            imei=imei or "abc1234",
        )
        if ret is None or ret.get("stat") != "Ok":
            raise RuntimeError(f"Finvasia login failed: {ret}")

        self._logged_in = True
        log.info("finvasia.login.success", paper=self.paper_trading)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
    def _live_place_order(self, order: Order) -> Order:
        ret = self._api.place_order(
            buy_or_sell="B" if order.side == OrderSide.BUY else "S",
            product_type=_PRODUCT_MAP[order.product],
            exchange=order.exchange,
            tradingsymbol=order.symbol,
            quantity=order.qty,
            discloseqty=0,
            price_type=_ORDER_TYPE_MAP[order.order_type.value],
            price=order.price if order.price > 0 else 0,
            trigger_price=order.trigger_price if order.trigger_price > 0 else None,
            retention="DAY",
            remarks=order.tag[:20] if order.tag else None,
        )
        if ret is None or ret.get("stat") != "Ok":
            raise RuntimeError(f"Finvasia order failed: {ret}")

        order.broker_order_id = ret["norenordno"]
        order.status = OrderStatus.OPEN
        log.info("finvasia.order.placed", broker_id=order.broker_order_id, symbol=order.symbol)
        return order

    def _live_cancel_order(self, order_id: str) -> bool:
        ret = self._api.cancel_order(orderno=order_id)
        return ret is not None and ret.get("stat") == "Ok"

    def get_positions(self) -> list[Position]:
        positions_data = self._api.get_positions() or []
        positions = []
        for p in positions_data:
            qty = int(p.get("netqty", 0))
            if qty == 0:
                continue
            positions.append(Position(
                symbol=p["tsym"],
                exchange=p["exch"],
                product=ProductType.MIS,
                qty=qty,
                avg_price=float(p.get("netavgprc", 0)),
                ltp=float(p.get("lp", 0)),
                pnl=float(p.get("urmtom", 0)),
            ))
        return positions

    def get_margins(self) -> dict:
        limits = self._api.get_limits() or {}
        return {
            "available_cash": float(limits.get("cash", 0)),
            "utilised": float(limits.get("marginused", 0)),
            "net": float(limits.get("net", 0)),
        }

    def get_ltp(self, symbol: str, exchange: str) -> float:
        quote = self._api.get_quotes(exchange=exchange, token=self.get_instrument_token(symbol, exchange))
        return float(quote.get("lp", 0))

    def get_instrument_token(self, symbol: str, exchange: str) -> str:
        results = self._api.searchscrip(exchange=exchange, searchtext=symbol) or {}
        for scrip in results.get("values", []):
            if scrip["tsym"] == symbol:
                return scrip["token"]
        raise ValueError(f"Instrument not found: {symbol} on {exchange}")
