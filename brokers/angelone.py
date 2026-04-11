from __future__ import annotations

import pyotp
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from brokers.base import BrokerBase, Order, OrderSide, OrderStatus, Position, ProductType
from config.settings import settings

log = structlog.get_logger(__name__)

_PRODUCT_MAP = {
    ProductType.MIS: "INTRADAY",
    ProductType.CNC: "DELIVERY",
    ProductType.NRML: "CARRYFORWARD",
}


class AngelOneBroker(BrokerBase):
    """Angel One SmartAPI broker integration."""

    def __init__(self, paper_trading: bool = True) -> None:
        super().__init__(paper_trading)
        self._smart = None

    def login(self) -> None:
        from SmartApi import SmartConnect

        api_key = settings.angelone_api_key
        client_id = settings.angelone_client_id
        password = settings.angelone_password
        totp_secret = settings.angelone_totp_secret

        if not all([api_key, client_id, password, totp_secret]):
            raise ValueError("AngelOne requires API_KEY, CLIENT_ID, PASSWORD, TOTP_SECRET")

        totp = pyotp.TOTP(totp_secret).now()
        self._smart = SmartConnect(api_key=api_key)
        data = self._smart.generateSession(client_id, password, totp)

        if data["status"] is False:
            raise RuntimeError(f"AngelOne login failed: {data.get('message')}")

        self._logged_in = True
        log.info("angelone.login.success", paper=self.paper_trading)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
    def _live_place_order(self, order: Order) -> Order:
        params = {
            "variety": "NORMAL",
            "tradingsymbol": order.symbol,
            "symboltoken": self.get_instrument_token(order.symbol, order.exchange),
            "transactiontype": order.side.value,
            "exchange": order.exchange,
            "ordertype": order.order_type.value,
            "producttype": _PRODUCT_MAP[order.product],
            "duration": "DAY",
            "price": str(order.price) if order.price > 0 else "0",
            "squareoff": "0",
            "stoploss": "0",
            "quantity": str(order.qty),
            "ordertag": order.tag[:20] if order.tag else "",
        }
        response = self._smart.placeOrder(params)
        if response["status"] is False:
            raise RuntimeError(f"AngelOne order failed: {response.get('message')}")

        order.broker_order_id = response["data"]["orderid"]
        order.status = OrderStatus.OPEN
        log.info("angelone.order.placed", broker_id=order.broker_order_id, symbol=order.symbol)
        return order

    def _live_cancel_order(self, order_id: str) -> bool:
        response = self._smart.cancelOrder(order_id, "NORMAL")
        return response.get("status") is True

    def get_positions(self) -> list[Position]:
        response = self._smart.position()
        positions = []
        for p in (response.get("data") or []):
            qty = int(p.get("netqty", 0))
            if qty == 0:
                continue
            positions.append(Position(
                symbol=p["tradingsymbol"],
                exchange=p["exchange"],
                product=ProductType.MIS,
                qty=qty,
                avg_price=float(p.get("netprice", 0)),
                ltp=float(p.get("ltp", 0)),
                pnl=float(p.get("unrealised", 0)),
            ))
        return positions

    def get_margins(self) -> dict:
        data = self._smart.rmsLimit().get("data", {})
        return {
            "available_cash": float(data.get("availablecash", 0)),
            "utilised": float(data.get("utiliseddebits", 0)),
            "net": float(data.get("net", 0)),
        }

    def get_ltp(self, symbol: str, exchange: str) -> float:
        token = self.get_instrument_token(symbol, exchange)
        data = self._smart.ltpData(exchange, symbol, token)
        return float(data["data"]["ltp"])

    def get_instrument_token(self, symbol: str, exchange: str) -> str:
        # Search instrument master for token
        # Full implementation should cache the instruments JSON from SmartAPI
        instruments = self._smart.searchScrip(exchange, symbol)
        for inst in (instruments.get("data") or []):
            if inst["tradingsymbol"] == symbol:
                return inst["symboltoken"]
        raise ValueError(f"Instrument not found: {symbol} on {exchange}")
