from __future__ import annotations

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from brokers.base import BrokerBase, Order, OrderSide, OrderStatus, Position, ProductType
from config.settings import settings

log = structlog.get_logger(__name__)


class UpstoxBroker(BrokerBase):
    """Upstox API v2 broker integration via upstox-python-sdk."""

    def __init__(self, paper_trading: bool = True) -> None:
        super().__init__(paper_trading)
        self._config = None
        self._order_api = None
        self._portfolio_api = None
        self._market_api = None

    def login(self) -> None:
        import upstox_client
        from upstox_client.rest import ApiException  # noqa: F401

        access_token = settings.upstox_access_token
        if not access_token:
            raise ValueError(
                "UPSTOX_ACCESS_TOKEN is required. Generate it via the OAuth flow at "
                f"{settings.upstox_redirect_uri} and set in .env"
            )

        configuration = upstox_client.Configuration()
        configuration.access_token = access_token

        api_client = upstox_client.ApiClient(configuration)
        self._order_api = upstox_client.OrderApi(api_client)
        self._portfolio_api = upstox_client.PortfolioApi(api_client)
        self._market_api = upstox_client.MarketQuoteApi(api_client)

        self._logged_in = True
        log.info("upstox.login.success", paper=self.paper_trading)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
    def _live_place_order(self, order: Order) -> Order:
        import upstox_client

        body = upstox_client.PlaceOrderRequest(
            quantity=order.qty,
            product=order.product.value,
            validity="DAY",
            price=order.price,
            tag=order.tag[:20] if order.tag else "",
            instrument_token=self.get_instrument_token(order.symbol, order.exchange),
            order_type=order.order_type.value,
            transaction_type=order.side.value,
            disclosed_quantity=0,
            trigger_price=order.trigger_price,
            is_amo=False,
        )

        response = self._order_api.place_order(body, api_version="2.0")
        order.broker_order_id = response.data.order_id
        order.status = OrderStatus.OPEN
        log.info("upstox.order.placed", broker_id=order.broker_order_id, symbol=order.symbol)
        return order

    def _live_cancel_order(self, order_id: str) -> bool:
        self._order_api.cancel_order(order_id=order_id, api_version="2.0")
        log.info("upstox.order.cancelled", order_id=order_id)
        return True

    def get_positions(self) -> list[Position]:
        response = self._portfolio_api.get_positions(api_version="2.0")
        positions = []
        for p in (response.data or []):
            if p.quantity == 0:
                continue
            positions.append(Position(
                symbol=p.tradingsymbol,
                exchange=p.exchange,
                product=ProductType(p.product),
                qty=p.quantity,
                avg_price=p.average_price,
                ltp=p.last_price,
                pnl=p.pnl,
            ))
        return positions

    def get_margins(self) -> dict:
        import upstox_client
        api = upstox_client.UserApi(self._order_api.api_client)
        response = api.get_user_fund_margin(api_version="2.0", segment="SEC")
        data = response.data
        return {
            "available_cash": data.available_margin if data else 0,
            "utilised": data.used_margin if data else 0,
            "net": (data.available_margin or 0) + (data.used_margin or 0),
        }

    def get_ltp(self, symbol: str, exchange: str) -> float:
        token = self.get_instrument_token(symbol, exchange)
        response = self._market_api.ltp(symbol=token, api_version="2.0")
        return response.data[token].last_price

    def get_instrument_token(self, symbol: str, exchange: str) -> str:
        # Upstox instrument token format: "NSE_EQ|INE009A01021"
        # For simplicity, look up from a pre-cached instrument master
        # Full implementation should query the instruments CSV
        return f"{exchange}_EQ|{symbol}"
