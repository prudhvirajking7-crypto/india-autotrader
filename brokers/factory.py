from __future__ import annotations

from config.settings import BrokerName, settings
from brokers.base import BrokerBase


def get_broker() -> BrokerBase:
    """Return the configured broker instance."""
    name = settings.active_broker

    if name == BrokerName.zerodha:
        from brokers.zerodha import ZerodhaBroker
        return ZerodhaBroker(paper_trading=settings.paper_trading)

    if name == BrokerName.upstox:
        from brokers.upstox import UpstoxBroker
        return UpstoxBroker(paper_trading=settings.paper_trading)

    if name == BrokerName.angelone:
        from brokers.angelone import AngelOneBroker
        return AngelOneBroker(paper_trading=settings.paper_trading)

    if name == BrokerName.finvasia:
        from brokers.finvasia import FinvasiaBroker
        return FinvasiaBroker(paper_trading=settings.paper_trading)

    raise ValueError(f"Unknown broker: {name}")
