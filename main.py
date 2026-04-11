"""India AutoTrader — main entrypoint."""
import uvicorn
from utils.logger import setup_logging
from webhook.server import create_app

setup_logging()
app = create_app()

if __name__ == "__main__":
    from config.settings import settings
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.app_env.value == "development",
        log_config=None,  # Use structlog instead
    )
