"""
Vercel serverless entry point.

Scheduler background jobs are skipped in serverless mode.
All webhook, API, and dashboard endpoints remain functional.
"""
import sys
import os
import traceback

# Add project root to path so imports work from Vercel's build context
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Create a fallback app that reports the real import error
_import_error = None
app = None

try:
    from utils.logger import setup_logging
    setup_logging()
    from webhook.server import create_app
    app = create_app()
except Exception as _e:
    _import_error = traceback.format_exc()

if app is None:
    app = FastAPI(title="India AutoTrader")

    @app.get("/")
    @app.get("/{path:path}")
    async def startup_error(path: str = ""):
        return JSONResponse(
            status_code=503,
            content={"error": "startup_failed", "detail": _import_error},
        )
