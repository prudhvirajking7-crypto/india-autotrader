"""
Vercel serverless entry point.

Note: APScheduler background jobs are disabled in serverless mode.
Webhook, API, and dashboard endpoints remain fully functional.
For scheduled analysis, trigger /scheduler/run manually or use an
external cron (GitHub Actions, Vercel Cron, etc.).
"""
import sys
import os

# Add project root to path so imports work from Vercel's build context
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logging
from webhook.server import create_app

setup_logging()
app = create_app()
