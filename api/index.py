"""
Vercel Serverless Function Entry Point for LexAI FastAPI Application.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path so services and modules import seamlessly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import app  # ASGI app for Vercel Python runtime
