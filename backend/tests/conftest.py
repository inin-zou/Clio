"""Pytest conftest — load .env so tests can construct clients that read API keys."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Walk up from this file to find the project root .env
_HERE = Path(__file__).resolve()
for parent in [_HERE.parent] + list(_HERE.parents):
    candidate = parent / ".env"
    if candidate.exists():
        load_dotenv(candidate)
        break

# Defensive default: if a key is still missing, set a dummy so module imports
# don't crash. Tests that actually call the API (none in this dir) would fail.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
