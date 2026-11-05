"""Minimal client for the official Brawl Stars API.

The API key is read from .env (BRAWL_API_KEY) and is never printed or logged.
"""
from __future__ import annotations

import os
import time
import urllib.parse
from pathlib import Path

import requests
from dotenv import load_dotenv

BASE = "https://api.brawlstars.com/v1"
ROOT = Path(__file__).resolve().parent


class ApiError(Exception):
    def __init__(self, status: int, reason: str, message: str = ""):
        super().__init__(f"{status} {reason} {message}".strip())
        self.status = status
        self.reason = reason
        self.message = message


def _load_key() -> str:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("BRAWL_API_KEY", "").strip()
    if not key:
        raise RuntimeError("BRAWL_API_KEY missing from .env")
    return key


def enc_tag(tag: str) -> str:
    tag = tag.strip().upper()
    if not tag.startswith("#"):
        tag = "#" + tag
    return urllib.parse.quote(tag, safe="")


class Client:
    """Rate-limited client with backoff on 429/5xx."""

    def __init__(self, rps: float = 4.0, timeout: float = 15.0):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {_load_key()}",
            "Accept": "application/json",
        })
        self.min_interval = 1.0 / rps
        self.timeout = timeout
        self._last = 0.0
        self.requests_made = 0

    def _throttle(self):
        wait = self._last + self.min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def get(self, path: str, params: dict | None = None, max_retries: int = 5):
        backoff = 2.0
        for attempt in range(max_retries + 1):
            self._throttle()
            try:
                r = self.session.get(BASE + path, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                if attempt == max_retries:
                    raise ApiError(0, "network", type(e).__name__)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            self.requests_made += 1
            if r.status_code == 200:
                return r.json()
            try:
                body = r.json()
            except ValueError:
                body = {}
            reason = body.get("reason", "")
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == max_retries:
                    raise ApiError(r.status_code, reason, body.get("message", ""))
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            raise ApiError(r.status_code, reason, body.get("message", ""))
        raise ApiError(0, "exhausted")

    # Convenience wrappers
    def rankings(self, country: str = "global", limit: int = 200):
        return self.get(f"/rankings/{country}/players", {"limit": limit})

    def battlelog(self, tag: str):
        return self.get(f"/players/{enc_tag(tag)}/battlelog")

    def player(self, tag: str):
        return self.get(f"/players/{enc_tag(tag)}")

    def brawlers(self):
        return self.get("/brawlers")
