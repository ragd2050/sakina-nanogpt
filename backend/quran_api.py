# -*- coding: utf-8 -*-
"""
Quran Foundation API integration for verse recitation audio.

*** IMPORTANT, READ BEFORE DEPLOYING ***
v13 change: switched to the endpoint pattern given directly in the v13
brief as the previously-*working* flow --

    GET {QF_API_BASE}/quran/recitations/{recitation_id}?chapter_number={chapter_number}
    then locate the verse by audio_file["verse_key"] == verse_id

replacing v12's `/recitations/{id}/by_chapter/{chapter}`, which the brief
explicitly flagged as untested and NOT the confirmed-working path. Reciter
listing was similarly switched from `/resources/recitations` to
`/quran/recitations` to match the same `quran/` prefix family.

This sandbox's network allowlist still does not include quran.foundation
(or any Quran API domain) -- only pypi/npm/github-class package-install
domains. That means **this module has still never actually been executed
against the real API in this environment**, in v12 or now. It's written
exactly to the pattern given in the brief, but that must still be verified
live before production use:
  1. Set QF_CLIENT_ID / QF_CLIENT_SECRET as real environment variables
     (never hardcoded -- see _get_access_token() below).
  2. Run it against the real API yourself and fix whatever the real
     response shapes don't match (API response field names are a best
     guess from the brief + public documentation, not verified live).
  3. Check current Quran Foundation API docs -- endpoint paths/versions
     may have changed since this was written.

No secret values are ever placed in frontend code -- the frontend only
ever calls OUR /api/reciters and /api/audio endpoints; it never sees
QF_CLIENT_ID/QF_CLIENT_SECRET or talks to Quran Foundation directly.
"""
import os
import time
import requests

QF_CLIENT_ID = os.environ.get("QF_CLIENT_ID")
QF_CLIENT_SECRET = os.environ.get("QF_CLIENT_SECRET")
QF_ENV = os.getenv("QF_ENV", "prelive").lower()

if QF_ENV == "production":
    QF_OAUTH_URL = "https://oauth2.quran.foundation/oauth2/token"
    QF_API_BASE = "https://apis.quran.foundation/content/api/v4"

else:
    QF_OAUTH_URL = "https://prelive-oauth2.quran.foundation/oauth2/token"
    QF_API_BASE = "https://apis-prelive.quran.foundation/content/api/v4"
_token_cache = {"access_token": None, "expires_at": 0}


class QuranFoundationError(Exception):
    pass


def _get_access_token() -> str:
    """Client-credentials OAuth2 flow, cached until expiry. Raises
    QuranFoundationError with a clear message if credentials are missing
    (rather than silently failing), so the caller can degrade gracefully."""
    if not QF_CLIENT_ID or not QF_CLIENT_SECRET:
        raise QuranFoundationError(
            "QF_CLIENT_ID / QF_CLIENT_SECRET not set. Audio features need "
            "real Quran Foundation credentials as environment variables."
        )

    if _token_cache["access_token"] and time.time() < _token_cache["expires_at"] - 30:
        return _token_cache["access_token"]

    resp = requests.post(
        QF_OAUTH_URL,
        data={"grant_type": "client_credentials", "scope": "content"},
        auth=(QF_CLIENT_ID, QF_CLIENT_SECRET),
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    _token_cache["access_token"] = data["access_token"]
    _token_cache["expires_at"] = time.time() + data.get("expires_in", 3600)
    return _token_cache["access_token"]


def _headers():
    return {
        "Authorization": f"Bearer {_get_access_token()}",
        "x-auth-token": _get_access_token(),
        "x-client-id": QF_CLIENT_ID,
    }


def list_reciters():
    """GET /quran/recitations -- returns available reciters.

    v13: path corrected to the `quran/` prefix family per the brief
    (previously `/resources/recitations`, never confirmed working)."""
    resp = requests.get(
        f"{QF_API_BASE}/resources/recitations",
        headers=_headers(),
        timeout=10
    )
    resp.raise_for_status()
    return resp.json().get("recitations", [])


def get_verse_audio_url(verse_id: str, recitation_id: int) -> str:
    """
    verse_id like "13:28" -> chapter 13, verse 28.

    v13: uses the exact pattern specified in the brief as the previously
    *working* flow (replacing v12's untested `/recitations/{id}/by_chapter/{chapter}`):

        GET {QF_API_BASE}/quran/recitations/{recitation_id}?chapter_number={chapter_number}
        then find audio_file["verse_key"] == verse_id in the response.
    """
    if ":" not in verse_id:
        raise ValueError(f"verse_id must look like '13:28', got {verse_id!r}")
    chapter_number = verse_id.split(":")[0]

    resp = requests.get(
        f"{QF_API_BASE}/quran/recitations/{recitation_id}",
        params={"chapter_number": chapter_number},
        headers=_headers(),
        timeout=10,
    )
    resp.raise_for_status()
    files = resp.json().get("audio_files", [])

    match = next((f for f in files if f.get("verse_key") == verse_id), None)
    if not match:
        raise QuranFoundationError(f"No audio file found for verse_key={verse_id}")

    url = match.get("url") or match.get("audio_url")
    if url and not url.startswith("http"):
        url = f"https://verses.quran.foundation/{url.lstrip('/')}"
    return url
