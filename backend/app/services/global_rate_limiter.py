"""Global daily chat-request limiter backed by SQLite.

Complements the in-memory per-IP limiter (app/services/rate_limiter.py): it
caps TOTAL accepted chat requests application-wide per UTC calendar day, so
distributed abuse from many IPs cannot drain Groq/Tavily quota. It survives
process restarts because the counter lives in a local SQLite file.

Design properties:
- Default: 500 accepted chat requests per UTC calendar day (configurable via
  MAX_CHAT_REQUESTS_PER_DAY).
- UTC calendar days. The cap resets at 00:00 UTC.
- Atomic "read-check-increment" inside a single BEGIN IMMEDIATE transaction:
  concurrent requests are serialized by the database write lock and the cap
  cannot be exceeded even with simultaneous traffic.
- Minimal storage: one row per day (day TEXT PRIMARY KEY, accepted INTEGER).
  No prompts, answers, IPs, user identities, headers, or secrets are stored.
  Rows older than 31 days are pruned automatically on access.
- Fail-closed: if SQLite is unavailable/corrupt/locked, requests are REFUSED
  (caller returns a safe 503) — the protection is never silently disabled.
- Invalid MAX_CHAT_REQUESTS_PER_DAY values clamp to the safe default (500)
  with one WARNING per process; a non-numeric env value fails fast at boot
  (pydantic) — protection is never silently switched off.
- Logs one anonymous INFO/ERROR line on refusal or storage failure; nothing
  else is logged. No IPs, prompts, keys or content.
"""

import logging
import sqlite3
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

DEFAULT_MAX_PER_DAY = 500

_clamped_warned = False


def _clamped_limit() -> int:
    """Read the configured daily limit; fall back to the safe default."""
    global _clamped_warned
    raw = getattr(settings, "max_chat_requests_per_day", DEFAULT_MAX_PER_DAY)
    try:
        limit = int(raw)
    except (TypeError, ValueError, OverflowError):
        limit = 0
    if limit < 1:
        if not _clamped_warned:
            logger.warning(
                "Invalid MAX_CHAT_REQUESTS_PER_DAY=%r; falling back to %d.",
                raw,
                DEFAULT_MAX_PER_DAY,
            )
            _clamped_warned = True
        return DEFAULT_MAX_PER_DAY
    return limit


def _today_utc(now: datetime) -> str:
    return now.date().isoformat()


def _seconds_until_utc_midnight(now: datetime) -> int:
    tomorrow = now.date() + timedelta(days=1)
    midnight = datetime.combine(tomorrow, time.min, tzinfo=timezone.utc)
    return max(1, int((midnight - now).total_seconds()))


class GlobalRateLimiter:
    """Persistent, thread- and process-safe daily counter (SQLite)."""

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS daily_counter (
            day TEXT PRIMARY KEY,
            accepted INTEGER NOT NULL
        )
    """

    def __init__(self, db_path: str | Path | None = None, now=None) -> None:
        configured = getattr(settings, "rate_limit_global_db_path", None)
        path = db_path if db_path is not None else configured or (
            Path(__file__).resolve().parent.parent.parent / "data" / "global_rate_limit.sqlite3"
        )
        self.db_path = Path(path)
        self._now = now or (lambda: datetime.now(timezone.utc))

    @staticmethod
    def _connect(path: Path) -> sqlite3.Connection:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), timeout=10.0, isolation_level=None)
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _prune(self, conn: sqlite3.Connection, today: str) -> None:
        cutoff = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=31)).strftime("%Y-%m-%d")
        conn.execute("DELETE FROM daily_counter WHERE day < ?", (cutoff,))

    def try_consume(self) -> tuple[bool | None, float]:
        """Atomically consume one unit of today's allowance.

        Returns (True, 0.0) if the request may proceed;
                (False, retry_after_seconds) when the daily cap is reached;
                (None, 0.0) when storage failed (fail-closed: caller must refuse).
        """
        limit = _clamped_limit()
        now = self._now()
        day = _today_utc(now)
        try:
            conn = self._connect(self.db_path)
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(self._SCHEMA)
                self._prune(conn, day)
                row = conn.execute(
                    "SELECT accepted FROM daily_counter WHERE day = ?", (day,)
                ).fetchone()
                current = row[0] if row else 0
                if current >= limit:
                    conn.execute("ROLLBACK")
                    return False, float(_seconds_until_utc_midnight(now))
                if row is None:
                    conn.execute(
                        "INSERT INTO daily_counter(day, accepted) VALUES (?, 1)", (day,)
                    )
                else:
                    conn.execute(
                        "UPDATE daily_counter SET accepted = accepted + 1 WHERE day = ?",
                        (day,),
                    )
                conn.execute("COMMIT")
                return True, 0.0
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            finally:
                conn.close()
        except (sqlite3.Error, OSError) as exc:
            # Fail CLOSED — never allow unlimited traffic when the counter
            # cannot be read/written. Do not leak paths/secrets in the log.
            logger.error("global rate limiter unavailable: %s", type(exc).__name__)
            return None, 0.0

    def today_count(self) -> int:
        """Current counter value for today (read-only; 0 on any error)."""
        now = self._now()
        day = _today_utc(now)
        try:
            conn = self._connect(self.db_path)
            try:
                row = conn.execute(
                    "SELECT accepted FROM daily_counter WHERE day = ?", (day,)
                ).fetchone()
                return int(row[0]) if row else 0
            finally:
                conn.close()
        except (sqlite3.Error, OSError):
            return 0

    def reset(self) -> None:
        """Clear the counter for today. Used by tests only."""
        now = self._now()
        day = _today_utc(now)
        try:
            conn = self._connect(self.db_path)
            try:
                conn.execute("DELETE FROM daily_counter WHERE day = ?", (day,))
            finally:
                conn.close()
        except (sqlite3.Error, OSError):
            pass


# Shared instance used by the /api/chat route.
global_rate_limiter = GlobalRateLimiter()
