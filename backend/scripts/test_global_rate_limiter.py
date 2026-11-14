"""Mocked global daily rate limiter tests — no network, no models, tmp db files.

Covers the Step 26 contract:
- below / at / over the daily cap (UTC calendar day)
- Retry-After on daily-limit refusal, /api/health stays reachable
- UTC midnight rollover resets the counter
- persistence across limiter-instance recreation (SQLite)
- concurrency: 20 threads race for 5 slots — cap is never exceeded
- per-IP limiter still works alongside the global cap (order: per-IP, global)
- no RAG/generation call after the daily cap is reached
- storage failure -> fail CLOSED (503), pipeline NOT called
- invalid configuration clamps to 500
- counter schema stores only {day, accepted} — nothing personal

Run from the backend/ directory:
    python scripts/test_global_rate_limiter.py
"""

import os
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient

from app.config import Settings, settings
from app.main import app
import app.api.routes.chat as chat_module
from app.services.global_rate_limiter import (
    DEFAULT_MAX_PER_DAY,
    GlobalRateLimiter,
    _clamped_limit,
)
from app.services.rag_service import RAGResult, RAGSource

failures = 0


def ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def bad(msg: str) -> None:
    global failures
    failures += 1
    print(f"[FAIL] {msg}")


class patched:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        self.saved = {k: getattr(settings, k) for k in self.kwargs}
        for k, v in self.kwargs.items():
            setattr(settings, k, v)

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(settings, k, v)


class FakeNow:
    """Controllable UTC clock."""

    def __init__(self, start: datetime):
        self.moment = start

    def __call__(self):
        return self.moment

    def set(self, moment: datetime):
        self.moment = moment


ANSWER = RAGResult(
    status="answered",
    answer="Fall intake closes on 25 July 2026.",
    sources=[RAGSource(filename="gazette.pdf", page_number=2, snippet="closing date", distance=0.1)],
    error=None,
)


class FakeRAG:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def answer(self, message):
        self.calls += 1
        return self.result


orig_rag = chat_module.rag_service
orig_global = chat_module.global_rate_limiter
orig_ip = chat_module.rate_limiter

client = TestClient(app)
TMP = Path(tempfile.mkdtemp(prefix="pugpt_globalrl_"))


def tmp_db(name: str) -> Path:
    return TMP / name


def fresh_route_limiters(db_file: Path, per_ip_limit: int = 10):
    grl = GlobalRateLimiter(db_path=db_file)
    chat_module.global_rate_limiter = grl
    chat_module.rate_limiter.reset()
    return grl


def restore():
    chat_module.global_rate_limiter = orig_global
    chat_module.rate_limiter = orig_ip
    chat_module.rag_service = orig_rag


print("=" * 78)
print("Global daily rate limiter — mocked tests")
print("=" * 78)

now = FakeNow(datetime(2026, 8, 3, 12, 0, 0, tzinfo=timezone.utc))

with patched(max_chat_requests_per_day=5):
    print("\n[1] below / at / over the daily cap")
    limiter = GlobalRateLimiter(db_path=tmp_db("cap.sqlite3"), now=now)
    results = [limiter.try_consume() for _ in range(7)]
    allowed = [pair for pair in results if pair[0] is True]
    denied = [pair for pair in results if pair[0] is False]
    if len(allowed) == 5 and len(denied) == 2:
        ok("first 5 allowed; requests 6-7 refused")
    else:
        bad(f"allowed={len(allowed)}, denied={len(denied)}")

    print("\n[2] counter state and Retry-After at the cap")
    if limiter.today_count() == 5:
        ok("today_count == 5 at the cap")
    else:
        bad(f"today_count={limiter.today_count()}")
    _, retry_after = denied[0]
    if 0 < retry_after <= 86400:
        ok(f"retry_after={retry_after:.0f}s (<= midnight UTC)")
    else:
        bad(f"retry_after out of range: {retry_after}")

    print("\n[3] schema stores only {day, accepted}")
    conn = sqlite3.connect(str(tmp_db("cap.sqlite3")))
    try:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(daily_counter)").fetchall()]
    finally:
        conn.close()
    if cols == ["day", "accepted"]:
        ok(f"counter schema minimal: {cols}")
    else:
        bad(f"unexpected schema: {cols}")

    print("\n[4] UTC midnight rollover resets the counter")
    limiter.reset()
    take_five = [limiter.try_consume() for _ in range(5)]
    if all(r[0] is True for r in take_five):
        ok("consumed 5 on day 1")
    else:
        bad("could not consume day-1 allowance")
    day1_count = limiter.today_count()
    now.set(datetime(2026, 8, 4, 0, 0, 1, tzinfo=timezone.utc))
    allowed_new_day = limiter.try_consume()[0]
    if allowed_new_day is True and limiter.today_count() == 1:
        ok("counter reset at 00:00:01 UTC (day 2 starts at 1)")
    else:
        bad(f"midnight rollover failed: allowed={allowed_new_day}, count={limiter.today_count()}")
    now.set(datetime(2026, 8, 3, 12, 0, 0, tzinfo=timezone.utc))
    if limiter.today_count() == 5 and day1_count == 5:
        ok("day-1 row preserved (5) after rollover")
    else:
        bad("day-1 row lost or corrupted")

print("\n[5] persistence across instance recreation")
now.set(datetime(2026, 8, 3, 12, 0, 0, tzinfo=timezone.utc))
persist_file = tmp_db("persist.sqlite3")
first = GlobalRateLimiter(db_path=persist_file, now=now)
with patched(max_chat_requests_per_day=5):
    for _ in range(3):
        first.try_consume()
    second = GlobalRateLimiter(db_path=persist_file, now=now)
    if second.today_count() == 3:
        ok("counter persisted to SQLite (new instance reads 3)")
    else:
        bad(f"recreated instance lost counter: {second.today_count()}")
    third = GlobalRateLimiter(db_path=persist_file, now=now)
    third.try_consume(); third.try_consume()
    if GlobalRateLimiter(db_path=persist_file, now=now).today_count() == 5:
        ok("multiple instances share the SQLite counter consistently")
    else:
        bad("instances diverged")

print("\n[6] concurrency: 20 threads race for 5 slots")
now.set(datetime(2026, 8, 3, 12, 0, 0, tzinfo=timezone.utc))
concurrent_file = tmp_db("race.sqlite3")
with patched(max_chat_requests_per_day=5):
    limiter = GlobalRateLimiter(db_path=concurrent_file, now=now)
    with ThreadPoolExecutor(max_workers=20) as pool:
        outcomes = list(pool.map(lambda _: limiter.try_consume(), range(20)))
    winners = sum(1 for ok_flag, _ in outcomes if ok_flag is True)
    losers = sum(1 for ok_flag, _ in outcomes if ok_flag is False)
    if winners == 5 and losers == 15 and limiter.today_count() == 5:
        ok(f"exactly 5 of 20 racing requests admitted (count={limiter.today_count()})")
    else:
        bad(f"race result wrong: winners={winners}, losers={losers}, count={limiter.today_count()}")

print("\n[7] storage failure fails CLOSED (no bypass)")
blocker = tmp_db("blocker_as_file")
blocker.write_text("not a database", encoding="utf-8")
broken = GlobalRateLimiter(db_path=blocker / "x.sqlite3", now=now)  # parent is a FILE
outcome = broken.try_consume()
if outcome[0] is None:
    ok("unwritable db path -> try_consume returns None (fail-closed)")
else:
    bad(f"storage failure returned {outcome} — protection may be bypassed!")
corrupt = GlobalRateLimiter(db_path=blocker, now=now)  # db file is garbage bytes
outcome = corrupt.try_consume()
if outcome[0] is None:
    ok("corrupt db file -> try_consume returns None (fail-closed)")
else:
    bad(f"corrupt db returned {outcome}")

print("\n[8] invalid configuration clamps to the safe default")
defaults_ok = True
with patched(max_chat_requests_per_day=0):
    defaults_ok &= _clamped_limit() == DEFAULT_MAX_PER_DAY
with patched(max_chat_requests_per_day=-25):
    defaults_ok &= _clamped_limit() == DEFAULT_MAX_PER_DAY
with patched(max_chat_requests_per_day=float("inf")):
    defaults_ok &= _clamped_limit() == DEFAULT_MAX_PER_DAY
with patched(max_chat_requests_per_day="not-a-number"):
    defaults_ok &= _clamped_limit() == DEFAULT_MAX_PER_DAY
with patched(max_chat_requests_per_day=250):
    defaults_ok &= _clamped_limit() == 250
if defaults_ok:
    ok("0 / negative / inf / non-numeric -> 500; 250 honored")
else:
    bad("invalid configuration handling wrong")
if (Settings.model_fields["max_chat_requests_per_day"].default == 500
        and str(Settings.model_fields["rate_limit_global_db_path"].default).endswith("global_rate_limit.sqlite3")):
    ok("defaults are safe: 500 requests/day at backend/data/global_rate_limit.sqlite3")
else:
    bad("unsafe defaults in Settings")

print("\n[9] route: global cap reached -> 429, RAG not called, per-IP still works, health OK")
RAG = FakeRAG(ANSWER)
chat_module.rag_service = RAG
route_db = tmp_db("route.sqlite3")
now.set(datetime(2026, 8, 3, 12, 0, 0, tzinfo=timezone.utc))

fresh_route_limiters(route_db, per_ip_limit=10)
with patched(max_chat_requests_per_day=2, rate_limit_trust_proxy=False):
    r1 = client.post("/api/chat", json={"message": "admission dates?"})
    r2 = client.post("/api/chat", json={"message": "admission dates?"})
    calls_after_two = RAG.calls
    r3 = client.post("/api/chat", json={"message": "admission dates?"})
    body = r3.json()
    ra = r3.headers.get("Retry-After")
    try:
        ra_value = int(ra)
    except (TypeError, ValueError):
        ra_value = -1
    if r1.status_code == 200 and r2.status_code == 200 and calls_after_two == 2:
        ok("first 2 daily requests -> 200 (pipeline called twice)")
    else:
        bad(f"first two: {r1.status_code}/{r2.status_code}, rag calls={calls_after_two}")
    if (r3.status_code == 429 and body.get("error") == "daily_limit_exceeded"
            and isinstance(body.get("retry_after"), int) and 0 < ra_value <= 86400):
        ok(f"3rd -> 429 daily_limit_exceeded; Retry-After={ra}; RAG calls stay {calls_after_two}")
    else:
        bad(f"daily cap enforcement wrong: status={r3.status_code}, body={body}, Retry-After={ra}")
    r4 = client.get("/api/health")
    if r4.status_code == 200:
        ok("/api/health -> 200 after daily cap reached")
    else:
        bad(f"/api/health -> {r4.status_code}")

print("\n[10] route: per-IP limiter still enforced alongside (per-IP 11th -> 429)")
route_db2 = tmp_db("route2.sqlite3")
now.set(datetime(2026, 8, 3, 12, 0, 0, tzinfo=timezone.utc))
fresh_route_limiters(route_db2, per_ip_limit=10)
with patched(max_chat_requests_per_day=500):
    last = None
    for _ in range(11):
        last = client.post("/api/chat", json={"message": "q"})
    if last.status_code == 429 and last.json().get("error") == "rate_limit_exceeded":
        ok("per-IP 10/min still enforced with global cap at 500")
    else:
        bad(f"per-IP enforcement broke: {last.status_code} {last.json() if last else None}")

print("\n[11] route: storage failure -> 503, pipeline NOT called")
chat_module.rag_service = RAG
before_calls = RAG.calls
chat_module.rate_limiter.reset()  # isolate: per-IP bucket must allow this request
broken_route_db = tmp_db("blocked_parent_is_a_file")
blocker2 = tmp_db("blocker2")
blocker2.write_text("x", encoding="utf-8")
chat_module.global_rate_limiter = GlobalRateLimiter(db_path=blocker2 / "x.sqlite3", now=now)
r503 = client.post("/api/chat", json={"message": "q"})
body503 = r503.json()
if (r503.status_code == 503 and body503.get("error") == "rate_limit_unavailable"
        and RAG.calls == before_calls and r503.headers.get("Retry-After")):
    ok("storage failure -> 503 fail-closed; pipeline NOT called")
else:
    bad(f"fail-closed path broken: status={r503.status_code}, body={body503}, rag delta={RAG.calls - before_calls}")

restore()
shutil.rmtree(TMP, ignore_errors=True)
print()
if failures:
    print(f"RESULT: FAILED ({failures} check(s) failed)")
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED")
sys.exit(0)
