"""Mocked rate limiter tests — no network, no models, controllable clock.

Covers the Step 25 contract:
- below the limit / at the limit / over the limit
- HTTP 429 + JSON body + Retry-After header, RAG pipeline NOT called
- separate client IPs with separate limits
- /api/health is never rate limited
- fixed-window expiry resets correctly
- X-Forwarded-For spoofing protection (ignored by default; rightmost-entry
  trust only when RATE_LIMIT_TRUST_PROXY=true)
- configuration defaults + invalid configuration falls back safely

Run from the backend/ directory:
    python scripts/test_rate_limiter.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.config import Settings, settings
from app.main import app
from app.schemas.chat import Source
from app.services.rag_service import RAGResult, RAGSource
import app.api.routes.chat as chat_module
from app.services.rate_limiter import (
    DEFAULT_MAX_REQUESTS,
    DEFAULT_WINDOW_SECONDS,
    RateLimiter,
    _clamped_limits,
    client_ip,
)

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


class FakeClock:
    def __init__(self, start: float = 1_000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds: float):
        self.now += seconds


class StubRequest:
    class _Client:
        def __init__(self, host):
            self.host = host

    def __init__(self, host: str | None = "203.0.113.7", xff: str | None = None):
        self.client = self._Client(host) if host is not None else None
        self.headers = {} if xff is None else {"x-forwarded-for": xff}


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


client = TestClient(app)
RAG = FakeRAG(ANSWER)
orig_rag = chat_module.rag_service
orig_limiter = chat_module.rate_limiter
chat_module.rag_service = RAG


def fresh_limiter() -> RateLimiter:
    limiter = RateLimiter()
    chat_module.rate_limiter = limiter
    return limiter


def restore():
    chat_module.rate_limiter = orig_limiter
    chat_module.rag_service = orig_rag


print("=" * 78)
print("Rate limiter — mocked tests")
print("=" * 78)

# ================================================= 1. unit: limits
print("\n[1] below / at / over the limit (10 per 60s per IP)")
limiter = RateLimiter()
with patched(rate_limit_chat=10, rate_limit_window_seconds=60, rate_limit_trust_proxy=False):
    results = [limiter.check("203.0.113.1") for _ in range(11)]
    allowed = [r for r, _ in results if r]
    denied = [r for r, _ in results if not r]
    if len(allowed) == 10 and len(denied) == 1:
        ok("first 10 requests allowed; 11th rejected")
    else:
        bad(f"allowed={len(allowed)}, denied={len(denied)}")
    retry_after = results[-1][1]
    if 0 < retry_after <= 60:
        ok(f"Retry-After {retry_after:.1f}s (within 60s window)")
    else:
        bad(f"retry_after out of range: {retry_after}")

print("\n[2] separate IPs have separate buckets")
limiter = RateLimiter()
with patched(rate_limit_chat=10, rate_limit_window_seconds=60):
    for _ in range(10):
        limiter.check("10.0.0.1")
    denied_ip1 = limiter.check("10.0.0.1")[0]
    allowed_ip2 = limiter.check("10.0.0.2")[0]
    if not denied_ip1 and allowed_ip2:
        ok("10.0.0.1 blocked; 10.0.0.2 still allowed")
    else:
        bad(f"ip separation broken: ip1_blocked={denied_ip1}, ip2_allowed={allowed_ip2}")

print("\n[3] expired window resets correctly (fake clock)")
clock = FakeClock()
limiter = RateLimiter(clock=clock)
with patched(rate_limit_chat=3, rate_limit_window_seconds=60):
    for _ in range(3):
        limiter.check("10.0.0.3")
    if not limiter.check("10.0.0.3")[0]:
        ok("limit reached within window")
    else:
        bad("limit not enforced")
    clock.advance(59)
    if not limiter.check("10.0.0.3")[0]:
        ok("still blocked at t=59s")
    else:
        bad("window reset too early")
    clock.advance(1)  # t=60s
    if limiter.check("10.0.0.3")[0]:
        ok("window expired at t=60s; counter reset")
    else:
        bad("window did not reset at expiry")

print("\n[4] retry_after decreases within the window")
clock = FakeClock()
limiter = RateLimiter(clock=clock)
with patched(rate_limit_chat=1, rate_limit_window_seconds=60):
    limiter.check("10.0.0.4")
    ra1 = limiter.check("10.0.0.4")[1]
    clock.advance(30)
    ra2 = limiter.check("10.0.0.4")[1]
    if 55 < ra1 <= 60 and 25 < ra2 <= 30:
        ok(f"retry_after decreases: {ra1:.0f}s -> {ra2:.0f}s")
    else:
        bad(f"retry_after not decreasing as expected: {ra1}, {ra2}")

print("\n[5] configuration defaults and invalid configuration")
with patched(rate_limit_chat=10, rate_limit_window_seconds=60, rate_limit_trust_proxy=False):
    mx, win = _clamped_limits()
    if mx == 10 and win == 60.0:
        ok("valid configuration respected")
    else:
        bad(f"valid configuration not respected: {mx}/{win}")
with patched(rate_limit_chat=0, rate_limit_window_seconds=60):
    mx, win = _clamped_limits()
    ok("rate_limit_chat=0 falls back safely") if (mx, win) == (10, 60.0) else bad(f"got {mx}/{win}")
with patched(rate_limit_chat=-3, rate_limit_window_seconds=60):
    mx, win = _clamped_limits()
    ok("negative rate_limit_chat falls back safely") if (mx, win) == (10, 60.0) else bad(f"got {mx}/{win}")
with patched(rate_limit_chat=10, rate_limit_window_seconds=0):
    mx, win = _clamped_limits()
    ok("zero window falls back safely") if (mx, win) == (10, 60.0) else bad(f"got {mx}/{win}")
with patched(rate_limit_chat=10, rate_limit_window_seconds=-60):
    mx, win = _clamped_limits()
    ok("negative window falls back safely") if (mx, win) == (10, 60.0) else bad(f"got {mx}/{win}")
with patched(rate_limit_chat=10, rate_limit_window_seconds=float("inf")):
    mx, win = _clamped_limits()
    ok("non-finite window falls back safely") if (mx, win) == (10, 60.0) else bad(f"got {mx}/{win}")
if (Settings.model_fields["rate_limit_chat"].default == 10
        and Settings.model_fields["rate_limit_window_seconds"].default == 60.0
        and Settings.model_fields["rate_limit_trust_proxy"].default is False):
    ok("defaults are safe: 10 / 60s / trust_proxy=false")
else:
    bad("unsafe defaults in Settings")

print("\n[6] X-Forwarded-For spoofing protection")
with patched(rate_limit_trust_proxy=False):
    if client_ip(StubRequest("203.0.113.7", "6.6.6.6")) == "203.0.113.7":
        ok("trust_proxy=false: XFF ignored, peer used (spoof defeated)")
    else:
        bad("XFF honored despite trust_proxy=false")
    if client_ip(StubRequest("203.0.113.7", None)) == "203.0.113.7":
        ok("no XFF: peer used")
    else:
        bad("no-XFF path broken")
with patched(rate_limit_trust_proxy=True):
    if client_ip(StubRequest("10.0.0.1", "6.6.6.6")) == "6.6.6.6":
        ok("trust_proxy=true: single XFF entry trusted")
    else:
        bad("trusted XFF single-entry not honored")
    if client_ip(StubRequest("10.0.0.1", "1.1.1.1, 2.2.2.2")) == "2.2.2.2":
        ok("trust_proxy=true: rightmost XFF entry trusted (proxy-appended)")
    else:
        bad("rightmost-entry selection broken")
    if client_ip(StubRequest("10.0.0.1", "1.1.1.1, garbage-entry")) == "10.0.0.1":
        ok("unparseable rightmost XFF falls back to peer")
    else:
        bad("invalid rightmost XFF did not fall back to peer")
    if client_ip(StubRequest("10.0.0.1", "")) == "10.0.0.1":
        ok("empty XFF falls back to peer")
    else:
        bad("empty XFF broken")
    if client_ip(StubRequest("10.0.0.1", "2001:4860:4860::8888, 10.1.1.1")) == "10.1.1.1":
        ok("IPv6 leading entries do not affect rightmost selection")
    else:
        bad("IPv6 XFF handling wrong")

# ================================================= 7. route-level tests
print("\n[7] /api/chat route: 429 path and peer counting")

def send_n(n, xff=None):
    resp = None
    calls_before = RAG.calls
    for _ in range(n):
        headers = {"X-Forwarded-For": xff} if xff else {}
        resp = client.post("/api/chat", json={"message": "When is the admission deadline?"}, headers=headers)
    return resp, RAG.calls - calls_before

fresh_limiter()  # 10 per 60s default
resp, rag_calls = send_n(10)
if resp.status_code == 200 and rag_calls == 10:
    ok("10 requests within limit -> 200, pipeline called 10 times")
else:
    bad(f"first 10: status={resp.status_code}, rag_calls={rag_calls}")
resp429, rag_calls = send_n(1)
body = resp429.json()
if resp429.status_code == 429 and rag_calls == 0:
    ok("11th request -> 429, pipeline NOT called (quota protected)")
else:
    bad(f"11th: status={resp429.status_code}, rag_calls={rag_calls}")
if body.get("error") == "rate_limit_exceeded" and "detail" in body and isinstance(body.get("retry_after"), int):
    ok(f"429 body shape correct: {body}")
else:
    bad(f"429 body wrong: {body}")
retry_after = resp429.headers.get("Retry-After")
try:
    retry_after_value = int(retry_after)
    ok(f"Retry-After header present and integer ({retry_after_value})") if 0 < retry_after_value <= 60 else bad(f"Retry-After out of range: {retry_after_value}")
except (TypeError, ValueError):
    bad(f"Retry-After missing/not an integer: {retry_after!r}")

print("\n[8] /api/health remains accessible after limit exhaustion")
r = client.get("/api/health")
if r.status_code == 200:
    ok("/api/health -> 200 with exhausted chat limiter")
else:
    bad(f"/api/health -> {r.status_code} (should be 200)")

print("\n[9] spoofing XFF cannot bypass the limit (trust_proxy=false)")
fresh_limiter()
r = None
rag_calls_before = RAG.calls
for i in range(11):
    r = client.post("/api/chat", json={"message": "q"}, headers={"X-Forwarded-For": f"6.6.6.{i}"})
if r.status_code == 429 and RAG.calls - rag_calls_before == 10:
    ok("attacker rotating XFF values all counted against the peer bucket; 429 after 10")
else:
    bad(f"XFF spoofing bypassed limit: last={r.status_code}, rag_calls_delta={RAG.calls - rag_calls_before}")

print("\n[10] with trust_proxy=true, XFF client IPs get independent buckets")
fresh_limiter()
with patched(rate_limit_trust_proxy=True):
    r1, _ = send_n(10, xff="9.9.9.9")  # fresh client A exhausts its bucket
    r2 = client.post("/api/chat", json={"message": "q"}, headers={"X-Forwarded-For": "9.9.9.9"})
    r3 = client.post("/api/chat", json={"message": "q"}, headers={"X-Forwarded-For": "8.8.8.8"})
    if r1.status_code == 200 and r2.status_code == 429 and r3.status_code == 200:
        ok("9.9.9.9 exhausted (429) while 8.8.8.8 unaffected (200)")
    else:
        bad(f"trusted-proxy buckets broken: r1={r1.status_code}, r2={r2.status_code}, r3={r3.status_code}")

restore()
print()
if failures:
    print(f"RESULT: FAILED ({failures} check(s) failed)")
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED")
sys.exit(0)
