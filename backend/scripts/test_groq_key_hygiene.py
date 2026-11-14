"""Groq API-key hygiene tests — no network, no real key.

Guards the LocalProtocolError class of failure fixed in Step 13:
- outer whitespace/newlines on the stored key are normalized before the
  Authorization header is built;
- embedded control characters / non-ASCII bytes in the key fail fast with a
  clear, key-free GroqError and never reach the network;
- when h11-style failures arrive via transport, the raised error text and the
  logged output contain NO key material (h11 embeds the offending value in its
  exception message — the service must not surface it).

Run from the backend/ directory:
    python scripts/test_groq_key_hygiene.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.config import Settings
from app.services.groq_service import GroqError, GroqService, GroqUpstreamError

TEST_KEY = "sk-test-SECRETKEY123"  # distinctive so leaks are easy to detect
failures = 0


def ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def bad(msg: str) -> None:
    global failures
    failures += 1
    print(f"[FAIL] {msg}")


def make_svc(api_key, captured=None, raise_exc=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(request)
        if raise_exc is not None:
            raise raise_exc
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}]}
        )

    return GroqService(
        api_key=api_key,
        base_url="https://api.groq.test/chat/completions",
        model="test-model",
        timeout=5.0,
        transport=httpx.MockTransport(handler),
    )


log_records: list[str] = []


class CaptureHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        log_records.append(record.getMessage())


root_logger = logging.getLogger()
_capture = CaptureHandler()
root_logger.addHandler(_capture)
root_logger.setLevel(logging.DEBUG)

print("=" * 78)
print("Groq API-key hygiene — mocked tests")
print("=" * 78)

# ------------------------------------------------------------- normal key
print("\n[normal key]")
captured: list = []
svc = make_svc(TEST_KEY, captured)
out = svc.generate("hello")
req = captured[0]
if req.headers["authorization"] == f"Bearer {TEST_KEY}":
    ok("header carries the key verbatim")
else:
    bad("authorization header mismatch")

# ------------------------------ outer whitespace/newlines stripped at wire
print("\n[outer whitespace / newlines on stored key]")
padded_variants = [
    f"  {TEST_KEY}  ",
    f"\n{TEST_KEY}\n",
    f"\r\n{TEST_KEY}\r\n",
    f"\t{TEST_KEY}\t",
    f"{TEST_KEY}\n ",
]
for i, padded in enumerate(padded_variants):
    captured = []
    svc = make_svc(padded, captured)
    try:
        svc.generate("hello")
    except Exception as exc:  # noqa: BLE001
        bad(f"variant {i}: unexpected raise: {type(exc).__name__}")
        continue
    header = captured[0].headers["authorization"]
    if header == f"Bearer {TEST_KEY}":
        ok(f"variant {i}: key stripped -> header is legal")
    else:
        bad(f"variant {i}: header still carries whitespace")

# --------------------- embedded control chars / non-printable -> fail fast
print("\n[embedded invalid characters -> fail fast, zero network]")
bad_keys = [
    f"{TEST_KEY[:4]}\x00{TEST_KEY[4:]}",
    f"{TEST_KEY[:4]}\tremainder{TEST_KEY[6:]}",
    f"{TEST_KEY}\x7f",
    f"{TEST_KEY[:6]}\n{TEST_KEY[6:]}",
    f"{TEST_KEY} embedded spaces",
    "secrét-KEY-İçer-123",
    f"{TEST_KEY[:3]}\x0b{TEST_KEY[3:]}",
]
for i, bad_key in enumerate(bad_keys):
    captured = []
    svc = make_svc(bad_key, captured)
    try:
        svc.generate("hello")
        bad(f"bad key {i}: no error raised")
    except GroqError as exc:
        msg = str(exc)
        if captured:
            bad(f"bad key {i}: request reached transport (must fail fast)")
        elif bad_key in msg or TEST_KEY in msg:
            bad(f"bad key {i}: error message leaked the key")
        else:
            ok(f"bad key {i}: fail-fast GroqError, no key in message, no request")

# ---------------------------- whitespace-only key still fails very fast
print("\n[whitespace-only key]")
svc = make_svc("   \n", captured := [])
try:
    svc.generate("hello")
    bad("whitespace-only key did not raise")
except GroqError as exc:
    if captured:
        bad("whitespace-only key reached transport")
    elif TEST_KEY in str(exc):
        bad("whitespace-only error leaked something odd")
    else:
        ok("whitespace-only key -> configuration error, no request")

# --------------- h11-style transport failure must not leak key into logs
print("\n[h11-style failure via transport — no key in raised error or logs]")
poisoned = httpx.LocalProtocolError(f"Illegal header value b'Bearer {TEST_KEY}\\n'")
svc = make_svc(TEST_KEY, captured := [], raise_exc=poisoned)
log_records.clear()
try:
    svc.generate("hello")
    bad("poisoned transport did not raise")
except GroqUpstreamError as exc:
    msg = str(exc)
    if TEST_KEY in msg:
        bad("raised error leaked the key")
    else:
        ok("raised GroqUpstreamError carries no key material")
    leaked = [r for r in log_records if TEST_KEY in r]
    if leaked:
        bad("API key appeared in logged output")
    else:
        ok("no logged line contains the API key")

# ----------------------- configuration boundary normalization (validator)
print("\n[Settings boundary — outer whitespace normalized, Bearer untouched]")
s = Settings(_env_file=None, groq_api_key=f"\n  {TEST_KEY}  \r\n", tavily_api_key="  tv-test-abc\n")
if s.groq_api_key == TEST_KEY:
    ok("groq_api_key outer whitespace stripped at Settings boundary")
else:
    bad("groq_api_key NOT normalized by Settings")
if s.tavily_api_key == "tv-test-abc":
    ok("tavily_api_key outer whitespace stripped at Settings boundary")
else:
    bad("tavily_api_key NOT normalized by Settings")

# leading "Bearer " is deliberately preserved (reported, not silently fixed)
s = Settings(_env_file=None, groq_api_key=f"Bearer {TEST_KEY}")
if s.groq_api_key == f"Bearer {TEST_KEY}":
    ok("leading 'Bearer ' prefix preserved (not silently removed)")
else:
    bad("leading 'Bearer ' prefix was silently removed")

root_logger.removeHandler(_capture)

print()
if failures:
    print(f"RESULT: FAILED ({failures} check(s) failed)")
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED")
sys.exit(0)
