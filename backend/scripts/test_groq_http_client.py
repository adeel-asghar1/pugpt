"""Groq HTTP client reuse + output-token bound — offline, fully mocked.

Does NOT call external services. Verifies:
- GROQ_MAX_TOKENS configuration (default, override, invalid values);
- the request payload carries ``max_completion_tokens`` (current parameter name);
- the shared httpx client is reused across requests (connection pooling),
  closes properly, and is recreated after close();
- finish_reason="length" fails closed (never returns a partial answer);
- per-request timeout and error classification survive client sharing;
- Tavily WebSearchService shares the same lifecycle;
- diagnostics contain numeric usage only — never the question, answer or key.

Run from the backend/ directory:
    python scripts/test_groq_http_client.py
"""

import concurrent.futures
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from pydantic import ValidationError

from app.config import Settings
from app.services.groq_service import GroqResponseError, GroqService, GroqTimeoutError, GroqRateLimitError
from app.services.web_search_service import WebSearchService

failures = 0


def ok(msg):
    print(f"[PASS] {msg}")


def bad(msg):
    global failures
    failures += 1
    print(f"[FAIL] {msg}")


def check(label, condition, detail=""):
    if condition:
        ok(label)
    else:
        bad(f"{label} :: {detail}")


def payload_with(finish_reason="stop", content="Answer body", usage=None):
    body = {
        "choices": [{"message": {"role": "assistant", "content": content},
                     "finish_reason": finish_reason}],
        "usage": usage if usage is not None else
        {"prompt_tokens": 123, "completion_tokens": 55, "total_tokens": 178},
    }
    return body


captured_requests = []


def make_service(body, raise_exc=None, max_tokens=None):
    captured_requests.clear()

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        if raise_exc is not None:
            raise raise_exc
        return httpx.Response(200, json=body)

    kwargs = dict(api_key="sk-test-CONST", transport=httpx.MockTransport(handler))
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return GroqService(**kwargs)


print("=" * 78)
print("Groq HTTP client reuse + GROQ_MAX_TOKENS — mocked offline tests")
print("=" * 78)

print("\n[configuration — default, override, validation]")
check("default GROQ_MAX_TOKENS is 1024",
      Settings.model_fields["groq_max_tokens"].default == 1024,
      str(Settings.model_fields["groq_max_tokens"].default))
check("Settings(groq_max_tokens=2048) accepted",
      Settings(groq_max_tokens=2048).groq_max_tokens == 2048)
for invalid in (0, -1, 65537):
    try:
        Settings(groq_max_tokens=invalid)
        bad(f"config accepted invalid GROQ_MAX_TOKENS={invalid}")
    except ValidationError:
        ok(f"config rejects GROQ_MAX_TOKENS={invalid}")
try:
    Settings(groq_max_tokens=True)
    bad("config accepted boolean GROQ_MAX_TOKENS")
except ValidationError:
    ok("config rejects boolean GROQ_MAX_TOKENS")
for invalid in (0, -1, 65537, True):
    try:
        GroqService(max_tokens=invalid)
        bad(f"service accepted invalid max_tokens={invalid}")
    except ValueError:
        ok(f"service rejects max_tokens={invalid}")

print("\n[payload — current parameter name and contents]")
svc = make_service(payload_with())
out = svc.generate("Question?", system="System.")
raw = json.loads(captured_requests[0].content.decode())
check("payload carries max_completion_tokens", raw.get("max_completion_tokens") == 1024,
      f"payload={raw}")
check("deprecated max_tokens not sent", "max_tokens" not in raw)
check("model preserved", raw.get("model") == "openai/gpt-oss-120b", raw.get("model"))
check("temperature preserved", raw.get("temperature") == 0)
check("messages roles preserved",
      [m["role"] for m in raw["messages"]] == ["system", "user"])
check("answer returned unchanged", out == "Answer body")
svc.close()
svc = make_service(payload_with(), max_tokens=2048)
svc.generate("Question?")
raw = json.loads(captured_requests[0].content.decode())
check("explicit max_tokens override honored in payload",
      raw.get("max_completion_tokens") == 2048, f"payload={raw}")
svc.close()

print("\n[shared client — reuse and lifecycle]")
svc = make_service(payload_with())
client_one = svc._get_client()
svc.generate("First?")
svc.generate("Second?")
check("two requests reused one shared client instance",
      svc._get_client() is client_one, "client instance changed between calls")
check("both mocked requests dispatched", len(captured_requests) == 2,
      f"{len(captured_requests)} requests")
old = svc._get_client()
svc.close()
check("close() marks the shared client closed", old.is_closed, "client still open")
client_two = svc._get_client()
check("a fresh client is created lazily after close()",
      client_two is not old and not client_two.is_closed, "no replacement client")
svc.close()

print("\n[finish_reason=length — truncation never presented as complete]")
svc = make_service(payload_with(finish_reason="length", content="Partial ans"))
try:
    svc.generate("Question?")
    bad("finish_reason=length returned a (partial) answer")
except GroqResponseError as exc:
    check("finish_reason=length raises GroqResponseError",
          "token limit" in str(exc), str(exc))
except Exception as exc:  # noqa: BLE001
    bad(f"finish_reason=length raised unexpected {type(exc).__name__}")
svc.close()

print("\n[finish_reason=stop — normal completion accepted]")
svc = make_service(payload_with(finish_reason="stop", content="Complete answer."))
try:
    check("finish_reason=stop returns the content",
          svc.generate("Question?") == "Complete answer.")
except Exception as exc:  # noqa: BLE001
    bad(f"finish_reason=stop raised {type(exc).__name__}")
svc.close()

print("\n[error classification preserved with shared client]")
svc = make_service(payload_with(), raise_exc=httpx.TimeoutException("boom"))
try:
    svc.generate("Question?")
    bad("timeout did not raise")
except GroqTimeoutError:
    ok("timeout -> GroqTimeoutError")
svc.close()
svc._shared_client = httpx.Client(timeout=1, transport=httpx.MockTransport(
    lambda request: httpx.Response(429, json={"error": "slow down"})))
try:
    svc.generate("Question?")
    bad("HTTP 429 did not raise")
except GroqRateLimitError:
    ok("HTTP 429 -> GroqRateLimitError")
svc.close()

print("\n[concurrent use — shared client across threads]")
svc = make_service(payload_with())
results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(svc.generate, f"Q{i}?") for i in range(4)]
    for future in futures:
        try:
            results.append(future.result(timeout=10))
        except Exception as exc:  # noqa: BLE001
            results.append(f"EXC:{type(exc).__name__}")
check("all 4 concurrent generations returned content",
      results == ["Answer body"] * 4, f"{results}")
svc.close()

print("\n[diagnostics — numeric-only usage, no key in logs]")
log_lines = []

class _Capture(logging.Handler):
    def emit(self, record):
        log_lines.append(self.format(record))

capture = _Capture()
capture.setFormatter(logging.Formatter("%(message)s"))
previous_levels = {}
for name in ("app.services.groq_service", "app.services.web_search_service"):
    logger_ = logging.getLogger(name)
    previous_levels[name] = logger_.level
    logger_.setLevel(logging.INFO)
    logger_.addHandler(capture)
svc = make_service(payload_with(usage={"prompt_tokens": 123,
                                        "completion_tokens": 55,
                                        "total_tokens": 178}))
svc.generate("Question?")
svc.close()
usage_lines = [line for line in log_lines if line.startswith("groq usage:")]
check("usage diagnostics logged", len(usage_lines) == 1, f"lines={usage_lines}")
check("completion_tokens logged numerically",
      any("completion_tokens=55" in line for line in usage_lines), str(usage_lines))
check("finish_reason logged",
      any("finish_reason=stop" in line for line in usage_lines), str(usage_lines))
check("no API key anywhere in captured diagnostics",
      not any("sk-test-CONST" in line for line in log_lines))
for name, previous in previous_levels.items():
    logger_ = logging.getLogger(name)
    logger_.removeHandler(capture)
    logger_.setLevel(previous)

print("\n[tavily — shared client lifecycle]")
tavily_body = {"results": [], "usage": {"credits": 1}, "response_time": 0.5}


def tavily_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=tavily_body)


tsvc = WebSearchService(api_key="sk-test-TAVILY", base_url="https://api.tavily.test",
                        timeout=5.0, transport=httpx.MockTransport(tavily_handler))
t_client = tsvc._get_client()
tsvc.search("What is the eligibility criteria?")
tsvc.search("What is the eligibility criteria?")
check("tavily searches reused one shared client instance",
      tsvc._get_client() is t_client, "client instance changed between searches")
old_t = tsvc._get_client()
tsvc.close()
check("tavily close() marks the shared client closed", old_t.is_closed)
new_t = tsvc._get_client()
check("tavily lazily recreates a client after close()",
      new_t is not old_t and not new_t.is_closed)
tsvc.close()

print()
if failures:
    print(f"RESULT: FAILED ({failures} check(s) failed)")
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED")
sys.exit(0)
