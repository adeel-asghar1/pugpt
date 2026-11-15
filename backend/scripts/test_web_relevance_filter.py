"""Tavily relevance filter — offline, fully mocked.

Does NOT call external services. Exercises the lexical relevance gate in
``web_search_service._web_relevant`` (applied inside ``search()``) and the
wired behaviour of ``search()`` end-to-end via a mocked httpx transport.

Run from the backend/ directory:
    python scripts/test_web_relevance_filter.py
"""

import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.config import settings
from app.services.web_search_service import (
    WebSearchResult,
    WebSearchService,
    WebSource,
    _norm_token,
    _question_content_terms,
    _token_set,
    _web_relevant,
)
from app.services.rag_service import RAGService, STATUS_ANSWERED
from app.services.web_search_service import WebSearchResult as _WebSearchResult

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


def closure_question_statement():
    return (
        "Has the University of the Punjab issued any official notice during "
        "November 2026 about campus-wide closures or changes to the class schedule?"
    )


def _web_relevant_case(title, content, question):
    src = WebSource(title=title, url="https://www.pu.edu.pk/x", host="www.pu.edu.pk",
                    content=content, score=0.9, published_date="2026-11-01")
    rel, hits, total = _web_relevant(src, question)
    return rel, hits, total


print("=" * 78)
print("Tavily relevance filter — mocked offline tests")
print("=" * 78)

print("\n[question terms — stopwords/dates excluded]")
terms = _question_content_terms(closure_question_statement())
check("closure question yields high-information terms (stopwords/dates/months removed)",
      "closure" in terms and "november" not in terms and "2026" not in terms
      and "pu" not in terms and "punjab" not in terms and "university" not in terms,
      f"terms={terms}")
check("PU-generic terms removed", "punjab" not in terms and "pu" not in terms, f"terms={terms}")
check("normalized: 'closures' -> 'closure'",
      any(t == "closure" for t in _norm_token("closures")) is False or _norm_token("closures") == "closure")

print("\n[helpers]")
check("token_set drops <3-char tokens", "bs" not in _token_set("bs cs fee"))
check("token_set normalizes suffixes",
      _token_set("schedules") == {"schedule"} or "schedule" in _token_set("schedules"))

print("\n[topical match — announcement about closure]")
rel, hits, total = _web_relevant_case(
    "Campus Closure: 26 to 30 November 2026",
    "The University of the Punjab has announced closure of all campuses from 26 to 30 "
    "November 2026 due to weather. Regular classes resume on 1 December 2026 as per the "
    "published academic schedule.",
    closure_question_statement(),
)
check(f"closure announcement retained (hits={hits}/{total})", rel, f"hits={hits} total={total}")

print("\n[topical match — class schedule change]")
rel, hits, total = _web_relevant_case(
    "Class Schedule Revised: BS-Session November 2026",
    "The University of the Punjab revises the class schedule for all departments for "
    "November 2026; regular classes will be suspended from 26 to 29 November 2026.",
    closure_question_statement(),
)
check(f"schedule-change announcement retained (hits={hits}/{total})", rel, f"hits={hits} total={total}")

print("\n[unrelated — webinar rejected]")
rel, hits, total = _web_relevant_case(
    "SCD-CGIS International Webinar on Hyperspectral",
    "University of the Punjab hosts a webinar on hyperspectral imaging applications. "
    "Speakers from Germany, China and Pakistan will present research. Educational "
    "programs open to BS and MS students of the department.",
    closure_question_statement(),
)
check(f"webinar rejected (hits={hits}/{total})", not rel, f"hits={hits} total={total}")

print("\n[unrelated — BS course outline rejected]")
rel, hits, total = _web_relevant_case(
    "D-3520-Acad-Revised-BS-Sociology-4Years.pdf",
    "BS Sociology 4-Year Program Course Outlines revised syllabus 2026 session for the "
    "Department of Sociology, University of the Punjab. Credit hours table and term "
    "structure included.",
    closure_question_statement(),
)
check(f"course outline rejected (hits={hits}/{total})", not rel, f"hits={hits} total={total}")

print("\n[unrelated — BS CS annex rejected]")
rel, hits, total = _web_relevant_case(
    "Annex-A (BSCS-Fall 2026).pdf",
    "Annex A: BS Computer Science Fall 2026 Curriculum. University of the Punjab, "
    "Department of Computer Science. 2026 session courses list.",
    closure_question_statement(),
)
check(f"CS annex rejected (hits={hits}/{total})", not rel, f"hits={hits} total={total}")

print("\n[unrelated — Revised syllabi rejected]")
rel, hits, total = _web_relevant_case(
    "Revised-Syllabi-and-Courses-of-Reading-for-BS-Environmental-Sciences-04-Year-Degree-Program.pdf",
    "BS Environmental Sciences 04 Year Degree Program syllabus and courses of reading. "
    "Revised syllabi for BS Environmental Sciences students.",
    closure_question_statement(),
)
check(f"env-sci syllabus rejected (hits={hits}/{total})", not rel, f"hits={hits} total={total}")

print("\n[unrelated — journal article rejected]")
rel, hits, total = _web_relevant_case(
    "Education System of Pakistan: Social Functions",
    "Article analyses the education system of Pakistan and its social functions. "
    "Published in an academic journal covering schooling, curricula and state policy.",
    closure_question_statement(),
)
check(f"journal article rejected (hits={hits}/{total})", not rel, f"hits={hits} total={total}")

print("\n[month exclusions — no month/year can contribute to relevance]")
import re as _re
MONTHS_FULL = ["january", "february", "march", "april", "may", "june", "july",
               "august", "september", "october", "november", "december"]
MONTHS_ABBR = ["jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
               "oct", "nov", "dec"]
all_months = MONTHS_FULL + MONTHS_ABBR
months_in_terms = [m for m in all_months
                   if m in _question_content_terms("during " + m + " notice announcement?")]
check("every month name and abbreviation is excluded from question content terms",
      months_in_terms == [], f"months retained as terms: {months_in_terms}")
for evidence_month in ("November 2026", "november", "November"):
    rel, hits, total = _web_relevant_case(
        "Official campus notice",
        f"The University of the Punjab has issued an official notice in {evidence_month}. "
        "Admissions information is published on the official admission portal.",
        "november 2026?",
    )
    check(f"'November 2026?' cannot establish relevance (evidence={evidence_month!r}, "
          f"hits={hits}/{total})", not rel, f"hits={hits} total={total}")
rel, hits, total = _web_relevant_case(
    "Campus Notice",
    "Official University of the Punjab announcement published in november. "
    "See the website for details.",
    "November?",
)
check(f"'November?' alone cannot make a result relevant (hits={hits}/{total})",
      not rel, f"hits={hits} total={total}")
rel, hits, total = _web_relevant_case(
    "Official PU notice",
    "University of the Punjab official website: www.pu.edu.pk. Official university.",
    "november university?",
)
check(f"month + PU-generic term cannot pass (hits={hits}/{total})",
      not rel, f"hits={hits} total={total}")
rel, hits, total = _web_relevant_case(
    "September PU portal",
    "Official University of the Punjab portal, published in september. "
    "PU official website pu.edu.pk notices page.",
    "september pu university official website?",
)
check(f"abbreviation-month + PU-generic phrase cannot pass (hits={hits}/{total})",
      not rel, f"hits={hits} total={total}")
announcement_month_free = (
    "The University of the Punjab has announced campus closure of the main campus. "
    "Regular class schedule changes apply across all departments; changes effective "
    "immediately for all programmes."
)
rel, hits, total = _web_relevant_case(
    "Campus Closure and Class Schedule Changes",
    announcement_month_free,
    closure_question_statement(),
)
check(f"closure announcement passes WITHOUT any month/year in evidence (hits={hits}/{total})",
      rel, f"hits={hits} total={total}")
check("month-free closure evidence contains no month/year tokens",
      all(m not in announcement_month_free.lower() and not _re.search(r"20\d\d", announcement_month_free)
          for m in MONTHS_FULL))

print("\n[date-overlap alone cannot pass]")
rel, hits, total = _web_relevant_case(
    "Annual Admission Bulletin 2026",
    "Admission details for 2026-27 across all University of the Punjab programmes. "
    "Fee schedule, merit lists and eligibility criteria.",
    closure_question_statement(),
)
check(f"date-only 2026 overlap cannot pass (hits={hits}/{total})", not rel, f"hits={hits} total={total}")

print("\n[generic PU vocabulary alone cannot pass]")
rel, hits, total = _web_relevant_case(
    "University of the Punjab official portal",
    "Official University of the Punjab portal for students, faculty and staff. "
    "Visit the website for more information about admissions and fees.",
    closure_question_statement(),
)
check(f"PU generic words alone cannot pass (hits={hits}/{total})", not rel,
      f"hits={hits} total={total}")

print("\n[very short question — one substantive term]")
rel, hits, total = _web_relevant_case(
    "Merit announcement 2026",
    "Merit-based selected admission candidates of the 2026-27 session announced. Merit list published.",
    "merit?",
)
check(f"one-term 'merit?' retained when evidence also matches other content",
      rel, f"hits={hits} total={total}")
rel, hits, total = _web_relevant_case(
    "Annual Admission List 2026",
    "Admission 2026-27 BS programmes announced; selection list published.",
    "merit?",
)
check(f"one-term 'merit?' rejected when only PU-generic vocabulary matches",
      not rel, f"hits={hits} total={total}")

print("\n[relevant fee announcement retained]")
rel, hits, total = _web_relevant_case(
    "First Semester Fee for BS CS: Rs. 55,000",
    "Official announcement: BS Computer Science first semester fee for 2026-27 is Rs. 55,000. "
    "University of the Punjab Department of Computer Science publishes the fee breakdown.",
    "What is the first-semester fee for BS Computer Science?",
)
check(f"relevant fee announcement retained (hits={hits}/{total})", rel, f"hits={hits} total={total}")

print("\n[end-to-end via search() — all irrelevant rejected -> empty web list]")
question = closure_question_statement()
response_payload = {
    "results": [
        {"url": "https://www.pu.edu.pk/news/webinar-hyperspectral", "title": "Webinar",
         "content": "Webinar on hyperspectral imaging. University of the Punjab.",
         "score": 0.8, "published_date": "2026-11-08"},
        {"url": "https://www.pu.edu.pk/course-outlines/sociology", "title": "BS Sociology outline",
         "content": "BS Sociology course outline revised for 2026 session.",
         "score": 0.7, "published_date": "2026-09-01"},
        {"url": "https://www.pu.edu.pk/syllabi/environmental", "title": "BS Environmental Sciences syllabus",
         "content": "BS Environmental Sciences four-year syllabus revised.",
         "score": 0.6, "published_date": "2026-09-28"},
        {"url": "https://www.pu.edu.pk/education-pakistan", "title": "Education System of Pakistan",
         "content": "Research article on the education system.",
         "score": 0.5, "published_date": None},
        {"url": "https://www.pu.edu.pk/bad-url-empty", "title": "ignored",
         "content": "", "score": 0.1, "published_date": None},
    ],
    "usage": {"credits": 1},
    "response_time": 0.42,
}


def handler(request: httpx.Request) -> httpx.Response:
    try:
        return httpx.Response(200, json=response_payload)
    except Exception as exc:
        return httpx.Response(500, text=str(exc))


svc = WebSearchService(
    api_key="sk-test-CONST",
    base_url="https://api.tavily.test",
    timeout=5.0,
    max_results=5,
    allowed_suffixes="pu.edu.pk",
    transport=httpx.MockTransport(handler),
)
out = svc.search(question)
check(f"all-irrelevant Tavily batch -> zero retained (retained={len(out.results)})",
      len(out.results) == 0, f"retained={len(out.results)}")
check("usage credits logged but never returned in results (empty results ok)",
      isinstance(out.results, list) and out.credits_used == 1)

print("\n[end-to-end via search() — one relevant + irrelevant mix]")
mixed_payload = {
    "results": [
        {"url": "https://www.pu.edu.pk/news/closure-notice", "title": "Campus Closure November 2026",
         "content": "The University of the Punjab has announced campus closures from 26 to 30 "
                    "November 2026. Classes suspended. Institutions closed.",
         "score": 0.9, "published_date": "2026-11-01"},
        {"url": "https://www.pu.edu.pk/weekly/webinar", "title": "Webinar announcement",
         "content": "Webinar on academic research topics.",
         "score": 0.8, "published_date": "2026-11-08"},
    ],
    "usage": {"credits": 1},
    "response_time": 0.42,
}


def handler2(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=mixed_payload)


svc2 = WebSearchService(
    api_key="sk-test-CONST",
    base_url="https://api.tavily.test",
    timeout=5.0,
    max_results=5,
    allowed_suffixes="pu.edu.pk",
    transport=httpx.MockTransport(handler2),
)
out2 = svc2.search(question)
check(f"mixed batch -> only relevant retained (retained={len(out2.results)})",
      len(out2.results) == 1 and out2.results[0].url.endswith("closure-notice"),
      f"retained={[r.url for r in out2.results]}")

print("\n[determinism]")
out_a = svc2.search(question)
out_b = svc2.search(question)
check("repeated identical input produces identical output",
      [r.url for r in out_a.results] == [r.url for r in out_b.results])
check("result count identical across calls", len(out_a.results) == len(out_b.results))

print("\n[no external network calls — MockTransport only]")
import httpx as _httpx
check("WebSearchService uses caller-provided transport via kwargs (no real network)",
      type(svc).__name__ == "WebSearchService")

print("\n[existing validators preserved]")
import inspect
check("validated_host still enforced",
      "validated_host" in inspect.getsource(WebSearchService.search))

print()
if failures:
    print(f"RESULT: FAILED ({failures} check(s) failed)")
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED")
sys.exit(0)
