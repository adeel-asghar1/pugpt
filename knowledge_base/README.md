# PUGPT Knowledge Base

This directory holds the **official University of the Punjab source documents** that PUGPT will eventually answer from. It is the only place PUGPT documents should come from.

> **Golden rule: only reliable, official University of the Punjab material may enter this knowledge base.**
> Official URLs alone do not prove a document is current or authoritative. A human must confirm genuineness *and* currency before `verification_status` is set to `verified`. PUGPT must never learn from unofficial, stale, or invented content.

## Layout

```
knowledge_base/
├── documents/            # One folder per category; PDFs live here, named exactly as in the registry
│   ├── admissions/         # Admission policies, schedules, prospectuses, entry tests
│   ├── academics/          # Academic regulations, semester system, degree requirements
│   ├── examinations/       # Exam rules, supplementary papers, result notifications
│   ├── fees/               # Fee schedules, challan info, dues notifications
│   ├── scholarships/       # Scholarships, financial aid, waivers
│   ├── faculties_programs/ # Faculty/department/program information, curricula
│   └── student_services/   # Hostels, transport, health, library, sports
├── metadata/
│   └── sources.json        # The registry: one entry per official document
└── reports/
    └── validation_latest.md  # Latest validator output (generated)
```

Keep filenames stable once registered — the validator, the ingestion pipeline and chat citations all trace back to `<category>/<filename>`.

## How to register a new document

1. Confirm the PDF came from an official PU source (see "Verifying a document").
2. Choose the category folder and save the PDF there, e.g. `documents/admissions/<filename>.pdf`.
3. Add an entry to `metadata/sources.json` (do **not** invent values — leave unknown fields empty):

```json
{
  "document_id": "pu-admissions-prospectus-2025-001",
  "title": "Punjab University Prospectus 2025",
  "filename": "prospectus-2025.pdf",
  "category": "admissions",
  "source_url": "https://www.pu.edu.pk/...",
  "publisher": "University of the Punjab, Directorate of Public Relations",
  "published_date": "2025-04-15",
  "last_updated": "",
  "effective_session": "Autumn 2025",
  "campus": "All campuses",
  "program": "BS/BSc (Hons) admissions",
  "document_type": "prospectus",
  "verification_status": "pending",
  "notes": ""
}
```

4. Run the validator. Fix every error before ingesting anything.
5. Have a human review the document and set `verification_status` to `verified`.
6. Only then ingest with `python scripts/ingest_documents.py`.

## Verifying that a document is genuinely official

- The URL host must be `pu.edu.pk` or a subdomain (e.g. `cs.pu.edu.pk`). Anything else — including `hec.gov.pk`, blogs, Facebook pages, or generic news sites — is **not** official PU material and will fail validation.
- Prefer PDFs hosted on `pu.edu.pk` over reposts elsewhere.
- Check the publisher line and any official circular/letterhead numbers inside the document.
- Check `published_date`, `last_updated` and `effective_session` — official notification PDFs usually state them.
- If you cannot establish authenticity or currency, leave `verification_status` as `pending` or mark it `rejected`. When in doubt, exclude the document.

## Why publication dates and sessions matter

University rules change with every session. A fee schedule for Autumn 2023 or an admission deadline for a past year is **not** a current fact, even if it is official. PUGPT could otherwise give a student dangerously outdated advice. Always record:

- `published_date` / `last_updated` — when the document was issued;
- `effective_session` — the session it applies to (e.g. `Autumn 2025`);
- `campus` / `program` — where it applies, when stated.

Historical documents may be kept in the registry with their session noted, but they must never be presented as currently applicable. PUGPT's context will tell it these details exist for a reason — treat unspecified applicability as **unknown**, not as universal.

## Conflicting or ambiguous documents

- Two official documents can conflict (or one can partially amend another). **Newer is not automatically superior**, and a broad circular does not silently repeal a specific rule.
- When conflicts are found, do not ingest silently. Flag both documents, record the conflict in `notes`, and get human review before choosing what PUGPT may answer.
- Ambiguous documents stay `pending` until clarified.

## Running the validator

```powershell
cd C:\Stoic\pu-gpt\backend
.\venv\Scripts\Activate.ps1
python scripts\validate_knowledge_base.py
```

Exit code **0** = pass, **1** = critical errors (must fix). The script checks: registry JSON, required fields, allowed categories, filename safety, statuses, official PU domains, duplicate IDs/filenames, referenced PDF existence, extractable text, missing dates (warnings), and non-PU domains (critical). A Markdown copy is written to `reports/validation_latest.md`.

## Scanned PDFs and OCR

Many official PU documents are scanned images. Signatures, stamps and seals prove little to no text layer. PyMuPDF extracts **nothing** from them. Unread text means PUGPT cannot cite the document — the validator reports this as a warning, and ingestion must refuse them. Until OCR (e.g. Tesseract) is deliberately added, scanned PDFs cannot join the knowledge base.

## Data quality rules (binding)

- Prefer official PU sources over everything else.
- Preserve document filenames and page references end-to-end.
- Record campus, program and academic session whenever available.
- Never present historical deadlines or fee schedules as current.
- Never assume newer overrides older without review.
- Flag conflicts/ambiguities for manual review — never resolve them silently.
- Never invent missing dates, fees, policies or eligibility requirements; if unknown, say unknown.

## What does NOT happen here

- No document is downloaded or scraped by the tooling in this repository.
- No document enters ChromaDB without registry validation and human verification.
- No document is marked `verified` automatically.
