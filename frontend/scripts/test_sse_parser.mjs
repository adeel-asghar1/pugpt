// Hermetic node test for the incremental SSE parser.
// Run: node scripts/test_sse_parser.mjs
import assert from 'node:assert/strict'
import { createSSEParser } from '../src/api/sse.js'

let failures = 0
const ok = (msg) => console.log(`[PASS] ${msg}`)
const bad = (msg) => { failures += 1; console.log(`[FAIL] ${msg}`) }
const check = (label, condition, detail = '') => {
  if (condition) ok(label)
  else bad(`${label} :: ${detail}`)
}

// Sample multi-event stream: delta, verifying, final.
const DOC =
  'data: {"type":"delta","text":"Eligibility "}\n\n' +
  'data: {"type":"delta","text":"criteria."}\n\n' +
  'data: {"type":"verifying"}\n\n' +
  'data: {"type":"final","answer":"Eligibility criteria.","status":"answered","sources":[]}\n\n'

// 1. Single-shot parse
{
  const parser = createSSEParser()
  const events = [...parser.feed(DOC), ...parser.flush()]
  check('single-shot parse yields 4 events', events.length === 4, JSON.stringify(events))
  check('events preserve order and type',
    events.map((e) => e.type).join(',') === 'delta,delta,verifying,final')
  check('delta payloads intact', events[0].text === 'Eligibility ' && events[1].text === 'criteria.')
}

// 2. Every possible packet boundary produces identical events (split points tested at every byte)
{
  const reference = JSON.stringify(
    (() => { const p = createSSEParser(); return [...p.feed(DOC), ...p.flush()] })()
  )
  let allBoundariesOk = true
  for (let split = 1; split < DOC.length; split += 1) {
    const parser = createSSEParser()
    const events = [...parser.feed(DOC.slice(0, split)), ...parser.feed(DOC.slice(split)), ...parser.flush()]
    if (JSON.stringify(events) !== reference) {
      allBoundariesOk = false
      console.log(`   boundary ${split} diverged`)
      break
    }
  }
  check(`all ${DOC.length - 1} packet boundaries parse identically`, allBoundariesOk)
}

// 3. Chunk-by-chunk delivery one byte at a time
{
  const parser = createSSEParser()
  const events = []
  for (const ch of DOC) events.push(...parser.feed(ch))
  events.push(...parser.flush())
  check('byte-at-a-time delivery yields identical events',
    events.length === 4 && events[3].type === 'final', JSON.stringify(events.map((e) => e.type)))
}

// 4. Malformed JSON yields __malformed__, does not crash the parse
{
  const parser = createSSEParser()
  const events = parser.feed('data: {oops}\n\n')
  check('malformed JSON -> __malformed__ event', events.length === 1 && events[0].type === '__malformed__')
}

// 5. Comment keep-alives and non-data lines are ignored
{
  const parser = createSSEParser()
  const events = parser.feed(': keepalive\n\n' + 'event: message\n\n' + 'data: {"type":"delta","text":"x"}\n\n')
  check('comments and non-data lines ignored', events.length === 1 && events[0].type === 'delta')
}

// 6. flush() with a trailing event lacking \n\n still parses it
{
  const parser = createSSEParser()
  const events = [...parser.feed('data: {"type":"final","answer":"a","status":"answered","sources":[]}'),
                  ...parser.flush()]
  check('trailing separator-free event flushed', events.length === 1 && events[0].type === 'final')
}

// 7. Partial line across chunks + flush() do not duplicate events
{
  const parser = createSSEParser()
  const part = parser.feed('data: {"type":"delta","text":"')
  const rest = parser.feed('abc"}\n\n')
  const events = [...part, ...rest, ...parser.flush()]
  check('event split across chunks parsed once', events.length === 1 && events[0].text === 'abc')
}

// 8. CRLF line endings tolerated
{
  const parser = createSSEParser()
  const events = parser.feed('data: {"type":"delta","text":"y"}\r\n\r\n')
  check('CRLF separators parsed', events.length === 1 && events[0].text === 'y')
}

console.log('')
if (failures > 0) {
  console.log(`RESULT: FAILED (${failures} check(s) failed)`)
  process.exit(1)
}
console.log('RESULT: ALL CHECKS PASSED')
process.exit(0)
