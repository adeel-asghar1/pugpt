// Incremental Server-Sent Events parser.
// The browser may deliver network chunks at arbitrary packet boundaries —
// a chunk can split a line, or split an event *separator* ("\n\n") across
// two chunks. feed() buffers until complete lines/events are available.
//
// Lines starting with "data:" are parsed as JSON events (this is what the
// PUGPT backend emits). Malformed JSON yields { type: '__malformed__' } so
// callers can fail closed instead of trusting partial data. Everything else
// (comment keep-alives, event:/id: lines, blank separators) is ignored.
export function createSSEParser() {
  let buffer = ''

  function extractCompleteEvents(final = false) {
    const events = []
    for (;;) {
      const boundary = buffer.indexOf('\n\n')
      if (boundary === -1) {
        if (final && buffer.trim().length > 0) {
          // Trailing event without a separator at end-of-stream — parse it.
          const rest = buffer
          buffer = ''
          events.push(...parseBlock(rest))
        }
        break
      }
      const block = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      events.push(...parseBlock(block))
    }
    return events
  }

  function parseBlock(block) {
    const events = []
    for (const rawLine of block.split('\n')) {
      const line = rawLine.replace(/\r$/, '')
      if (!line || line.startsWith(':')) continue
      if (!line.startsWith('data:')) continue
      const payload = line.slice(5).trim()
      if (!payload) continue
      try {
        events.push(JSON.parse(payload))
      } catch {
        events.push({ type: '__malformed__' })
      }
    }
    return events
  }

  return {
    feed(chunk) {
      buffer += String(chunk).replace(/\r\n/g, '\n').replace(/\r/g, '\n')
      return extractCompleteEvents(false)
    },
    flush() {
      return extractCompleteEvents(true)
    },
  }
}
