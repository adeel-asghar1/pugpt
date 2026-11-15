// Thin wrapper around the FastAPI backend.
// In development, requests to /api are proxied to http://localhost:8000
// by Vite (see vite.config.js), so relative URLs are used.
import { createSSEParser } from './sse'

const BASE = import.meta.env.VITE_API_BASE_URL ?? '';

export async function sendChatMessage(message) {
  const response = await fetch(`${BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  })

  if (!response.ok) {
    throw new Error(`The server responded with status ${response.status}.`)
  }

  return response.json()
}

/**
 * Progressive (SSE) variant of /api/chat.
 *
 * onEvent is called once per parsed backend event:
 *   { type: 'delta', text }              — provisional token chunk
 *   { type: 'verifying' }                — generation complete, verification pending
 *   { type: 'final', answer, sources, status }     — verified completion
 *   { type: 'replace', answer, status }             — verification rejected; full replacement
 *   { type: 'error', message }                      — generic failure; full replacement
 *   { type: '__malformed__' }                       — unparseable server event
 *
 * Throws (with .status set) when the endpoint rejects the request
 * (429/503) or the connection fails. No automatic retry is performed —
 * callers decide what to do; this function never resends.
 */
export async function streamChatMessage(message, { signal, onEvent } = {}) {
  const response = await fetch(`${BASE}/api/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
    signal,
  })

  if (!response.ok) {
    let detail = null
    try {
      const body = await response.json()
      detail = body?.detail ?? null
    } catch {
      // Non-JSON error body — fall back to a generic message below.
    }
    const error = new Error(detail ?? `The server responded with status ${response.status}.`)
    error.status = response.status
    throw error
  }

  if (!response.body) {
    const error = new Error('Streaming is not supported by this browser.')
    error.status = 0
    throw error
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  const parser = createSSEParser()

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      for (const event of parser.feed(decoder.decode(value, { stream: true }))) {
        onEvent?.(event)
      }
    }
    for (const event of parser.feed(decoder.decode())) {
      onEvent?.(event)
    }
    for (const event of parser.flush()) {
      onEvent?.(event)
    }
  } finally {
    // Release the stream promptly on any exit (including AbortError).
    try {
      await reader.cancel()
    } catch {
      // The reader may already be released — harmless.
    }
  }
}
