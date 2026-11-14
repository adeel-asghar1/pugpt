// Thin wrapper around the FastAPI backend.
// In development, requests to /api are proxied to http://localhost:8000
// by Vite (see vite.config.js), so relative URLs are used.
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
