// Thin wrapper around the FastAPI backend.
// In development, requests to /api are proxied to http://localhost:8000
// by Vite (see vite.config.js), so relative URLs are used.
export async function sendChatMessage(message) {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  })

  if (!response.ok) {
    throw new Error(`The server responded with status ${response.status}.`)
  }

  return response.json()
}
