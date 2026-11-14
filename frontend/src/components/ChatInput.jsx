import { useState } from 'react'

export default function ChatInput({ onSend, isLoading }) {
  const [value, setValue] = useState('')

  const handleSubmit = (event) => {
    event.preventDefault()
    const trimmed = value.trim()
    if (!trimmed || isLoading) return
    onSend(trimmed)
    setValue('')
  }

  const canSend = !isLoading && value.trim().length > 0
  const showCount = value.length >= 1800

  return (
    <form onSubmit={handleSubmit} className="composer">
      <input
        type="text"
        value={value}
        onChange={(event) => setValue(event.target.value)}
        disabled={isLoading}
        maxLength={2000}
        placeholder="Ask about admissions, fees, semesters…"
        aria-label="Your question"
        className="composer-input"
      />
      {showCount && (
        <span className="composer-count on" aria-hidden="true">
          {value.length}/2000
        </span>
      )}
      <button
        type="submit"
        disabled={!canSend}
        aria-label="Send question"
        className="composer-send"
      >
        <span>Ask</span>
        <svg viewBox="0 0 16 16" width="15" height="15" fill="none" aria-hidden="true">
          <path d="M13.5 2.5 7 9M13.5 2.5 9.8 13.5 7 9m6.5-6.5L2.5 7 7 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
    </form>
  )
}
