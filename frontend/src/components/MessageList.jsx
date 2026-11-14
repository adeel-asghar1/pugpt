import { useEffect, useRef } from 'react'
import CapIcon from './CapIcon'
import Message from './Message'

const SUGGESTED_QUESTIONS = [
  { label: 'What are the admission requirements for a BS programme?', question: 'What are the eligibility criteria for undergraduate admissions?' },
  { label: 'What is the fee structure for BS Computer Science?', question: 'What is the fee structure of BS Computer Science?' },
  { label: 'What are the closing merits for BS admissions in 2025?', question: 'What were the closing merits for BS admissions in 2025?' },
  { label: 'How are GPA and CGPA calculated for Semester Rules?', question: 'What are the rules for calculating CGPA?' },
]

const INDEX = [
  ['I', 'Admissions & Eligibility, 2026-27', 'eligibility'],
  ['II', 'Fee / Dues Schedule, 2026-27', 'fees'],
  ['III', 'Semester Rules, Undergraduate', 'rules'],
  ['IV', 'Merit of Last Year — 2025', 'merit'],
  ['V', 'Admission Regulations', 'policy'],
]

function WelcomePanel({ onAsk }) {
  return (
    <div className="welcome-grid">
      <div className="welcome-main">
        <div className="eyebrow rise rise-1">The University of the Punjab · Official assistant</div>
        <h1 className="welcome-title rise rise-2">
          Ask anything about <em>Punjab University.</em>
        </h1>
        <p className="welcome-sub rise rise-3">
          Admissions, fee schedules, semester rules, eligibility and closing
          merits — answered only from the University’s own official documents,
          with every claim cited.
        </p>

        <div className="welcome-rule rise rise-3" aria-hidden="true">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
            <path d="M7 1l1.6 3.9L12.6 6.4 8.8 8.3 7 12.4 5.2 8.3 1.4 6.4 5.4 4.9 7 1z"
              stroke="currentColor" strokeWidth="1.1" strokeLinejoin="round" fill="var(--gold-tint)" />
          </svg>
        </div>

        <div className="suggestion-list rise rise-4">
          {SUGGESTED_QUESTIONS.map((item, index) => (
            <button
              key={index}
              type="button"
              className="suggestion-row"
              onClick={() => onAsk?.(item.question)}
            >
              <span className="suggestion-index">{String(index + 1).padStart(2, '0')}</span>
              <span className="suggestion-text">{item.label}</span>
              <span className="suggestion-arrow" aria-hidden="true">
                <svg width="17" height="17" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                  <path d="M2.5 8h10M8.5 3.5 13 8l-4.5 4.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </span>
            </button>
          ))}
        </div>

        <div className="welcome-footnote rise rise-5">Every answer cites its source document and page</div>
      </div>

      <aside className="welcome-aside rise rise-5" aria-hidden="true">
        <div className="index-card">
          <div className="index-head">
            <span className="index-title">INDEX</span>
            <span className="index-meta">pugpt.knowledge</span>
          </div>
          {INDEX.map(([numeral, title, doc]) => (
            <div key={numeral} className="index-item">
              <span className="index-numeral">{numeral}</span>
              <span>{title}</span>
              <span className="index-dots" />
              <span className="index-doc">{doc}</span>
            </div>
          ))}
          <div className="index-foot">
            <span>official pu.edu.pk</span>
            <span>↓</span>
          </div>
        </div>
      </aside>
    </div>
  )
}

function TypingIndicator() {
  return (
    <div className="msg-row">
      <div className="msg-avatar" aria-hidden="true">
        <CapIcon className="h-4 w-4" />
      </div>
      <div>
        <div className="typing-bubble" role="status" aria-label="PUGPT is thinking">
          <span className="typing-dot" />
          <span className="typing-dot" />
          <span className="typing-dot" />
        </div>
        <div className="typing-note">Consulting official sources…</div>
      </div>
    </div>
  )
}

export default function MessageList({ messages, isLoading, onAsk }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  if (messages.length === 0 && !isLoading) {
    return <WelcomePanel onAsk={onAsk} />
  }

  return (
    <div
      className="stream"
      aria-live="polite"
      aria-relevant="additions"
      aria-atomic="false"
    >
      {messages.map((message) => (
        <Message
          key={message.id}
          role={message.role}
          content={message.content}
          sources={message.sources}
        />
      ))}
      {isLoading && <TypingIndicator />}
      <div ref={bottomRef} />
    </div>
  )
}
