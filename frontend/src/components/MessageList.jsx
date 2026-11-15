import { useEffect, useRef } from 'react'
import CapIcon from './CapIcon'
import Message from './Message'

const SUGGESTED_QUESTIONS = [
  { tag: 'ADMISSIONS', label: 'What are the admission requirements for a BS programme?', question: 'What are the eligibility criteria for undergraduate admissions?' },
  { tag: 'FEES', label: 'What is the fee structure for BS Computer Science?', question: 'What is the fee structure of BS Computer Science?' },
  { tag: 'MERIT', label: 'What are the closing merits for BS admissions in 2025?', question: 'What were the closing merits for BS admissions in 2025?' },
  { tag: 'ACADEMIC RULES', label: 'How are GPA and CGPA calculated for Semester Rules?', question: 'What are the rules for calculating CGPA?' },
]

function Landing({ onAsk }) {
  return (
    <div className="landing">
      <p className="landing-pill rise rise-1">University of the Punjab · Official assistant</p>

      <h1 className="landing-title rise rise-2">
        Ask anything about<br /><span className="landing-accent">Punjab University</span>.
      </h1>
      <p className="landing-sub rise rise-3">
        Find answers about admissions, fees, academic rules and merit — grounded in official university documents.
      </p>

      <p className="suggestion-heading rise rise-4">Explore popular questions</p>
      <div className="suggestion-grid rise rise-4" role="list">
        {SUGGESTED_QUESTIONS.map((item, index) => (
          <button
            key={index}
            type="button"
            role="listitem"
            className="suggestion-card"
            onClick={() => onAsk?.(item.question)}
          >
            <span className="suggestion-body">
              <span className="suggestion-tag">{item.tag}</span>
              <span className="suggestion-label">{item.label}</span>
            </span>
            <span className="suggestion-caret" aria-hidden="true">
              <svg viewBox="0 0 16 16" width="15" height="15" fill="none">
                <path d="M3 8h9M8.5 3.5 13 8l-4.5 4.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </span>
          </button>
        ))}
      </div>

      <p className="landing-foot rise rise-5">
        <svg viewBox="0 0 14 14" width="13" height="13" fill="none" aria-hidden="true">
          <circle cx="7" cy="7" r="6.2" stroke="currentColor" strokeWidth="1.3" />
          <path d="M4.6 7.2 6.3 8.9 9.6 5.2" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        Every answer cites its source document and page
      </p>
    </div>
  )
}

function TypingIndicator() {
  return (
    <div className="msg msg-assistant">
      <div className="msg-avatar" aria-hidden="true">
        <CapIcon className="h-4 w-4" />
      </div>
      <div className="msg-body">
        <div className="typing-sheet" role="status" aria-label="PUGPT is consulting official sources">
          <span className="typing-dot" />
          <span className="typing-dot" />
          <span className="typing-dot" />
          <span className="typing-label">Consulting official sources…</span>
        </div>
      </div>
    </div>
  )
}

function VerifyingIndicator() {
  return (
    <div className="msg msg-assistant">
      <div className="msg-avatar" aria-hidden="true">
        <CapIcon className="h-4 w-4" />
      </div>
      <div className="msg-body">
        <div className="typing-sheet" role="status" aria-label="Checking answer against sources">
          <span className="typing-label">Checking answer against sources…</span>
        </div>
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
    return <Landing onAsk={onAsk} />
  }

  const lastMessage = messages[messages.length - 1]
  const lastIsAssistant = lastMessage?.role === 'assistant'
  const lastAssistantVerifying = Boolean(lastIsAssistant && lastMessage.verifying)
  const typingVisible = isLoading && !(
    lastIsAssistant && (lastMessage.content?.length > 0 || lastMessage.verifying)
  )

  return (
    <div className="stream" aria-live="polite" aria-relevant="additions" aria-atomic="false">
      {messages.map((message) => {
        // An assistant bubble with no visible content while streaming would
        // render empty — suppress it (the typing indicator covers this state).
        if (
          message.role === 'assistant' &&
          message.streaming &&
          !message.content
        ) {
          return null
        }
        return (
          <Message
            key={message.id}
            role={message.role}
            content={message.content}
            sources={message.sources}
          />
        )
      })}
      {typingVisible && <TypingIndicator />}
      {lastAssistantVerifying && <VerifyingIndicator />}
      <div ref={bottomRef} />
    </div>
  )
}
