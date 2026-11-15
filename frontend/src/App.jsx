import { useEffect, useRef, useState } from 'react'
import ChatInput from './components/ChatInput'
import ErrorBanner from './components/ErrorBanner'
import Header from './components/Header'
import MessageList from './components/MessageList'
import { streamChatMessage } from './api/client'

// Mirrors backend ERROR_ANSWER — used when the stream dies before the
// server can deliver a terminal event (interrupted connection, etc.).
const GENERIC_ERROR = 'PUGPT is temporarily unable to answer. Please try again in a moment.'

export default function App() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const abortRef = useRef(null)

  // Cancel any in-flight stream if the component unmounts.
  useEffect(() => () => abortRef.current?.abort(), [])

  const appendDelta = (assistantId, text) => {
    setMessages((previous) =>
      previous.map((message) =>
        message.id === assistantId
          ? { ...message, content: message.content + text }
          : message
      )
    )
  }

  const finishAssistant = (assistantId, patch) => {
    setMessages((previous) =>
      previous.map((message) =>
        message.id === assistantId ? { ...message, ...patch } : message
      )
    )
  }

  const handleSend = async (text) => {
    setError(null)
    const assistantId = crypto.randomUUID()
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setMessages((previous) => [
      ...previous,
      { id: crypto.randomUUID(), role: 'user', content: text },
      {
        id: assistantId,
        role: 'assistant',
        content: '',
        sources: [],
        status: null,
        streaming: true,
        verifying: false,
      },
    ])
    setIsLoading(true)

    const startedAt = performance.now()
    let firstTextAt = null
    let terminalReceived = false

    const settleTerminal = (patch) => {
      terminalReceived = true
      finishAssistant(assistantId, { ...patch, streaming: false, verifying: false })
    }

    try {
      await streamChatMessage(text, {
        signal: controller.signal,
        onEvent: (event) => {
          if (event.type === 'delta') {
            if (firstTextAt === null) {
              firstTextAt = performance.now()
              performance.mark('pugpt:first-text')
              console.info(`pugpt: first visible text in ${Math.round(firstTextAt - startedAt)}ms`)
            }
            appendDelta(assistantId, event.text)
            return
          }
          if (event.type === 'verifying') {
            setMessages((previous) =>
              previous.map((message) =>
                message.id === assistantId ? { ...message, verifying: true } : message
              )
            )
            return
          }
          if (event.type === 'final') {
            performance.mark('pugpt:done')
            if (firstTextAt !== null) {
              performance.measure('pugpt:stream', 'pugpt:first-text', 'pugpt:done')
            }
            console.info(
              `pugpt: completion in ${Math.round(performance.now() - startedAt)}ms (stream total)`
            )
            settleTerminal({
              content: event.answer,
              sources: event.sources ?? [],
              status: event.status ?? 'answered',
            })
            return
          }
          if (event.type === 'replace') {
            settleTerminal({
              content: event.answer ?? GENERIC_ERROR,
              sources: [],
              status: event.status ?? 'no_relevant_context',
            })
            return
          }
          if (event.type === 'error' || event.type === '__malformed__') {
            settleTerminal({
              content: event.type === 'error' ? event.message ?? GENERIC_ERROR : GENERIC_ERROR,
              sources: [],
              status: 'error',
            })
            setError('Could not reach PUGPT right now. Please try again in a moment.')
            return
          }
          // Unknown event type — ignore; the server contract is additive-safe.
        },
      })
      if (!terminalReceived) {
        // Connection closed without a terminal event — fail closed.
        settleTerminal({ content: GENERIC_ERROR, sources: [], status: 'error' })
        setError('The connection was interrupted before PUGPT could finish. Please try again.')
      }
    } catch (err) {
      if (controller.signal.aborted) {
        // Intentional cancellation (unmount or superseded send) — settle
        // silently; no error banner.
        settleTerminal({ content: GENERIC_ERROR, sources: [], status: 'error' })
        setIsLoading(false)
        return
      }
      settleTerminal({ content: GENERIC_ERROR, sources: [], status: 'error' })
      console.error('Failed to stream message:', err)
      setError('Could not reach PUGPT right now. Please try again in a moment.')
    } finally {
      setIsLoading(false)
    }
  }

  const isLanding = messages.length === 0 && !isLoading && !error

  const composer = (
    <>
      <ChatInput onSend={handleSend} isLoading={isLoading} />
      <p className="composer-note">Official university documents · Page-linked citations</p>
    </>
  )

  return (
    <div className="app-shell">
      <Header />
      <main className="main-region">
        {isLanding ? (
          <div className="landing-flow">
            <MessageList messages={messages} isLoading={isLoading} onAsk={handleSend} />
            <div className="landing-composer">{composer}</div>
          </div>
        ) : (
          <MessageList messages={messages} isLoading={isLoading} onAsk={handleSend} />
        )}
      </main>
      {!isLanding && (
        <footer className="composer-wrap">
          <div className="composer-inner">{composer}</div>
        </footer>
      )}
      {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}
    </div>
  )
}
