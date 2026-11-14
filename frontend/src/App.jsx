import { useState } from 'react'
import ChatInput from './components/ChatInput'
import ErrorBanner from './components/ErrorBanner'
import Header from './components/Header'
import MessageList from './components/MessageList'
import { sendChatMessage } from './api/client'

export default function App() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSend = async (text) => {
    setError(null)
    setMessages((previous) => [
      ...previous,
      { id: crypto.randomUUID(), role: 'user', content: text },
    ])
    setIsLoading(true)

    try {
      const data = await sendChatMessage(text)
      setMessages((previous) => [
        ...previous,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: data.answer,
          sources: data.sources ?? [],
        },
      ])
    } catch (err) {
      console.error('Failed to send message:', err)
      setError(
        'Could not reach PUGPT right now. Please try again in a moment.'
      )
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
