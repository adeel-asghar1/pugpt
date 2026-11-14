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

  return (
    <div className="app-shell">
      <Header />
      <main className="flex-1 overflow-y-auto">
        <MessageList messages={messages} isLoading={isLoading} onAsk={handleSend} />
      </main>
      {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}
      <footer className="composer-wrap">
        <div className="composer-inner">
          <ChatInput onSend={handleSend} isLoading={isLoading} />
          <p className="composer-note">
            PUGPT answers only about the University of the Punjab · Verify critical details with official sources · 10 questions/min · 500 questions/day · When live answers are needed, your question may be sent to Tavily (pu.edu.pk only)
          </p>
        </div>
      </footer>
    </div>
  )
}
