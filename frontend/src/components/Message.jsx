import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import CapIcon from './CapIcon'
import SourceList from './SourceList'

export default function Message({ role, content, sources }) {
  const isUser = role === 'user'
  const hasAssistantBody = typeof content === 'string' && content.trim().length > 0

  if (isUser) {
    return (
      <div className="msg msg-user">
        <div className="user-bubble">
          <p>{content}</p>
        </div>
      </div>
    )
  }

  return (
    <article className="msg msg-assistant">
      <div className="msg-avatar" aria-hidden="true">
        <CapIcon className="h-4 w-4" />
      </div>
      <div className="msg-body">
        <div className="assistant-panel">
          {hasAssistantBody ? (
            <div className="markdown-body">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
            </div>
          ) : (
            <p className="answer-empty">PUGPT didn’t return an answer. Please try asking again.</p>
          )}
          <SourceList sources={sources} />
        </div>
      </div>
    </article>
  )
}
