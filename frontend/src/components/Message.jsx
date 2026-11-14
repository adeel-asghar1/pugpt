import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import CapIcon from './CapIcon'
import SourceList from './SourceList'

export default function Message({ role, content, sources }) {
  const isUser = role === 'user'
  const hasAssistantBody = typeof content === 'string' && content.trim().length > 0

  return (
    <div className={`msg-row ${isUser ? 'msg-user-row' : ''}`}>
      {!isUser && (
        <div className="msg-avatar" aria-hidden="true">
          <CapIcon className="h-4 w-4" />
        </div>
      )}
      <div className={`bubble ${isUser ? 'bubble-user' : 'bubble-assistant'}`}>
        {isUser ? (
          <p>{content}</p>
        ) : (
          <div className="markdown-body">
            {hasAssistantBody ? (
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
            ) : (
              <p className="italic text-stone-500">
                PUGPT didn’t return an answer. Please try asking again.
              </p>
            )}
          </div>
        )}
        {!isUser && <SourceList sources={sources} />}
      </div>
    </div>
  )
}
