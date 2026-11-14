function hostOf(url) {
  try {
    return new URL(url).hostname
  } catch {
    return url
  }
}

export default function SourceList({ sources }) {
  if (!sources || sources.length === 0) {
    return null
  }

  return (
    <div className="sources">
      <p className="sources-label">Sources</p>
      <ul className="space-y-1">
        {sources.map((source, index) => {
          const isWeb = source.source_type === 'web' || Boolean(source.url)
          if (isWeb) {
            return (
              <li key={index} className="source-row">
                <span className="source-numeral">{String(index + 1).padStart(2, '0')}</span>
                <span className="live-chip">Live web</span>
                <a href={source.url} target="_blank" rel="noreferrer noopener">
                  {source.title || source.url}
                </a>
                <span className="source-page">
                  {hostOf(source.url)} · {source.published_date || 'date unknown'}
                </span>
              </li>
            )
          }
          return (
            <li key={index} className="source-row">
              <span className="source-numeral">{String(index + 1).padStart(2, '0')}</span>
              <span>{source.filename}</span>
              {source.page_number != null && (
                <span className="source-page">p. {source.page_number}</span>
              )}
            </li>
          )
        })}
      </ul>
    </div>
  )
}
