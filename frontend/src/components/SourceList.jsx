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
      <ol className="sources-list">
        {sources.map((source, index) => {
          const isWeb = source.source_type === 'web' || Boolean(source.url)
          if (isWeb) {
            return (
              <li key={index} className="source-item">
                <span className="source-numeral" aria-hidden="true">
                  {String(index + 1).padStart(2, '0')}
                </span>
                <span className="source-body">
                  <span className="live-chip">Live web</span>{' '}
                  <a href={source.url} target="_blank" rel="noreferrer noopener">
                    {source.title || source.url}
                  </a>
                  <span className="source-meta">
                    {hostOf(source.url)} · {source.published_date || 'date unknown'}
                  </span>
                </span>
              </li>
            )
          }
          return (
            <li key={index} className="source-item">
              <span className="source-numeral" aria-hidden="true">
                {String(index + 1).padStart(2, '0')}
              </span>
              <span className="source-body">
                <span className="source-file">{source.filename}</span>
                {source.page_number != null && (
                  <span className="source-page">p.&thinsp;{source.page_number}</span>
                )}
              </span>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
