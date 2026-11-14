export default function ErrorBanner({ message, onDismiss }) {
  return (
    <div role="alert" className="error-banner">
      <div className="error-inner">
        <span>{message}</span>
        <button onClick={onDismiss} aria-label="Dismiss error" className="error-dismiss">
          ×
        </button>
      </div>
    </div>
  )
}
