import CapIcon from './CapIcon'

export default function Header() {
  return (
    <header className="masthead">
      <div className="masthead-inner">
        <div className="masthead-mark">
          <span className="masthead-seal" aria-hidden="true">
            <CapIcon className="h-6 w-6" />
          </span>
          <span>
            <span className="masthead-wordmark">PUGPT</span>
            <span className="masthead-sub">University of the Punjab · Est. 1882</span>
          </span>
        </div>
        <div className="masthead-right">
          <span className="live-dot" aria-hidden="true"></span>
          Grounded in official
          <br />
          PU documents
        </div>
      </div>
    </header>
  )
}
