import ThemeToggle from './ThemeToggle'

export default function Header() {
  return (
    <header className="topbar">
      <div className="topbar-inner">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="17" height="17" fill="none">
              <path d="M5 10.2 12 6l7 4.2-7 4.2-7-4.2Z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
              <path d="M7.6 11.8v3.6c2.5 1.6 6.3 1.6 8.8 0v-3.6M19 10.8v4.4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </span>
          <span className="brand-text">
            <span className="brand-word">PUGPT</span>
            <span className="brand-sub">University of the Punjab · Est. 1882</span>
          </span>
        </div>
        <ThemeToggle />
      </div>
    </header>
  )
}
