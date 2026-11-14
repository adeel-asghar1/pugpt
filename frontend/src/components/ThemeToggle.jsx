import { useEffect, useState } from 'react'

const KEY = 'pugpt-theme'

function readInitialTheme() {
  try {
    const saved = localStorage.getItem(KEY)
    if (saved === 'dark' || saved === 'light') return saved
  } catch {
    /* storage unavailable */
  }
  try {
    if (document.documentElement.getAttribute('data-theme')) {
      return document.documentElement.getAttribute('data-theme')
    }
  } catch {
    /* ignore */
  }
  try {
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  } catch {
    return 'light'
  }
}

function SunIcon() {
  return (
    <svg viewBox="0 0 20 20" width="17" height="17" fill="none" aria-hidden="true">
      <circle cx="10" cy="10" r="3.6" stroke="currentColor" strokeWidth="1.4" />
      <path
        d="M10 1.8v1.9M10 16.3v1.9M1.8 10h1.9M16.3 10h1.9M4.2 4.2l1.35 1.35M14.45 14.45L15.8 15.8M4.2 15.8l1.35-1.35M14.45 5.55L15.8 4.2"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
    </svg>
  )
}

function MoonIcon() {
  return (
    <svg viewBox="0 0 20 20" width="17" height="17" fill="none" aria-hidden="true">
      <path
        d="M16.6 12.3A7.1 7.1 0 0 1 7.7 3.4a7.1 7.1 0 1 0 8.9 8.9Z"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
    </svg>
  )
}

export default function ThemeToggle() {
  const [theme, setTheme] = useState(readInitialTheme)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    const meta = document.querySelector('meta[name="theme-color"]')
    if (meta) {
      meta.setAttribute('content', theme === 'dark' ? '#0f1512' : '#f7f9f8')
    }
  }, [theme])

  const toggle = () => {
    const next = theme === 'dark' ? 'light' : 'dark'
    setTheme(next)
    try {
      localStorage.setItem(KEY, next)
    } catch {
      /* storage unavailable — theme still applies for this session */
    }
  }

  const isDark = theme === 'dark'
  return (
    <button
      type="button"
      className="theme-toggle"
      aria-pressed={isDark}
      aria-label={isDark ? 'Switch to light theme' : 'Switch to dark theme'}
      title={isDark ? 'Switch to light theme' : 'Switch to dark theme'}
      onClick={toggle}
    >
      {isDark ? <SunIcon /> : <MoonIcon />}
    </button>
  )
}
