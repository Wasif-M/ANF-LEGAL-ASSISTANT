import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkBreaks from 'remark-breaks'
import remarkGfm from 'remark-gfm'
import { synthesizeSpeech } from '../utils/api'
import '../styles/MessageBubble.css'

// Detect Urdu/Arabic script so we can render RTL and pick the right voice even
// for older messages that were saved without a `language` field.
const URDU_RE = /[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]/
const detectIsUrdu = (message) => {
  if (message.language === 'ur') return true
  if (message.language === 'en') return false
  return URDU_RE.test(message.content || '')
}

// Per-answer "play the response aloud" control (text-to-speech).
function SpeakButton({ message, isUrdu }) {
  const [state, setState] = useState('idle') // idle | loading | playing
  const audioRef = useRef(null)
  const urlRef = useRef(null)

  // Clean up the audio + object URL when the message unmounts
  useEffect(() => () => {
    if (audioRef.current) audioRef.current.pause()
    if (urlRef.current) URL.revokeObjectURL(urlRef.current)
  }, [])

  const stop = () => {
    if (audioRef.current) { audioRef.current.pause(); audioRef.current.currentTime = 0 }
    setState('idle')
  }

  const play = async () => {
    if (state === 'playing') { stop(); return }
    // Reuse already-synthesised audio if present
    if (audioRef.current && urlRef.current) {
      audioRef.current.play(); setState('playing'); return
    }
    setState('loading')
    try {
      const url = await synthesizeSpeech(message.content, isUrdu ? 'ur' : 'en')
      urlRef.current = url
      const audio = new Audio(url)
      audioRef.current = audio
      audio.onended = () => setState('idle')
      audio.onerror = () => setState('idle')
      await audio.play()
      setState('playing')
    } catch (err) {
      console.error('TTS failed:', err)
      setState('idle')
    }
  }

  return (
    <button
      type="button"
      className={`speak-btn ${state}`}
      onClick={play}
      disabled={state === 'loading'}
      title={state === 'playing' ? 'Stop' : 'Play response'}
      aria-label={state === 'playing' ? 'Stop audio' : 'Play response aloud'}
    >
      {state === 'loading' ? (
        <span className="speak-spinner" />
      ) : state === 'playing' ? (
        <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1"></rect><rect x="14" y="5" width="4" height="14" rx="1"></rect></svg>
      ) : (
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path><path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path></svg>
      )}
      <span className="speak-btn-label">{state === 'playing' ? 'Stop' : 'Listen'}</span>
    </button>
  )
}

function StarRating({ message, onRate }) {
  const [hovered, setHovered] = useState(0)
  const rating = message.rating || 0
  const display = hovered || rating

  return (
    <div className="star-rating" onMouseLeave={() => setHovered(0)}>
      <span className="star-rating-label">
        {rating > 0 ? `You rated ${rating}/5` : 'Rate this answer'}
      </span>
      <div className="star-rating-stars">
        {[1, 2, 3, 4, 5].map((star) => (
          <button
            key={star}
            type="button"
            className={`star-btn ${star <= display ? 'filled' : ''} ${hovered > 0 ? 'previewing' : ''}`}
            onMouseEnter={() => setHovered(star)}
            onClick={() => onRate?.(message, star)}
            title={`${star} star${star > 1 ? 's' : ''}`}
            aria-label={`Rate ${star} out of 5 stars`}
          >
            <svg width="16" height="16" viewBox="0 0 24 24"
              fill={star <= display ? 'currentColor' : 'none'}
              stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round">
              <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
            </svg>
          </button>
        ))}
      </div>
    </div>
  )
}

function ThinkingBlock({ message }) {
  const steps = message.thinking || []
  if (steps.length === 0) return null

  // Live while the model is still reasoning (no answer text yet)
  if (message.isThinking && !message.content) {
    return (
      <div className="thinking-block active">
        <div className="thinking-header">
          <span className="thinking-spinner" />
          <span className="thinking-label">Thinking…</span>
        </div>
        <ul className="thinking-steps">
          {steps.map((step, i) => (
            <li key={i} className={i === steps.length - 1 ? 'current' : 'done'}>{step}</li>
          ))}
        </ul>
      </div>
    )
  }

  // Once the answer streams, collapse the trail into an expandable summary
  return (
    <details className="thinking-block collapsed">
      <summary>Thought process · {steps.length} step{steps.length > 1 ? 's' : ''}</summary>
      <ul className="thinking-steps">
        {steps.map((step, i) => <li key={i} className="done">{step}</li>)}
      </ul>
    </details>
  )
}

function MessageBubble({ message, onRate }) {
  const isUser = message.role === 'user'
  const isError = message.isError
  // Ratable once the answer is fully streamed and persisted (has a DB id)
  const canRate = !isUser && !isError && !message.isStreaming && message.dbId
  const isUrdu = detectIsUrdu(message)
  // Offer "Listen" on any non-empty, non-error message once it stops streaming
  const canSpeak = !isError && !message.isStreaming && (message.content || '').trim().length > 0

  return (
    <div className={`message-bubble ${message.role} ${isError ? 'error' : ''}`}>
      <div className="message-avatar">
        {isUser ? 'U' : 'AI'}
      </div>
      <div className="message-content">
        {!isUser && <ThinkingBlock message={message} />}
        <div
          className={`message-text ${isUrdu ? 'urdu-text' : ''}`}
          dir={isUrdu ? 'rtl' : 'ltr'}
        >
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkBreaks]}
            components={{
              p: ({node, ...props}) => <p style={{ color: isUser ? 'white' : 'inherit' }} {...props} />,
              h1: ({node, ...props}) => <h1 className="markdown-heading markdown-h1" {...props} />,
              h2: ({node, ...props}) => <h2 className="markdown-heading markdown-h2" {...props} />,
              h3: ({node, ...props}) => <h3 className="markdown-heading markdown-h3" {...props} />,
              code: ({node, inline, ...props}) =>
                inline ? <code className="inline-code" {...props} /> : <pre className="code-block"><code {...props} /></pre>,
              ul: ({node, ...props}) => <ul className="markdown-list" {...props} />,
              ol: ({node, ...props}) => <ol className="markdown-list" {...props} />,
              blockquote: ({node, ...props}) => <blockquote className="markdown-quote" {...props} />,
              a: ({node, ...props}) => <a className="markdown-link" target="_blank" rel="noopener noreferrer" {...props} />,
              hr: ({node, ...props}) => <hr className="markdown-hr" {...props} />,
              table: ({node, ...props}) => (
                <div className="markdown-table-wrap">
                  <table className="markdown-table" {...props} />
                </div>
              ),
              thead: ({node, ...props}) => <thead className="markdown-thead" {...props} />,
              th: ({node, ...props}) => <th className="markdown-th" {...props} />,
              td: ({node, ...props}) => <td className="markdown-td" {...props} />,
            }}
          >
            {message.content || ''}
          </ReactMarkdown>
        </div>
        <div className="message-footer">
          {canSpeak && <SpeakButton message={message} isUrdu={isUrdu} />}
          {canRate && <StarRating message={message} onRate={onRate} />}
          <span className="message-time">
            {new Date(message.timestamp).toLocaleTimeString('en-US', {hour: '2-digit', minute: '2-digit'})}
          </span>
        </div>
      </div>
    </div>
  )
}

export default MessageBubble
