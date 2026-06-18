import { useState, useEffect, useRef } from 'react'
import { queryAPI, fetchDocuments, rateMessageAPI, transcribeAudio } from '../utils/api'
import MessageBubble from './MessageBubble'
import '../styles/ChatArea.css'

// Answer-language options for the toggle. "auto" follows the question language.
const LANG_OPTIONS = [
  { key: 'auto', label: 'Auto' },
  { key: 'en', label: 'EN' },
  { key: 'ur', label: 'اردو' },
]

function ChatArea({ conversation, onAddMessage, onUpdateTitle, onCreateConversation }) {
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [retrievedSources, setRetrievedSources] = useState([])
  const [documents, setDocuments] = useState(null) // null = loading
  const [language, setLanguage] = useState('auto') // 'auto' | 'en' | 'ur'
  const [isRecording, setIsRecording] = useState(false)
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [voiceError, setVoiceError] = useState('')
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])

  // Load the list of indexed documents for the welcome screen
  useEffect(() => {
    if (conversation) return
    let cancelled = false
    fetchDocuments()
      .then((data) => { if (!cancelled) setDocuments(data) })
      .catch((err) => {
        console.error('Failed to load documents:', err)
        if (!cancelled) setDocuments({ documents: [], total_documents: 0, total_chunks: 0 })
      })
    return () => { cancelled = true }
  }, [conversation])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [conversation?.messages])

  useEffect(() => {
    inputRef.current?.focus()
  }, [conversation?.id])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const handleSendMessage = async (e) => {
    if (e) e.preventDefault()
    if (!inputValue.trim() || !conversation || isLoading) return

    const currentInput = inputValue;
    const timestamp = new Date().toISOString();
    const uniqueId = "user-" + Date.now() + "-" + Math.random().toString(36).substr(2, 9);
    
    console.log('ChatArea: User input captured:', currentInput, 'ID:', uniqueId);
    
    const userMessage = {
      id: uniqueId,
      role: 'user',
      content: currentInput,
      timestamp: timestamp
    }

    console.log('ChatArea: User message object:', userMessage);
    // Clear input immediately for better UX
    setInputValue('')
    
    // Add user message first - ensure it's visible
    console.log('ChatArea: Adding user message to conversation');
    onAddMessage(userMessage, conversation.id)
    
    // Scroll and set loading after state updates
    setTimeout(() => {
      console.log('ChatArea: Scrolling to bottom and setting loading state');
      scrollToBottom()
      setIsLoading(true)
    }, 100);

    try {
      // Create assistant message that will be updated as we stream
      const assistantMessageId = "assistant-" + Date.now() + "-" + Math.random().toString(36).substr(2, 9);
      const assistantMessage = {
        id: assistantMessageId,
        role: 'assistant',
        content: '',
        timestamp: new Date().toISOString(),
        sources: [],
        chunks: [],
        isStreaming: true
      }

      console.log('ChatArea: Adding empty assistant message with ID:', assistantMessageId);
      onAddMessage(assistantMessage, conversation.id)
      
      // Update conversation title if it's still "New Chat"
      if (conversation.title === 'New Chat') {
        const titlePreview = currentInput.substring(0, 30)
        onUpdateTitle(conversation.id, titlePreview + (currentInput.length > 30 ? '...' : ''))
      }

      scrollToBottom()

      let fullAnswer = ''
      let sources = []
      let thinkingLines = []
      // Default the rendered language to the toggle; the server confirms it via
      // a 'language' event (important when the toggle is on "auto").
      let answerLang = language === 'ur' ? 'ur' : (language === 'en' ? 'en' : 'auto')

      const pushUpdate = (overrides) => {
        onAddMessage({
          id: assistantMessageId,
          role: 'assistant',
          content: fullAnswer,
          timestamp: assistantMessage.timestamp,
          sources: sources,
          chunks: assistantMessage.chunks,
          thinking: thinkingLines,
          language: answerLang,
          isThinking: false,
          isStreaming: true,
          ...overrides
        }, conversation.id)
      }

      // Use streaming API with callback; conversation id makes the server
      // persist both the question and the final answer to SQLite
      await queryAPI(currentInput, (event) => {
        if (event.type === 'language') {
          answerLang = event.content
          pushUpdate({ isThinking: fullAnswer.length === 0 })
        } else if (event.type === 'thinking') {
          // Model/retriever progress: show as a live "Thinking…" trail until
          // the first answer token arrives
          thinkingLines = [...thinkingLines, event.content]
          pushUpdate({ isThinking: fullAnswer.length === 0 })
          scrollToBottom()
        } else if (event.type === 'chunk' || event.type === 'answer') {
          fullAnswer += event.content
          console.log('ChatArea: Streaming update, fullAnswer length:', fullAnswer.length);
          pushUpdate()
          scrollToBottom()
        } else if (event.type === 'sources') {
          sources = event.content
          console.log('ChatArea: Sources received');
          pushUpdate()
        } else if (event.type === 'done') {
          console.log('ChatArea: Streaming done');
          pushUpdate({ isStreaming: false })
          setRetrievedSources([])
        } else if (event.type === 'saved') {
          // Server persisted the answer — attach its DB id so it can be rated
          pushUpdate({ isStreaming: false, dbId: event.messageId })
        }
      }, conversation.id, language)
    } catch (error) {
      console.error('Error querying API:', error)
      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: `Error: ${error.message}. Please try again.`,
        timestamp: new Date(),
        isError: true
      }
      onAddMessage(errorMessage, conversation.id)
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage(e)
    }
  }

  // ─── Voice input (record → transcribe → fill the input box) ───

  const startRecording = async () => {
    setVoiceError('')
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setVoiceError('Voice input is not supported in this browser.')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      // Pick a mime type the browser actually supports
      const mime = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg']
        .find((t) => MediaRecorder.isTypeSupported(t)) || ''
      const recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined)
      audioChunksRef.current = []
      recorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunksRef.current.push(e.data) }
      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop())
        const blob = new Blob(audioChunksRef.current, { type: mime || 'audio/webm' })
        if (blob.size === 0) { setIsTranscribing(false); return }
        setIsTranscribing(true)
        try {
          const { text } = await transcribeAudio(blob, language)
          if (text) {
            setInputValue((prev) => (prev ? prev + ' ' : '') + text)
            inputRef.current?.focus()
          } else {
            setVoiceError("Couldn't hear anything — please try again.")
          }
        } catch (err) {
          console.error('Transcription failed:', err)
          setVoiceError(err.message || 'Transcription failed.')
        } finally {
          setIsTranscribing(false)
        }
      }
      mediaRecorderRef.current = recorder
      recorder.start()
      setIsRecording(true)
    } catch (err) {
      console.error('Mic access failed:', err)
      setVoiceError('Microphone access was blocked. Please allow it and try again.')
    }
  }

  const stopRecording = () => {
    const recorder = mediaRecorderRef.current
    if (recorder && recorder.state !== 'inactive') {
      recorder.stop()
    }
    setIsRecording(false)
  }

  const toggleRecording = () => {
    if (isRecording) stopRecording()
    else startRecording()
  }

  const handleRateMessage = async (message, rating) => {
    if (!message.dbId) return
    // Optimistic update; revert if the server rejects it
    const previous = message.rating ?? null
    onAddMessage({ id: message.id, rating }, conversation.id)
    try {
      await rateMessageAPI(message.dbId, rating)
    } catch (err) {
      console.error('Failed to save rating:', err)
      onAddMessage({ id: message.id, rating: previous }, conversation.id)
    }
  }

  if (!conversation) {
    return (
      <div className="chat-area empty">
        <div className="empty-state-chat">
          <div className="logo">⚖️</div>
          <h1>ANF AI Legal Assistant</h1>
          <p>Ask questions about Pakistani law and get grounded, section-cited answers.</p>
          <button className="start-chat-btn" onClick={onCreateConversation}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
              <line x1="12" y1="5" x2="12" y2="19"></line>
              <line x1="5" y1="12" x2="19" y2="12"></line>
            </svg>
            Start a new chat
          </button>

          <div className="documents-panel">
            <div className="documents-panel-header">
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
                <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
              </svg>
              <span>
                Indexed legal documents
                {documents && documents.total_documents > 0 && ` (${documents.total_documents})`}
              </span>
            </div>
            {documents === null ? (
              <div className="documents-loading">
                <span className="documents-spinner" />
                Loading document list…
              </div>
            ) : documents.documents.length === 0 ? (
              <div className="documents-empty">No documents indexed yet.</div>
            ) : (
              <div className="documents-grid">
                {documents.documents.map((doc) => (
                  <div key={doc.file} className="document-card" title={doc.file}>
                    <div className="document-icon">
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                        <polyline points="14 2 14 8 20 8"></polyline>
                      </svg>
                    </div>
                    <div className="document-meta">
                      <span className="document-title">{doc.title}</span>
                      <span className="document-file">{doc.file}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="chat-area">
      <div className="chat-header">
        <h2>{conversation.title}</h2>
        <div className="header-actions">
          <button className="header-btn" title="Settings">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="1"></circle>
              <circle cx="12" cy="5" r="1"></circle>
              <circle cx="12" cy="19" r="1"></circle>
            </svg>
          </button>
        </div>
      </div>

      <div className="chat-content">
        <div className="messages-container">
          {conversation.messages && conversation.messages.map((msg) => {
            console.log('Rendering message:', msg.role, msg.id, 'content:', msg.content?.substring(0, 50));
            return <MessageBubble key={msg.id} message={msg} onRate={handleRateMessage} />;
          })}
          {isLoading && !conversation.messages.some(m => m.role === 'assistant' && m.isStreaming) && (
            <div className="message-bubble assistant loading">
              <div className="message-avatar">AI</div>
              <div className="message-content">
                <div className="message-text">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {retrievedSources.length > 0 && (
          <div className="sources-panel">
            <details className="sources-details">
              <summary>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
                  <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
                </svg>
                <span>Retrieved Sources ({retrievedSources.length})</span>
              </summary>
              <div className="sources-list">
                {retrievedSources.slice(0, 3).map((chunk, idx) => (
                  <div key={idx} className="source-item">
                    <div className="source-rank">#{chunk.rank}</div>
                    <div className="source-content">
                      <p className="source-path">{chunk.source_path}</p>
                      <p className="source-text">{chunk.text.substring(0, 100)}...</p>
                      <div className="source-scores">
                        <span className="score-badge">Dense: {(chunk.dense_score || 0).toFixed(2)}</span>
                        <span className="score-badge">Fused: {(chunk.fused_score || 0).toFixed(2)}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </details>
          </div>
        )}
      </div>

      <form className="input-area" onSubmit={handleSendMessage}>
        <div className="input-toolbar">
          <div className="lang-toggle" role="group" aria-label="Answer language">
            <span className="lang-toggle-label">Answer in:</span>
            {LANG_OPTIONS.map((opt) => (
              <button
                key={opt.key}
                type="button"
                className={`lang-btn ${language === opt.key ? 'active' : ''} ${opt.key === 'ur' ? 'urdu' : ''}`}
                onClick={() => setLanguage(opt.key)}
                title={`Answer in ${opt.label}`}
              >
                {opt.label}
              </button>
            ))}
          </div>
          {voiceError && <span className="voice-error">{voiceError}</span>}
        </div>
        <div className="input-wrapper">
          <textarea
            ref={inputRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isTranscribing ? 'Transcribing your voice…' : 'Ask about legal documents… (Shift+Enter for new line)'}
            rows="1"
            disabled={isLoading || isTranscribing}
            className="message-input"
          />
          <button
            type="button"
            onClick={toggleRecording}
            disabled={isLoading || isTranscribing}
            className={`mic-btn ${isRecording ? 'recording' : ''}`}
            title={isRecording ? 'Stop recording' : 'Ask by voice'}
            aria-label={isRecording ? 'Stop recording' : 'Ask by voice'}
          >
            {isTranscribing ? (
              <span className="mic-spinner" />
            ) : isRecording ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <rect x="6" y="6" width="12" height="12" rx="2"></rect>
              </svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                <line x1="12" y1="19" x2="12" y2="23"></line>
                <line x1="8" y1="23" x2="16" y2="23"></line>
              </svg>
            )}
          </button>
          <button
            type="submit"
            disabled={isLoading || isTranscribing || !inputValue.trim()}
            className="send-btn"
            title="Send (Enter)"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <path d="M16.6915026,12.4744748 L3.50612381,13.2599618 C3.19218622,13.2599618 3.03521743,13.4170592 3.03521743,13.5741566 L1.15159189,20.0151496 C0.8376543,20.8006365 0.99,21.89 1.77946707,22.52 C2.41,22.99 3.50612381,23.1 4.13399899,22.8429026 L21.714504,14.0454487 C22.6563168,13.5741566 23.1272231,12.6315722 22.9702544,11.6889879 L4.13399899,1.16346276 C3.34915502,0.9 2.40734225,1.00636533 1.77946707,1.4776575 C0.994623095,2.10604706 0.837654326,3.0486314 1.15159189,3.99701575 L3.03521743,10.4380088 C3.03521743,10.5951061 3.34915502,10.7522035 3.50612381,10.7522035 L16.6915026,11.5376905 C16.6915026,11.5376905 17.1624089,11.5376905 17.1624089,12.0089827 C17.1624089,12.4744748 16.6915026,12.4744748 16.6915026,12.4744748 Z"></path>
            </svg>
          </button>
        </div>
        <p className="input-hint">
          {isRecording ? '● Recording… click the stop button when you finish speaking' : 'AI Legal Assistant • Always verify important information'}
        </p>
      </form>
    </div>
  )
}

export default ChatArea
