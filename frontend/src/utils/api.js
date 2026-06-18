import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// ─── Auth token handling ───

export const getToken = () => localStorage.getItem('auth_token')
export const setToken = (token) => localStorage.setItem('auth_token', token)
export const clearToken = () => localStorage.removeItem('auth_token')

apiClient.interceptors.request.use((config) => {
  const token = getToken()
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

const extractError = (error, fallback) =>
  new Error(error.response?.data?.detail || error.message || fallback)

// ─── Auth API ───

export const signupAPI = async (username, email, password) => {
  try {
    const res = await apiClient.post('/auth/signup', { username, email, password })
    setToken(res.data.token)
    return res.data.user
  } catch (error) {
    throw extractError(error, 'Signup failed')
  }
}

export const loginAPI = async (username, password) => {
  try {
    const res = await apiClient.post('/auth/login', { username, password })
    setToken(res.data.token)
    return res.data.user
  } catch (error) {
    throw extractError(error, 'Login failed')
  }
}

export const logoutAPI = async () => {
  try {
    await apiClient.post('/auth/logout')
  } finally {
    clearToken()
  }
}

export const meAPI = async () => {
  const res = await apiClient.get('/auth/me')
  return res.data.user
}

// ─── Conversations API ───

export const fetchConversations = async () => {
  const res = await apiClient.get('/conversations')
  return res.data.conversations
}

export const createConversationAPI = async (title = 'New Chat') => {
  const res = await apiClient.post('/conversations', { title })
  return res.data
}

export const renameConversationAPI = async (id, title) => {
  await apiClient.patch(`/conversations/${id}`, { title })
}

export const deleteConversationAPI = async (id) => {
  await apiClient.delete(`/conversations/${id}`)
}

export const fetchMessages = async (conversationId) => {
  const res = await apiClient.get(`/conversations/${conversationId}/messages`)
  return res.data.messages
}

// ─── Ratings ───

export const rateMessageAPI = async (messageId, rating) => {
  try {
    await apiClient.post(`/messages/${messageId}/rating`, { rating })
  } catch (error) {
    throw extractError(error, 'Failed to save rating')
  }
}

// ─── Indexed documents ───

export const fetchDocuments = async () => {
  const res = await apiClient.get('/documents')
  return res.data
}

// ─── Chat (streaming) ───

export const queryAPI = async (question, onChunk, conversationId = null, language = 'auto') => {
  try {
    // Use fetch for streaming support
    const headers = { 'Content-Type': 'application/json' }
    const token = getToken()
    if (token) headers.Authorization = `Bearer ${token}`

    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        question: question,
        max_chars: 15000,
        conversation_id: conversationId,
        language: language,
      }),
    })

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let answer = ''
    // sources are no longer passed to the UI; server logs them to terminal
    let sources = []
    let sections = []
    let thinking = []

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const chunk = decoder.decode(value, { stream: true })
      const lines = chunk.split('\n')

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6))

            if (data.type === 'answer') {
              answer += data.content
              // Convert 'answer' type to 'chunk' for backward compatibility
              onChunk?.({ type: 'chunk', content: data.content })
            } else if (data.type === 'language') {
              // The server resolved which language it will answer in (en/ur)
              onChunk?.({ type: 'language', content: data.content })
            } else if (data.type === 'thinking') {
              thinking.push(data.content)
              onChunk?.({ type: 'thinking', content: data.content })
            } else if (data.type === 'sections') {
              sections = data.content
              onChunk?.({ type: 'sections', content: data.content })
            } else if (data.type === 'sources') {
              // ignore sources events in the UI; they are logged on the server
              sources = data.content
            } else if (data.type === 'done') {
              onChunk?.({ type: 'done' })
            } else if (data.type === 'saved') {
              // DB id of the persisted assistant message (enables rating)
              onChunk?.({ type: 'saved', messageId: data.message_id })
            } else if (data.type === 'error') {
              throw new Error(data.content)
            }
          } catch (e) {
            // Skip invalid JSON
            if (!line.includes('[DONE]')) {
              console.error('Failed to parse SSE data:', line, e)
            }
          }
        }
      }
    }

    return {
      answer: answer,
      sources: sources,
      sections: sections,
      thinking: thinking,
      chunks: [],
    }
  } catch (error) {
    console.error('API Error:', error)
    throw new Error(error.message || 'Failed to query API')
  }
}

export const queryStreamAPI = async (question, onChunk, conversationId = null, language = 'auto') => {
  return queryAPI(question, onChunk, conversationId, language)
}

// ─── Voice: speech-to-text (ask by audio) ───

export const transcribeAudio = async (audioBlob, language = 'auto') => {
  const form = new FormData()
  // Name the file so the server can sniff the format from the extension
  const ext = (audioBlob.type && audioBlob.type.includes('ogg')) ? 'ogg' : 'webm'
  form.append('audio', audioBlob, `recording.${ext}`)
  form.append('language', language)

  const headers = {}
  const token = getToken()
  if (token) headers.Authorization = `Bearer ${token}`

  const res = await fetch(`${API_BASE_URL}/transcribe`, {
    method: 'POST',
    headers,
    body: form,
  })
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try { detail = (await res.json()).detail || detail } catch { /* ignore */ }
    throw new Error(detail)
  }
  return res.json() // { text, language }
}

// ─── Voice: text-to-speech (play the response) ───
// Returns an object URL for an <audio> element; caller must revoke it when done.

export const synthesizeSpeech = async (text, language = 'auto') => {
  const headers = { 'Content-Type': 'application/json' }
  const token = getToken()
  if (token) headers.Authorization = `Bearer ${token}`

  const res = await fetch(`${API_BASE_URL}/tts`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ text, language }),
  })
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try { detail = (await res.json()).detail || detail } catch { /* ignore */ }
    throw new Error(detail)
  }
  const blob = await res.blob()
  return URL.createObjectURL(blob)
}

export const healthCheck = async () => {
  try {
    const response = await apiClient.get('/health')
    return response.data
  } catch (error) {
    console.error('Health check failed:', error)
    throw error
  }
}

export default apiClient
