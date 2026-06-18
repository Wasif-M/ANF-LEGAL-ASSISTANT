import { useState, useEffect } from 'react'
import Sidebar from './components/Sidebar'
import ChatArea from './components/ChatArea'
import AuthPage from './components/AuthPage'
import {
  getToken,
  clearToken,
  meAPI,
  logoutAPI,
  fetchConversations,
  fetchMessages,
  createConversationAPI,
  renameConversationAPI,
  deleteConversationAPI,
} from './utils/api'
import './styles/App.css'

// Map a server conversation row to the UI shape
const toUiConversation = (conv) => ({
  id: conv.id,
  title: conv.title,
  messages: [],
  messagesLoaded: false,
  createdAt: conv.created_at,
})

// Map a stored DB message to the UI message shape
const toUiMessage = (m) => ({
  id: `db-${m.id}`,
  dbId: m.id,
  role: m.role,
  content: m.content,
  timestamp: m.created_at,
  thinking: m.thinking || [],
  sections: m.sections || [],
  rating: m.rating ?? null,
  isStreaming: false,
})

function App() {
  const [user, setUser] = useState(null)
  const [isAuthChecking, setIsAuthChecking] = useState(true)
  const [conversations, setConversations] = useState([])
  const [currentConversationId, setCurrentConversationId] = useState(null)
  const [theme, setTheme] = useState('light')

  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') || 'light'
    setTheme(savedTheme)
    document.documentElement.setAttribute('data-theme', savedTheme)
  }, [])

  // Restore session from saved token
  useEffect(() => {
    const restore = async () => {
      if (!getToken()) {
        setIsAuthChecking(false)
        return
      }
      try {
        const me = await meAPI()
        setUser(me)
      } catch {
        clearToken()
      } finally {
        setIsAuthChecking(false)
      }
    }
    restore()
  }, [])

  // Load the user's saved conversations once logged in
  useEffect(() => {
    if (!user) return
    const load = async () => {
      try {
        const convs = await fetchConversations()
        setConversations(convs.map(toUiConversation))
      } catch (err) {
        console.error('Failed to load conversations:', err)
      }
    }
    load()
  }, [user])

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light'
    setTheme(newTheme)
    localStorage.setItem('theme', newTheme)
    document.documentElement.setAttribute('data-theme', newTheme)
  }

  const handleAuthenticated = (loggedInUser) => {
    setUser(loggedInUser)
    setConversations([])
    setCurrentConversationId(null)
  }

  const handleLogout = async () => {
    try {
      await logoutAPI()
    } catch (err) {
      console.error('Logout failed:', err)
    }
    setUser(null)
    setConversations([])
    setCurrentConversationId(null)
  }

  const createNewConversation = async () => {
    try {
      const conv = await createConversationAPI('New Chat')
      const uiConv = { ...toUiConversation(conv), messagesLoaded: true }
      setConversations(prev => [uiConv, ...prev])
      setCurrentConversationId(uiConv.id)
    } catch (err) {
      console.error('Failed to create conversation:', err)
    }
  }

  const selectConversation = async (id) => {
    setCurrentConversationId(id)
    const conv = conversations.find(c => c.id === id)
    if (conv && !conv.messagesLoaded) {
      try {
        const messages = await fetchMessages(id)
        setConversations(prev =>
          prev.map(c =>
            c.id === id
              ? { ...c, messages: messages.map(toUiMessage), messagesLoaded: true }
              : c
          )
        )
      } catch (err) {
        console.error('Failed to load messages:', err)
      }
    }
  }

  const deleteConversation = async (id) => {
    try {
      await deleteConversationAPI(id)
    } catch (err) {
      console.error('Failed to delete conversation:', err)
    }
    setConversations(prev => prev.filter(c => c.id !== id))
    if (currentConversationId === id) {
      const remaining = conversations.filter(c => c.id !== id)
      setCurrentConversationId(remaining.length > 0 ? remaining[0].id : null)
    }
  }

  const updateConversationTitle = (id, newTitle) => {
    setConversations(prev =>
      prev.map(c => (c.id === id ? { ...c, title: newTitle } : c))
    )
    renameConversationAPI(id, newTitle).catch(err =>
      console.error('Failed to rename conversation:', err)
    )
  }

  const addMessageToConversation = (message, conversationId = currentConversationId) => {
    setConversations(prev => {
      return prev.map(conv => {
        if (conv.id !== conversationId) {
          return conv;
        }

        // Get the current messages array
        const currentMessages = Array.isArray(conv.messages) ? [...conv.messages] : [];

        // Find if message already exists
        const existingIndex = currentMessages.findIndex(m => m.id === message.id);

        let updatedMessages;
        if (existingIndex >= 0) {
          // Update existing message - preserve all other messages
          updatedMessages = currentMessages.map((m, idx) =>
            idx === existingIndex ? { ...m, ...message } : m
          );
        } else {
          updatedMessages = [...currentMessages, message];
        }

        return {
          ...conv,
          messages: updatedMessages
        };
      });
    })
  }

  if (isAuthChecking) {
    return (
      <div className="app-loading">
        <div className="app-loading-spinner" />
      </div>
    )
  }

  if (!user) {
    return <AuthPage onAuthenticated={handleAuthenticated} />
  }

  const currentConversation = conversations.find(c => c.id === currentConversationId)

  return (
    <div className="app-container">
      <Sidebar
        conversations={conversations}
        currentConversationId={currentConversationId}
        onSelectConversation={selectConversation}
        onNewConversation={createNewConversation}
        onDeleteConversation={deleteConversation}
        theme={theme}
        onToggleTheme={toggleTheme}
        user={user}
        onLogout={handleLogout}
      />
      <ChatArea
        conversation={currentConversation}
        onAddMessage={addMessageToConversation}
        onUpdateTitle={updateConversationTitle}
        onCreateConversation={createNewConversation}
      />
    </div>
  )
}

export default App
