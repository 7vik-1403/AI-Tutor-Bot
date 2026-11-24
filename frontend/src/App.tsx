import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { BookOpen, Bot, Check, ChevronDown, FileText, GraduationCap, LoaderCircle, MessageCircle, Paperclip, Plus, Send, Settings2, ShieldCheck, Sparkles, Trash2, UploadCloud, User, X, Zap } from 'lucide-react'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'
type Source = { page: number; text: string; similarity: number; source_type?: 'text' | 'visual' }
type Message = { role: 'user' | 'assistant'; content: string; sources?: Source[]; seconds?: number }
type Health = { api: string; ollama: { running: boolean; model_installed: boolean; model: string }; embedding_model: string; ocr: boolean; vision_model?: string; vision_model_installed?: boolean }

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [docId, setDocId] = useState('')
  const [docInfo, setDocInfo] = useState<{page_count:number;chunk_count:number;visual_count:number;visual_pages_skipped:number;filename:string}|null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [question, setQuestion] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [showSettings, setShowSettings] = useState(false)
  const [showSources, setShowSources] = useState<number | null>(null)
  const [topK, setTopK] = useState(5)
  const [threshold, setThreshold] = useState(0.30)
  const [context, setContext] = useState(4096)
  const [maxTokens, setMaxTokens] = useState(300)
  const [reranker, setReranker] = useState(false)
  const [dragging, setDragging] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)
  const chatEnd = useRef<HTMLDivElement>(null)
  const questionInput = useRef<HTMLTextAreaElement>(null)

  const refreshHealth = async () => {
    try { const r = await fetch(`${API}/api/health`); setHealth(await r.json()) }
    catch { setHealth(null) }
  }
  useEffect(() => { refreshHealth(); const id = setInterval(refreshHealth, 12000); return () => clearInterval(id) }, [])
  useEffect(() => { chatEnd.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, busy])

  const selectFile = (f?: File) => {
    if (!f) return
    if (f.type !== 'application/pdf' && !f.name.toLowerCase().endsWith('.pdf')) { setError('Please choose a PDF file.'); return }
    setError(''); setFile(f)
  }
  const upload = async () => {
    if (!file) return
    setUploading(true); setError(''); setUploadStatus('Reading PDF and preparing its text…')
    const form = new FormData(); form.append('file', file)
    try {
      const r = await fetch(`${API}/api/upload`, { method: 'POST', body: form })
      const data = await r.json()
      if (!r.ok) throw new Error(data.detail || 'Could not process this PDF.')
      setDocId(data.document_id)
      setDocInfo({ page_count: data.page_count, chunk_count: data.chunk_count, visual_count: data.visual_count || 0, visual_pages_skipped: data.visual_pages_skipped || 0, filename: data.filename })
      setMessages([{ role: 'assistant', content: `I've indexed **${data.filename}**. It contains ${data.page_count} pages and ${data.chunk_count} searchable passages, from selectable PDF text. Ask me anything about the text.` }])
      setUploadStatus(data.cached ? 'Loaded from this session’s index.' : `Indexed ${data.chunk_count} text passages.`)
      setFile(null)
      if (fileInput.current) fileInput.current.value = ''
    } catch (e) { setError(e instanceof Error ? e.message : 'Upload failed.'); setUploadStatus('') }
    finally { setUploading(false) }
  }

  const send = async () => {
    const q = question.trim()
    if (!q || !docId || busy) return
    setError(''); setQuestion('')
    const prior = messages.slice(-6).map(m => ({ role: m.role, content: m.content }))
    const userMessage: Message = { role: 'user', content: q }
    const assistantMessage: Message = { role: 'assistant', content: '' }
    setMessages(prev => [...prev, userMessage, assistantMessage])
    setBusy(true)
    const started = performance.now()
    try {
      const response = await fetch(`${API}/api/chat`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ document_id: docId, question: q, history: prior, top_k: topK, threshold, num_ctx: context, max_tokens: maxTokens, use_reranker: reranker })
      })
      if (!response.ok) { const d = await response.json(); throw new Error(d.detail || 'Question request failed.') }
      if (!response.body) throw new Error('Streaming is not supported by this browser.')
      const reader = response.body.getReader(); const decoder = new TextDecoder()
      let buffer = ''; let answer = ''; let sources: Source[] = []
      const updateAssistant = (patch: Partial<Message>) => setMessages(prev => {
        const next = [...prev]; const idx = next.length - 1
        next[idx] = { ...next[idx], ...patch }; return next
      })
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const blocks = buffer.split('\n\n'); buffer = blocks.pop() || ''
        for (const block of blocks) {
          const event = block.match(/^event: (.+)$/m)?.[1]
          const raw = block.match(/^data: (.+)$/m)?.[1]
          if (!raw) continue
          const payload = JSON.parse(raw)
          if (event === 'meta') { sources = payload.sources || []; updateAssistant({ sources }) }
          if (event === 'token') { answer += payload.text; updateAssistant({ content: answer, sources }) }
          if (event === 'error') throw new Error(payload.message)
          if (event === 'done') updateAssistant({ seconds: (performance.now() - started) / 1000 })
        }
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Something went wrong.'
      setMessages(prev => {
        const next = [...prev]
        if (next[next.length - 1]?.role === 'assistant' && !next[next.length - 1].content) next.pop()
        return next
      })
      setError(msg)
    } finally { setBusy(false); questionInput.current?.focus() }
  }

  const resetDocument = async () => {
    if (docId) fetch(`${API}/api/document/${docId}`, { method: 'DELETE' }).catch(() => {})
    setDocId(''); setDocInfo(null); setMessages([]); setFile(null); setUploadStatus(''); setError('')
  }
  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
  }
  const ready = !!docId
  const localOk = !!health?.ollama.running && !!health?.ollama.model_installed

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-icon"><BookOpen size={20}/></div><div><b>pagewise</b><span>LOCAL STUDY SPACE</span></div></div>
      <button className="new-chat" onClick={resetDocument}><Plus size={17}/> New textbook</button>
      <div className="side-label">WORKSPACE</div>
      {docInfo ? <div className="document-chip"><FileText size={17}/><div className="doc-chip-text"><b>{docInfo.filename}</b><span>{docInfo.page_count} pages · {docInfo.chunk_count} passages</span></div><button className="icon-button tiny" title="Remove textbook" onClick={resetDocument}><X size={15}/></button></div> : <div className="empty-workspace">Your uploaded textbook will appear here.</div>}
      <div className="sidebar-bottom">
        <div className="privacy-note"><ShieldCheck size={16}/><span><b>Private by design</b><small>AI runs locally on your computer.</small></span></div>
        <div className="profile"><div className="avatar"><GraduationCap size={18}/></div><div><b>Study workspace</b><small>Offline-capable after setup</small></div></div>
      </div>
    </aside>

    <main className="main">
      <header className="topbar">
        <div className="crumb"><span>Workspace</span><span className="slash">/</span><b>{docInfo ? 'Textbook tutor' : 'New session'}</b></div>
        <div className="top-actions">
          <div className={`status-pill ${localOk ? 'online' : 'offline'}`}><i/>{localOk ? 'Local AI ready' : 'AI connection issue'}</div>
          <button className={`icon-button ${showSettings ? 'active' : ''}`} onClick={() => setShowSettings(!showSettings)} title="Settings"><Settings2 size={18}/></button>
        </div>
      </header>

      <div className="content-wrap">
        {showSettings && <section className="settings-panel">
          <div className="settings-head"><div><b>Answer settings</b><small>Balance response quality and speed</small></div><button className="icon-button" onClick={() => setShowSettings(false)}><X size={17}/></button></div>
          <div className="settings-grid">
            <label>Search passages <strong>{topK}</strong><input type="range" min="3" max="10" value={topK} onChange={e=>setTopK(+e.target.value)}/></label>
            <label>Similarity threshold <strong>{threshold.toFixed(2)}</strong><input type="range" min="0" max="0.8" step="0.01" value={threshold} onChange={e=>setThreshold(+e.target.value)}/></label>
            <label>Context window<select value={context} onChange={e=>setContext(+e.target.value)}><option value={2048}>2048 (faster)</option><option value={3072}>3072</option><option value={4096}>4096 (balanced)</option><option value={6144}>6144</option></select></label>
            <label>Answer length<select value={maxTokens} onChange={e=>setMaxTokens(+e.target.value)}><option value={150}>Short</option><option value={300}>Balanced</option><option value={500}>Detailed</option></select></label>
            <label className="toggle-setting"><input type="checkbox" checked={reranker} onChange={e=>setReranker(e.target.checked)}/><span><b>CrossEncoder reranking</b><small>May improve ranking; slower on CPU</small></span></label>
          </div>
        </section>}

        {!ready ? <div className="welcome">
          <div className="welcome-orb"><BookOpen size={28}/><Sparkles size={15}/></div>
          <div className="eyebrow">YOUR PERSONAL TEXTBOOK TUTOR</div>
          <h1>Understand your textbook.<br/><em>One question at a time.</em></h1>
          <p>Upload a PDF to ask questions about its selectable text. Answers include textbook page references. Scanned images and diagrams are not analyzed.</p>
          <div className={`connection-card ${localOk ? 'good' : 'bad'}`}>
            <div className="connection-dot"><i/></div><div><b>{localOk ? 'Your local AI is ready' : 'Connect your local AI'}</b><small>{health?.ollama.running ? (health.ollama.model_installed ? `Ollama · ${health.ollama.model}` : `Model missing · run ollama pull ${health.ollama.model}`) : 'Open the Ollama desktop app before asking questions.'}{health?.ollama.running && health.vision_model && !health.vision_model_installed ? ` · Vision model missing: ollama pull ${health.vision_model}` : ''}</small></div><button onClick={refreshHealth}>Refresh</button>
          </div>
          <div className={`upload-zone ${dragging ? 'dragging' : ''}`} onDragOver={e=>{e.preventDefault();setDragging(true)}} onDragLeave={()=>setDragging(false)} onDrop={e=>{e.preventDefault();setDragging(false);selectFile(e.dataTransfer.files[0])}}>
            <input ref={fileInput} type="file" accept="application/pdf,.pdf" hidden onChange={e=>selectFile(e.target.files?.[0])}/>
            <div className="upload-icon"><UploadCloud size={22}/></div>
            <b>{file ? file.name : 'Drop your textbook PDF here'}</b><span>{file ? `${(file.size/1024/1024).toFixed(1)} MB · ready to index` : 'or browse files from your computer'}</span>
            <div className="upload-actions"><button className="button-secondary" onClick={()=>fileInput.current?.click()}>{file ? 'Choose another' : 'Browse PDF'}</button><button className="button-primary" disabled={!file || uploading} onClick={upload}>{uploading ? <><LoaderCircle className="spin" size={16}/> Indexing…</> : <><Zap size={15}/> Upload & index</>}</button></div>
            {uploadStatus && <div className="upload-status"><Check size={14}/>{uploadStatus}</div>}
          </div>
          <div className="feature-row"><span><ShieldCheck size={15}/> Local & private</span><span><MessageCircle size={15}/> Grounded answers</span><span><FileText size={15}/> Page citations</span><span><Sparkles size={15}/> Diagram & chart insights</span></div>
        </div> : <div className="chat-layout">
          <div className="chat-heading"><div className="chat-title-icon"><GraduationCap size={20}/></div><div><div className="eyebrow">TEXTBOOK SESSION</div><h2>Let's study smarter.</h2><p>Ask a question. I'll find the relevant passages and explain them.</p></div><div className="chat-heading-spacer"/><button className="button-quiet" onClick={resetDocument}><Plus size={15}/> New textbook</button></div>
          <div className="messages">
            {messages.map((m, i)=><div className={`message-row ${m.role}`} key={i}>
              <div className={`message-avatar ${m.role}`}>{m.role === 'assistant' ? <Bot size={17}/> : <User size={17}/>}</div>
              <div className="message-body"><div className="message-author">{m.role === 'assistant' ? 'Pagewise tutor' : 'You'} <span>{m.role === 'assistant' && m.seconds ? `· ${m.seconds.toFixed(1)}s` : ''}</span></div>
                <div className="markdown"><ReactMarkdown>{m.content || (busy && i === messages.length-1 ? ' ' : '')}</ReactMarkdown>{busy && i === messages.length-1 && <span className="typing-cursor"/>}</div>
                {m.sources && m.sources.length > 0 && <div className="source-actions"><button onClick={()=>setShowSources(showSources===i?null:i)}><FileText size={14}/> {m.sources.length} textbook sources <ChevronDown size={14} className={showSources===i?'rotated':''}/></button></div>}
                {showSources===i && m.sources && <div className="source-list">{m.sources.map((s,j)=><div className="source-card" key={j}><div><b>Page {s.page}</b><span>Match {(s.similarity*100).toFixed(0)}%</span></div><p>{s.text}</p><small className={`source-kind ${s.source_type === 'visual' ? 'visual' : ''}`}>{s.source_type === 'visual' ? 'Visual insight · diagram / chart / image' : 'Extracted textbook text'}</small></div>)}</div>}
              </div>
            </div>)}
            {busy && messages[messages.length-1]?.content === '' && <div className="thinking"><span/><span/><span/> Searching the textbook and generating an answer</div>}
            <div ref={chatEnd}/>
          </div>
          <div className="composer-wrap">
            {error && <div className="error-banner"><span>{error}</span><button onClick={()=>setError('')}><X size={15}/></button></div>}
            <div className="composer"><textarea ref={questionInput} value={question} onChange={e=>setQuestion(e.target.value)} onKeyDown={onKeyDown} placeholder="Ask anything about your textbook…" rows={1} disabled={busy}/><button className="send-button" disabled={!question.trim() || busy} onClick={send}>{busy ? <LoaderCircle className="spin" size={18}/> : <Send size={18}/>}</button></div>
            <div className="composer-foot"><span><ShieldCheck size={13}/> Answers are grounded in your uploaded textbook</span><span>Enter to send · Shift + Enter for new line</span></div>
          </div>
        </div>}
        {!ready && error && <div className="global-error"><X size={15}/>{error}<button onClick={()=>setError('')}><X size={14}/></button></div>}
      </div>
      <footer className="footer"><span>PAGEWISE · LOCAL TEXTBOOK AI</span><span><i/> Powered by Ollama + BGE embeddings</span></footer>
    </main>
  </div>
}
