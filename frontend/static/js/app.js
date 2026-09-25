/**
 * LexAI — RAG Legal Assistant & Chatbot Logic
 */

(() => {
  'use strict';

  // ── State ──────────────────────────────────────────────────────────────────
  const state = {
    sessionId: null,
    filename: null,
    chatHistory: [], // [{ role: 'user'|'ai', content, citations, follow_ups }]
    isSpeaking: false,
  };

  // ── DOM References ─────────────────────────────────────────────────────────
  const $ = id => document.getElementById(id);
  const el = {
    // Upload view
    uploadView: $('uploadView'),
    dropzone: $('dropzone'),
    fileInput: $('fileInput'),
    browseBtn: $('browseBtn'),
    uploadProgress: $('uploadProgress'),
    progressFill: $('progressFill'),
    progressLabel: $('progressLabel'),

    // RAG Workspace
    ragWorkspace: $('ragWorkspace'),
    docName: $('docName'),
    statPages: $('statPages'),
    statWords: $('statWords'),
    statChunks: $('statChunks'),
    uploadNewBtn: $('uploadNewBtn'),
    starterChipsContainer: $('starterChipsContainer'),
    starterChips: $('starterChips'),

    // Chat
    chatMessages: $('chatMessages'),
    chatForm: $('chatForm'),
    chatInput: $('chatInput'),
    chatSendBtn: $('chatSendBtn'),

    // Navigation / Header
    statusDot: $('statusDot'),
    sessionLabel: $('sessionLabel'),
    clearSessionBtn: $('clearSessionBtn'),
    toastContainer: $('toastContainer'),
  };

  // ── Toast Notifications ────────────────────────────────────────────────────
  function toast(msg, type = 'info', duration = 4000) {
    const icons = { success: '✅', error: '❌', info: 'ℹ️', warning: '⚠️' };
    const div = document.createElement('div');
    div.className = `toast toast-${type}`;
    div.innerHTML = `<span>${icons[type] || 'ℹ️'}</span><span>${escapeHtml(msg)}</span>`;
    el.toastContainer.appendChild(div);
    setTimeout(() => {
      div.style.transition = 'opacity 0.3s, transform 0.3s';
      div.style.opacity = '0';
      div.style.transform = 'translateY(10px)';
      setTimeout(() => div.remove(), 350);
    }, duration);
  }

  function escapeHtml(str) {
    if (!str) return '';
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
  }

  function renderMarkdown(md) {
    if (!window.marked) return escapeHtml(md);
    return marked.parse(md, { gfm: true, breaks: true });
  }

  function setSessionStatus(active, label) {
    el.statusDot.classList.toggle('active', active);
    el.sessionLabel.textContent = label;
    el.clearSessionBtn.style.display = active ? 'inline-flex' : 'none';
  }

  // ── Auto-resize Textarea ───────────────────────────────────────────────────
  el.chatInput.addEventListener('input', () => {
    el.chatInput.style.height = 'auto';
    el.chatInput.style.height = Math.min(el.chatInput.scrollHeight, 120) + 'px';
  });

  el.chatInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!el.chatSendBtn.disabled) {
        el.chatForm.requestSubmit();
      }
    }
  });

  // ── File Upload & Dropzone ─────────────────────────────────────────────────
  el.browseBtn.addEventListener('click', e => {
    e.stopPropagation();
    el.fileInput.click();
  });

  el.dropzone.addEventListener('click', () => el.fileInput.click());

  el.dropzone.addEventListener('dragenter', e => {
    e.preventDefault();
    el.dropzone.classList.add('drag-over');
  });

  el.dropzone.addEventListener('dragover', e => {
    e.preventDefault();
  });

  el.dropzone.addEventListener('dragleave', e => {
    if (!el.dropzone.contains(e.relatedTarget)) {
      el.dropzone.classList.remove('drag-over');
    }
  });

  el.dropzone.addEventListener('drop', e => {
    e.preventDefault();
    el.dropzone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleUpload(file);
  });

  el.fileInput.addEventListener('change', () => {
    const file = el.fileInput.files[0];
    if (file) handleUpload(file);
  });

  el.uploadNewBtn.addEventListener('click', () => {
    el.fileInput.click();
  });

  async function handleUpload(file) {
    const allowed = ['.pdf', '.docx', '.txt'];
    const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
    if (!allowed.includes(ext)) {
      toast(`Unsupported file type "${ext}". Please upload PDF, DOCX, or TXT.`, 'error');
      return;
    }

    const maxMb = 20;
    if (file.size > maxMb * 1024 * 1024) {
      toast(`File is too large. Maximum size is ${maxMb} MB.`, 'error');
      return;
    }

    // Show Progress Bar
    el.uploadProgress.style.display = 'flex';
    el.browseBtn.disabled = true;
    el.progressLabel.textContent = `Uploading "${file.name}" & building vector index…`;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Upload processing failed');

      state.sessionId = data.session_id;
      state.filename = data.filename;
      state.chatHistory = [];

      // Transition to RAG Workspace
      el.uploadProgress.style.display = 'none';
      el.browseBtn.disabled = false;
      el.fileInput.value = '';

      initWorkspace(data);
      setSessionStatus(true, data.filename.length > 26 ? data.filename.slice(0, 24) + '…' : data.filename);
      toast(`"${data.filename}" indexed successfully with ${data.chunk_count} clauses!`, 'success');

    } catch (err) {
      el.uploadProgress.style.display = 'none';
      el.browseBtn.disabled = false;
      el.fileInput.value = '';
      toast(err.message || 'Upload error. Please try again.', 'error');
    }
  }

  // ── Initialize Workspace ───────────────────────────────────────────────────
  function initWorkspace(data) {
    el.uploadView.style.display = 'none';
    el.ragWorkspace.style.display = 'flex';

    el.docName.textContent = data.filename;
    el.statPages.textContent = data.pages ? `${data.pages} Page${data.pages > 1 ? 's' : ''}` : 'Document';
    el.statWords.textContent = `~${(data.word_count || 0).toLocaleString()} words`;
    el.statChunks.textContent = `${data.chunk_count || 0} Clauses Indexed`;

    // Render Starter Chips
    el.starterChips.innerHTML = '';
    const starters = data.starter_questions || [
      "What are the main obligations in this contract?",
      "What are the termination conditions?",
      "Are there any liability or indemnity traps?",
      "What are the payment terms and deadlines?",
    ];

    starters.forEach(q => {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'starter-chip';
      chip.textContent = q;
      chip.addEventListener('click', () => {
        askQuestion(q);
      });
      el.starterChips.appendChild(chip);
    });

    // Render Initial Welcome Message
    el.chatMessages.innerHTML = '';
    appendAiMessage({
      answer: `### 👋 Welcome to LexAI RAG Legal Copilot\n\nI have thoroughly reviewed and indexed **"${escapeHtml(data.filename)}"** across **${data.chunk_count} clauses** into vector memory.\n\nEvery answer I give is directly retrieved from your contract, verified against the actual clauses, and presented with citations. You can click any suggested question above or type your own question below!`,
      citations: [],
      follow_ups: starters.slice(0, 3),
    });

    el.chatInput.focus();
  }

  // ── Submit Question ────────────────────────────────────────────────────────
  el.chatForm.addEventListener('submit', e => {
    e.preventDefault();
    const query = el.chatInput.value.trim();
    if (!query) return;
    askQuestion(query);
  });

  async function askQuestion(questionText) {
    if (!state.sessionId || !questionText) return;

    el.chatInput.value = '';
    el.chatInput.style.height = 'auto';
    el.chatSendBtn.disabled = true;
    el.chatInput.disabled = true;

    // Append User Bubble
    appendUserBubble(questionText);
    state.chatHistory.push({ role: 'user', content: questionText });

    // Append Loading Indicator
    const typingBubble = appendTypingIndicator();

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: state.sessionId,
          question: questionText,
        }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Chat query failed');

      typingBubble.remove();
      appendAiMessage(data);
      state.chatHistory.push({
        role: 'ai',
        content: data.answer,
        citations: data.citations,
        follow_ups: data.follow_ups,
      });

    } catch (err) {
      typingBubble.remove();
      appendAiMessage({
        answer: `### ⚠️ Query Could Not Be Completed\n\n${err.message || 'An error occurred while analyzing the document clauses. Please try again.'}`,
        citations: [],
        follow_ups: [],
      });
      toast(err.message || 'Failed to get answer.', 'error');
    } finally {
      el.chatSendBtn.disabled = false;
      el.chatInput.disabled = false;
      el.chatInput.focus();
      scrollChatToBottom();
    }
  }

  // ── Message Rendering ──────────────────────────────────────────────────────
  function appendUserBubble(text) {
    const wrap = document.createElement('div');
    wrap.className = 'chat-bubble user';

    const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    wrap.innerHTML = `
      <div class="bubble-content">${escapeHtml(text)}</div>
      <div class="bubble-meta">${now}</div>
    `;

    el.chatMessages.appendChild(wrap);
    scrollChatToBottom();
    return wrap;
  }

  function appendAiMessage(data) {
    const wrap = document.createElement('div');
    wrap.className = 'chat-bubble ai';

    const header = document.createElement('div');
    header.className = 'bubble-header';
    header.innerHTML = `
      <div class="bubble-avatar">⚖️</div>
      <span class="bubble-author">LexAI Legal Copilot</span>
      <span class="bubble-badge">RAG Grounded</span>
    `;

    const contentBox = document.createElement('div');
    contentBox.className = 'bubble-content';

    // 1. Answer text in structured markdown
    const answerEl = document.createElement('div');
    answerEl.className = 'answer-body';
    answerEl.innerHTML = renderMarkdown(data.answer);
    contentBox.appendChild(answerEl);

    // 2. Citations Accordion (if citations are present)
    if (data.citations && data.citations.length > 0) {
      const citationsBox = document.createElement('div');
      citationsBox.className = 'citations-box';

      const toggleBtn = document.createElement('button');
      toggleBtn.type = 'button';
      toggleBtn.className = 'citations-toggle';
      toggleBtn.innerHTML = `
        <span>📑 ${data.citations.length} Retrieved Contract Clause${data.citations.length > 1 ? 's' : ''} (Click to inspect)</span>
        <span class="citations-arrow">▼</span>
      `;

      const list = document.createElement('div');
      list.className = 'citations-list';

      data.citations.forEach(c => {
        const item = document.createElement('div');
        item.className = 'citation-card';
        const matchPct = Math.round(c.score * 100);
        item.innerHTML = `
          <div class="citation-top">
            <span class="citation-heading">${escapeHtml(c.heading || `Clause #${c.id}`)}</span>
            <span class="citation-score">${matchPct > 0 ? matchPct + '% relevance' : 'Grounding Reference'}</span>
          </div>
          <div class="citation-snippet">"${escapeHtml(c.snippet)}"</div>
        `;
        list.appendChild(item);
      });

      toggleBtn.addEventListener('click', () => {
        citationsBox.classList.toggle('open');
      });

      citationsBox.appendChild(toggleBtn);
      citationsBox.appendChild(list);
      contentBox.appendChild(citationsBox);
    }

    // 3. Follow-up suggestion pills
    if (data.follow_ups && data.follow_ups.length > 0) {
      const followupsBox = document.createElement('div');
      followupsBox.className = 'followups-wrapper';
      followupsBox.innerHTML = `<span class="followups-label">Suggested Next Questions:</span>`;

      const chipsRow = document.createElement('div');
      chipsRow.className = 'followups-chips';

      data.follow_ups.forEach(q => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'followup-btn';
        btn.innerHTML = `<span>💡</span> ${escapeHtml(q)}`;
        btn.addEventListener('click', () => {
          askQuestion(q);
        });
        chipsRow.appendChild(btn);
      });

      followupsBox.appendChild(chipsRow);
      contentBox.appendChild(followupsBox);
    }

    // 4. Action toolbar (Copy, Read Aloud, Download)
    const actions = document.createElement('div');
    actions.className = 'bubble-actions';

    // Copy Button
    const copyBtn = document.createElement('button');
    copyBtn.type = 'button';
    copyBtn.className = 'btn-action';
    copyBtn.innerHTML = '📋 Copy';
    copyBtn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(data.answer);
        copyBtn.innerHTML = '✅ Copied!';
        setTimeout(() => { copyBtn.innerHTML = '📋 Copy'; }, 2000);
        toast('Answer copied to clipboard!', 'success', 2000);
      } catch {
        toast('Could not copy automatically.', 'warning');
      }
    });
    actions.appendChild(copyBtn);

    // Read Aloud Button
    if ('speechSynthesis' in window) {
      const speakBtn = document.createElement('button');
      speakBtn.type = 'button';
      speakBtn.className = 'btn-action';
      speakBtn.innerHTML = '🔊 Read Aloud';
      speakBtn.addEventListener('click', () => {
        if (state.isSpeaking) {
          window.speechSynthesis.cancel();
          state.isSpeaking = false;
          speakBtn.innerHTML = '🔊 Read Aloud';
        } else {
          window.speechSynthesis.cancel();
          const cleanText = data.answer.replace(/[#*`_~]/g, '');
          const utterance = new SpeechSynthesisUtterance(cleanText);
          utterance.rate = 1.0;
          utterance.onend = () => {
            state.isSpeaking = false;
            speakBtn.innerHTML = '🔊 Read Aloud';
          };
          utterance.onerror = () => {
            state.isSpeaking = false;
            speakBtn.innerHTML = '🔊 Read Aloud';
          };
          window.speechSynthesis.speak(utterance);
          state.isSpeaking = true;
          speakBtn.innerHTML = '⏹ Stop Audio';
        }
      });
      actions.appendChild(speakBtn);
    }

    // Save as text file
    const dlBtn = document.createElement('button');
    dlBtn.type = 'button';
    dlBtn.className = 'btn-action';
    dlBtn.innerHTML = '⬇ Save';
    dlBtn.addEventListener('click', () => {
      const blob = new Blob([data.answer], { type: 'text/markdown' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `lexai-analysis-${Date.now()}.md`;
      a.click();
      URL.revokeObjectURL(url);
      toast('Saved response file!', 'success', 2000);
    });
    actions.appendChild(dlBtn);

    contentBox.appendChild(actions);

    wrap.appendChild(header);
    wrap.appendChild(contentBox);
    el.chatMessages.appendChild(wrap);

    scrollChatToBottom();
    return wrap;
  }

  function appendTypingIndicator() {
    const wrap = document.createElement('div');
    wrap.className = 'chat-bubble ai';
    wrap.innerHTML = `
      <div class="bubble-header">
        <div class="bubble-avatar">⚖️</div>
        <span class="bubble-author">LexAI Legal Copilot</span>
      </div>
      <div class="bubble-content" style="padding:14px 20px;">
        <div class="typing-box">
          <div class="dot-flashing"></div>
          <span style="font-size:0.85rem;color:var(--clr-text-muted);margin-left:14px;">Searching clauses & generating legal analysis…</span>
        </div>
      </div>
    `;
    el.chatMessages.appendChild(wrap);
    scrollChatToBottom();
    return wrap;
  }

  function scrollChatToBottom() {
    setTimeout(() => {
      const viewport = document.querySelector('.chat-viewport');
      if (viewport) {
        viewport.scrollTop = viewport.scrollHeight;
      }
    }, 50);
  }

  // ── Clear Session ──────────────────────────────────────────────────────────
  el.clearSessionBtn.addEventListener('click', async () => {
    if (state.sessionId) {
      try {
        await fetch(`/api/session/${state.sessionId}`, { method: 'DELETE' });
      } catch {}
    }

    if (window.speechSynthesis) window.speechSynthesis.cancel();
    state.sessionId = null;
    state.filename = null;
    state.chatHistory = [];
    state.isSpeaking = false;

    el.ragWorkspace.style.display = 'none';
    el.uploadView.style.display = 'flex';
    setSessionStatus(false, 'No document loaded');
    toast('Session reset. Ready for next document.', 'info');
  });

  // ── Health Check ───────────────────────────────────────────────────────────
  (async () => {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
        console.info('[LexAI] System online. Model:', data.model, 'AI Ready:', data.ai_ready);
      }
    } catch {
      console.warn('[LexAI] Backend offline or warming up.');
    }
  })();

})();
