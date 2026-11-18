const fileInput = document.getElementById('fileInput');
const chooseBtn = document.getElementById('chooseBtn');
const dropzone = document.getElementById('dropzone');
const emptyState = document.getElementById('emptyState');
const sourcePreview = document.getElementById('sourcePreview');
const sourceTag = document.getElementById('sourceTag');
const changeFileBtn = document.getElementById('changeFileBtn');
const hint = document.getElementById('hint');
const runBtn = document.getElementById('runBtn');
const stopBtn = document.getElementById('stopBtn');
const saveBtn = document.getElementById('saveBtn');
const clearBtn = document.getElementById('clearBtn');
const gpuChip = document.getElementById('gpuChip');
const gpuLabel = document.getElementById('gpuLabel');
const ocrChip = document.getElementById('ocrChip');
const engineSelect = document.getElementById('engineSelect');
const layoutModeBtn = document.getElementById('layoutModeBtn');
const inferTime = document.getElementById('inferTime');
const historyCount = document.getElementById('historyCount');
const historyList = document.getElementById('historyList');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
const navWorkspace = document.getElementById('navWorkspace');
const navAsk = document.getElementById('navAsk');
const navHistory = document.getElementById('navHistory');
const workspacePage = document.getElementById('workspacePage');
const askPage = document.getElementById('askPage');
const historyPage = document.getElementById('historyPage');
const askForm = document.getElementById('askForm');
const askInput = document.getElementById('askInput');
const askBtn = document.getElementById('askBtn');
const reindexBtn = document.getElementById('reindexBtn');
const clearSearchBtn = document.getElementById('clearSearchBtn');
const askSearchAll = document.getElementById('askSearchAll');
const askScopeLabel = document.getElementById('askScopeLabel');
const askModes = document.getElementById('askModes');
const ragMeta = document.getElementById('ragMeta');
const chatThread = document.getElementById('chatThread');
const chatMessages = document.getElementById('chatMessages');
const chatEmpty = document.getElementById('chatEmpty');
const resultsEmpty = document.getElementById('resultsEmpty');
const resultsBody = document.getElementById('resultsBody');
const zoomInBtn = document.getElementById('zoomInBtn');
const zoomOutBtn = document.getElementById('zoomOutBtn');
const zoomFitBtn = document.getElementById('zoomFitBtn');
const zoomLabel = document.getElementById('zoomLabel');
const layoutZoomControls = document.getElementById('layoutZoomControls');
const layoutZoomInBtn = document.getElementById('layoutZoomInBtn');
const layoutZoomOutBtn = document.getElementById('layoutZoomOutBtn');
const layoutZoomFitBtn = document.getElementById('layoutZoomFitBtn');
const layoutZoomLabel = document.getElementById('layoutZoomLabel');
const saveModal = document.getElementById('saveModal');
const saveModalPath = document.getElementById('saveModalPath');
const openFolderBtn = document.getElementById('openFolderBtn');
const closeSaveModalBtn = document.getElementById('closeSaveModalBtn');
const ocrOptions = document.getElementById('ocrOptions');
const optOrientation = document.getElementById('optOrientation');
const optUnwarping = document.getElementById('optUnwarping');

const OCR_OPTIONS_KEY = 'ocr-run-options';
const CHAT_STORAGE_KEY = 'ocr-chat-by-document';
const CHAT_HISTORY_TURNS = 6;

const state = {
  file: null,
  previewUrl: null,
  currentJob: null,
  runJobId: null,
  jobs: [],
  health: null,
  pollTimer: null,
  inferTimer: null,
  inferStartedAt: null,
  page: 'workspace',
  jobCache: {},
  zoom: 1,
  fitZoom: 1,
  atFit: true,
  naturalWidth: 0,
  pan: null,
  layoutZoom: 1,
  layoutFitZoom: 1,
  layoutAtFit: true,
  layoutWidth: 0,
  layoutPan: null,
  savedFolder: null,
  engine: 'paddle',
  ragMode: 'hybrid',
  chats: loadChats(),
  chatKey: undefined,
  chat: null,
};

chooseBtn.addEventListener('click', () => openFilePicker());
changeFileBtn.addEventListener('click', () => openFilePicker());
engineSelect.addEventListener('change', () => selectEngine(engineSelect.value));
optOrientation.addEventListener('change', () => saveOcrOptions());
optUnwarping.addEventListener('change', () => saveOcrOptions());
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) selectFile(fileInput.files[0]);
});

['dragenter', 'dragover'].forEach((evt) => dropzone.addEventListener(evt, (e) => {
  e.preventDefault();
  dropzone.classList.add('dragging');
}));
['dragleave', 'drop'].forEach((evt) => dropzone.addEventListener(evt, (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragging');
}));
dropzone.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files[0];
  if (file) selectFile(file);
});

runBtn.addEventListener('click', () => runJob());
stopBtn.addEventListener('click', () => stopJob());
saveBtn.addEventListener('click', () => saveResult());
openFolderBtn.addEventListener('click', () => openSavedFolder());
closeSaveModalBtn.addEventListener('click', () => hideSaveModal());
saveModal.addEventListener('click', (event) => {
  if (event.target === saveModal) hideSaveModal();
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') hideSaveModal();
});
clearBtn.addEventListener('click', () => clearAll());
navWorkspace.addEventListener('click', () => showWorkspace());
navAsk.addEventListener('click', () => showAsk());
navHistory.addEventListener('click', () => showHistory());
clearHistoryBtn.addEventListener('click', () => clearHistory());
askForm.addEventListener('submit', (event) => {
  event.preventDefault();
  runAsk();
});
askInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    runAsk();
  }
});
askInput.addEventListener('input', () => autosizeAskInput());
reindexBtn.addEventListener('click', () => reindexRag());
clearSearchBtn.addEventListener('click', () => clearSearchResults());
askModes.querySelectorAll('.mode-btn').forEach((btn) => btn.addEventListener('click', () => {
  askModes.querySelectorAll('.mode-btn').forEach((tab) => tab.classList.remove('active'));
  btn.classList.add('active');
  state.ragMode = btn.dataset.mode || 'hybrid';
}));
askSearchAll.addEventListener('change', () => syncAskScope());
zoomInBtn.addEventListener('click', () => changeZoom(1.25));
zoomOutBtn.addEventListener('click', () => changeZoom(1 / 1.25));
zoomFitBtn.addEventListener('click', () => fitZoom());
layoutZoomInBtn.addEventListener('click', () => changeLayoutZoom(1.25));
layoutZoomOutBtn.addEventListener('click', () => changeLayoutZoom(1 / 1.25));
layoutZoomFitBtn.addEventListener('click', () => fitLayoutZoom());

document.querySelectorAll('.mode-btn[data-pane]').forEach((btn) => btn.addEventListener('click', () => {
  document.querySelectorAll('.mode-btn[data-pane]').forEach((tab) => tab.classList.remove('active'));
  document.querySelectorAll('.results-pane').forEach((pane) => pane.classList.add('hidden'));
  btn.classList.add('active');
  const pane = document.getElementById(btn.dataset.pane);
  if (!pane) return;
  pane.classList.remove('hidden');
  const isLayout = btn.dataset.pane === 'results-layout';
  layoutZoomControls.classList.toggle('hidden', !isLayout);
  if (isLayout) requestAnimationFrame(() => fitLayoutZoom());
}));

function showWorkspace() {
  state.page = 'workspace';
  workspacePage.classList.remove('hidden');
  askPage.classList.add('hidden');
  historyPage.classList.add('hidden');
  navWorkspace.classList.add('active');
  navAsk.classList.remove('active');
  navHistory.classList.remove('active');
}

function showAsk() {
  state.page = 'ask';
  workspacePage.classList.add('hidden');
  askPage.classList.remove('hidden');
  historyPage.classList.add('hidden');
  navWorkspace.classList.remove('active');
  navAsk.classList.add('active');
  navHistory.classList.remove('active');
  syncAskScope();
  refreshRagHealth();
  const jobId = state.currentJob && state.currentJob.job_id;
  if (jobId) ensureJobIndexed(jobId);
}

function syncAskScope() {
  const job = state.currentJob;
  if (job && job.job_id) {
    askScopeLabel.textContent = askSearchAll.checked
      ? 'Scope: all indexed docs'
      : `Scope: ${job.filename}`;
  } else {
    askScopeLabel.textContent = 'Scope: all indexed docs';
  }
  switchChat();
}

function chatKey() {
  if (askSearchAll.checked) return '__all__';
  const job = state.currentJob;
  if (!job) return '__none__';
  // A picked-but-unprocessed file has no job id yet; its chat is temporary.
  return job.job_id || null;
}

function switchChat(force = false) {
  const key = chatKey();
  if (!force && key === state.chatKey && (key !== null || state.chat)) return;
  state.chatKey = key;
  if (key) {
    if (!state.chats[key]) state.chats[key] = [];
    state.chat = state.chats[key];
  } else {
    state.chat = [];
  }
  renderChat();
}

function activeAskJobId() {
  if (askSearchAll.checked) return null;
  return state.currentJob && state.currentJob.job_id ? state.currentJob.job_id : null;
}

async function ensureJobIndexed(jobId) {
  if (!jobId) return;
  try {
    await fetchJSON('/api/rag/reindex', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: jobId, force: false }),
    });
    setTimeout(refreshRagHealth, 2500);
  } catch (err) {
    // Non-fatal: asking can still trigger indexing.
  }
}

function showHistory() {
  state.page = 'history';
  workspacePage.classList.add('hidden');
  askPage.classList.add('hidden');
  historyPage.classList.remove('hidden');
  navWorkspace.classList.remove('active');
  navAsk.classList.remove('active');
  navHistory.classList.add('active');
  renderHistory();
}

async function refreshRagHealth() {
  try {
    const health = await fetchJSON('/api/rag/health');
    const docs = health.document_count ?? health.documents ?? 0;
    const chunks = health.chunk_count ?? '—';
    const device = health.device || health.embedding_device || 'cpu';
    ragMeta.textContent = `${docs} docs · ${chunks} chunks · ${device}`;
  } catch (err) {
    ragMeta.textContent = 'RAG unavailable';
  }
}

function loadChats() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(CHAT_STORAGE_KEY) || '{}');
    return saved && typeof saved === 'object' && !Array.isArray(saved) ? saved : {};
  } catch (err) {
    return {};
  }
}

function saveChat() {
  const stored = {};
  Object.entries(state.chats).forEach(([key, messages]) => {
    const done = messages.filter((msg) => !msg.pending);
    if (done.length) stored[key] = done;
  });
  try {
    sessionStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(stored));
  } catch (err) {
    // Storage full or disabled: the conversation still lives in memory for this page.
  }
}

function forgetChats(jobIds) {
  jobIds.forEach((jobId) => { delete state.chats[jobId]; });
  saveChat();
}

function clearSearchResults() {
  if (askBtn.disabled) return;
  state.chat.length = 0;
  saveChat();
  renderChat();
  askInput.focus();
}

function autosizeAskInput() {
  askInput.style.height = 'auto';
  askInput.style.height = `${Math.min(askInput.scrollHeight, 180)}px`;
}

function chatSourceLabel(msg) {
  return msg.llmError ? `LLM unavailable, used extractive answer: ${msg.llmError}` : '';
}

function stripCitations(text) {
  return String(text || '')
    .replace(/\s*[[(][^[\]()]*\bp\.\s*\d+[^[\]()]*[\])]/g, '')
    .replace(/\s+([.,;:])/g, '$1')
    .trim();
}

function renderChatAnswer(text) {
  // Model output is untrusted: neutralise raw HTML, then render Markdown and math.
  if (typeof marked === 'undefined') return escapeHtml(text);
  return renderMarkdown(text, null, { escapeHtml: true });
}

function renderChat() {
  chatMessages.querySelectorAll('.chat-msg').forEach((node) => node.remove());
  chatEmpty.classList.toggle('hidden', state.chat.length > 0);
  state.chat.forEach((msg) => {
    const node = document.createElement('div');
    node.className = `chat-msg ${msg.role}${msg.pending ? ' pending' : ''}${msg.error ? ' error' : ''}`;
    if (msg.role === 'user') {
      node.innerHTML = `<div class="chat-bubble">${escapeHtml(msg.content)}</div>`;
    } else {
      const meta = chatSourceLabel(msg);
      const body = msg.pending
        ? '<span class="chat-typing"><i></i><i></i><i></i></span>'
        : renderChatAnswer(stripCitations(msg.content));
      node.innerHTML = `
        <div class="chat-answer">${body}</div>
        ${meta && !msg.pending ? `<div class="chat-meta">${escapeHtml(meta)}</div>` : ''}
      `;
    }
    chatMessages.appendChild(node);
  });
  requestAnimationFrame(() => { chatThread.scrollTop = chatThread.scrollHeight; });
}

function chatHistoryPayload() {
  return state.chat
    .filter((msg) => !msg.pending && !msg.error && msg.source !== 'none')
    .slice(-CHAT_HISTORY_TURNS)
    .map((msg) => ({ role: msg.role, content: msg.content }));
}

async function runAsk() {
  const query = askInput.value.trim();
  if (!query || askBtn.disabled) return;

  const jobId = activeAskJobId();
  const scope = jobId && state.currentJob ? state.currentJob.filename : 'All documents';
  const history = chatHistoryPayload();
  const reply = { role: 'assistant', content: '', pending: true };
  const chat = state.chat;
  chat.push({ role: 'user', content: query });
  chat.push(reply);
  askInput.value = '';
  autosizeAskInput();
  renderChat();

  askBtn.disabled = true;
  reindexBtn.disabled = true;
  try {
    if (jobId) await ensureJobIndexed(jobId);
    const payload = await fetchJSON('/api/rag/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        mode: state.ragMode,
        top_k: 12,
        job_id: jobId,
        history,
      }),
    });
    reply.content = payload.answer || payload.warning || 'No answer returned.';
    reply.source = payload.answer_source || null;
    reply.llmError = payload.llm_error || null;
    reply.scope = scope;
    refreshRagHealth();
  } catch (err) {
    reply.content = err.message || 'Request failed.';
    reply.error = true;
  } finally {
    reply.pending = false;
    saveChat();
    if (state.chat === chat) renderChat();
    askBtn.disabled = false;
    reindexBtn.disabled = false;
    askInput.focus();
  }
}

function pushAssistantNotice(text, error = false) {
  state.chat.push({ role: 'assistant', content: text, source: 'none', error });
  saveChat();
  renderChat();
}

async function reindexRag() {
  reindexBtn.disabled = true;
  try {
    const jobId = activeAskJobId();
    await fetchJSON('/api/rag/reindex', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: jobId, force: true }),
    });
    const target = jobId && state.currentJob ? state.currentJob.filename : 'all documents';
    pushAssistantNotice(`Indexing started for ${target}. Ask again once the index chip updates.`);
    refreshRagHealth();
    setTimeout(refreshRagHealth, 5000);
    setTimeout(refreshRagHealth, 15000);
  } catch (err) {
    pushAssistantNotice(err.message, true);
  } finally {
    reindexBtn.disabled = false;
  }
}

function openFilePicker() {
  if (isRunning()) return;
  fileInput.value = '';
  fileInput.click();
}

function selectFile(file) {
  if (isRunning()) return;
  state.file = file;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(file);
  state.currentJob = {
    job_id: null,
    filename: file.name,
    mime_type: file.type || guessMime(file.name),
    status: 'ready',
    page_count: 0,
    inference_ms: null,
    urls: { source: state.previewUrl },
    visualizations: [],
    local: true,
  };
  state.chatKey = undefined;
  syncAskScope();
  showWorkspace();
  showSource(state.currentJob);
  clearResultsPanel();
  setHint(`Ready · ${file.name}`, '');
  updateButtons();
}

async function runJob() {
  if (isRunning()) return;
  if (!state.file) {
    showWorkspace();
    setHint('Still needed an input source', 'warn');
    hint.animate([{ transform: 'translateX(0)' }, { transform: 'translateX(-6px)' }, { transform: 'translateX(6px)' }, { transform: 'translateX(0)' }], { duration: 280 });
    return;
  }

  const form = new FormData();
  form.append('file', state.file);
  if (state.engine === 'paddle') {
    Object.entries(currentOcrOptions()).forEach(([key, value]) => form.append(key, String(value)));
  }
  showWorkspace();
  setHint(`Starting ${state.file.name}…`, 'busy');
  startInferClock();
  updateButtons(true);

  try {
    const response = await fetch('/api/jobs', { method: 'POST', body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(detailMessage(payload));
    state.currentJob = payload;
    state.runJobId = payload.job_id;
    syncAskScope();
    await refreshJobs();
    setHint(`Parsing ${payload.filename}…`, 'busy');
    pollJob(payload.job_id);
  } catch (err) {
    stopInferClock();
    setHint(err.message, 'error');
    updateButtons();
  }
}

async function stopJob() {
  const jobId = state.runJobId || (state.currentJob && state.currentJob.job_id);
  if (!jobId || !isRunning()) return;
  setHint('Stopping…', 'warn');
  try {
    await fetch(`/api/jobs/${jobId}/cancel`, { method: 'POST' });
  } catch (err) {
    setHint(err.message, 'error');
  }
}

async function saveResult() {
  const job = state.currentJob;
  if (!job || job.status !== 'completed' || !job.job_id) return;
  setHint(`Saving ${job.filename}…`, 'busy');
  try {
    const saved = await fetchJSON(`/api/jobs/${job.job_id}/save`, { method: 'POST' });
    setHint(`Saved to results/${saved.name}`, 'ok');
    showSaveModal(saved);
  } catch (err) {
    setHint(err.message, 'error');
  }
}

function showSaveModal(saved) {
  state.savedFolder = saved;
  saveModalPath.textContent = `results/${saved.name}`;
  saveModal.classList.remove('hidden');
  openFolderBtn.focus();
}

function hideSaveModal() {
  saveModal.classList.add('hidden');
}

async function openSavedFolder() {
  const saved = state.savedFolder;
  if (!saved || !saved.name) return;
  try {
    await fetchJSON('/api/exports/open', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: saved.name }),
    });
    hideSaveModal();
    setHint(`Opened results/${saved.name}`, 'ok');
  } catch (err) {
    setHint(err.message, 'error');
  }
}

async function clearAll() {
  const hasWork = Boolean(state.file || state.currentJob || state.previewUrl);
  if (hasWork && !window.confirm('Clear the current file and results?')) return;
  const jobId = state.runJobId;
  stopPolling();
  stopInferClock();
  state.runJobId = null;
  if (jobId) {
    try {
      await fetch(`/api/jobs/${jobId}/cancel`, { method: 'POST' });
      await refreshJobs();
    } catch (err) {
      setHint(err.message, 'error');
    }
  }
  resetWorkspace();
}

function resetWorkspace() {
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.file = null;
  state.previewUrl = null;
  state.currentJob = null;
  state.runJobId = null;
  fileInput.value = '';
  inferTime.textContent = '—';
  emptyState.classList.remove('hidden');
  sourcePreview.classList.add('hidden');
  sourcePreview.innerHTML = '';
  sourceTag.textContent = 'No file loaded';
  clearResultsPanel();
  syncAskScope();
  setHint('Still needed an input source', '');
  showWorkspace();
  updateButtons();
}

function pollJob(jobId) {
  stopPolling();
  const tick = async () => {
    try {
      const job = await fetchJSON(`/api/jobs/${jobId}?include_result=true`);
      if (state.runJobId !== jobId) return;
      state.jobCache[jobId] = job;
      if (state.runJobId === jobId) state.currentJob = job;
      await refreshJobs();
      if (state.runJobId !== jobId) return;
      if (job.status === 'running' || job.status === 'queued') {
        setHint(job.status === 'queued' ? `Queued · ${job.filename}` : `Parsing ${job.filename}…`, 'busy');
        return;
      }
      stopPolling();
      stopInferClock(job.inference_ms);
      if (job.status === 'completed') {
        setHint(`Completed in ${formatMs(job.inference_ms)} · ${job.page_count} page(s)`, 'ok');
        showJob(job);
      } else if (job.status === 'cancelled') {
        setHint('Stopped', 'warn');
      } else {
        setHint(job.error || 'OCR failed', 'error');
      }
      state.runJobId = null;
      updateButtons();
    } catch (err) {
      stopPolling();
      stopInferClock();
      setHint(err.message, 'error');
      updateButtons();
    }
  };
  tick();
  state.pollTimer = setInterval(tick, 1200);
}

async function openHistoryItem(jobId) {
  const listed = state.jobs.find((item) => item.job_id === jobId);
  state.file = null;
  if (listed) {
    state.currentJob = listed;
    syncAskScope();
    showWorkspace();
    showSource(listed);
    setHint(`Opening ${listed.filename}…`, 'busy');
    updateButtons();
  } else {
    showWorkspace();
  }
  try {
    const cached = state.jobCache[jobId];
    const job = cached && (cached.markdown || cached.document)
      ? cached
      : await fetchJSON(`/api/jobs/${jobId}?include_result=true`);
    state.jobCache[jobId] = job;
    state.currentJob = job;
    showJob(job);
    syncAskScope();
    if (job.status === 'completed') ensureJobIndexed(jobId);
    setHint(job.status === 'completed' ? `Opened ${job.filename}` : `${job.filename} · ${job.status}`, job.status === 'failed' ? 'error' : 'ok');
    updateButtons();
  } catch (err) {
    setHint(err.message, 'error');
    updateButtons();
  }
}

function showJob(job) {
  showSource(job);
  if (job.status === 'completed') renderResultsPanel(job);
  else clearResultsPanel();
}

function showSource(data) {
  emptyState.classList.add('hidden');
  sourcePreview.classList.remove('hidden');
  sourceTag.textContent = data.filename || 'Loaded';
  renderPreviewInto(sourcePreview, data);
}

function renderPreviewInto(preview, data) {
  preview.innerHTML = '';
  const url = data.urls && data.urls.source;
  if (!url) return;
  const stage = document.createElement('div');
  stage.className = 'source-stage';
  stage.id = 'sourceStage';
  const canvas = document.createElement('div');
  canvas.className = 'source-canvas';
  canvas.id = 'sourceCanvas';
  if ((data.mime_type || '').includes('pdf') || /\.pdf$/i.test(data.filename || '')) {
    stage.classList.add('is-pdf');
    const frame = document.createElement('iframe');
    frame.src = url;
    frame.title = 'Source PDF';
    canvas.appendChild(frame);
    stage.appendChild(canvas);
    preview.appendChild(stage);
    zoomLabel.textContent = 'Fit';
    return;
  }
  const img = document.createElement('img');
  img.id = 'sourceImage';
  img.src = url;
  img.alt = 'Source document';
  img.draggable = false;
  canvas.appendChild(img);
  stage.appendChild(canvas);
  preview.appendChild(stage);
  bindSourceZoom(stage, img);
}

function bindSourceZoom(stage, img) {
  const apply = () => {
    if (!state.naturalWidth) return;
    img.style.width = `${Math.round(state.naturalWidth * state.zoom)}px`;
    zoomLabel.textContent = state.atFit ? 'Fit' : `${Math.round(state.zoom * 100)}%`;
    const stageEl = currentStage();
    if (stageEl) stageEl.style.overflow = state.atFit ? 'hidden' : 'auto';
  };
  img._applyZoom = apply;

  img.addEventListener('load', () => {
    state.naturalWidth = img.naturalWidth;
    state.naturalHeight = img.naturalHeight;
    fitZoom();
  });
  if (img.complete && img.naturalWidth) {
    state.naturalWidth = img.naturalWidth;
    state.naturalHeight = img.naturalHeight;
    fitZoom();
  }

  stage.addEventListener('wheel', (event) => {
    event.preventDefault();
    const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
    changeZoom(factor, event);
  }, { passive: false });

  stage.addEventListener('pointerdown', (event) => {
    if (event.button !== 0) return;
    state.pan = {
      x: event.clientX,
      y: event.clientY,
      left: stage.scrollLeft,
      top: stage.scrollTop,
    };
    stage.classList.add('panning');
    stage.setPointerCapture(event.pointerId);
  });
  stage.addEventListener('pointermove', (event) => {
    if (!state.pan) return;
    stage.scrollLeft = state.pan.left - (event.clientX - state.pan.x);
    stage.scrollTop = state.pan.top - (event.clientY - state.pan.y);
  });
  const stopPan = () => {
    state.pan = null;
    stage.classList.remove('panning');
  };
  stage.addEventListener('pointerup', stopPan);
  stage.addEventListener('pointercancel', stopPan);

  if (state.resizeObserver) state.resizeObserver.disconnect();
  state.resizeObserver = new ResizeObserver(() => {
    if (!state.atFit) return;
    const rect = stage.getBoundingClientRect();
    const key = `${Math.round(rect.width)}x${Math.round(rect.height)}`;
    if (state.fitSize === key) return;
    state.fitSize = key;
    fitZoom();
  });
  state.resizeObserver.observe(stage);
}

function currentStage() {
  return document.getElementById('sourceStage');
}

function currentSourceImage() {
  return document.getElementById('sourceImage');
}

function fitZoom() {
  const stage = currentStage();
  const img = currentSourceImage();
  if (!stage || !img || !img.naturalWidth) {
    zoomLabel.textContent = 'Fit';
    return;
  }
  const rect = stage.getBoundingClientRect();
  const pad = 24;
  const fit = Math.min(
    Math.max(rect.width - pad, 1) / img.naturalWidth,
    Math.max(rect.height - pad, 1) / img.naturalHeight,
  );
  state.fitZoom = Math.max(fit, 0.05);
  state.zoom = state.fitZoom;
  state.atFit = true;
  applyZoom();
  stage.scrollLeft = 0;
  stage.scrollTop = 0;
}

function changeZoom(factor, event) {
  const stage = currentStage();
  const img = currentSourceImage();
  if (!stage || !img || !img.naturalWidth) return;
  const prev = state.zoom;
  const next = Math.min(8, Math.max(0.2, prev * factor));
  if (Math.abs(next - prev) < 0.0001) return;
  state.atFit = Math.abs(next - state.fitZoom) < 0.01;
  let focusX = stage.scrollLeft + stage.clientWidth / 2;
  let focusY = stage.scrollTop + stage.clientHeight / 2;
  if (event) {
    const rect = stage.getBoundingClientRect();
    focusX = event.clientX - rect.left + stage.scrollLeft;
    focusY = event.clientY - rect.top + stage.scrollTop;
  }
  state.zoom = next;
  applyZoom();
  const ratio = next / prev;
  stage.scrollLeft = focusX * ratio - (event ? event.clientX - stage.getBoundingClientRect().left : stage.clientWidth / 2);
  stage.scrollTop = focusY * ratio - (event ? event.clientY - stage.getBoundingClientRect().top : stage.clientHeight / 2);
}

function applyZoom() {
  const img = currentSourceImage();
  if (img && img._applyZoom) img._applyZoom();
}

function layoutStage() {
  return document.getElementById('layoutStage');
}

function layoutImages() {
  return [...document.querySelectorAll('#layoutStage img')];
}

function bindLayoutZoom(stage, images) {
  const apply = () => {
    images.forEach((img) => {
      if (!img.naturalWidth) return;
      img.style.width = `${Math.round(img.naturalWidth * state.layoutZoom)}px`;
    });
    layoutZoomLabel.textContent = state.layoutAtFit ? 'Fit' : `${Math.round(state.layoutZoom * 100)}%`;
    stage.style.overflow = state.layoutAtFit ? 'hidden' : 'auto';
  };
  state.layoutApply = apply;

  const onReady = () => {
    const first = images.find((img) => img.naturalWidth) || images[0];
    if (!first || !first.naturalWidth) return;
    state.layoutWidth = first.naturalWidth;
    state.layoutHeight = first.naturalHeight;
    fitLayoutZoom();
  };
  images.forEach((img) => {
    img.addEventListener('load', onReady);
    if (img.complete && img.naturalWidth) onReady();
  });

  stage.addEventListener('wheel', (event) => {
    event.preventDefault();
    const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
    changeLayoutZoom(factor, event);
  }, { passive: false });

  stage.addEventListener('pointerdown', (event) => {
    if (event.button !== 0) return;
    state.layoutPan = {
      x: event.clientX,
      y: event.clientY,
      left: stage.scrollLeft,
      top: stage.scrollTop,
    };
    stage.classList.add('panning');
    stage.setPointerCapture(event.pointerId);
  });
  stage.addEventListener('pointermove', (event) => {
    if (!state.layoutPan) return;
    stage.scrollLeft = state.layoutPan.left - (event.clientX - state.layoutPan.x);
    stage.scrollTop = state.layoutPan.top - (event.clientY - state.layoutPan.y);
  });
  const stopPan = () => {
    state.layoutPan = null;
    stage.classList.remove('panning');
  };
  stage.addEventListener('pointerup', stopPan);
  stage.addEventListener('pointercancel', stopPan);

  if (state.layoutResizeObserver) state.layoutResizeObserver.disconnect();
  state.layoutResizeObserver = new ResizeObserver(() => {
    if (!state.layoutAtFit) return;
    const rect = stage.getBoundingClientRect();
    const key = `${Math.round(rect.width)}x${Math.round(rect.height)}`;
    if (state.layoutFitSize === key) return;
    state.layoutFitSize = key;
    fitLayoutZoom();
  });
  state.layoutResizeObserver.observe(stage);
}

function fitLayoutZoom() {
  const stage = layoutStage();
  const images = layoutImages();
  const img = images.find((item) => item.naturalWidth) || images[0];
  if (!stage || !img || !img.naturalWidth) {
    layoutZoomLabel.textContent = 'Fit';
    return;
  }
  const rect = stage.getBoundingClientRect();
  if (rect.width < 8 || rect.height < 8) return;
  const pad = 24;
  const fit = Math.min(
    Math.max(rect.width - pad, 1) / img.naturalWidth,
    Math.max(rect.height - pad, 1) / img.naturalHeight,
  );
  state.layoutFitZoom = Math.max(fit, 0.05);
  state.layoutZoom = state.layoutFitZoom;
  state.layoutAtFit = true;
  if (state.layoutApply) state.layoutApply();
  stage.scrollLeft = 0;
  stage.scrollTop = 0;
}

function changeLayoutZoom(factor, event) {
  const stage = layoutStage();
  const images = layoutImages();
  const img = images.find((item) => item.naturalWidth) || images[0];
  if (!stage || !img || !img.naturalWidth) return;
  const prev = state.layoutZoom;
  const next = Math.min(8, Math.max(0.2, prev * factor));
  if (Math.abs(next - prev) < 0.0001) return;
  state.layoutAtFit = Math.abs(next - state.layoutFitZoom) < 0.01;
  let focusX = stage.scrollLeft + stage.clientWidth / 2;
  let focusY = stage.scrollTop + stage.clientHeight / 2;
  if (event) {
    const rect = stage.getBoundingClientRect();
    focusX = event.clientX - rect.left + stage.scrollLeft;
    focusY = event.clientY - rect.top + stage.scrollTop;
  }
  state.layoutZoom = next;
  if (state.layoutApply) state.layoutApply();
  const ratio = next / prev;
  const rect = stage.getBoundingClientRect();
  stage.scrollLeft = focusX * ratio - (event ? event.clientX - rect.left : stage.clientWidth / 2);
  stage.scrollTop = focusY * ratio - (event ? event.clientY - rect.top : stage.clientHeight / 2);
}

function layoutAvailable() {
  const jobEngine = state.currentJob && state.currentJob.engine;
  if (jobEngine && jobEngine !== 'paddle') return false;
  return state.engine === 'paddle';
}

function syncLayoutTab() {
  const show = layoutAvailable();
  layoutModeBtn.classList.toggle('hidden', !show);
  const emptyCopy = resultsEmpty.querySelector('p');
  if (emptyCopy) {
    emptyCopy.textContent = show
      ? 'Run OCR to see markdown, blocks, and layout here.'
      : 'Run OCR to see markdown and blocks here.';
  }
  if (show || !layoutModeBtn.classList.contains('active')) return;
  document.querySelectorAll('.mode-btn').forEach((tab) => tab.classList.remove('active'));
  document.querySelector('.mode-btn[data-pane="results-rendered"]').classList.add('active');
  document.querySelectorAll('.results-pane').forEach((pane) => pane.classList.add('hidden'));
  document.getElementById('results-rendered').classList.remove('hidden');
  layoutZoomControls.classList.add('hidden');
}

function clearResultsPanel() {
  resultsEmpty.classList.remove('hidden');
  resultsBody.classList.add('hidden');
  layoutZoomControls.classList.add('hidden');
  syncLayoutTab();
  setStats({ filename: state.currentJob && state.currentJob.filename, page_count: null, inference_ms: null, blocks: 0, status: 'Idle' });
}

function renderResultsPanel(job) {
  resultsEmpty.classList.add('hidden');
  resultsBody.classList.remove('hidden');
  const blocks = extractBlocks(job);
  setStats({ ...job, blocks: blocks.length });

  document.getElementById('results-table').innerHTML = `
    <div class="results-head"><span>Label</span><span>Text</span><span>Page</span></div>
    ${blocks.length ? blocks.map((block) => `
      <div class="result-row">
        <span class="clip">${escapeHtml(block.label)}</span>
        <span class="clip" title="${escapeHtml(block.text)}">${escapeHtml(block.text)}</span>
        <span>${escapeHtml(String(block.page))}</span>
      </div>
    `).join('') : '<p>No structure blocks in this run.</p>'}
  `;
  document.getElementById('results-rendered').innerHTML = renderMarkdown(job.markdown || '*No Markdown generated.*', job);
  renderLayoutVisualizations(job.visualizations || []);
  document.querySelectorAll('.mode-btn').forEach((tab, index) => tab.classList.toggle('active', index === 0));
  document.querySelectorAll('.results-pane').forEach((pane, index) => pane.classList.toggle('hidden', index !== 0));
  layoutZoomControls.classList.add('hidden');
  syncLayoutTab();
}

function cleanTex(tex) {
  return String(tex || '')
    .trim()
    .replace(/\{r\s+l\}/g, '{rl}')
    .replace(/\{l\s+r\}/g, '{lr}');
}

function renderTex(block) {
  if (!block) return '';
  if (typeof katex === 'undefined') {
    return `<code>${escapeHtml(block.tex)}</code>`;
  }
  try {
    return katex.renderToString(block.tex, {
      displayMode: Boolean(block.display),
      throwOnError: false,
      strict: 'ignore',
      errorColor: '#ff9aa0',
    });
  } catch (err) {
    return `<code class="tex-fallback">${escapeHtml(block.tex)}</code>`;
  }
}

function renderMarkdown(markdown, job, options = {}) {
  const blocks = [];
  let text = String(markdown || '');
  text = text.replace(/\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]/g, (_, dollars, brackets) => {
    const index = blocks.length;
    blocks.push({ display: true, tex: cleanTex(dollars ?? brackets) });
    return `\n\n@@MATH${index}@@\n\n`;
  });
  text = text.replace(/\\\(([\s\S]+?)\\\)/g, (_, tex) => {
    const index = blocks.length;
    blocks.push({ display: false, tex: cleanTex(tex) });
    return `@@MATH${index}@@`;
  });
  text = text.replace(/(^|[^$])\$([^$\n]+?)\$(?!\$)/g, (_, pre, tex) => {
    const index = blocks.length;
    blocks.push({ display: false, tex: cleanTex(tex) });
    return `${pre}@@MATH${index}@@`;
  });
  if (options.escapeHtml) {
    text = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  // Math is already swapped for placeholders, so a remaining backslash-n is a VLM line break, not \nu or \neq.
  text = text.replace(/\\n/g, '<br>');
  let html = marked.parse(text);
  html = html.replace(/@@MATH(\d+)@@/g, (_, index) => renderTex(blocks[Number(index)]));
  const base = job && job.job_id ? `/results/${job.job_id}/` : '';
  if (!base) return html;
  return html.replace(/(<img\b[^>]*?\bsrc=["'])([^"']+)(["'])/gi, (match, open, src, close) => {
    if (/^(https?:|data:|\/)/i.test(src)) return match;
    const relative = src.replace(/^\.\//, '').replace(/\\/g, '/');
    return `${open}${base}${relative}${close}`;
  });
}

function renderLayoutVisualizations(viz) {
  const layout = document.getElementById('results-layout');
  layout.className = 'results-pane hidden layout-pane';
  if (!viz.length) {
    layout.innerHTML = '<p class="results-pane">No layout visualizations for this run.</p>';
    return;
  }
  const stage = document.createElement('div');
  stage.className = 'source-stage';
  stage.id = 'layoutStage';
  const canvas = document.createElement('div');
  canvas.className = 'source-canvas layout-canvas';
  const images = viz.map((src) => {
    const img = document.createElement('img');
    img.className = 'layout-zoom-img';
    img.src = src;
    img.alt = 'Layout visualization';
    img.draggable = false;
    canvas.appendChild(img);
    return img;
  });
  if (images[0]) images[0].id = 'layoutImage';
  stage.appendChild(canvas);
  layout.innerHTML = '';
  layout.appendChild(stage);
  bindLayoutZoom(stage, images);
}

function setStats(job) {
  document.getElementById('statFile').textContent = job.filename || '—';
  document.getElementById('statPages').textContent = job.page_count ? String(job.page_count) : '—';
  document.getElementById('statTime').textContent = formatMs(job.inference_ms);
  document.getElementById('statBlocks').textContent = job.blocks != null ? String(job.blocks) : '—';
}

function extractBlocks(job) {
  const rows = [];
  const pages = (job.document && job.document.pages) || [];
  pages.forEach((page, index) => {
    const payload = page.res || page;
    const list = payload.parsing_res_list || [];
    const pageNumber = page._app_page_number || payload.page_index || index + 1;
    list.forEach((block) => {
      rows.push({
        label: block.block_label || 'block',
        text: String(block.block_content || '')
          .split(/(\$\$[\s\S]+?\$\$|\$[^$\n]+?\$)/)
          .map((part, i) => (i % 2 ? part : part.replace(/\\n/g, ' ')))
          .join('')
          .replace(/\s+/g, ' ')
          .trim(),
        page: pageNumber,
      });
    });
  });
  return rows;
}

function recentJobs() {
  return state.jobs.slice(0, 10);
}

function fileFormat(job) {
  const name = String(job.filename || '');
  const dot = name.lastIndexOf('.');
  if (dot > 0 && dot < name.length - 1) return name.slice(dot + 1).toUpperCase();
  const mime = String(job.mime_type || '').toLowerCase();
  if (mime === 'application/pdf') return 'PDF';
  if (mime.startsWith('image/')) return mime.slice(6).replace('jpeg', 'jpg').toUpperCase();
  return '—';
}

function historyStatus(job) {
  return job.status === 'completed' ? 'Completed' : 'Incomplete';
}

function historyTime(job) {
  const when = formatDate(job.finished_at || job.created_at);
  if (job.inference_ms == null) return when;
  return `${formatMs(job.inference_ms)} · ${when}`;
}

function renderHistory() {
  const jobs = recentJobs();
  historyCount.textContent = String(jobs.length);
  clearHistoryBtn.disabled = state.jobs.length === 0;
  if (!jobs.length) {
    historyList.innerHTML = '<div class="history-empty">No history yet. Run OCR from Workspace to create one.</div>';
    return;
  }
  const head = `
    <div class="history-row head">
      <span>File name</span>
      <span>Format</span>
      <span>Model</span>
      <span>Status</span>
      <span>Pages</span>
      <span>Time</span>
      <span></span>
    </div>`;
  const rows = jobs.map((job) => {
    const done = job.status === 'completed';
    return `
      <div class="history-row">
        <button class="history-open" data-job="${escapeHtml(job.job_id)}" type="button">
          <span class="history-file">${escapeHtml(job.filename)}</span>
          <span>${escapeHtml(fileFormat(job))}</span>
          <span class="history-model">${escapeHtml(job.model_name || 'PaddleOCR-VL-1.6')}</span>
          <span class="history-status ${done ? 'completed' : 'incomplete'}">${historyStatus(job)}</span>
          <span>${job.page_count ? escapeHtml(job.page_count) : '—'}</span>
          <span>${escapeHtml(historyTime(job))}</span>
        </button>
        <button class="history-delete" data-job="${escapeHtml(job.job_id)}" data-name="${escapeHtml(job.filename)}" type="button" title="Delete this entry" aria-label="Delete ${escapeHtml(job.filename)}">×</button>
      </div>`;
  }).join('');
  historyList.innerHTML = head + rows;
  historyList.querySelectorAll('.history-open').forEach((btn) => {
    btn.addEventListener('click', () => openHistoryItem(btn.dataset.job));
  });
  historyList.querySelectorAll('.history-delete').forEach((btn) => {
    btn.addEventListener('click', (event) => {
      event.stopPropagation();
      deleteHistoryItem(btn.dataset.job, btn.dataset.name);
    });
  });
}

function dropOpenJob(jobId) {
  const open = (state.currentJob && state.currentJob.job_id === jobId) || state.runJobId === jobId;
  if (!open) return;
  if (state.runJobId === jobId) {
    stopPolling();
    stopInferClock();
    state.runJobId = null;
  }
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.file = null;
  state.previewUrl = null;
  state.currentJob = null;
  fileInput.value = '';
  inferTime.textContent = '—';
  emptyState.classList.remove('hidden');
  sourcePreview.classList.add('hidden');
  sourcePreview.innerHTML = '';
  sourceTag.textContent = 'No file loaded';
  clearResultsPanel();
  syncAskScope();
  updateButtons();
}

async function deleteHistoryItem(jobId, filename) {
  const label = filename || 'this entry';
  if (!window.confirm(`Delete ${label} from history?`)) return;
  try {
    await fetchJSON(`/api/jobs/${jobId}`, { method: 'DELETE' });
    state.jobs = state.jobs.filter((job) => job.job_id !== jobId);
    delete state.jobCache[jobId];
    forgetChats([jobId]);
    dropOpenJob(jobId);
    renderHistory();
  } catch (err) {
    setHint(err.message, 'error');
  }
}

async function clearHistory() {
  if (!state.jobs.length) return;
  if (!window.confirm('Delete all history? Saved folders are kept.')) return;
  try {
    await fetchJSON('/api/jobs', { method: 'DELETE' });
    const openId = (state.currentJob && state.currentJob.job_id) || state.runJobId;
    state.jobs = [];
    state.jobCache = {};
    forgetChats(Object.keys(state.chats).filter((key) => !key.startsWith('__')));
    if (openId) dropOpenJob(openId);
    renderHistory();
  } catch (err) {
    setHint(err.message, 'error');
  }
}

async function refreshJobs() {
  const payload = await fetchJSON('/api/jobs');
  state.jobs = payload.jobs || [];
  historyCount.textContent = String(Math.min(state.jobs.length, 10));
  if (state.page === 'history') renderHistory();
}

async function loadHealth() {
  try {
    const health = await fetchJSON('/api/health');
    state.health = health;
    gpuLabel.textContent = health.device_label || (health.gpu_available ? 'GPU (CUDA)' : 'CPU');
    gpuChip.classList.toggle('gpu-on', Boolean(health.gpu_available));
    gpuChip.classList.toggle('gpu-off', !health.gpu_available);
    gpuChip.title = health.device_name || health.device_label;
    initOcrOptions(health.default_options);
    applyEngine(health.engine, health.engines);
  } catch (err) {
    gpuLabel.textContent = 'Device unknown';
    gpuChip.classList.add('gpu-off');
  }
}

function applyEngine(engine, engines) {
  if (Array.isArray(engines) && engines.length) {
    engineSelect.innerHTML = engines.map((item) => (
      `<option value="${escapeHtml(item.id)}">${escapeHtml(item.label)}</option>`
    )).join('');
  }
  state.engine = engine || state.engine || 'paddle';
  engineSelect.value = state.engine;
  ocrChip.title = state.engine === 'paddle'
    ? 'PaddlePaddle/PaddleOCR-VL-1.6'
    : 'xiaomi-research/Xiaomi-OCR-0';
  syncLayoutTab();
  syncOcrOptions();
}

async function selectEngine(engine) {
  if (isRunning()) {
    engineSelect.value = state.engine;
    return;
  }
  const previous = state.engine;
  engineSelect.disabled = true;
  try {
    const payload = await fetchJSON('/api/engine', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ engine }),
    });
    applyEngine(payload.engine, payload.engines);
    clearForNewEngine();
    setHint(`${payload.label} is ready. The other model stays unloaded until you switch back.`, 'ok');
  } catch (err) {
    engineSelect.value = previous;
    state.engine = previous;
    setHint(err.message, 'error');
  } finally {
    updateButtons();
  }
}

function clearForNewEngine() {
  inferTime.textContent = '—';
  if (state.currentJob) {
    state.currentJob = {
      ...state.currentJob,
      job_id: null,
      status: state.file ? 'ready' : 'Idle',
      page_count: 0,
      inference_ms: null,
      markdown: '',
      document: null,
      visualizations: [],
      urls: state.previewUrl
        ? { source: state.previewUrl }
        : (state.currentJob.urls || {}),
    };
  }
  clearResultsPanel();
  syncAskScope();
}

function updateButtons(starting = false) {
  const running = starting || isRunning();
  const job = state.currentJob;
  const completed = job && job.status === 'completed' && Boolean(job.job_id);
  runBtn.disabled = running;
  stopBtn.disabled = !running;
  saveBtn.disabled = !completed;
  changeFileBtn.disabled = running;
  engineSelect.disabled = running;
  syncOcrOptions(running);
}

function syncOcrOptions(running = isRunning()) {
  // Orientation and unwarping are PaddleOCR-VL preprocessors; Xiaomi-OCR-0 ignores them.
  const unsupported = state.engine !== 'paddle';
  const locked = running || unsupported;
  optOrientation.disabled = locked;
  optUnwarping.disabled = locked;
  ocrOptions.classList.toggle('disabled', locked);
  ocrOptions.title = unsupported ? 'Only used by PaddleOCR-VL' : '';
}

function initOcrOptions(defaults) {
  let saved = null;
  try {
    saved = JSON.parse(localStorage.getItem(OCR_OPTIONS_KEY) || 'null');
  } catch (err) {
    saved = null;
  }
  const source = saved || defaults || {};
  optOrientation.checked = source.use_doc_orientation_classify !== false;
  optUnwarping.checked = source.use_doc_unwarping !== false;
}

function saveOcrOptions() {
  try {
    localStorage.setItem(OCR_OPTIONS_KEY, JSON.stringify(currentOcrOptions()));
  } catch (err) {
    // Preference only; runs still use the checkbox state.
  }
}

function currentOcrOptions() {
  return {
    use_doc_orientation_classify: optOrientation.checked,
    use_doc_unwarping: optUnwarping.checked,
  };
}

function isRunning() {
  const job = state.jobs.find((item) => item.job_id === state.runJobId) || state.currentJob;
  const status = job && job.status;
  return Boolean(state.runJobId) || status === 'queued' || status === 'running';
}

function startInferClock() {
  stopInferClock();
  state.inferStartedAt = Date.now();
  inferTime.textContent = '0.0s';
  state.inferTimer = setInterval(() => {
    inferTime.textContent = formatMs(Date.now() - state.inferStartedAt);
  }, 100);
}

function stopInferClock(finalMs) {
  if (state.inferTimer) {
    clearInterval(state.inferTimer);
    state.inferTimer = null;
  }
  if (finalMs != null) inferTime.textContent = formatMs(finalMs);
}

function stopPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

function setHint(text, kind) {
  hint.textContent = text;
  hint.className = `hint ${kind || ''}`.trim();
}

function formatMs(ms) {
  if (ms == null || Number.isNaN(Number(ms))) return '—';
  const seconds = Number(ms) / 1000;
  if (seconds < 10) return `${seconds.toFixed(2)}s`;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = seconds - minutes * 60;
  return `${minutes}m ${rest.toFixed(0)}s`;
}

function formatDate(value) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

function guessMime(name) {
  const ext = (name.split('.').pop() || '').toLowerCase();
  if (ext === 'pdf') return 'application/pdf';
  if (['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'webp'].includes(ext)) return `image/${ext === 'jpg' ? 'jpeg' : ext}`;
  return 'application/octet-stream';
}

function stripExt(name) {
  return String(name || 'document').replace(/\.[^.]+$/, '');
}

function escapeHtml(value) {
  return String(value || '').replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[ch]));
}

function detailMessage(payload) {
  if (!payload) return 'Request failed';
  if (typeof payload.detail === 'string') return payload.detail;
  return payload.detail ? JSON.stringify(payload.detail) : 'Request failed';
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(detailMessage(payload));
  return payload;
}

async function boot() {
  syncAskScope();
  await loadHealth();
  await refreshJobs();
  const running = state.jobs.find((job) => job.status === 'running' || job.status === 'queued');
  if (running) {
    state.currentJob = running;
    state.runJobId = running.job_id;
    syncAskScope();
    startInferClock();
    pollJob(running.job_id);
  }
  updateButtons();
}

boot();
