let wasmModule = null;
let activeTranscriber = null;
let activeWhisperModelId = null;
let webLlmEngine = null;
let activeLlmModelId = null;

const MODEL_URLS = {
  'tiny': 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin',
  'base': 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin',
  'small-q5_1': 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q5_1.bin',
  'small': 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin'
};

async function getModelsDir() {
  const root = await navigator.storage.getDirectory();
  return await root.getDirectoryHandle('transcriber-models', { create: true });
}

async function readOpfsModel(modelId) {
  try {
    const dir = await getModelsDir();
    const fileHandle = await dir.getFileHandle(`ggml-${modelId}.bin`);
    const file = await fileHandle.getFile();
    if (file.size < 1024 * 1024) return null;
    return new Uint8Array(await file.arrayBuffer());
  } catch { return null; }
}

async function listOpfsModels() {
  try {
    const dir = await getModelsDir();
    const installed = [];
    for await (const [name, handle] of dir.entries()) {
      if (handle.kind === 'file' && name.startsWith('ggml-') && name.endsWith('.bin')) {
        const f = await handle.getFile();
        if (f.size > 1024 * 1024) installed.push({ id: name.slice(5, -4), sizeBytes: f.size });
      }
    }
    return installed;
  } catch { return []; }
}

async function downloadAndCacheModel(modelId) {
  const url = MODEL_URLS[modelId];
  if (!url) throw new Error(`Modello "${modelId}" non presente nel catalogo.`);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Download fallito (HTTP ${response.status})`);
  const total = Number(response.headers.get('content-length') || 0);
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    self.postMessage({ type: 'download-progress', modelId, received, total, percent: total > 0 ? Math.round((received / total) * 100) : 0 });
  }
  const fullBuffer = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { fullBuffer.set(chunk, offset); offset += chunk.byteLength; }
  const dir = await getModelsDir();
  const fileHandle = await dir.getFileHandle(`ggml-${modelId}.bin`, { create: true });
  const writable = await fileHandle.createWritable();
  await writable.write(fullBuffer);
  await writable.close();
  return fullBuffer;
}

async function ensureWasmLoaded() {
  if (wasmModule) return wasmModule;
  const mod = await import('./pkg/transcriber_wasm.js');
  await mod.default();
  wasmModule = mod;
  return wasmModule;
}

async function ensureTranscriberReady(modelId) {
  const wasm = await ensureWasmLoaded();
  if (activeTranscriber && activeWhisperModelId === modelId) return activeTranscriber;
  if (activeTranscriber) { activeTranscriber.free(); activeTranscriber = null; activeWhisperModelId = null; }
  let modelBytes = await readOpfsModel(modelId);
  if (!modelBytes) modelBytes = await downloadAndCacheModel(modelId);
  activeTranscriber = new wasm.WasmTranscriber(modelBytes);
  activeWhisperModelId = modelId;
  return activeTranscriber;
}

async function summarizeWithLocalWebLLM(text, style, llmModelId) {
  const webllm = await import('https://esm.run/@mlc-ai/web-llm');
  if (!webLlmEngine || activeLlmModelId !== llmModelId) {
    if (webLlmEngine) { await webLlmEngine.unload(); webLlmEngine = null; }
    webLlmEngine = await webllm.CreateMLCEngine(llmModelId, {
      initProgressCallback: (r) => self.postMessage({ type: 'llm-progress', percent: Math.round((r.progress || 0) * 100), message: r.text || 'Download LLM...' })
    });
    activeLlmModelId = llmModelId;
  }
  const sys = style === 'Brief'
    ? 'Riassumi in italiano in massimo 2 o 3 frasi in terza persona.'
    : 'Riassumi in italiano con elenco puntato ("- ") e sezione finale "Da fare:".';
  const reply = await webLlmEngine.chat.completions.create({
    messages: [{ role: 'system', content: sys }, { role: 'user', content: text }],
    temperature: 0.2,
    max_tokens: style === 'Brief' ? 250 : 500
  });
  return (reply.choices?.[0]?.message?.content || '').replace(/<think>[\s\S]*?<\/think>/g, '').trim();
}

self.onmessage = async (event) => {
  const msg = event.data;
  try {
    if (msg.type === 'list-models') { self.postMessage({ type: 'models-list', installed: await listOpfsModels() }); return; }
    if (msg.type === 'download-model') {
      await downloadAndCacheModel(msg.modelId);
      self.postMessage({ type: 'models-list', installed: await listOpfsModels() });
      self.postMessage({ type: 'download-complete', modelId: msg.modelId });
      return;
    }
    if (msg.type === 'transcribe') {
      const transcriber = await ensureTranscriberReady(msg.modelId);
      self.postMessage({ type: 'models-list', installed: await listOpfsModels() });
      const transcript = transcriber.transcribe_whatsapp(
        new Uint8Array(msg.audioBuffer), msg.extensionHint || 'opus', msg.language || 'it',
        msg.initialPrompt || null, msg.threads || 0,
        (segment) => self.postMessage({ type: 'live-segment', segment })
      );
      self.postMessage({ type: 'transcription-done', transcript });
      return;
    }
    if (msg.type === 'summarize') {
      if (msg.engine === 'webllm' && msg.llmModelId) {
        const s = await summarizeWithLocalWebLLM(msg.text, msg.style, msg.llmModelId);
        self.postMessage({ type: 'summary-done', summary: s, engineUsed: `WebLLM Locale (${msg.llmModelId})` });
      } else {
        const wasm = await ensureWasmLoaded();
        const s = wasm.summarize_extractive_rust(msg.text, msg.style === 'KeyPoints');
        self.postMessage({ type: 'summary-done', summary: s, engineUsed: 'Kernel Rust TextRank (0 MB RAM)' });
      }
      return;
    }
    if (msg.type === 'release-memory') {
      if (activeTranscriber) { activeTranscriber.free(); activeTranscriber = null; activeWhisperModelId = null; }
      if (webLlmEngine) { await webLlmEngine.unload(); webLlmEngine = null; activeLlmModelId = null; }
      self.postMessage({ type: 'memory-released' });
    }
  } catch (err) { self.postMessage({ type: 'error', message: err?.message || String(err) }); }
};
