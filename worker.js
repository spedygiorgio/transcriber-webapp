// Transcriber Web — Web Worker (modulo)
// Trascrizione: transformers.js (Whisper ONNX) su WebGPU con fallback WASM.
// Riassunto: WebLLM (WebGPU) oppure TextRank estrattivo in puro JS.
// Tutto 100% nel browser: nessun dato lascia il dispositivo.

import {
  pipeline,
  env,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';

// Niente modelli locali: li prendiamo da Hugging Face e li teniamo nella
// Cache del browser (comportamento di default), così l'app resta offline
// dopo il primo download.
env.allowLocalModels = false;

// Senza cross-origin isolation i thread WASM non sono disponibili: limitiamo
// a 1 per evitare errori. Con il COI service worker attivo ne usa di più.
try {
  if (!self.crossOriginIsolated && env.backends?.onnx?.wasm) {
    env.backends.onnx.wasm.numThreads = 1;
  }
} catch (_) { /* no-op */ }

// Mappa degli ID modello ONNX (multilingua).
const MODEL_IDS = {
  tiny: 'onnx-community/whisper-tiny',
  base: 'onnx-community/whisper-base',
  small: 'onnx-community/whisper-small',
};

let asr = null;
let asrModelId = null;
let asrDevice = null;

let webLlmEngine = null;
let activeLlmModelId = null;

// Righe che Whisper "inventa" su silenzio o in coda (apprese dai sottotitoli).
// Portate dal kernel Rust (transcriber-core).
const INVENTED_LINES = [
  'sottotitoli e revisione a cura di qtss',
  'sottotitoli creati dalla comunità amara.org',
  'sottotitoli a cura di qtss',
  'subtitles by the amara.org community',
];

function isInventedText(text) {
  const t = (text || '')
    .replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '')
    .toLowerCase();
  return INVENTED_LINES.includes(t);
}

async function getTranscriber(modelId, preferWebgpu) {
  if (asr && asrModelId === modelId) return asr;
  if (asr) {
    try { await asr.dispose?.(); } catch (_) { /* no-op */ }
    asr = null;
    asrModelId = null;
  }

  const hasWebgpu = typeof navigator !== 'undefined' && !!navigator.gpu;
  const wantDevice = preferWebgpu && hasWebgpu ? 'webgpu' : 'wasm';

  const build = (device) =>
    pipeline('automatic-speech-recognition', MODEL_IDS[modelId], {
      device,
      dtype: 'q8', // buon compromesso dimensione/qualità, valido su WebGPU e WASM
      progress_callback: (p) => {
        if (p.status === 'progress') {
          self.postMessage({
            type: 'model-progress',
            file: p.file,
            percent: Math.round(p.progress || 0),
          });
        }
      },
    });

  try {
    asr = await build(wantDevice);
    asrDevice = wantDevice;
  } catch (err) {
    // Se WebGPU fallisce (driver, flag, memoria), ripieghiamo su WASM.
    if (wantDevice === 'webgpu') {
      self.postMessage({
        type: 'notice',
        message: 'WebGPU non disponibile, uso il backend WASM (più lento).',
      });
      asr = await build('wasm');
      asrDevice = 'wasm';
    } else {
      throw err;
    }
  }

  asrModelId = modelId;
  self.postMessage({ type: 'model-ready', modelId, device: asrDevice });
  return asr;
}

function chunksToSegments(chunks) {
  if (!Array.isArray(chunks)) return [];
  const segments = [];
  for (const c of chunks) {
    const text = (c.text || '').trim();
    if (!text || isInventedText(text)) continue;
    const start = Array.isArray(c.timestamp) ? c.timestamp[0] : null;
    const end = Array.isArray(c.timestamp) ? c.timestamp[1] : null;
    segments.push({
      start_ms: start == null ? 0 : Math.round(start * 1000),
      end_ms: end == null ? 0 : Math.round(end * 1000),
      text,
    });
  }
  return segments;
}

async function transcribe(msg) {
  const transcriber = await getTranscriber(msg.modelId, msg.preferWebgpu !== false);

  // L'audio arriva già decodificato a 16 kHz mono (Float32Array) dal main thread.
  const audio = msg.audio instanceof Float32Array ? msg.audio : new Float32Array(msg.audio);
  const durationMs = Math.round((audio.length / 16000) * 1000);

  self.postMessage({ type: 'transcribe-started' });
  const t0 = performance.now();

  const out = await transcriber(audio, {
    chunk_length_s: 30,
    stride_length_s: 5,
    language: msg.language && msg.language !== 'auto' ? msg.language : undefined,
    task: 'transcribe',
    return_timestamps: true,
  });

  const inferenceMs = Math.round(performance.now() - t0);
  const segments = chunksToSegments(out.chunks);
  const text = segments.length
    ? segments.map((s) => s.text).join(' ')
    : (out.text || '').trim();

  self.postMessage({
    type: 'transcription-done',
    transcript: {
      text,
      segments,
      language: msg.language || 'auto',
      audio_duration_ms: durationMs,
      inference_ms: inferenceMs,
      device: asrDevice,
    },
  });
}

// ---- Riassunto estrattivo TextRank (porting fedele da transcriber-core) ----

const SUMM_STOPWORDS = new Set([
  'che', 'non', 'per', 'con', 'una', 'uno', 'del', 'della', 'delle', 'dei', 'degli',
  'nel', 'nella', 'nelle', 'sul', 'sulla', 'come', 'sono', 'siamo', 'abbiamo', 'quindi',
  'allora', 'comunque', 'praticamente', 'diciamo', 'così', 'questo', 'questa', 'questi',
  'quella', 'quello', 'anche', 'pero', 'però', 'poi', 'gia', 'già', 'ancora', 'tutto',
]);

const SUMM_ACTION_MARKERS = [
  'sabato', 'domenica', 'lunedì', 'martedì', 'mercoledì', 'giovedì', 'venerdì',
  'domani', 'entro', 'alle', 'ore', 'bisogna', 'dobbiamo', 'devi', 'chiama',
  'chiamato', 'viene', 'verrà', 'fare', 'controllare', 'mandami', 'scadenza',
];

function summarizeExtractive(text, keyPoints) {
  const sentences = (text || '')
    .split(/[.!?]/)
    .map((s) => s.trim())
    .filter((s) => s.split(/\s+/).filter(Boolean).length >= 4);

  if (sentences.length === 0) return (text || '').trim();
  if (sentences.length <= 2) return sentences.join('. ') + '.';

  const freq = new Map();
  for (const s of sentences) {
    for (const raw of s.split(/[^\p{L}\p{N}]+/u)) {
      const w = raw.toLowerCase();
      if ([...w].length >= 4 && !SUMM_STOPWORDS.has(w)) {
        freq.set(w, (freq.get(w) || 0) + 1);
      }
    }
  }

  let scored = sentences.map((s, idx) => {
    const lower = s.toLowerCase();
    let score = 0;
    for (const w of lower.split(/[^\p{L}\p{N}]+/u)) {
      if (freq.has(w)) score += freq.get(w);
    }
    const isAction = SUMM_ACTION_MARKERS.some((m) => lower.includes(m));
    if (isAction) score += 6;
    if (idx === 0 || idx === sentences.length - 1) score += 2;
    return { idx, score, s, isAction };
  });

  scored.sort((a, b) => b.score - a.score);
  let top = scored.slice(0, 4);
  top.sort((a, b) => a.idx - b.idx);

  if (!keyPoints) {
    return top.slice(0, 3).map((t) => `${t.s}.`).join(' ');
  }

  let out = '';
  const todos = [];
  for (const t of top) {
    out += `- ${t.s}.\n`;
    if (t.isAction) todos.push(`- ${t.s}.`);
  }
  if (todos.length) {
    out += '\nDa fare:\n' + todos.join('\n');
  }
  return out.trim();
}

async function summarizeWithLocalWebLLM(text, style, llmModelId) {
  // jsDelivr (+esm) si comporta correttamente sotto COEP/CORP; esm.run no
  // (l'import falliva con "Failed to fetch" sul sito pubblicato).
  const webllm = await import('https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2/+esm');
  if (!webLlmEngine || activeLlmModelId !== llmModelId) {
    if (webLlmEngine) { await webLlmEngine.unload(); webLlmEngine = null; }
    webLlmEngine = await webllm.CreateMLCEngine(llmModelId, {
      initProgressCallback: (r) =>
        self.postMessage({
          type: 'llm-progress',
          percent: Math.round((r.progress || 0) * 100),
          message: r.text || 'Download LLM...',
        }),
    });
    activeLlmModelId = llmModelId;
  }
  const sys = style === 'Brief'
    ? 'Riassumi in italiano in massimo 2 o 3 frasi in terza persona.'
    : 'Riassumi in italiano con elenco puntato ("- ") e sezione finale "Da fare:".';
  const reply = await webLlmEngine.chat.completions.create({
    messages: [
      { role: 'system', content: sys },
      { role: 'user', content: text },
    ],
    temperature: 0.2,
    max_tokens: style === 'Brief' ? 250 : 500,
  });
  return (reply.choices?.[0]?.message?.content || '')
    .replace(/<think>[\s\S]*?<\/think>/g, '')
    .trim();
}

self.onmessage = async (event) => {
  const msg = event.data;
  try {
    if (msg.type === 'transcribe') {
      await transcribe(msg);
      return;
    }

    if (msg.type === 'summarize') {
      if (msg.engine === 'webllm' && msg.llmModelId) {
        const s = await summarizeWithLocalWebLLM(msg.text, msg.style, msg.llmModelId);
        self.postMessage({
          type: 'summary-done',
          summary: s,
          engineUsed: `WebLLM locale (${msg.llmModelId})`,
        });
      } else {
        const s = summarizeExtractive(msg.text, msg.style === 'KeyPoints');
        self.postMessage({
          type: 'summary-done',
          summary: s,
          engineUsed: 'TextRank estrattivo (0 MB RAM)',
        });
      }
      return;
    }

    if (msg.type === 'release-memory') {
      if (asr) { try { await asr.dispose?.(); } catch (_) {} asr = null; asrModelId = null; }
      if (webLlmEngine) { try { await webLlmEngine.unload(); } catch (_) {} webLlmEngine = null; activeLlmModelId = null; }
      self.postMessage({ type: 'memory-released' });
      return;
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: err?.message || String(err) });
  }
};
