# transcriber-webapp

Web app / PWA **100% locale, offline-capable e zero-API** per trascrivere e
riassumere vocali WhatsApp (e altre registrazioni) interamente nel browser:
nessun audio e nessun dato lascia il dispositivo.

**Live:** https://spedygiorgio.github.io/transcriber-webapp/

## Architettura

Tutta l'elaborazione avviene nel browser, in un Web Worker isolato.

| Funzione | Motore | Note |
|---|---|---|
| Trascrizione (speech-to-text) | [transformers.js](https://github.com/huggingface/transformers.js) — Whisper ONNX | WebGPU con fallback automatico a WASM |
| Decodifica audio → 16 kHz mono | Web Audio API (`AudioContext` + `OfflineAudioContext`) | sul main thread |
| Riassunto estrattivo | TextRank in puro JS | istantaneo, 0 MB extra |
| Riassunto generativo (opz.) | [WebLLM](https://github.com/mlc-ai/web-llm) — SmolLM2 / Qwen2.5 | WebGPU, modello scaricato una volta |

I modelli Whisper (`onnx-community/whisper-tiny|base|small`) vengono scaricati
da Hugging Face al primo uso e tenuti nella **Cache del browser**, così l'app
funziona offline dopo il primo avvio. Il COI service worker inietta gli header
`COOP`/`COEP`/`CORP` (GitHub Pages non permette header lato server) per abilitare
`SharedArrayBuffer` e il multithread WASM, e intercetta i vocali condivisi dal
menu **Condividi** di Android/WhatsApp (`share_target`).

> **Nota sul kernel Rust.** Questa webapp **non** usa
> [`transcriber-core`](https://github.com/spedygiorgio/transcriber-core): il
> kernel dipende da `whisper.cpp` (C++) e `libopus` (C), che non compilano per
> il target `wasm32-unknown-unknown` di `wasm-bindgen` (manca una libc). Nel
> browser la via praticabile è un motore di inferenza nativo-web (qui Whisper
> ONNX via transformers.js). Il kernel Rust resta il cuore comune per la CLI e
> l'app Android, dove whisper.cpp gira nativamente.

## Funzionalità

- Caricamento file (`.opus`, `.ogg`, `.m4a`, `.mp3`, `.wav`, `.flac`, `.webm`), drag & drop, o **registrazione dal microfono**
- Profilo hardware adattivo (sceglie modello e motore in base a RAM/GPU)
- Trascrizione con **timestamp per segmento**, player audio sincronizzato (clic sul testo → salto nel punto)
- Riassunto **Breve** o a **Punti chiave** (con sezione "Da fare:")
- Export **`.txt` / `.md` / `.srt`** e copia rapida
- **Storico locale** in IndexedDB
- Installabile come PWA, con `share_target` per ricevere i vocali da WhatsApp

## Sviluppo

È un sito statico: nessuna build. Per provarlo in locale serve un server con
cross-origin isolation (per i thread WASM); con WebGPU non è strettamente
necessaria. Esempio:

```bash
npx serve .   # oppure qualsiasi static server; apri http://localhost:3000
```

Il deploy su GitHub Pages è automatico a ogni push su `main`
(`.github/workflows/deploy-wasm-pages.yml`).

## Privacy

Audio e trascrizioni non vengono mai inviati a server esterni. Le uniche
richieste di rete sono il **download una tantum** dei modelli (Hugging Face, CDN
jsDelivr per le librerie); dopo, l'app funziona offline.
