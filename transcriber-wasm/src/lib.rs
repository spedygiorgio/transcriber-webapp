use serde::Serialize;
use std::collections::{HashMap, HashSet};
use transcriber_core::{Segment, TranscribeOptions, Transcriber};
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
struct WasmSegment {
    start_ms: u64,
    end_ms: u64,
    text: String,
}

impl From<Segment> for WasmSegment {
    fn from(s: Segment) -> Self {
        Self { start_ms: s.start_ms, end_ms: s.end_ms, text: s.text }
    }
}

#[derive(Serialize)]
struct WasmTranscript {
    text: String,
    segments: Vec<WasmSegment>,
    language: String,
    audio_duration_ms: u64,
    speech_duration_ms: u64,
    decode_ms: u64,
    inference_ms: u64,
}

#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct WasmTranscriber {
    inner: Transcriber,
}

#[wasm_bindgen]
impl WasmTranscriber {
    #[wasm_bindgen(constructor)]
    pub fn new(model_bytes: &[u8]) -> Result<WasmTranscriber, JsError> {
        let inner = Transcriber::from_bytes(model_bytes)
            .map_err(|e| JsError::new(&format!("Errore caricamento modello GGML: {e}")))?;
        Ok(Self { inner })
    }

    #[wasm_bindgen]
    pub fn transcribe_whatsapp(
        &self,
        audio_bytes: Vec<u8>,
        extension_hint: Option<String>,
        language: Option<String>,
        initial_prompt: Option<String>,
        threads: u32,
        on_segment: Option<js_sys::Function>,
    ) -> Result<JsValue, JsError> {
        let mut opts = TranscribeOptions {
            language,
            initial_prompt,
            threads,
            fast_short_audio: true,
            on_segment: on_segment.map(|cb| {
                Box::new(move |seg: Segment| {
                    if let Ok(js_seg) = serde_wasm_bindgen::to_value(&WasmSegment::from(seg)) {
                        let _ = cb.call1(&JsValue::NULL, &js_seg);
                    }
                }) as transcriber_core::SegmentCallback
            }),
            ..Default::default()
        };

        let transcript = self
            .inner
            .transcribe_bytes(audio_bytes, extension_hint.as_deref(), &mut opts, None)
            .map_err(|e| JsError::new(&format!("Errore trascrizione: {e}")))?;

        let out = WasmTranscript {
            text: transcript.text,
            segments: transcript.segments.into_iter().map(Into::into).collect(),
            language: transcript.language,
            audio_duration_ms: transcript.audio_duration_ms,
            speech_duration_ms: transcript.speech_duration_ms,
            decode_ms: transcript.decode_ms,
            inference_ms: transcript.inference_ms,
        };

        serde_wasm_bindgen::to_value(&out)
            .map_err(|e| JsError::new(&format!("Errore serializzazione JS: {e}")))
    }
}

#[wasm_bindgen]
pub fn summarize_extractive_rust(text: &str, key_points: bool) -> String {
    let sentences: Vec<&str> = text
        .split(['.', '!', '?'])
        .map(str::trim)
        .filter(|s| s.split_whitespace().count() >= 4)
        .collect();

    if sentences.is_empty() {
        return text.trim().to_string();
    }
    if sentences.len() <= 2 {
        return sentences.join(". ") + ".";
    }

    let stopwords: HashSet<&str> = [
        "che", "non", "per", "con", "una", "uno", "del", "della", "delle", "dei", "degli",
        "nel", "nella", "nelle", "sul", "sulla", "come", "sono", "siamo", "abbiamo", "quindi",
        "allora", "comunque", "praticamente", "diciamo", "così", "questo", "questa", "questi",
        "quella", "quello", "anche", "pero", "però", "poi", "gia", "già", "ancora", "tutto",
    ]
    .into_iter()
    .collect();

    let mut freq: HashMap<String, usize> = HashMap::new();
    for s in &sentences {
        for word in s.split(|c: char| !c.is_alphanumeric()) {
            let w = word.to_lowercase();
            if w.chars().count() >= 4 && !stopwords.contains(w.as_str()) {
                *freq.entry(w).or_insert(0) += 1;
            }
        }
    }

    let action_markers = [
        "sabato", "domenica", "lunedì", "martedì", "mercoledì", "giovedì", "venerdì",
        "domani", "entro", "alle", "ore", "bisogna", "dobbiamo", "devi", "chiama",
        "chiamato", "viene", "verrà", "fare", "controllare", "mandami", "scadenza",
    ];

    let mut scored: Vec<(usize, usize, &str, bool)> = sentences
        .iter()
        .enumerate()
        .map(|(idx, &s)| {
            let lower = s.to_lowercase();
            let mut score = 0usize;
            for word in lower.split(|c: char| !c.is_alphanumeric()) {
                if let Some(&f) = freq.get(word) {
                    score += f;
                }
            }
            let is_action = action_markers.iter().any(|m| lower.contains(m));
            if is_action {
                score += 6;
            }
            if idx == 0 || idx == sentences.len() - 1 {
                score += 2;
            }
            (idx, score, s, is_action)
        })
        .collect();

    scored.sort_by(|a, b| b.1.cmp(&a.1));
    let mut top: Vec<(usize, usize, &str, bool)> = scored.into_iter().take(4).collect();
    top.sort_by_key(|item| item.0);

    if !key_points {
        return top
            .into_iter()
            .take(3)
            .map(|(_, _, s, _)| format!("{s}."))
            .collect::<Vec<_>>()
            .join(" ");
    }

    let mut out = String::new();
    let mut todos = Vec::new();
    for (_, _, s, is_action) in top {
        out.push_str(&format!("- {s}.\n"));
        if is_action {
            todos.push(format!("- {s}."));
        }
    }
    if !todos.is_empty() {
        out.push_str("\nDa fare:\n");
        out.push_str(&todos.join("\n"));
    }
    out.trim().to_string()
}
