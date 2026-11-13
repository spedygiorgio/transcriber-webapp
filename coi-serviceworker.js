// Service worker unico per la PWA:
//  1) inietta gli header COOP/COEP (+CORP) per abilitare SharedArrayBuffer e il
//     multithread WASM su GitHub Pages, che non permette header lato server;
//  2) intercetta i vocali condivisi dal menu "Condividi" di Android/WhatsApp
//     (share_target POST) e li consegna alla pagina.
// Adattato dall'approccio di coi-serviceworker (gzuidhof), MIT.

const SHARE_CACHE = 'transcriber-shared-audio';
const SHARE_KEY = './shared-audio-file';

if (typeof window === 'undefined') {
  self.addEventListener('install', () => self.skipWaiting());
  self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

  self.addEventListener('fetch', (event) => {
    const req = event.request;
    const url = new URL(req.url);

    // --- Share target: ricezione del vocale condiviso (POST) ---
    if (req.method === 'POST' && url.pathname.endsWith('/share-target')) {
      event.respondWith((async () => {
        try {
          const form = await req.formData();
          const file = form.get('audio_file') || form.get('files');
          if (file) {
            const cache = await caches.open(SHARE_CACHE);
            await cache.put(SHARE_KEY, new Response(file, {
              headers: { 'Content-Type': file.type || 'audio/ogg' },
            }));
          }
        } catch (_) { /* no-op */ }
        return Response.redirect(new URL('./?share-target=1', req.url).href, 303);
      })());
      return;
    }

    // --- Consegna / cancellazione del file condiviso ---
    if (url.pathname.endsWith('/shared-audio-file')) {
      if (req.method === 'DELETE') {
        event.respondWith((async () => {
          const cache = await caches.open(SHARE_CACHE);
          await cache.delete(SHARE_KEY);
          return new Response('', { status: 204 });
        })());
        return;
      }
      event.respondWith((async () => {
        const cache = await caches.open(SHARE_CACHE);
        return (await cache.match(SHARE_KEY)) || new Response('', { status: 404 });
      })());
      return;
    }

    // --- Iniezione header COOP/COEP/CORP su tutte le altre richieste ---
    if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;
    event.respondWith(
      fetch(req)
        .then((response) => {
          if (response.status === 0) return response;
          const headers = new Headers(response.headers);
          headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
          headers.set('Cross-Origin-Opener-Policy', 'same-origin');
          // Permette il caricamento delle risorse cross-origin (CDN, modelli HF)
          // sotto COEP require-corp.
          headers.set('Cross-Origin-Resource-Policy', 'cross-origin');
          return new Response(response.body, {
            status: response.status,
            statusText: response.statusText,
            headers,
          });
        })
        .catch((e) => console.error(e))
    );
  });
} else {
  (() => {
    const reloaded = sessionStorage.getItem('coiReloadedBySelf');
    sessionStorage.removeItem('coiReloadedBySelf');
    if (window.crossOriginIsolated || reloaded) return;
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('./coi-serviceworker.js').then((reg) => {
        if (reg.active && !navigator.serviceWorker.controller) {
          sessionStorage.setItem('coiReloadedBySelf', 'true');
          window.location.reload();
        }
      }).catch(() => { /* no-op */ });
    }
  })();
}
