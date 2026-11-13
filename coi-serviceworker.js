if (typeof window === 'undefined') {
  self.addEventListener('install', () => self.skipWaiting());
  self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));
  self.addEventListener('fetch', (event) => {
    const req = event.request;
    if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;
    event.respondWith(
      fetch(req).then((response) => {
        if (response.status === 0) return response;
        const newHeaders = new Headers(response.headers);
        newHeaders.set('Cross-Origin-Embedder-Policy', 'require-corp');
        newHeaders.set('Cross-Origin-Opener-Policy', 'same-origin');
        return new Response(response.body, { status: response.status, statusText: response.statusText, headers: newHeaders });
      }).catch((e) => console.error(e))
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
      });
    }
  })();
}
