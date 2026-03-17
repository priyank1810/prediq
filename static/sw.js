const CACHE_NAME = 'stockai-v2';
const RUNTIME_CACHE = 'stockai-runtime-v1';
const MAX_RUNTIME_CACHE_ENTRIES = 100;

// Pre-cache the app shell + critical lazy-loaded modules
const APP_SHELL = [
    '/',
    '/static/css/style.css?v=4',
    '/static/js/app.js?v=14',
    '/static/js/api.js?v=12',
    '/static/js/chart.js?v=11',
    '/static/js/search.js?v=11',
    '/static/js/shimmer.js?v=1',
    '/static/js/lazy.js?v=2',
    '/static/js/notifications.js?v=11',
    // Pre-cache critical lazy modules (signals is loaded at init)
    '/static/js/signals.js?v=13',
    '/static/js/watchlist.js?v=12',
    '/static/js/predictions.js?v=12',
];

// Install: pre-cache the app shell
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL))
    );
    self.skipWaiting();
});

// Activate: clean up old caches
self.addEventListener('activate', (event) => {
    const validCaches = new Set([CACHE_NAME, RUNTIME_CACHE]);
    event.waitUntil(
        caches.keys().then((keys) =>
            Promise.all(
                keys.filter((key) => !validCaches.has(key)).map((key) => caches.delete(key))
            )
        )
    );
    self.clients.claim();
});

// Trim runtime cache to prevent unbounded growth
async function trimCache(cacheName, maxEntries) {
    const cache = await caches.open(cacheName);
    const keys = await cache.keys();
    if (keys.length > maxEntries) {
        // Delete oldest entries (FIFO)
        const toDelete = keys.slice(0, keys.length - maxEntries);
        await Promise.all(toDelete.map((key) => cache.delete(key)));
    }
}

// Fetch: route requests by strategy
self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);

    // Skip non-GET requests
    if (event.request.method !== 'GET') return;

    // API requests: network-only (no caching)
    if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/ws/')) {
        return;
    }

    // Navigation requests: network-first with cache fallback
    if (event.request.mode === 'navigate') {
        event.respondWith(
            fetch(event.request).then((response) => {
                if (response.ok) {
                    const clone = response.clone();
                    caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
                }
                return response;
            }).catch(() => caches.match(event.request))
        );
        return;
    }

    // Static assets: cache-first (versioned URLs are immutable)
    const isStatic = url.pathname.startsWith('/static/');
    if (isStatic) {
        event.respondWith(
            caches.match(event.request).then((cached) => {
                if (cached) return cached;
                return fetch(event.request).then((response) => {
                    if (response.ok) {
                        const clone = response.clone();
                        caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
                    }
                    return response;
                });
            })
        );
        return;
    }

    // CDN resources: stale-while-revalidate (serve cached, refresh in background)
    const isCDN = url.hostname.includes('cdn.tailwindcss.com') ||
                  url.hostname.includes('unpkg.com') ||
                  url.hostname.includes('accounts.google.com');

    if (isCDN) {
        event.respondWith(
            caches.match(event.request).then((cached) => {
                const fetchPromise = fetch(event.request).then((response) => {
                    if (response.ok) {
                        const clone = response.clone();
                        caches.open(RUNTIME_CACHE).then((cache) => {
                            cache.put(event.request, clone);
                            trimCache(RUNTIME_CACHE, MAX_RUNTIME_CACHE_ENTRIES);
                        });
                    }
                    return response;
                }).catch(() => cached);

                return cached || fetchPromise;
            })
        );
        return;
    }

    // Everything else: network with cache fallback
    event.respondWith(
        fetch(event.request).catch(() => caches.match(event.request))
    );
});
