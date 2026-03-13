const CACHE_NAME = 'climalens-v1';
const ASSETS = [
    '/',
    '/static/css/style.css',
    '/static/js/charts.js',
    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap',
    'https://unpkg.com/@phosphor-icons/web',
    'https://cdn.plot.ly/plotly-2.27.0.min.js'
];

self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => cache.addAll(ASSETS))
    );
});

self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request)
            .then(response => response || fetch(event.request))
    );
});
