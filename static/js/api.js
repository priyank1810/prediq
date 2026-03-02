const API = {
    baseUrl: '',
    ws: null,
    _wsBackoff: 1000,
    _wsMaxBackoff: 30000,

    // --- Client-side GET request cache ---
    _cache: new Map(),
    _cacheTTLs: {
        '/api/stocks/': 5000,           // quotes: 5s
        '/api/stocks/market-': 10000,   // market status/movers: 10s
        '/api/signals/': 30000,         // signals: 30s
        '/api/indicators/': 60000,      // indicators: 60s
        '/history': 60000,              // history: 60s
        '/api/fii-dii/': 60000,         // FII/DII: 60s
        '/api/sectors/': 60000,         // sectors: 60s
    },
    _cacheMaxEntries: 200,

    _getCacheTTL(url) {
        for (const [pattern, ttl] of Object.entries(this._cacheTTLs)) {
            if (url.includes(pattern)) return ttl;
        }
        return 0; // Don't cache by default
    },

    _cacheGet(url) {
        const entry = this._cache.get(url);
        if (!entry) return null;
        if (Date.now() > entry.expires) {
            this._cache.delete(url);
            return null;
        }
        return entry.data;
    },

    _cacheSet(url, data, ttl) {
        if (this._cache.size >= this._cacheMaxEntries) {
            // Evict oldest entries
            const keys = [...this._cache.keys()];
            for (let i = 0; i < 50 && i < keys.length; i++) {
                this._cache.delete(keys[i]);
            }
        }
        this._cache.set(url, { data, expires: Date.now() + ttl });
    },

    async request(url, options = {}) {
        const method = (options.method || 'GET').toUpperCase();

        // Check client-side cache for GET requests
        if (method === 'GET') {
            const cached = this._cacheGet(url);
            if (cached !== null) return cached;
        }

        try {
            const headers = {
                'Content-Type': 'application/json',
            };

            const fetchOptions = { headers, ...options };
            // Forward AbortController signal
            if (options.signal) {
                fetchOptions.signal = options.signal;
            }

            const res = await fetch(this.baseUrl + url, fetchOptions);

            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: res.statusText }));
                throw new Error(err.detail || 'Request failed');
            }
            const data = await res.json();

            // Cache GET responses
            if (method === 'GET') {
                const ttl = this._getCacheTTL(url);
                if (ttl > 0) this._cacheSet(url, data, ttl);
            }

            return data;
        } catch (e) {
            if (e.name === 'AbortError') throw e;
            console.error(`API error [${url}]:`, e);
            throw e;
        }
    },

    // Stocks (public)
    searchStocks(query, signal) { return this.request(`/api/stocks/search?q=${encodeURIComponent(query)}`, signal ? { signal } : {}); },
    getQuote(symbol) { return this.request(`/api/stocks/${encodeURIComponent(symbol)}/quote`); },
    getHistory(symbol, period = '1y') { return this.request(`/api/stocks/${encodeURIComponent(symbol)}/history?period=${period}`); },
    getMarketStatus() { return this.request('/api/stocks/market-status'); },
    getDataSource() { return this.request('/api/stocks/data-source'); },
    getMultipleQuotes(symbols) {
        return this.request('/api/stocks/quotes/bulk', {
            method: 'POST',
            body: JSON.stringify({ symbols })
        });
    },

    // Market Movers (public)
    getMarketMovers(count = 10) { return this.request(`/api/stocks/market-movers?count=${count}`); },

    // Predictions
    getPredictions(symbol, horizon = '1d') {
        return this.request(`/api/predictions/${encodeURIComponent(symbol)}`, {
            method: 'POST',
            body: JSON.stringify({ horizon, models: ['lstm', 'prophet', 'xgboost'] })
        });
    },

    // Indicators (public)
    getIndicators(symbol) { return this.request(`/api/indicators/${encodeURIComponent(symbol)}`); },

    // Watchlist
    getWatchlist() { return this.request('/api/watchlist'); },
    getWatchlistOverview() { return this.request('/api/watchlist/overview'); },
    addToWatchlist(data) { return this.request('/api/watchlist', { method: 'POST', body: JSON.stringify(data) }); },
    removeFromWatchlist(symbol) { return this.request(`/api/watchlist/${encodeURIComponent(symbol)}`, { method: 'DELETE' }); },

    // Signals (public)
    getIntradaySignal(symbol) { return this.request(`/api/signals/${encodeURIComponent(symbol)}`); },
    getSignalHistory(symbol, limit = 20) { return this.request(`/api/signals/${encodeURIComponent(symbol)}/history?limit=${limit}`); },
    scanHighConfidence(threshold = 60) { return this.request(`/api/signals/scan/high-confidence?threshold=${threshold}`); },
    getSignalAccuracy() { return this.request('/api/signals/stats/accuracy'); },
    onSignalUpdate: null,
    setSignalUpdateHandler(handler) { this.onSignalUpdate = handler; },

    // FII/DII
    getFIIDIIDaily() { return this.request('/api/fii-dii/daily'); },
    getFIIDIIHistory(days = 30) { return this.request(`/api/fii-dii/history?days=${days}`); },

    // Market Mood
    getMarketMood() { return this.request('/api/signals/market-mood'); },

    // Sectors
    getSectorHeatmap() { return this.request('/api/sectors/heatmap'); },

    // Chart Patterns
    getPatterns(symbol) { return this.request(`/api/indicators/${encodeURIComponent(symbol)}/patterns`); },

    // Smart Alerts
    getSmartAlerts() { return this.request('/api/alerts/smart'); },
    createSmartAlert(data) { return this.request('/api/alerts/smart', { method: 'POST', body: JSON.stringify(data) }); },
    deleteSmartAlert(id) { return this.request(`/api/alerts/smart/${id}`, { method: 'DELETE' }); },

    // Accuracy Stats
    getStatsBySector() { return this.request('/api/signals/stats/by-sector'); },
    getStatsByHorizon() { return this.request('/api/signals/stats/by-horizon'); },
    getStatsByRegime() { return this.request('/api/signals/stats/by-regime'); },
    getBacktestPnL() { return this.request('/api/signals/stats/backtest-pnl'); },

    // High-confidence alert handler
    onHighConfidenceAlert: null,
    setHighConfidenceHandler(handler) { this.onHighConfidenceAlert = handler; },

    // Market mood handler
    onMarketMoodUpdate: null,
    setMarketMoodHandler(handler) { this.onMarketMoodUpdate = handler; },

    // WebSocket with exponential backoff
    connectWebSocket(symbols, onPriceUpdate, onAlert) {
        this._wsBackoff = 1000; // Reset on fresh connect
        this._wsSymbols = symbols;
        this._wsOnPrice = onPriceUpdate;
        this._wsOnAlert = onAlert;
        this._connectWS();
    },

    _connectWS() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.ws = new WebSocket(`${protocol}//${location.host}/ws/prices`);
        this.ws.onopen = () => {
            this._wsBackoff = 1000; // Reset backoff on successful connection
            if (this._wsSymbols && this._wsSymbols.length > 0) {
                this.ws.send(JSON.stringify({ subscribe: this._wsSymbols }));
            }
        };
        this.ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === 'price_update' && this._wsOnPrice) this._wsOnPrice(msg.data);
            if (msg.type === 'alert_triggered' && this._wsOnAlert) this._wsOnAlert(msg.data);
            if (msg.type === 'signal_update' && this.onSignalUpdate) this.onSignalUpdate(msg.data);
            if (msg.type === 'high_confidence_alert' && this.onHighConfidenceAlert) this.onHighConfidenceAlert(msg.data);
            if (msg.type === 'market_mood_update' && this.onMarketMoodUpdate) this.onMarketMoodUpdate(msg.data);
            if (msg.type === 'smart_alert_triggered' && this.onHighConfidenceAlert) this.onHighConfidenceAlert(msg.data);
        };
        this.ws.onclose = () => {
            setTimeout(() => this._connectWS(), this._wsBackoff);
            this._wsBackoff = Math.min(this._wsBackoff * 2, this._wsMaxBackoff);
        };
    },

    subscribeTo(symbols) {
        // Track symbols for reconnection
        if (symbols && symbols.length > 0) {
            this._wsSymbols = [...new Set([...(this._wsSymbols || []), ...symbols])];
        }
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ subscribe: symbols }));
        }
    }
};
