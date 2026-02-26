const API = {
    baseUrl: '',
    ws: null,

    async request(url, options = {}) {
        try {
            const headers = {
                'Content-Type': 'application/json',
                ...(typeof Auth !== 'undefined' ? Auth.getAuthHeaders() : {}),
            };

            const res = await fetch(this.baseUrl + url, {
                headers,
                ...options
            });

            // Handle 401 — try token refresh + retry once
            if (res.status === 401 && typeof Auth !== 'undefined' && Auth.refreshToken) {
                const refreshed = await Auth.tryRefresh();
                if (refreshed) {
                    const retryRes = await fetch(this.baseUrl + url, {
                        headers: {
                            'Content-Type': 'application/json',
                            ...Auth.getAuthHeaders(),
                        },
                        ...options
                    });
                    if (retryRes.ok) return await retryRes.json();
                    if (retryRes.status === 401) {
                        Auth.logout();
                        throw new Error('Session expired. Please log in again.');
                    }
                    const err = await retryRes.json().catch(() => ({ detail: retryRes.statusText }));
                    throw new Error(err.detail || 'Request failed');
                }
                Auth.logout();
                throw new Error('Session expired. Please log in again.');
            }

            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: res.statusText }));
                throw new Error(err.detail || 'Request failed');
            }
            return await res.json();
        } catch (e) {
            console.error(`API error [${url}]:`, e);
            throw e;
        }
    },

    // Stocks (public)
    searchStocks(query) { return this.request(`/api/stocks/search?q=${encodeURIComponent(query)}`); },
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

    // Predictions (auth required)
    getPredictions(symbol, horizon = '1d') {
        return this.request(`/api/predictions/${encodeURIComponent(symbol)}`, {
            method: 'POST',
            body: JSON.stringify({ horizon, models: ['lstm', 'prophet', 'xgboost'] })
        });
    },

    // Indicators (public)
    getIndicators(symbol) { return this.request(`/api/indicators/${encodeURIComponent(symbol)}`); },

    // Watchlist (auth required)
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

    // Option Chain (public)
    getOptionExpiries(symbol) { return this.request(`/api/options/${encodeURIComponent(symbol)}/expiries`); },
    getOptionChain(symbol, expiry = '') {
        const q = expiry ? `?expiry=${encodeURIComponent(expiry)}` : '';
        return this.request(`/api/options/${encodeURIComponent(symbol)}/chain${q}`);
    },

    // High-confidence alert handler
    onHighConfidenceAlert: null,
    setHighConfidenceHandler(handler) { this.onHighConfidenceAlert = handler; },

    // WebSocket
    connectWebSocket(symbols, onPriceUpdate, onAlert) {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.ws = new WebSocket(`${protocol}//${location.host}/ws/prices`);
        this.ws.onopen = () => {
            if (symbols.length > 0) {
                this.ws.send(JSON.stringify({ subscribe: symbols }));
            }
        };
        this.ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === 'price_update' && onPriceUpdate) onPriceUpdate(msg.data);
            if (msg.type === 'alert_triggered' && onAlert) onAlert(msg.data);
            if (msg.type === 'signal_update' && this.onSignalUpdate) this.onSignalUpdate(msg.data);
            if (msg.type === 'high_confidence_alert' && this.onHighConfidenceAlert) this.onHighConfidenceAlert(msg.data);
        };
        this.ws.onclose = () => {
            setTimeout(() => this.connectWebSocket(symbols, onPriceUpdate, onAlert), 5000);
        };
    },

    subscribeTo(symbols) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ subscribe: symbols }));
        }
    }
};
