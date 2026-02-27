const API = {
    baseUrl: '',
    ws: null,

    async request(url, options = {}) {
        try {
            const headers = {
                'Content-Type': 'application/json',
            };

            const res = await fetch(this.baseUrl + url, {
                headers,
                ...options
            });

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

    // Option Chain (public)
    getOptionExpiries(symbol) { return this.request(`/api/options/${encodeURIComponent(symbol)}/expiries`); },
    getOptionChain(symbol, expiry = '') {
        const q = expiry ? `?expiry=${encodeURIComponent(expiry)}` : '';
        return this.request(`/api/options/${encodeURIComponent(symbol)}/chain${q}`);
    },

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
            if (msg.type === 'market_mood_update' && this.onMarketMoodUpdate) this.onMarketMoodUpdate(msg.data);
            if (msg.type === 'smart_alert_triggered' && this.onHighConfidenceAlert) this.onHighConfidenceAlert(msg.data);
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
