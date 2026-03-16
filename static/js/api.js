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

    _defaultTimeout: 30000,  // 30s default timeout for all requests

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
            // Attach auth token if available
            if (typeof Auth !== 'undefined' && Auth.token) {
                headers['Authorization'] = `Bearer ${Auth.token}`;
            }

            const fetchOptions = { headers, ...options };

            // Set up timeout via AbortController (unless caller provided their own signal)
            let timeoutId;
            if (!options.signal) {
                const timeout = options.timeout || this._defaultTimeout;
                const controller = new AbortController();
                fetchOptions.signal = controller.signal;
                timeoutId = setTimeout(() => controller.abort(), timeout);
            } else {
                fetchOptions.signal = options.signal;
            }

            let res;
            try {
                res = await fetch(this.baseUrl + url, fetchOptions);
            } finally {
                if (timeoutId) clearTimeout(timeoutId);
            }

            if (res.status === 401 && typeof Auth !== 'undefined' && Auth.refreshToken) {
                const refreshed = await Auth.tryRefresh();
                if (refreshed) {
                    fetchOptions.headers['Authorization'] = `Bearer ${Auth.token}`;
                    const retryRes = await fetch(this.baseUrl + url, fetchOptions);
                    if (!retryRes.ok) {
                        const err = await retryRes.json().catch(() => ({ detail: retryRes.statusText }));
                        throw new Error(err.detail || 'Request failed');
                    }
                    return await retryRes.json();
                }
            }
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

    // Earnings Calendar
    getUpcomingEarnings(symbols = '') { return this.request(`/api/stocks/earnings/upcoming?symbols=${encodeURIComponent(symbols)}`); },

    // Fundamentals & News
    getFundamentals(symbol) { return this.request(`/api/stocks/${encodeURIComponent(symbol)}/fundamentals`); },
    getStockNews(symbol) { return this.request(`/api/stocks/${encodeURIComponent(symbol)}/news`); },

    // Predictions
    getPredictions(symbol, horizon = '1d', signal = null) {
        const opts = {
            method: 'POST',
            body: JSON.stringify({ horizon }),
            timeout: 95000,  // 95s to match backend's 90s prediction timeout
        };
        if (signal) opts.signal = signal;
        return this.request(`/api/predictions/${encodeURIComponent(symbol)}`, opts);
    },

    // Indicators (public)
    getIndicators(symbol, period = '1y') { return this.request(`/api/indicators/${encodeURIComponent(symbol)}?period=${period}`); },

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
    getMultiTimeframeSignals(symbol) { return this.request(`/api/signals/multi-timeframe/${encodeURIComponent(symbol)}`); },
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

    // Correlation Analysis
    getCorrelationMatrix(symbols, period = '6mo') { return this.request(`/api/indicators/correlation-matrix?symbols=${encodeURIComponent(symbols.join(','))}&period=${period}`); },
    getSectorCorrelation(period = '6mo') { return this.request(`/api/indicators/sector-correlation?period=${period}`); },
    getStockCorrelations(symbol, n = 10, period = '6mo') { return this.request(`/api/indicators/${encodeURIComponent(symbol)}/correlations?n=${n}&period=${period}`); },

    // Portfolio
    getPortfolio() { return this.request('/api/portfolio'); },
    getPortfolioSummary() { return this.request('/api/portfolio/summary'); },
    getPortfolioAnalytics() { return this.request('/api/portfolio/analytics'); },
    addHolding(data) { return this.request('/api/portfolio', { method: 'POST', body: JSON.stringify(data) }); },
    deleteHolding(id) { return this.request(`/api/portfolio/${id}`, { method: 'DELETE' }); },
    async exportPortfolioCSV() {
        const headers = {};
        if (typeof Auth !== 'undefined' && Auth.token) {
            headers['Authorization'] = `Bearer ${Auth.token}`;
        }
        const res = await fetch('/api/portfolio/export/csv', { headers });
        if (!res.ok) throw new Error('Export failed');
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = res.headers.get('Content-Disposition')?.match(/filename="(.+)"/)?.[1] || 'portfolio.csv';
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    },
    async exportPortfolioHTML() {
        const headers = {};
        if (typeof Auth !== 'undefined' && Auth.token) {
            headers['Authorization'] = `Bearer ${Auth.token}`;
        }
        const res = await fetch('/api/portfolio/export/html', { headers });
        if (!res.ok) throw new Error('Export failed');
        const html = await res.text();
        const w = window.open('', '_blank');
        w.document.write(html);
        w.document.close();
    },

    // Trade Journal
    getTradeJournal(limit = 50) { return this.request(`/api/journal?limit=${limit}`); },
    createTrade(data) { return this.request('/api/journal', { method: 'POST', body: JSON.stringify(data) }); },
    deleteTrade(id) { return this.request(`/api/journal/${id}`, { method: 'DELETE' }); },
    getTradeStats() { return this.request('/api/journal/stats'); },

    // Strategies
    getStrategies(sort = 'newest', limit = 20, offset = 0) { return this.request(`/api/strategies/?sort=${sort}&limit=${limit}&offset=${offset}`); },
    getStrategyLeaderboard() { return this.request('/api/strategies/leaderboard'); },
    getMyStrategies() { return this.request('/api/strategies/my'); },
    getStrategy(id) { return this.request(`/api/strategies/${id}`); },
    createStrategy(data) { return this.request('/api/strategies/', { method: 'POST', body: JSON.stringify(data) }); },
    upvoteStrategy(id) { return this.request(`/api/strategies/${id}/upvote`, { method: 'POST' }); },
    followStrategy(id) { return this.request(`/api/strategies/${id}/follow`, { method: 'POST' }); },
    deleteStrategy(id) { return this.request(`/api/strategies/${id}`, { method: 'DELETE' }); },

    // Price Alerts
    getAlerts() { return this.request('/api/alerts'); },
    createAlert(data) { return this.request('/api/alerts', { method: 'POST', body: JSON.stringify(data) }); },
    deleteAlert(id) { return this.request(`/api/alerts/${id}`, { method: 'DELETE' }); },

    // Smart Alerts
    getSmartAlerts() { return this.request('/api/alerts/smart'); },
    createSmartAlert(data) { return this.request('/api/alerts/smart', { method: 'POST', body: JSON.stringify(data) }); },
    deleteSmartAlert(id) { return this.request(`/api/alerts/smart/${id}`, { method: 'DELETE' }); },

    // Accuracy Stats
    getStatsBySector() { return this.request('/api/signals/stats/by-sector'); },
    getStatsByHorizon() { return this.request('/api/signals/stats/by-horizon'); },
    getStatsByRegime() { return this.request('/api/signals/stats/by-regime'); },
    getBacktestPnL() { return this.request('/api/signals/stats/backtest-pnl'); },
    getPredictionLeaderboard() { return this.request('/api/signals/stats/prediction-leaderboard'); },
    getBacktestSignal(symbol, testDays = 60) { return this.request(`/api/signals/stats/backtest-signal?symbol=${encodeURIComponent(symbol)}&test_days=${testDays}`); },
    runVisualBacktest(params) {
        return this.request('/api/signals/stats/visual-backtest', {
            method: 'POST',
            body: JSON.stringify(params),
            timeout: 120000,
        });
    },
    runMonteCarlo(params) {
        return this.request('/api/signals/stats/monte-carlo', {
            method: 'POST',
            body: JSON.stringify(params),
            timeout: 60000,
        });
    },

    // High-confidence alert handler
    onHighConfidenceAlert: null,
    setHighConfidenceHandler(handler) { this.onHighConfidenceAlert = handler; },

    // Market mood handler
    onMarketMoodUpdate: null,
    setMarketMoodHandler(handler) { this.onMarketMoodUpdate = handler; },

    // OI/MTF push handlers
    onOIUpdate: null,
    setOIUpdateHandler(handler) { this.onOIUpdate = handler; },
    onMTFUpdate: null,
    setMTFUpdateHandler(handler) { this.onMTFUpdate = handler; },

    // WebSocket with exponential backoff
    connectWebSocket(symbols, onPriceUpdate, onAlert) {
        this._wsBackoff = 1000; // Reset on fresh connect
        this._wsSymbols = symbols;
        this._wsOnPrice = onPriceUpdate;
        this._wsOnAlert = onAlert;
        this._connectWS();
    },

    _wsReconnectTimer: null,
    _wsConnected: false,

    _connectWS() {
        // Clear any pending reconnect timer
        if (this._wsReconnectTimer) {
            clearTimeout(this._wsReconnectTimer);
            this._wsReconnectTimer = null;
        }

        try {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            this.ws = new WebSocket(`${protocol}//${location.host}/ws/prices`);
        } catch (e) {
            this._scheduleReconnect();
            return;
        }

        this.ws.onopen = () => {
            this._wsBackoff = 1000; // Reset backoff on successful connection
            this._wsConnected = true;
            console.log('[WS] Connected');
            if (this._wsSymbols && this._wsSymbols.length > 0) {
                this.ws.send(JSON.stringify({ subscribe: this._wsSymbols }));
            }
        };
        this.ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                if (msg.type === 'price_update' && this._wsOnPrice) this._wsOnPrice(msg.data);
                if (msg.type === 'alert_triggered' && this._wsOnAlert) this._wsOnAlert(msg.data);
                if (msg.type === 'signal_update' && this.onSignalUpdate) this.onSignalUpdate(msg.data);
                if (msg.type === 'high_confidence_alert' && this.onHighConfidenceAlert) this.onHighConfidenceAlert(msg.data);
                if (msg.type === 'market_mood_update' && this.onMarketMoodUpdate) this.onMarketMoodUpdate(msg.data);
                if (msg.type === 'smart_alert_triggered' && this.onHighConfidenceAlert) this.onHighConfidenceAlert(msg.data);
                if (msg.type === 'oi_update' && this.onOIUpdate) this.onOIUpdate(msg.data);
                if (msg.type === 'mtf_update' && this.onMTFUpdate) this.onMTFUpdate(msg.data);
                if (msg.type === 'news_alert') {
                    // Show toast notification for news alerts
                    if (typeof App !== 'undefined') {
                        const d = msg.data;
                        const sentiment = d.score > 0 ? 'positive' : 'negative';
                        const headline = d.top_headline ? `: ${d.top_headline.substring(0, 60)}...` : '';
                        App.showToast(`${d.symbol} news ${sentiment} (score: ${d.score})${headline}`, 'alert');
                    }
                    // Also push to notifications
                    if (typeof Notifications !== 'undefined' && Notifications.addNotification) {
                        const d = msg.data;
                        Notifications.addNotification({
                            type: 'news',
                            symbol: d.symbol,
                            message: d.type === 'news_sentiment_change'
                                ? `${d.symbol} sentiment shifted ${d.change > 0 ? '+' : ''}${d.change} (now ${d.score})`
                                : `${d.symbol} extreme sentiment: ${d.score}`,
                        });
                    }
                }
                if (msg.type === 'scanner_alert') {
                    if (typeof App !== 'undefined') {
                        const d = msg.data;
                        const filters = (d.matched_filters || []).join(', ');
                        App.showToast(`Scanner: ${d.symbol} matched [${filters}]`, 'alert');
                    }
                    if (typeof Notifications !== 'undefined' && Notifications.addNotification) {
                        const d = msg.data;
                        Notifications.addNotification({
                            type: 'scanner',
                            symbol: d.symbol,
                            message: `${d.symbol} matched: ${(d.matched_filters || []).join(', ')}`,
                        });
                    }
                    // Update live scanner feed in screener tab
                    const feed = document.getElementById('liveScannerFeed');
                    if (feed) {
                        const d = msg.data;
                        const filters = (d.matched_filters || []).join(', ');
                        const color = d.change_pct >= 0 ? 'text-green-400' : 'text-red-400';
                        const sign = d.change_pct >= 0 ? '+' : '';
                        const time = new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' });
                        const item = document.createElement('div');
                        item.className = 'flex items-center justify-between py-1 px-2 bg-dark-700 rounded text-xs cursor-pointer hover:bg-dark-600 transition';
                        item.onclick = () => Search.select(d.symbol, d.symbol);
                        item.innerHTML = `
                            <div class="flex items-center gap-2">
                                <span class="text-white font-medium">${d.symbol}</span>
                                <span class="${color}">${sign}${(d.change_pct || 0).toFixed(2)}%</span>
                            </div>
                            <div class="flex items-center gap-2">
                                <span class="text-gray-500 text-[10px]">${filters}</span>
                                <span class="text-gray-600 text-[10px]">${time}</span>
                            </div>
                        `;
                        // Remove the "Waiting..." placeholder
                        const placeholder = feed.querySelector('.text-center.text-gray-500');
                        if (placeholder && placeholder.textContent.includes('Waiting')) {
                            feed.innerHTML = '';
                        }
                        feed.prepend(item);
                        // Keep max 20 items
                        while (feed.children.length > 20) {
                            feed.removeChild(feed.lastChild);
                        }
                    }
                }
            } catch (e) {
                console.error('[WS] Failed to parse message:', e);
            }
        };
        this.ws.onerror = () => {
            // onerror is always followed by onclose, so just log
            console.warn('[WS] Error — will reconnect');
        };
        this.ws.onclose = (event) => {
            this._wsConnected = false;
            console.log(`[WS] Closed (code=${event.code}). Reconnecting in ${this._wsBackoff}ms...`);
            this._scheduleReconnect();
        };
    },

    _scheduleReconnect() {
        if (this._wsReconnectTimer) return; // Already scheduled
        this._wsReconnectTimer = setTimeout(() => {
            this._wsReconnectTimer = null;
            this._connectWS();
        }, this._wsBackoff);
        this._wsBackoff = Math.min(this._wsBackoff * 2, this._wsMaxBackoff);
    },

    subscribeTo(symbols) {
        // Track symbols for reconnection
        if (symbols && symbols.length > 0) {
            this._wsSymbols = [...new Set([...(this._wsSymbols || []), ...symbols])];
        }
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ subscribe: symbols }));
        }
    },

    // Broker
    placeBrokerOrder(data) {
        return this.request('/api/broker/order', { method: 'POST', body: JSON.stringify(data) });
    },
    cancelBrokerOrder(orderId) {
        return this.request(`/api/broker/order/${orderId}`, { method: 'DELETE' });
    },
    getBrokerPositions() { return this.request('/api/broker/positions'); },
    getBrokerOrders() { return this.request('/api/broker/orders'); },
    getBrokerRecent(limit) { return this.request(`/api/broker/recent?limit=${limit || 5}`); },
    syncBrokerPortfolio() { return this.request('/api/broker/sync', { method: 'POST' }); },
    getBrokerStatus() { return this.request('/api/broker/status'); },

    // Telegram
    getTelegramStatus() { return this.request('/api/telegram/status'); },
    linkTelegram(chatId) { return this.request('/api/telegram/link', { method: 'POST', body: JSON.stringify({ chat_id: chatId }) }); },
    unlinkTelegram() { return this.request('/api/telegram/unlink', { method: 'DELETE' }); },
    updateTelegramPreferences(alertTypes) { return this.request('/api/telegram/preferences', { method: 'PUT', body: JSON.stringify({ alert_types: alertTypes }) }); },
    sendTelegramTest() { return this.request('/api/telegram/test', { method: 'POST' }); }
};
