const App = {
    currentSymbol: null,
    currentPeriod: '1d',
    chart: null,
    rsiChart: null,
    macdChart: null,
    showIndicators: false,
    _overviewTimer: null,
    _overviewSymbols: [],
    _overviewQuotes: {},        // symbol -> latest quote for in-place updates
    _lastOverviewUpdate: null,

    async init() {
        this.chart = new StockChart('priceChart');
        this.rsiChart = new IndicatorChart('rsiChart');
        this.macdChart = new IndicatorChart('macdChart');

        Search.init();
        Predictions.init();
        Watchlist.init();
        Insights.init();
        Signals.init();
        Notifications.init();
        this.setupNavigation();
        this.setupChartControls();

        this.loadMarketStatus();
        this.loadDataSource();
        this.loadMarketOverview();

        // Back to Overview button
        document.getElementById('btnBackToOverview').addEventListener('click', () => {
            this.showMarketOverview();
        });

        // WebSocket — subscribe to all overview symbols immediately
        try {
            API.connectWebSocket(this._overviewSymbols, this.onPriceUpdate.bind(this), (data) => {
                this.onAlertTriggered(data);
            });
            API.setHighConfidenceHandler((data) => {
                Notifications.addNotification(data);
            });
            API.setMarketMoodHandler((data) => {
                this.renderMoodData(data);
            });
            API.setSignalUpdateHandler((data) => {
                if (data.symbol === this.currentSymbol) {
                    this.displaySignalBadge(data);
                }
            });
        } catch (e) { /* WS not available yet */ }

        // Auto-refresh market overview every 60s, only if WebSocket hasn't updated in 30s
        this._overviewTimer = setInterval(() => {
            if (!this.currentSymbol) {
                const stale = Date.now() - (this._lastOverviewUpdate || 0) > 30000;
                if (stale) this.refreshOverviewQuotes();
            }
        }, 60000);
    },

    setupNavigation() {
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
                tab.classList.add('active');
                document.getElementById(`tab-${tab.dataset.tab}`).classList.remove('hidden');

                if (tab.dataset.tab === 'watchlist') Watchlist.load();
                if (tab.dataset.tab === 'insights') Insights.load();
            });
        });
    },

    setupChartControls() {
        // Period buttons
        document.querySelectorAll('.period-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.period-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.currentPeriod = btn.dataset.period;
                if (this.currentSymbol) this.loadHistory(this.currentSymbol, this.currentPeriod);
            });
        });

        // Chart type buttons
        document.getElementById('btnCandlestick').addEventListener('click', () => {
            document.getElementById('btnCandlestick').classList.add('active');
            document.getElementById('btnLine').classList.remove('active');
            this.chart.setChartType('candlestick');
            if (this.currentSymbol) this.loadHistory(this.currentSymbol, this.currentPeriod);
        });

        document.getElementById('btnLine').addEventListener('click', () => {
            document.getElementById('btnLine').classList.add('active');
            document.getElementById('btnCandlestick').classList.remove('active');
            this.chart.setChartType('line');
            if (this.currentSymbol) this.loadHistory(this.currentSymbol, this.currentPeriod);
        });

        // Predict button
        document.getElementById('btnPredict').addEventListener('click', () => {
            if (this.currentSymbol) Predictions.loadPredictions(this.currentSymbol);
        });

        // Indicators toggle
        document.getElementById('btnIndicators').addEventListener('click', () => {
            this.showIndicators = !this.showIndicators;
            const btn = document.getElementById('btnIndicators');
            const panels = document.getElementById('indicatorPanels');

            if (this.showIndicators) {
                btn.classList.add('bg-accent-blue', 'text-white');
                btn.classList.remove('bg-dark-600', 'text-gray-300');
                panels.classList.remove('hidden');
                if (this.currentSymbol) this.loadIndicators(this.currentSymbol);
            } else {
                btn.classList.remove('bg-accent-blue', 'text-white');
                btn.classList.add('bg-dark-600', 'text-gray-300');
                panels.classList.add('hidden');
                this.chart.clearOverlays();
            }
        });
    },

    // --- Market Overview ---

    _allIndices: [
        "NIFTY 50", "NIFTY BANK", "SENSEX", "NIFTY IT",
        "NIFTY FINANCIAL", "INDIA VIX"
    ],
    _popularStocks: [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC"
    ],

    async loadMarketOverview() {
        this._overviewSymbols = [...this._allIndices, ...this._popularStocks];

        // Fetch quotes, smart alerts, and market movers in parallel
        try {
            const [quotes, smartAlerts] = await Promise.all([
                API.getMultipleQuotes(this._overviewSymbols),
                API.scanHighConfidence(60).catch(() => []),
            ]);
            // Fire market movers and new panels non-blocking
            this.loadMarketMovers();
            this.loadMarketMood();
            this.loadFIIDII();
            this.loadSectorHeatmap();

            // Store quotes for in-place updates
            quotes.forEach(q => { this._overviewQuotes[q.symbol] = q; });
            this._lastOverviewUpdate = Date.now();

            const indexQuotes = quotes.slice(0, this._allIndices.length);
            const stockQuotes = quotes.slice(this._allIndices.length);

            this.renderIndicesTicker(indexQuotes);
            this.renderGrid('indicesGrid', indexQuotes);
            this.renderGrid('stocksGrid', stockQuotes);
            this.renderSmartAlerts(smartAlerts);
            this.updateOverviewTimestamp();


            // Subscribe to all overview symbols on WebSocket
            API.subscribeTo(this._overviewSymbols);
        } catch (e) {
            console.error('Market overview failed:', e);
            document.getElementById('indicesTicker').innerHTML =
                '<div class="text-center py-4 text-gray-600 text-sm w-full">Failed to load market data. Please refresh.</div>';
        }
    },

    async refreshOverviewQuotes() {
        try {
            const quotes = await API.getMultipleQuotes(this._overviewSymbols);
            quotes.forEach(q => { this._overviewQuotes[q.symbol] = q; });
            this._lastOverviewUpdate = Date.now();

            const indexQuotes = quotes.slice(0, this._allIndices.length);
            const stockQuotes = quotes.slice(this._allIndices.length);

            this.renderIndicesTicker(indexQuotes);
            this.renderGrid('indicesGrid', indexQuotes);
            this.renderGrid('stocksGrid', stockQuotes);
            this.updateOverviewTimestamp();
            this.loadMarketMovers();
        } catch (e) {
            console.error('Overview refresh failed:', e);
        }
    },

    updateOverviewTimestamp() {
        const el = document.getElementById('overviewLastUpdated');
        if (el && this._lastOverviewUpdate) {
            const time = new Date(this._lastOverviewUpdate).toLocaleTimeString('en-IN', {
                timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', second: '2-digit'
            });
            el.textContent = `Updated ${time}`;
        }
    },

    renderIndicesTicker(quotes) {
        const container = document.getElementById('indicesTicker');
        if (!quotes || quotes.length === 0) {
            container.innerHTML = '<div class="text-center py-4 text-gray-600 text-sm w-full">No data available</div>';
            return;
        }
        container.innerHTML = quotes.map(q => {
            const up = (q.pct_change || 0) >= 0;
            const borderColor = up ? 'border-green-800' : 'border-red-800';
            const textColor = up ? 'text-green-400' : 'text-red-400';
            const sign = up ? '+' : '';
            const arrow = up ? '&#9650;' : '&#9660;';
            return `
                <div class="flex-shrink-0 bg-dark-800 border ${borderColor} rounded-lg px-3 sm:px-4 py-2 cursor-pointer hover:bg-dark-700 transition min-w-[130px] sm:min-w-[160px]"
                     data-live-symbol="${q.symbol}"
                     onclick="Search.select('${q.symbol}', '${q.symbol}')">
                    <div class="text-xs text-gray-400 truncate">${q.symbol}</div>
                    <div class="text-sm text-white font-bold" data-live-price>${q.ltp ? '₹' + q.ltp.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2}) : '-'}</div>
                    <div class="text-xs ${textColor}"><span data-live-arrow>${arrow}</span> <span data-live-change>${sign}${(q.pct_change || 0).toFixed(2)}%</span></div>
                </div>
            `;
        }).join('');
    },

    renderGrid(containerId, quotes) {
        const container = document.getElementById(containerId);
        if (!quotes || quotes.length === 0) {
            container.innerHTML = '<div class="col-span-2 text-center py-4 text-gray-600 text-sm">No data</div>';
            return;
        }
        container.innerHTML = quotes.map(q => {
            const up = (q.pct_change || 0) >= 0;
            const textColor = up ? 'text-green-400' : 'text-red-400';
            const sign = up ? '+' : '';
            const arrow = up ? '&#9650;' : '&#9660;';
            return `
                <div class="bg-dark-800 rounded-lg px-3 py-2 cursor-pointer hover:bg-dark-700 transition"
                     data-live-symbol="${q.symbol}"
                     onclick="Search.select('${q.symbol}', '${q.symbol}')">
                    <div class="flex items-center justify-between">
                        <span class="text-xs text-white font-medium truncate">${q.symbol}</span>
                        <span class="text-xs ${textColor}" data-live-arrow>${arrow}</span>
                    </div>
                    <div class="flex items-center justify-between mt-1">
                        <span class="text-xs text-gray-300" data-live-price>${q.ltp ? '₹' + q.ltp.toLocaleString('en-IN', {maximumFractionDigits: 0}) : '-'}</span>
                        <span class="text-xs ${textColor}" data-live-change>${sign}${(q.pct_change || 0).toFixed(2)}%</span>
                    </div>
                </div>
            `;
        }).join('');
    },

    renderSmartAlerts(signals) {
        const container = document.getElementById('smartAlertsList');
        const countEl = document.getElementById('smartAlertCount');

        if (!signals || signals.length === 0) {
            container.innerHTML = '<div class="text-center py-4 text-gray-600 text-sm">No high-confidence signals yet. The scanner runs every 2 minutes.</div>';
            countEl.textContent = '0 signals';
            return;
        }

        countEl.textContent = `${signals.length} signal${signals.length > 1 ? 's' : ''}`;
        container.innerHTML = signals.map(s => {
            const isBull = s.direction === 'BULLISH';
            const borderClass = isBull ? 'border-l-4 border-green-500 bg-green-900/10' : 'border-l-4 border-red-500 bg-red-900/10';
            const color = isBull ? 'text-green-400' : 'text-red-400';
            const arrow = isBull ? '&#9650;' : '&#9660;';
            const time = s.created_at ? new Date(s.created_at).toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit' }) : '';
            return `
                <div class="${borderClass} rounded-lg p-3 cursor-pointer hover:bg-dark-700 transition flex items-center gap-4"
                     onclick="Search.select('${s.symbol}', '${s.symbol}')">
                    <span class="${color} text-2xl">${arrow}</span>
                    <div class="flex-1">
                        <span class="text-white font-bold">${s.symbol}</span>
                        <span class="${color} text-sm ml-2">${s.direction}</span>
                    </div>
                    <div class="text-right">
                        <div class="text-white font-bold">${s.confidence.toFixed(1)}%</div>
                        <div class="text-xs text-gray-500">${s.price_at_signal ? '₹' + s.price_at_signal.toFixed(2) : ''} ${time}</div>
                    </div>
                </div>
            `;
        }).join('');
    },

    async loadMarketMovers() {
        try {
            const data = await API.getMarketMovers(10);
            this.renderMovers('gainersGrid', data.gainers, true);
            this.renderMovers('losersGrid', data.losers, false);
            const srcEl = document.getElementById('moversSource');
            if (srcEl && data.source) {
                srcEl.textContent = `(${data.source} \u2022 ${data.total_stocks} stocks)`;
            }
        } catch (e) {
            console.error('Market movers failed:', e);
            const g = document.getElementById('gainersGrid');
            const l = document.getElementById('losersGrid');
            if (g) g.innerHTML = '<div class="text-center py-4 text-gray-600 text-sm">Unavailable</div>';
            if (l) l.innerHTML = '<div class="text-center py-4 text-gray-600 text-sm">Unavailable</div>';
        }
    },

    renderMovers(containerId, stocks, isGainer) {
        const container = document.getElementById(containerId);
        if (!container) return;
        if (!stocks || stocks.length === 0) {
            container.innerHTML = '<div class="text-center py-4 text-gray-600 text-sm">No data</div>';
            return;
        }
        container.innerHTML = stocks.map((s, i) => {
            const pct = s.pct_change || 0;
            const color = isGainer ? 'text-green-400' : 'text-red-400';
            const bgHover = isGainer ? 'hover:border-green-800' : 'hover:border-red-800';
            const sign = pct >= 0 ? '+' : '';
            const arrow = pct >= 0 ? '&#9650;' : '&#9660;';
            const rank = i + 1;
            const ltp = s.ltp ? '\u20b9' + s.ltp.toLocaleString('en-IN', { maximumFractionDigits: 2 }) : '-';
            return `
                <div class="flex items-center gap-2 bg-dark-800 rounded-lg px-3 py-2 cursor-pointer hover:bg-dark-700 border border-transparent ${bgHover} transition"
                     onclick="Search.select('${s.symbol}', '${s.symbol}')">
                    <span class="text-[10px] text-gray-600 w-4 text-right">${rank}</span>
                    <span class="text-xs text-white font-medium flex-1 truncate">${s.symbol}</span>
                    <span class="text-xs text-gray-400">${ltp}</span>
                    <span class="text-xs ${color} min-w-[60px] text-right">${arrow} ${sign}${pct.toFixed(2)}%</span>
                </div>
            `;
        }).join('');
    },

    showMarketOverview() {
        this.currentSymbol = null;
        document.getElementById('stockInfoBar').classList.add('hidden');
        document.getElementById('chartControls').classList.add('hidden');
        document.getElementById('chartContainer').classList.add('hidden');
        document.getElementById('signalPanel').classList.add('hidden');
        document.getElementById('predictionPanel').classList.add('hidden');
        document.getElementById('indicatorPanels').classList.add('hidden');
        document.getElementById('marketOverview').classList.remove('hidden');
        const liveBadge = document.getElementById('liveBadge');
        if (liveBadge) liveBadge.classList.add('hidden');
        document.getElementById('stockSignalBadge')?.classList.add('hidden');
        this.loadMarketOverview();
    },

    // --- Stock View ---

    _switchToDashboardTab() {
        document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
        const dashTab = document.querySelector('.nav-tab[data-tab="dashboard"]');
        if (dashTab) dashTab.classList.add('active');
        document.getElementById('tab-dashboard').classList.remove('hidden');
    },

    async loadStock(symbol, name = '') {
        this.currentSymbol = symbol;

        // Always switch to dashboard tab first
        this._switchToDashboardTab();

        // Hide overview, show stock view
        document.getElementById('marketOverview').classList.add('hidden');
        document.getElementById('stockInfoBar').classList.remove('hidden');
        document.getElementById('chartControls').classList.remove('hidden');
        document.getElementById('chartContainer').classList.remove('hidden');

        document.getElementById('stockSymbol').textContent = symbol;
        document.getElementById('stockName').textContent = name;

        try {
            const [quote, history] = await Promise.all([
                API.getQuote(symbol),
                API.getHistory(symbol, this.currentPeriod),
            ]);

            this.displayQuote(quote);
            this.chart.init(this.currentPeriod === '1d' || this.currentPeriod === '5d');
            this.chart.setData(history);

            // Subscribe to live updates
            API.subscribeTo([symbol]);

            // Auto-enable and load indicators for full dashboard view
            if (!this.showIndicators) {
                this.showIndicators = true;
                const btn = document.getElementById('btnIndicators');
                const panels = document.getElementById('indicatorPanels');
                btn.classList.add('bg-accent-blue', 'text-white');
                btn.classList.remove('bg-dark-600', 'text-gray-300');
                panels.classList.remove('hidden');
            }
            this.loadIndicators(symbol);

            // Auto-load 15-min signal
            Signals.loadSignal(symbol);

        } catch (e) {
            this.showToast('Failed to load stock data: ' + e.message, 'error');
        }
    },

    displayQuote(quote) {
        if (quote.ltp != null) {
            document.getElementById('stockPrice').textContent = `₹${quote.ltp.toLocaleString('en-IN', { minimumFractionDigits: 2 })}`;
        }

        if (quote.change != null && quote.pct_change != null) {
            const changeEl = document.getElementById('stockChange');
            const sign = quote.change >= 0 ? '+' : '';
            changeEl.textContent = `${sign}${quote.change.toFixed(2)} (${sign}${quote.pct_change.toFixed(2)}%)`;
            changeEl.className = `text-sm ${quote.change >= 0 ? 'text-green-400' : 'text-red-400'}`;
        }

        if (quote.open != null) document.getElementById('stockOpen').textContent = `₹${quote.open.toFixed(2)}`;
        if (quote.high != null) document.getElementById('stockHigh').textContent = `₹${quote.high.toFixed(2)}`;
        if (quote.low != null) document.getElementById('stockLow').textContent = `₹${quote.low.toFixed(2)}`;
        if (quote.volume != null) document.getElementById('stockVolume').textContent = quote.volume.toLocaleString('en-IN');

        const volAvgEl = document.getElementById('stockVolAvg');
        if (volAvgEl && quote.volume && quote.avg_volume && quote.avg_volume > 0) {
            const ratio = (quote.volume / quote.avg_volume).toFixed(1);
            const color = ratio >= 1.2 ? 'text-green-400' : (ratio <= 0.8 ? 'text-red-400' : 'text-gray-500');
            volAvgEl.innerHTML = `<span class="${color} font-medium">${ratio}x avg</span>`;
        } else if (volAvgEl && !quote.avg_volume) {
            // Don't clear vol avg on WebSocket ticks that lack it
        }
    },

    async loadHistory(symbol, period) {
        try {
            const history = await API.getHistory(symbol, period);
            if (!history || !Array.isArray(history)) {
                this.showToast('No chart data available for this period', 'error');
                return;
            }
            // Reinit chart with time axis visible for intraday (1D / 1W) periods
            this.chart.init(period === '1d' || period === '5d');
            this.chart.setData(history);
            if (this.showIndicators) this.loadIndicators(symbol);
        } catch (e) {
            console.error('loadHistory error:', e);
            this.showToast('Failed to load history: ' + e.message, 'error');
        }
    },

    async loadIndicators(symbol) {
        try {
            const data = await API.getIndicators(symbol, this.currentPeriod);

            this.chart.clearOverlays();

            if (data.bollinger_upper && data.bollinger_lower) {
                const upper = data.bollinger_upper.dates.map((d, i) => ({ time: d, value: data.bollinger_upper.values[i] }));
                const lower = data.bollinger_lower.dates.map((d, i) => ({ time: d, value: data.bollinger_lower.values[i] }));
                const mid = data.bollinger_middle.dates.map((d, i) => ({ time: d, value: data.bollinger_middle.values[i] }));
                this.chart.addLineOverlay(upper, 'rgba(156,163,175,0.4)', 'BB Upper');
                this.chart.addLineOverlay(mid, 'rgba(156,163,175,0.3)', 'BB Mid');
                this.chart.addLineOverlay(lower, 'rgba(156,163,175,0.4)', 'BB Lower');
            }

            if (data.sma_20) {
                const sma = data.sma_20.dates.map((d, i) => ({ time: d, value: data.sma_20.values[i] }));
                this.chart.addLineOverlay(sma, '#ff9800', 'SMA 20');
            }
            if (data.sma_50) {
                const sma = data.sma_50.dates.map((d, i) => ({ time: d, value: data.sma_50.values[i] }));
                this.chart.addLineOverlay(sma, '#e91e63', 'SMA 50');
            }

            if (data.rsi) {
                this.rsiChart.init();
                const rsiData = data.rsi.dates.map((d, i) => ({ time: d, value: data.rsi.values[i] }));
                this.rsiChart.setRSIData(rsiData);
            }

            if (data.macd_line) {
                this.macdChart.init();
                const macdLine = data.macd_line.dates.map((d, i) => ({ time: d, value: data.macd_line.values[i] }));
                const signalLine = data.macd_signal.dates.map((d, i) => ({ time: d, value: data.macd_signal.values[i] }));
                const histogram = data.macd_histogram.dates.map((d, i) => ({ time: d, value: data.macd_histogram.values[i] }));
                this.macdChart.setMACDData(macdLine, signalLine, histogram);
            }

            // Re-fit chart to selected period after overlays are added
            if (this.chart && this.chart.chart) {
                this.chart.chart.timeScale().fitContent();
            }
        } catch (e) {
            console.error('Failed to load indicators:', e);
        }
    },

    async loadDataSource() {
        try {
            const info = await API.getDataSource();
            const badge = document.getElementById('dataSourceBadge');
            if (info.source === 'angel_one') {
                badge.textContent = 'Real-Time';
                badge.className = 'px-2 py-0.5 rounded text-[10px] font-medium bg-green-900/50 text-green-400 border border-green-800/50';
            } else {
                badge.textContent = 'Delayed ~15 min';
                badge.className = 'px-2 py-0.5 rounded text-[10px] font-medium bg-yellow-900/50 text-yellow-400 border border-yellow-800/50';
            }
        } catch (e) { /* ignore */ }
    },

    async loadMarketStatus() {
        try {
            const { status } = await API.getMarketStatus();
            const dot = document.getElementById('marketDot');
            const text = document.getElementById('marketText');
            const labels = {
                'open': ['Market Open', 'bg-green-500'],
                'pre_market': ['Pre-Market', 'bg-yellow-500'],
                'post_market': ['Market Closed', 'bg-red-500'],
                'closed_weekend': ['Weekend - Closed', 'bg-red-500'],
            };
            const [label, color] = labels[status] || ['Unknown', 'bg-gray-500'];
            dot.className = `w-2 h-2 rounded-full ${color}`;
            text.textContent = label;
        } catch (e) { /* ignore */ }
    },

    onPriceUpdate(data) {
        // Update stock detail view (price bar + chart)
        if (data.symbol === this.currentSymbol) {
            this.displayQuote(data);

            // Push live price into chart's last candle
            if (this.chart && data.ltp) {
                this.chart.updateLastCandle(data.ltp, data.volume || null);
            }

            // Update live indicator
            const liveEl = document.getElementById('liveBadge');
            if (liveEl) {
                liveEl.classList.remove('hidden');
                const ts = new Date().toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', second: '2-digit' });
                const tsEl = document.getElementById('liveTimestamp');
                if (tsEl) tsEl.textContent = ts;
            }
        }

        // Update market overview cards in-place (without full re-render)
        if (this._overviewQuotes[data.symbol]) {
            this._overviewQuotes[data.symbol] = { ...this._overviewQuotes[data.symbol], ...data };
            this.updateOverviewCard(data);
        }

        // Update watchlist cards in-place
        Watchlist.updateCard(data);
    },

    updateOverviewCard(data) {
        // Update any visible card for this symbol (ticker + grid)
        const cards = document.querySelectorAll(`[data-live-symbol="${data.symbol}"]`);
        cards.forEach(card => {
            const priceEl = card.querySelector('[data-live-price]');
            const changeEl = card.querySelector('[data-live-change]');
            const arrowEl = card.querySelector('[data-live-arrow]');
            const pct = data.pct_change || 0;
            const up = pct >= 0;

            if (priceEl && data.ltp) {
                priceEl.textContent = '₹' + data.ltp.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
                // Flash animation
                priceEl.classList.add('text-white');
                card.classList.add('ring-1', up ? 'ring-green-700' : 'ring-red-700');
                setTimeout(() => card.classList.remove('ring-1', 'ring-green-700', 'ring-red-700'), 600);
            }
            if (changeEl) {
                const sign = up ? '+' : '';
                changeEl.textContent = `${sign}${pct.toFixed(2)}%`;
                changeEl.className = `text-xs ${up ? 'text-green-400' : 'text-red-400'}`;
            }
            if (arrowEl) {
                arrowEl.innerHTML = up ? '&#9650;' : '&#9660;';
                arrowEl.className = `text-xs ${up ? 'text-green-400' : 'text-red-400'}`;
            }
        });
    },

    // --- Market Mood ---
    async loadMarketMood() {
        try {
            const data = await API.getMarketMood();
            this.renderMoodData(data);
        } catch (e) {
            console.error('Market mood failed:', e);
        }
    },

    renderMoodData(data) {
        const scoreEl = document.getElementById('moodScore');
        const labelEl = document.getElementById('moodLabel');
        const barEl = document.getElementById('moodBar');
        if (!scoreEl || !data) return;

        const score = data.score != null ? data.score : data.mood_score;
        if (score == null) return;

        scoreEl.textContent = Math.round(score);
        const label = data.label || data.mood_label || '';
        labelEl.textContent = label;
        barEl.style.width = score + '%';

        const colorMap = {
            'Extreme Fear': 'text-red-400',
            'Fear': 'text-red-300',
            'Neutral': 'text-yellow-400',
            'Greed': 'text-green-300',
            'Extreme Greed': 'text-green-400',
        };
        scoreEl.className = `text-4xl font-bold ${colorMap[label] || 'text-yellow-400'}`;
    },

    // --- FII/DII ---
    async loadFIIDII() {
        try {
            const data = await API.getFIIDIIDaily();
            const fiiEl = document.getElementById('fiiNet');
            const diiEl = document.getElementById('diiNet');
            const totalEl = document.getElementById('totalNet');
            if (!data) return;

            const fmt = (v) => {
                if (v == null) return '--';
                const abs = Math.abs(v);
                const sign = v >= 0 ? '+' : '-';
                if (abs >= 10000000) return sign + '₹' + (abs / 10000000).toFixed(1) + 'Cr';
                if (abs >= 100000) return sign + '₹' + (abs / 100000).toFixed(1) + 'L';
                return sign + '₹' + abs.toLocaleString('en-IN');
            };
            const color = (v) => v >= 0 ? 'text-green-400' : 'text-red-400';

            const fiiNet = data.fii_net ?? (data.fii && data.fii.net) ?? 0;
            const diiNet = data.dii_net ?? (data.dii && data.dii.net) ?? 0;
            const total = fiiNet + diiNet;

            fiiEl.textContent = fmt(fiiNet);
            fiiEl.className = `text-sm font-bold ${color(fiiNet)}`;
            diiEl.textContent = fmt(diiNet);
            diiEl.className = `text-sm font-bold ${color(diiNet)}`;
            totalEl.textContent = fmt(total);
            totalEl.className = `text-sm font-bold ${color(total)}`;
        } catch (e) {
            console.error('FII/DII failed:', e);
        }
    },

    // --- Sector Heatmap ---
    async loadSectorHeatmap() {
        try {
            const data = await API.getSectorHeatmap();
            const container = document.getElementById('sectorHeatmap');
            const sectors = Array.isArray(data) ? data : (data.sectors || []);
            if (!container || !sectors.length) return;

            container.innerHTML = sectors.map(s => {
                const pct = s.avg_change || 0;
                const intensity = Math.min(1, Math.abs(pct) / 3);
                let bg, text;
                if (pct >= 0) {
                    bg = `rgba(0, 200, 83, ${0.15 + intensity * 0.6})`;
                    text = 'text-green-300';
                } else {
                    bg = `rgba(255, 23, 68, ${0.15 + intensity * 0.6})`;
                    text = 'text-red-300';
                }
                const sign = pct >= 0 ? '+' : '';
                return `
                    <div class="rounded p-1.5 text-center cursor-default" style="background:${bg}">
                        <div class="text-[10px] text-white font-medium truncate">${s.sector}</div>
                        <div class="text-[10px] ${text} font-bold">${sign}${pct.toFixed(2)}%</div>
                    </div>
                `;
            }).join('');
        } catch (e) {
            console.error('Sector heatmap failed:', e);
            const c = document.getElementById('sectorHeatmap');
            if (c) c.innerHTML = '<div class="col-span-3 text-center py-2 text-gray-600 text-[10px]">Unavailable</div>';
        }
    },

    displaySignalBadge(signal) {
        const badge = document.getElementById('stockSignalBadge');
        if (!badge || !signal) { badge?.classList.add('hidden'); return; }
        badge.classList.remove('hidden');

        const dir = signal.direction; // "BULLISH" | "BEARISH" | "NEUTRAL"
        const conf = signal.confidence; // 0-100

        const arrowEl = document.getElementById('signalArrowBadge');
        const dirEl = document.getElementById('signalDirBadge');
        const confEl = document.getElementById('signalConfBadge');

        if (dir === 'BULLISH') {
            arrowEl.innerHTML = '&#9650;'; arrowEl.className = 'text-lg text-green-400';
            dirEl.textContent = 'BULLISH'; dirEl.className = 'text-xs font-bold text-green-400';
        } else if (dir === 'BEARISH') {
            arrowEl.innerHTML = '&#9660;'; arrowEl.className = 'text-lg text-red-400';
            dirEl.textContent = 'BEARISH'; dirEl.className = 'text-xs font-bold text-red-400';
        } else {
            arrowEl.innerHTML = '&#9654;'; arrowEl.className = 'text-lg text-gray-400';
            dirEl.textContent = 'NEUTRAL'; dirEl.className = 'text-xs font-bold text-gray-400';
        }

        if (signal.source === 'ai') {
            dirEl.textContent = dir + ' ';
            const aiTag = document.createElement('span');
            aiTag.className = 'text-[9px] px-1 py-0.5 rounded bg-purple-900 text-purple-300 ml-1 font-medium';
            aiTag.textContent = 'AI';
            dirEl.appendChild(aiTag);
        }

        confEl.textContent = conf != null ? conf.toFixed(0) + '% conf' : '';
    },

    onAlertTriggered(data) {
        const cond = data.condition === 'above' ? 'crossed above' : 'dropped below';
        const msg = `Price Alert: ${data.symbol} ${cond} ₹${data.target_price}`;
        this.showToast(msg, 'alert');

        // Browser notification if permitted
        if ('Notification' in window && Notification.permission === 'granted') {
            new Notification('StockAI Price Alert', { body: msg, icon: '/static/favicon.ico' });
        } else if ('Notification' in window && Notification.permission !== 'denied') {
            Notification.requestPermission();
        }

        // Refresh watchlist alerts if tab is visible
        const tab = document.getElementById('tab-watchlist');
        if (tab && !tab.classList.contains('hidden')) {
            Watchlist.load(true);
        }
    },

    showToast(message, type = 'success') {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        container.appendChild(toast);
        setTimeout(() => toast.remove(), 5000);
    }
};

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => App.init());
