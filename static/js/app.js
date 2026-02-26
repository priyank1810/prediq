const App = {
    currentSymbol: null,
    currentPeriod: '1y',
    chart: null,
    rsiChart: null,
    macdChart: null,
    showIndicators: false,
    _overviewTimer: null,
    _overviewSymbols: [],
    _overviewQuotes: {},        // symbol -> latest quote for in-place updates
    _lastOverviewUpdate: null,

    async init() {
        // Initialize auth first
        Auth.init();

        this.chart = new StockChart('priceChart');
        this.rsiChart = new IndicatorChart('rsiChart');
        this.macdChart = new IndicatorChart('macdChart');

        Search.init();
        Predictions.init();
        Watchlist.init();
        Insights.init();
        Signals.init();
        Notifications.init();
        Options.init();

        this.setupNavigation();
        this.setupChartControls();

        // Public data loads regardless of auth
        this.loadMarketStatus();
        this.loadDataSource();
        this.loadMarketOverview();

        // Back to Overview button
        document.getElementById('btnBackToOverview').addEventListener('click', () => {
            this.showMarketOverview();
        });

        // WebSocket — subscribe to all overview symbols immediately
        try {
            API.connectWebSocket(this._overviewSymbols, this.onPriceUpdate.bind(this), () => {});
            API.setHighConfidenceHandler((data) => {
                Notifications.addNotification(data);
            });
        } catch (e) { /* WS not available yet */ }

        // Auto-refresh market overview every 10 seconds
        this._overviewTimer = setInterval(() => {
            if (!this.currentSymbol) {
                this.refreshOverviewQuotes();
            }
        }, 10000);
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
        "NIFTY 50", "NIFTY BANK", "SENSEX", "NIFTY IT", "NIFTY NEXT 50",
        "NIFTY MIDCAP 100", "NIFTY FINANCIAL", "NIFTY AUTO", "NIFTY PHARMA",
        "NIFTY METAL", "NIFTY ENERGY", "NIFTY FMCG", "NIFTY REALTY", "INDIA VIX"
    ],
    _popularStocks: [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC",
        "LT", "AXISBANK", "BAJFINANCE", "MARUTI", "TITAN",
        "ASIANPAINT", "SUNPHARMA", "HCLTECH", "WIPRO", "ULTRACEMCO"
    ],

    async loadMarketOverview() {
        this._overviewSymbols = [...this._allIndices, ...this._popularStocks];

        // Fetch quotes and smart alerts in parallel
        try {
            const [quotes, smartAlerts] = await Promise.all([
                API.getMultipleQuotes(this._overviewSymbols),
                API.scanHighConfidence(60).catch(() => []),
            ]);

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
        } catch (e) {
            console.error('Overview refresh failed:', e);
        }
    },

    updateOverviewTimestamp() {
        const el = document.getElementById('overviewLastUpdated');
        if (el && this._lastOverviewUpdate) {
            const time = new Date(this._lastOverviewUpdate).toLocaleTimeString('en-IN', {
                hour: '2-digit', minute: '2-digit', second: '2-digit'
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
                <div class="flex-shrink-0 bg-dark-800 border ${borderColor} rounded-lg px-4 py-2 cursor-pointer hover:bg-dark-700 transition min-w-[160px]"
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
            const time = s.created_at ? new Date(s.created_at).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' }) : '';
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
        this.loadMarketOverview();
    },

    // --- Stock View ---

    async loadStock(symbol, name = '') {
        this.currentSymbol = symbol;

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
                API.getHistory(symbol, this.currentPeriod)
            ]);

            this.displayQuote(quote);
            this.chart.init(this.currentPeriod === '1d' || this.currentPeriod === '5d');
            this.chart.setData(history);

            // Subscribe to live updates
            API.subscribeTo([symbol]);

            // Load indicators if enabled
            if (this.showIndicators) this.loadIndicators(symbol);

            // Auto-load 15-min signal
            Signals.loadSignal(symbol);
        } catch (e) {
            this.showToast('Failed to load stock data: ' + e.message, 'error');
        }
    },

    displayQuote(quote) {
        document.getElementById('stockPrice').textContent = `₹${quote.ltp.toLocaleString('en-IN', { minimumFractionDigits: 2 })}`;

        const changeEl = document.getElementById('stockChange');
        const sign = quote.change >= 0 ? '+' : '';
        changeEl.textContent = `${sign}${quote.change.toFixed(2)} (${sign}${quote.pct_change.toFixed(2)}%)`;
        changeEl.className = `text-sm ${quote.change >= 0 ? 'text-green-400' : 'text-red-400'}`;

        document.getElementById('stockOpen').textContent = `₹${quote.open.toFixed(2)}`;
        document.getElementById('stockHigh').textContent = `₹${quote.high.toFixed(2)}`;
        document.getElementById('stockLow').textContent = `₹${quote.low.toFixed(2)}`;
        document.getElementById('stockVolume').textContent = quote.volume.toLocaleString('en-IN');
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
            const data = await API.getIndicators(symbol);

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
                const ts = new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
                const tsEl = document.getElementById('liveTimestamp');
                if (tsEl) tsEl.textContent = ts;
            }
        }

        // Update market overview cards in-place (without full re-render)
        if (this._overviewQuotes[data.symbol]) {
            this._overviewQuotes[data.symbol] = { ...this._overviewQuotes[data.symbol], ...data };
            this.updateOverviewCard(data);
        }
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
