const Watchlist = {
    _items: [],
    _alerts: [],
    _sortBy: 'name',
    _refreshTimer: null,
    _searchTimer: null,
    _searchAbort: null,
    _lastLoadTime: 0,

    init() {
        const input = document.getElementById('watchlistSymbolInput');
        const resultsDiv = document.getElementById('watchlistSearchResults');

        document.getElementById('addWatchlistForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.addSymbol();
        });
        document.getElementById('watchlistSort').addEventListener('change', (e) => {
            this._sortBy = e.target.value;
            this.renderCards();
        });

        // Autocomplete search on the watchlist input
        input.addEventListener('input', () => {
            clearTimeout(this._searchTimer);
            this._searchTimer = setTimeout(() => this._searchSymbols(), 300);
        });
        input.addEventListener('focus', () => {
            if (input.value.trim().length >= 1) this._searchSymbols();
        });
        // Close dropdown when clicking outside
        document.addEventListener('click', (e) => {
            if (!input.contains(e.target) && !resultsDiv.contains(e.target)) {
                resultsDiv.classList.add('hidden');
            }
        });
    },

    async _searchSymbols() {
        const input = document.getElementById('watchlistSymbolInput');
        const resultsDiv = document.getElementById('watchlistSearchResults');
        const query = input.value.trim();
        if (!query) { resultsDiv.classList.add('hidden'); return; }

        if (this._searchAbort) this._searchAbort.abort();
        this._searchAbort = new AbortController();

        try {
            const results = await API.searchStocks(query, this._searchAbort.signal);
            if (!results || results.length === 0) {
                resultsDiv.classList.add('hidden');
                return;
            }
            resultsDiv.innerHTML = results.map(r => {
                const badge = r.type === 'index'
                    ? '<span class="text-[10px] px-1.5 py-0.5 rounded bg-purple-900 text-purple-300 ml-2">INDEX</span>'
                    : (r.type === 'etf' ? '<span class="text-[10px] px-1.5 py-0.5 rounded bg-teal-900 text-teal-300 ml-2">ETF</span>' : '');
                return `
                    <div class="px-3 py-2 hover:bg-dark-600 cursor-pointer flex justify-between items-center"
                         onclick="Watchlist._selectSearchResult('${r.symbol}')">
                        <span class="text-white font-medium text-sm">${r.symbol}${badge}</span>
                        <span class="text-gray-500 text-xs truncate ml-3">${r.name}</span>
                    </div>`;
            }).join('');
            resultsDiv.classList.remove('hidden');
        } catch (e) {
            if (e.name === 'AbortError') return;
            resultsDiv.classList.add('hidden');
        }
    },

    _selectSearchResult(symbol) {
        const input = document.getElementById('watchlistSymbolInput');
        const resultsDiv = document.getElementById('watchlistSearchResults');
        input.value = symbol;
        resultsDiv.classList.add('hidden');
        this.addSymbol();
    },

    async load(force = false) {
        // Skip if data is fresh (loaded within last 15s) unless forced
        if (!force && this._items.length > 0 && Date.now() - this._lastLoadTime < 15000) {
            return;
        }
        // Show shimmer while loading
        if (this._items.length === 0) {
            Shimmer.show('watchlistCards', 'grid', 8);
        }
        try {
            const [items, alerts] = await Promise.all([
                API.getWatchlistOverview(),
                API.getAlerts().catch(() => []),
            ]);
            this._items = items || [];
            this._alerts = (alerts || []).filter(a => !a.is_triggered);
            this._lastLoadTime = Date.now();
            this.renderCards();
            this.renderAlerts();
            this._startAutoRefresh();

            // Subscribe watchlist symbols to WebSocket for live price ticks
            const symbols = this._items.map(i => i.symbol).filter(Boolean);
            if (symbols.length > 0) API.subscribeTo(symbols);
        } catch (e) {
            console.error('Failed to load watchlist:', e);
        }
    },

    _startAutoRefresh() {
        this._stopAutoRefresh();
        this._refreshTimer = setInterval(() => {
            // Only refresh if watchlist tab is visible
            const tab = document.getElementById('tab-watchlist');
            if (tab && !tab.classList.contains('hidden') && !document.hidden) {
                this.load();
            }
        }, 30000);
    },

    _stopAutoRefresh() {
        if (this._refreshTimer) {
            clearInterval(this._refreshTimer);
            this._refreshTimer = null;
        }
    },

    _getSorted() {
        const items = [...this._items];
        switch (this._sortBy) {
            case 'pct_change':
                return items.sort((a, b) => Math.abs(b.pct_change || 0) - Math.abs(a.pct_change || 0));
            case 'confidence':
                return items.sort((a, b) => (b.signal_confidence || 0) - (a.signal_confidence || 0));
            case 'sentiment':
                return items.sort((a, b) => (b.sentiment_score || 0) - (a.sentiment_score || 0));
            case 'name':
            default:
                return items.sort((a, b) => a.symbol.localeCompare(b.symbol));
        }
    },

    renderCards() {
        const container = document.getElementById('watchlistCards');
        const empty = document.getElementById('emptyWatchlist');

        if (!this._items || this._items.length === 0) {
            container.innerHTML = '';
            empty.classList.remove('hidden');
            return;
        }

        empty.classList.add('hidden');
        const sorted = this._getSorted();

        container.innerHTML = sorted.map(item => {
            const pct = item.pct_change || 0;
            const up = pct >= 0;
            const sign = up ? '+' : '';
            const borderColor = item.signal_direction === 'BULLISH' ? 'border-green-600'
                : (item.signal_direction === 'BEARISH' ? 'border-red-600' : 'border-gray-700');
            const dirLabel = item.signal_direction || 'N/A';
            const dirColor = item.signal_direction === 'BULLISH' ? 'text-green-400'
                : (item.signal_direction === 'BEARISH' ? 'text-red-400' : 'text-gray-500');
            const arrow = item.signal_direction === 'BULLISH' ? '&#9650;'
                : (item.signal_direction === 'BEARISH' ? '&#9660;' : '');
            const changeColor = up ? 'text-green-400' : 'text-red-400';

            // Sentiment display
            const sent = item.sentiment_score;
            const sentLabel = sent != null ? (sent >= 0 ? '+' : '') + sent.toFixed(0) : '-';
            const sentColor = sent != null ? (sent >= 10 ? 'text-green-400' : (sent <= -10 ? 'text-red-400' : 'text-gray-400')) : 'text-gray-600';

            // Volume ratio
            const vol = item.volume_ratio;
            const volLabel = vol != null ? vol.toFixed(1) + 'x avg' : '-';
            const volColor = vol != null ? (vol >= 1.2 ? 'text-green-400' : (vol <= 0.8 ? 'text-red-400' : 'text-gray-400')) : 'text-gray-600';

            // Open price
            const openLabel = item.open ? '₹' + item.open.toLocaleString('en-IN', { maximumFractionDigits: 0 }) : '-';

            // Day range bar
            const dayHigh = item.day_high || 0;
            const dayLow = item.day_low || 0;
            const ltp = item.ltp || 0;
            const rangeSpan = dayHigh - dayLow;
            const rangePct = rangeSpan > 0 ? Math.max(0, Math.min(100, ((ltp - dayLow) / rangeSpan) * 100)) : 50;
            const hasRange = dayHigh > 0 && dayLow > 0 && rangeSpan > 0;

            // Confidence
            const conf = item.signal_confidence;
            const confLabel = conf != null ? conf.toFixed(1) + '%' : '-';

            // Active alerts for this symbol
            const symbolAlerts = this._alerts.filter(a => a.symbol === item.symbol);
            const alertBadges = symbolAlerts.map(a => {
                const icon = a.condition === 'above' ? '&#9650;' : '&#9660;';
                return `<span class="inline-flex items-center gap-0.5 text-[10px] px-1.5 py-0.5 rounded bg-orange-900/50 text-orange-300 border border-orange-800/50">${icon} ₹${a.target_price.toLocaleString('en-IN')}</span>`;
            }).join(' ');

            const typeBadge = item.item_type === 'index'
                ? '<span class="text-[9px] px-1 py-0.5 rounded bg-purple-900/50 text-purple-300">IDX</span>'
                : (item.item_type === 'etf' ? '<span class="text-[9px] px-1 py-0.5 rounded bg-teal-900/50 text-teal-300">ETF</span>' : '');

            return `
                <div class="bg-dark-800 border-l-4 ${borderColor} rounded-lg p-3 flex flex-col gap-2 hover:bg-dark-700 transition relative group"
                     data-wl-symbol="${item.symbol}">
                    <!-- Header: Symbol + Direction badge -->
                    <div class="flex items-center justify-between">
                        <div class="flex items-center gap-1.5">
                            <span class="text-white font-bold text-sm cursor-pointer hover:text-accent-blue"
                                  onclick="Search.select('${item.symbol}', '${item.symbol}')">${item.symbol}</span>
                            ${typeBadge}
                        </div>
                        <span class="text-[10px] font-medium ${dirColor}">${dirLabel} ${arrow}</span>
                    </div>
                    <!-- Price + Change -->
                    <div class="flex items-baseline justify-between">
                        <span data-wl-price class="text-white font-bold text-lg">${item.ltp ? '₹' + item.ltp.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '-'}</span>
                        <span data-wl-change class="${changeColor} font-semibold text-sm">${sign}${pct.toFixed(2)}%</span>
                    </div>
                    <!-- Day Range -->
                    ${hasRange ? `
                    <div class="text-[10px]">
                        <div class="flex justify-between text-gray-500 mb-0.5">
                            <span>L ₹${dayLow.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</span>
                            <span>H ₹${dayHigh.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</span>
                        </div>
                        <div class="relative h-1.5 bg-dark-600 rounded-full" data-wl-range>
                            <div class="absolute top-0 left-0 h-full rounded-full" style="width:${rangePct}%;background:linear-gradient(to right,#ef4444,#facc15,#22c55e)"></div>
                            <div class="absolute top-1/2 -translate-y-1/2 w-2 h-2 bg-white rounded-full shadow border border-gray-600" style="left:calc(${rangePct}% - 4px)"></div>
                        </div>
                    </div>` : ''}
                    <!-- Stats row -->
                    <div class="grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]">
                        <div class="flex justify-between"><span class="text-gray-500">Signal</span><span class="text-gray-300">${confLabel}</span></div>
                        <div class="flex justify-between"><span class="text-gray-500">Sent</span><span class="${sentColor}">${sentLabel}</span></div>
                        <div class="flex justify-between"><span class="text-gray-500">Vol</span><span class="${volColor}">${volLabel}</span></div>
                        <div class="flex justify-between"><span class="text-gray-500">Open</span><span class="text-gray-300">${openLabel}</span></div>
                    </div>
                    <!-- Alert badges -->
                    ${alertBadges ? `<div class="flex flex-wrap gap-1">${alertBadges}</div>` : ''}
                    <!-- Alert form (hidden by default) -->
                    <div id="alertForm-${item.symbol}" class="hidden border-t border-gray-700 pt-2 mt-1">
                        <div class="flex gap-1.5 items-center">
                            <input type="number" step="0.05" placeholder="Price" id="alertPrice-${item.symbol}"
                                   class="bg-dark-700 border border-gray-600 rounded px-2 py-1 text-xs text-white w-20 placeholder-gray-500 focus:outline-none focus:border-accent-blue">
                            <select id="alertCond-${item.symbol}" class="bg-dark-700 border border-gray-600 rounded px-1 py-1 text-xs text-white focus:outline-none">
                                <option value="above">Above</option>
                                <option value="below">Below</option>
                            </select>
                            <button onclick="Watchlist.createAlert('${item.symbol}')"
                                    class="px-2 py-1 bg-orange-600 text-white text-[10px] rounded hover:bg-orange-500 transition font-medium">Set</button>
                            <button onclick="Watchlist.toggleAlertForm('${item.symbol}')"
                                    class="px-2 py-1 text-gray-500 text-[10px] hover:text-gray-300">Cancel</button>
                        </div>
                    </div>
                    <!-- Action buttons -->
                    <div class="flex gap-2 mt-auto pt-1">
                        <button onclick="Watchlist.toggleAlertForm('${item.symbol}')"
                                class="flex-1 text-[10px] px-2 py-1 rounded bg-dark-600 text-orange-400 hover:bg-dark-700 transition font-medium">Set Alert</button>
                        <button onclick="Search.select('${item.symbol}', '${item.symbol}')"
                                class="flex-1 text-[10px] px-2 py-1 rounded bg-dark-600 text-accent-blue hover:bg-dark-700 transition font-medium">Analyze</button>
                        <button onclick="Watchlist.removeSymbol('${item.symbol}')"
                                class="text-[10px] px-2 py-1 rounded text-red-500 hover:bg-dark-700 transition" title="Remove">&#10005;</button>
                    </div>
                </div>
            `;
        }).join('');
    },

    renderAlerts() {
        const section = document.getElementById('alertsSection');
        const list = document.getElementById('alertsList');
        const countEl = document.getElementById('alertCount');

        if (!this._alerts || this._alerts.length === 0) {
            section.classList.add('hidden');
            return;
        }

        section.classList.remove('hidden');
        countEl.textContent = `${this._alerts.length} alert${this._alerts.length !== 1 ? 's' : ''}`;

        list.innerHTML = this._alerts.map(a => {
            const icon = a.condition === 'above' ? '&#9650;' : '&#9660;';
            const condLabel = a.condition === 'above' ? 'goes above' : 'drops below';
            const created = a.created_at ? new Date(a.created_at).toLocaleDateString('en-IN', { day: 'numeric', month: 'short' }) : '';
            return `
                <div class="flex items-center justify-between bg-dark-700 rounded px-3 py-2">
                    <div class="flex items-center gap-2">
                        <span class="text-orange-400 text-sm">${icon}</span>
                        <span class="text-white text-sm font-medium">${a.symbol}</span>
                        <span class="text-gray-400 text-xs">${condLabel}</span>
                        <span class="text-orange-300 text-sm font-bold">₹${a.target_price.toLocaleString('en-IN')}</span>
                    </div>
                    <div class="flex items-center gap-3">
                        <span class="text-gray-600 text-[10px]">${created}</span>
                        <button onclick="Watchlist.deleteAlert(${a.id})" class="text-red-500 hover:text-red-400 text-xs">Delete</button>
                    </div>
                </div>
            `;
        }).join('');
    },

    toggleAlertForm(symbol) {
        const form = document.getElementById(`alertForm-${symbol}`);
        if (form) form.classList.toggle('hidden');
    },

    async createAlert(symbol) {
        const priceInput = document.getElementById(`alertPrice-${symbol}`);
        const condSelect = document.getElementById(`alertCond-${symbol}`);
        const price = parseFloat(priceInput.value);
        if (!price || isNaN(price)) {
            App.showToast('Enter a valid price', 'error');
            return;
        }
        try {
            await API.createAlert({
                symbol: symbol,
                target_price: price,
                condition: condSelect.value,
            });
            App.showToast(`Alert set: ${symbol} ${condSelect.value} ₹${price}`, 'success');
            priceInput.value = '';
            this.toggleAlertForm(symbol);
            this.load(true); // Force refresh to show new alert badge
        } catch (e) {
            App.showToast('Failed to create alert: ' + e.message, 'error');
        }
    },

    async deleteAlert(id) {
        try {
            await API.deleteAlert(id);
            App.showToast('Alert deleted', 'success');
            // Remove from local state and re-render without API call
            this._alerts = this._alerts.filter(a => a.id !== id);
            this.renderCards();
            this.renderAlerts();
        } catch (e) {
            App.showToast('Failed to delete alert: ' + e.message, 'error');
        }
    },

    updateCard(data) {
        const card = document.querySelector(`[data-wl-symbol="${data.symbol}"]`);
        if (!card) return;

        const priceEl = card.querySelector('[data-wl-price]');
        const changeEl = card.querySelector('[data-wl-change]');

        if (priceEl && data.ltp) {
            priceEl.textContent = '₹' + data.ltp.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
            // Flash ring
            const up = (data.pct_change || 0) >= 0;
            card.classList.add('ring-1', up ? 'ring-green-700' : 'ring-red-700');
            setTimeout(() => card.classList.remove('ring-1', 'ring-green-700', 'ring-red-700'), 600);
        }

        if (changeEl && data.pct_change != null) {
            const pct = data.pct_change;
            const up = pct >= 0;
            const sign = up ? '+' : '';
            changeEl.textContent = `${sign}${pct.toFixed(2)}%`;
            changeEl.className = `${up ? 'text-green-400' : 'text-red-400'} font-semibold text-sm`;
        }

        // Update day range bar position (use tick high/low or fall back to cached values)
        const rangeBar = card.querySelector('[data-wl-range]');
        if (rangeBar && data.ltp) {
            const item = this._items.find(i => i.symbol === data.symbol);
            const high = data.high || (item && item.day_high) || 0;
            const low = data.low || (item && item.day_low) || 0;
            const span = high - low;
            if (span > 0) {
                const pct = Math.max(0, Math.min(100, ((data.ltp - low) / span) * 100));
                const fill = rangeBar.querySelector('div:first-child');
                const dot = rangeBar.querySelector('div:last-child');
                if (fill) fill.style.width = pct + '%';
                if (dot) dot.style.left = `calc(${pct}% - 4px)`;
            }
        }

        // Update cached item for sort consistency
        const item = this._items.find(i => i.symbol === data.symbol);
        if (item) {
            if (data.ltp) item.ltp = data.ltp;
            if (data.pct_change != null) item.pct_change = data.pct_change;
            if (data.high) item.day_high = data.high;
            if (data.low) item.day_low = data.low;
        }
    },

    async addSymbol() {
        const input = document.getElementById('watchlistSymbolInput');
        const symbol = input.value.trim().toUpperCase();
        if (!symbol) return;

        try {
            await API.addToWatchlist({ symbol, item_type: 'stock' });
            input.value = '';
            App.showToast(`${symbol} added to watchlist`, 'success');
            this.load(true);
        } catch (e) {
            App.showToast('Failed to add: ' + e.message, 'error');
        }
    },

    async removeSymbol(symbol) {
        try {
            await API.removeFromWatchlist(symbol);
            App.showToast(`${symbol} removed`, 'success');
            // Remove from local state and re-render without API call
            this._items = this._items.filter(i => i.symbol !== symbol);
            this._alerts = this._alerts.filter(a => a.symbol !== symbol);
            this.renderCards();
            this.renderAlerts();
        } catch (e) {
            App.showToast('Failed to remove: ' + e.message, 'error');
        }
    }
};
