/**
 * Screener module — scan popular stocks against technical filters.
 */
const Screener = {
    _scanning: false,

    init() {
        document.getElementById('btnScreenerScan').addEventListener('click', () => this.scan());
        document.getElementById('btnScreenerClear').addEventListener('click', () => this.clear());
    },

    _getFilters() {
        const filters = {};
        if (document.getElementById('scrRsiOversold').checked) filters.rsi_oversold = true;
        if (document.getElementById('scrRsiOverbought').checked) filters.rsi_overbought = true;
        if (document.getElementById('scrMacdBullish').checked) filters.macd_bullish = true;
        if (document.getElementById('scrMacdBearish').checked) filters.macd_bearish = true;
        if (document.getElementById('scrAboveSma20').checked) filters.above_sma_20 = true;
        if (document.getElementById('scrAboveSma50').checked) filters.above_sma_50 = true;
        if (document.getElementById('scrAboveSma200').checked) filters.above_sma_200 = true;
        if (document.getElementById('scrBelowSma200').checked) filters.below_sma_200 = true;
        if (document.getElementById('scrVolumeSpike').checked) filters.volume_spike = true;

        const priceMin = document.getElementById('scrPriceMin').value;
        const priceMax = document.getElementById('scrPriceMax').value;
        if (priceMin !== '') filters.price_change_min = parseFloat(priceMin);
        if (priceMax !== '') filters.price_change_max = parseFloat(priceMax);

        return filters;
    },

    async scan() {
        if (this._scanning) return;

        const filters = this._getFilters();
        const hasFilter = Object.keys(filters).length > 0;
        if (!hasFilter) {
            App.showToast('Please select at least one filter', 'error');
            return;
        }

        this._scanning = true;
        const btn = document.getElementById('btnScreenerScan');
        btn.disabled = true;
        btn.textContent = 'Scanning...';

        document.getElementById('screenerLoading').classList.remove('hidden');
        document.getElementById('screenerEmpty').classList.add('hidden');
        document.getElementById('screenerResults').classList.add('hidden');
        document.getElementById('screenerCount').textContent = '';
        document.getElementById('screenerStatus').textContent = '';

        try {
            const resp = await fetch('/api/screener/scan', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(filters),
            });

            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();

            document.getElementById('screenerLoading').classList.add('hidden');

            if (!data.results || data.results.length === 0) {
                document.getElementById('screenerEmpty').classList.remove('hidden');
                document.getElementById('screenerEmpty').textContent = 'No stocks matched the selected filters.';
                document.getElementById('screenerCount').textContent = `0 of ${data.scanned || 20} stocks matched`;
            } else {
                this.renderResults(data.results);
                document.getElementById('screenerResults').classList.remove('hidden');
                document.getElementById('screenerCount').textContent = `${data.count} of ${data.scanned || 20} stocks matched`;
            }

            document.getElementById('screenerStatus').textContent =
                'Last scan: ' + new Date().toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit' });

        } catch (e) {
            document.getElementById('screenerLoading').classList.add('hidden');
            document.getElementById('screenerEmpty').classList.remove('hidden');
            document.getElementById('screenerEmpty').textContent = 'Scan failed: ' + e.message;
            console.error('Screener scan error:', e);
        } finally {
            this._scanning = false;
            btn.disabled = false;
            btn.textContent = 'Scan Stocks';
        }
    },

    renderResults(results) {
        const tbody = document.getElementById('screenerTableBody');
        tbody.innerHTML = results.map(r => {
            const up = r.change_pct >= 0;
            const changeColor = up ? 'text-green-400' : 'text-red-400';
            const sign = up ? '+' : '';
            const arrow = up ? '&#9650;' : '&#9660;';

            // RSI color
            let rsiColor = 'text-gray-300';
            if (r.rsi != null) {
                if (r.rsi < 30) rsiColor = 'text-green-400';
                else if (r.rsi > 70) rsiColor = 'text-red-400';
            }

            // MACD badge
            let macdBadge = '<span class="text-gray-500">-</span>';
            if (r.macd_signal === 'bullish') {
                macdBadge = '<span class="px-1.5 py-0.5 rounded text-[10px] bg-green-900/50 text-green-400 border border-green-800/50">Bullish</span>';
            } else if (r.macd_signal === 'bearish') {
                macdBadge = '<span class="px-1.5 py-0.5 rounded text-[10px] bg-red-900/50 text-red-400 border border-red-800/50">Bearish</span>';
            }

            // SMA status badges
            const smaEntries = r.sma_status || {};
            const smaBadges = Object.entries(smaEntries).map(([key, val]) => {
                const label = key.replace('sma_', '');
                const color = val === 'above' ? 'text-green-400' : 'text-red-400';
                const symbol = val === 'above' ? '&#9650;' : '&#9660;';
                return `<span class="text-[10px] ${color}">${symbol}${label}</span>`;
            }).join(' ');

            // Volume ratio
            let volText = '-';
            let volColor = 'text-gray-400';
            if (r.volume_ratio != null) {
                volText = r.volume_ratio.toFixed(1) + 'x';
                if (r.volume_ratio > 2) volColor = 'text-green-400';
                else if (r.volume_ratio > 1.5) volColor = 'text-yellow-400';
            }

            // Matched filters
            const filterBadges = r.matched_filters.map(f =>
                `<span class="inline-block px-1.5 py-0.5 rounded text-[10px] bg-accent-blue/20 text-blue-300 border border-blue-800/30 mr-1 mb-1">${f}</span>`
            ).join('');

            return `
                <tr class="border-b border-gray-700/50 hover:bg-dark-700 cursor-pointer transition"
                    onclick="Search.select('${r.symbol}', '${r.symbol}')">
                    <td class="px-3 py-2 text-white font-medium">${r.symbol}</td>
                    <td class="px-3 py-2 text-right text-gray-300">${r.ltp ? '\u20b9' + r.ltp.toLocaleString('en-IN', { maximumFractionDigits: 2 }) : '-'}</td>
                    <td class="px-3 py-2 text-right ${changeColor}">${arrow} ${sign}${r.change_pct.toFixed(2)}%</td>
                    <td class="px-3 py-2 text-right ${rsiColor}">${r.rsi != null ? r.rsi.toFixed(1) : '-'}</td>
                    <td class="px-3 py-2 text-center">${macdBadge}</td>
                    <td class="px-3 py-2 text-center">${smaBadges || '-'}</td>
                    <td class="px-3 py-2 text-right ${volColor}">${volText}</td>
                    <td class="px-3 py-2">${filterBadges}</td>
                </tr>
            `;
        }).join('');
    },

    clear() {
        // Uncheck all checkboxes
        document.querySelectorAll('.screener-filter').forEach(cb => { cb.checked = false; });
        document.getElementById('scrPriceMin').value = '';
        document.getElementById('scrPriceMax').value = '';

        // Reset results
        document.getElementById('screenerResults').classList.add('hidden');
        document.getElementById('screenerEmpty').classList.remove('hidden');
        document.getElementById('screenerEmpty').textContent = 'Select filters above and click "Scan Stocks" to find matching stocks.';
        document.getElementById('screenerCount').textContent = '';
        document.getElementById('screenerStatus').textContent = '';
    },
};
