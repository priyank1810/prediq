/**
 * Multi-Timeframe Signal Dashboard
 *
 * Displays a grid of 3 timeframes (1h, 4h, 1D) with signal direction,
 * strength bar, RSI, MACD status, and SMA trend. Includes a consensus row.
 */
const Mtf = {
    _loading: false,

    init() { /* no-op — loaded on demand */ },

    async load(symbol) {
        if (!symbol) return;
        if (this._loading) return;
        this._loading = true;

        const container = document.getElementById('mtfDashboard');
        if (!container) { this._loading = false; return; }

        container.innerHTML = this._shimmer();

        try {
            const resp = await fetch(`/api/mtf/${encodeURIComponent(symbol)}`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            container.innerHTML = this._render(data);
        } catch (e) {
            console.error('MTF dashboard load failed:', e);
            container.innerHTML = `<div class="text-center py-8 text-gray-500 text-sm">Failed to load multi-timeframe data. ${e.message}</div>`;
        } finally {
            this._loading = false;
        }
    },

    _shimmer() {
        const rows = Array.from({ length: 5 }, () =>
            `<tr>
                <td class="px-3 py-3"><div class="h-4 w-12 bg-dark-600 rounded animate-pulse"></div></td>
                <td class="px-3 py-3"><div class="h-5 w-16 bg-dark-600 rounded animate-pulse"></div></td>
                <td class="px-3 py-3"><div class="h-3 w-full bg-dark-600 rounded animate-pulse"></div></td>
                <td class="px-3 py-3"><div class="h-4 w-10 bg-dark-600 rounded animate-pulse"></div></td>
                <td class="px-3 py-3"><div class="h-4 w-16 bg-dark-600 rounded animate-pulse"></div></td>
                <td class="px-3 py-3"><div class="h-4 w-14 bg-dark-600 rounded animate-pulse"></div></td>
            </tr>`
        ).join('');
        return `<table class="w-full text-sm"><tbody>${rows}</tbody></table>`;
    },

    _signalBadge(signal) {
        if (signal === 'BUY') {
            return `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-bold bg-green-900/40 text-green-400 border border-green-700/50">
                        <span class="text-base leading-none">&#9650;</span> BUY</span>`;
        }
        if (signal === 'SELL') {
            return `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-bold bg-red-900/40 text-red-400 border border-red-700/50">
                        <span class="text-base leading-none">&#9660;</span> SELL</span>`;
        }
        if (signal === 'N/A') {
            return `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-bold bg-dark-600 text-gray-500 border border-gray-700/50">
                        &#8212; N/A</span>`;
        }
        return `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-bold bg-gray-800/40 text-gray-400 border border-gray-700/50">
                    <span class="text-base leading-none">&#9654;</span> NEUTRAL</span>`;
    },

    _strengthBar(strength, signal) {
        let color = 'bg-gray-600';
        if (signal === 'BUY') color = 'bg-green-500';
        else if (signal === 'SELL') color = 'bg-red-500';

        return `<div class="flex items-center gap-2">
                    <div class="flex-1 h-2 bg-dark-600 rounded-full overflow-hidden">
                        <div class="${color} h-full rounded-full transition-all" style="width:${strength}%"></div>
                    </div>
                    <span class="text-xs text-gray-400 w-8 text-right">${strength}%</span>
                </div>`;
    },

    _rsiCell(rsi) {
        if (rsi == null) return `<span class="text-gray-600">--</span>`;
        let color = 'text-gray-300';
        let label = '';
        if (rsi <= 30) { color = 'text-green-400'; label = ' (OS)'; }
        else if (rsi >= 70) { color = 'text-red-400'; label = ' (OB)'; }
        else if (rsi <= 40) { color = 'text-green-300'; }
        else if (rsi >= 60) { color = 'text-red-300'; }
        return `<span class="${color}">${rsi.toFixed(1)}${label}</span>`;
    },

    _macdCell(direction) {
        if (direction === 'BULLISH') return `<span class="text-green-400 font-medium">Bullish</span>`;
        if (direction === 'BEARISH') return `<span class="text-red-400 font-medium">Bearish</span>`;
        if (direction === 'N/A') return `<span class="text-gray-600">N/A</span>`;
        return `<span class="text-gray-400">Flat</span>`;
    },

    _trendCell(trend) {
        if (trend === 'ABOVE') return `<span class="text-green-400">Above</span>`;
        if (trend === 'BELOW') return `<span class="text-red-400">Below</span>`;
        return `<span class="text-gray-600">N/A</span>`;
    },

    _render(data) {
        const tfs = data.timeframes || [];
        const consensus = data.consensus || {};

        const header = `
            <div class="flex items-center justify-between mb-3">
                <h3 class="text-sm font-bold text-white">Multi-Timeframe Signals</h3>
                ${data.current_price ? `<span class="text-xs text-gray-400">Price: <span class="text-white font-medium">\u20b9${data.current_price.toLocaleString('en-IN', { minimumFractionDigits: 2 })}</span></span>` : ''}
            </div>`;

        const tableHeader = `
            <thead>
                <tr class="text-xs text-gray-500 uppercase tracking-wider border-b border-gray-700/50">
                    <th class="px-3 py-2 text-left">Timeframe</th>
                    <th class="px-3 py-2 text-left">Signal</th>
                    <th class="px-3 py-2 text-left min-w-[120px]">Strength</th>
                    <th class="px-3 py-2 text-left">RSI</th>
                    <th class="px-3 py-2 text-left">MACD</th>
                    <th class="px-3 py-2 text-left">vs EMA 21</th>
                </tr>
            </thead>`;

        const rows = tfs.map(tf => {
            const ind = tf.indicators || {};
            return `
                <tr class="border-b border-gray-800/50 hover:bg-dark-700/30 transition">
                    <td class="px-3 py-2.5">
                        <span class="text-white font-medium text-xs">${tf.timeframe}</span>
                    </td>
                    <td class="px-3 py-2.5">${this._signalBadge(tf.signal)}</td>
                    <td class="px-3 py-2.5">${this._strengthBar(tf.strength, tf.signal)}</td>
                    <td class="px-3 py-2.5 text-xs">${this._rsiCell(ind.rsi)}</td>
                    <td class="px-3 py-2.5 text-xs">${this._macdCell(ind.macd_direction)}</td>
                    <td class="px-3 py-2.5 text-xs">${this._trendCell(ind.sma_trend)}</td>
                </tr>`;
        }).join('');

        // Consensus row
        const consBg = consensus.signal === 'BUY' ? 'bg-green-900/10 border-green-800/30'
                      : consensus.signal === 'SELL' ? 'bg-red-900/10 border-red-800/30'
                      : 'bg-dark-700/30 border-gray-700/30';
        const consensusRow = `
            <tr class="${consBg} border-t-2 border-gray-600">
                <td class="px-3 py-2.5">
                    <span class="text-white font-bold text-xs">CONSENSUS</span>
                </td>
                <td class="px-3 py-2.5">${this._signalBadge(consensus.signal)}</td>
                <td class="px-3 py-2.5">${this._strengthBar(consensus.strength, consensus.signal)}</td>
                <td class="px-3 py-2.5 text-xs text-gray-400" colspan="3">
                    ${consensus.agreement}/${consensus.total} timeframes agree
                </td>
            </tr>`;

        return `
            ${header}
            <div class="overflow-x-auto rounded-lg border border-gray-700/50">
                <table class="w-full text-sm">
                    ${tableHeader}
                    <tbody>
                        ${rows}
                        ${consensusRow}
                    </tbody>
                </table>
            </div>
            <p class="text-[10px] text-gray-600 mt-2">
                Indicators: RSI (14), MACD (12/26/9), EMA trend (price vs EMA-21).
                OS = Oversold, OB = Overbought.
            </p>`;
    },
};
