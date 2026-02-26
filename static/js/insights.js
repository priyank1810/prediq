const Insights = {
    init() {},

    async load() {
        await Promise.all([
            this.loadHighConfidence(),
            this.loadAccuracy(),
            this.loadRecentFeed(),
        ]);
    },

    async loadHighConfidence() {
        try {
            const signals = await API.scanHighConfidence(60);
            const container = document.getElementById('insightsHighConfidence');
            if (!signals || signals.length === 0) {
                container.innerHTML = '<div class="text-center py-6 text-gray-500 text-sm col-span-3">No high-confidence signals right now. The background scanner will populate this data over time.</div>';
                return;
            }
            container.innerHTML = signals.map(s => {
                const isBull = s.direction === 'BULLISH';
                const borderColor = isBull ? 'border-green-500/40' : 'border-red-500/40';
                const glow = isBull ? 'shadow-green-900/20 shadow-lg' : 'shadow-red-900/20 shadow-lg';
                const color = isBull ? 'text-green-400' : 'text-red-400';
                const arrow = isBull ? '&#9650;' : '&#9660;';
                const time = s.created_at ? new Date(s.created_at).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' }) : '-';
                return `
                    <div class="bg-dark-700 border ${borderColor} ${glow} rounded-lg p-4 cursor-pointer hover:bg-dark-600 transition"
                         onclick="Search.select('${s.symbol}', '${s.symbol}')">
                        <div class="flex items-center justify-between mb-2">
                            <span class="text-white font-bold">${s.symbol}</span>
                            <span class="${color} text-xl">${arrow}</span>
                        </div>
                        <div class="${color} text-lg font-bold">${s.direction}</div>
                        <div class="flex items-center justify-between mt-2">
                            <span class="text-white text-sm font-bold">${s.confidence.toFixed(1)}%</span>
                            <span class="text-xs text-gray-500">${s.price_at_signal ? '₹' + s.price_at_signal.toFixed(2) : ''}</span>
                        </div>
                        <div class="text-xs text-gray-500 mt-1">${time}</div>
                    </div>
                `;
            }).join('');
        } catch (e) {
            console.error('Failed to load high confidence signals:', e);
        }
    },

    async loadAccuracy() {
        try {
            const stats = await API.getSignalAccuracy();
            const tbody = document.getElementById('accuracyTable');
            if (!stats || stats.length === 0) {
                tbody.innerHTML = '<tr><td colspan="4" class="text-center py-4 text-gray-500">No accuracy data yet. Signals need time to be verified.</td></tr>';
                return;
            }
            tbody.innerHTML = stats.map(s => {
                const accColor = s.accuracy >= 60 ? 'text-green-400' : (s.accuracy >= 40 ? 'text-yellow-400' : 'text-red-400');
                return `
                    <tr class="border-b border-gray-800 hover:bg-dark-700 cursor-pointer transition"
                        onclick="Search.select('${s.symbol}', '${s.symbol}')">
                        <td class="px-4 py-2 text-accent-blue font-medium">${s.symbol}</td>
                        <td class="px-4 py-2 text-right text-gray-300">${s.total}</td>
                        <td class="px-4 py-2 text-right text-gray-300">${s.correct}</td>
                        <td class="px-4 py-2 text-right ${accColor} font-bold">${s.accuracy.toFixed(1)}%</td>
                    </tr>
                `;
            }).join('');
        } catch (e) {
            console.error('Failed to load accuracy stats:', e);
        }
    },

    async loadRecentFeed() {
        try {
            const signals = await API.scanHighConfidence(0);
            const container = document.getElementById('insightsRecentFeed');
            if (!signals || signals.length === 0) {
                container.innerHTML = '<div class="text-center py-4 text-gray-500 text-sm">No recent signals. The background scanner runs every 2 minutes.</div>';
                return;
            }
            container.innerHTML = signals.slice(0, 30).map(s => {
                const color = s.direction === 'BULLISH' ? 'text-green-400' : (s.direction === 'BEARISH' ? 'text-red-400' : 'text-yellow-400');
                const isHigh = s.confidence >= 60;
                const highlight = isHigh ? 'border-l-2 border-accent-blue pl-3' : 'pl-4';
                const time = s.created_at ? new Date(s.created_at).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' }) : '-';
                const arrow = s.direction === 'BULLISH' ? '&#9650;' : (s.direction === 'BEARISH' ? '&#9660;' : '&#9654;');
                return `
                    <div class="flex items-center gap-3 py-2 ${highlight} cursor-pointer hover:bg-dark-700 rounded transition"
                         onclick="Search.select('${s.symbol}', '${s.symbol}')">
                        <span class="${color} text-sm">${arrow}</span>
                        <span class="text-sm text-white font-medium w-28 truncate">${s.symbol}</span>
                        <span class="text-xs ${color} font-medium w-16">${s.direction}</span>
                        <span class="text-xs text-gray-300 w-12 text-right">${s.confidence.toFixed(1)}%</span>
                        <span class="text-xs text-gray-500 w-16 text-right">${s.price_at_signal ? '₹' + s.price_at_signal.toFixed(0) : ''}</span>
                        <span class="text-xs text-gray-500 ml-auto">${time}</span>
                        ${isHigh ? '<span class="text-[10px] px-1.5 py-0.5 rounded bg-accent-blue text-white font-bold">HIGH</span>' : ''}
                    </div>
                `;
            }).join('');
        } catch (e) {
            console.error('Failed to load recent feed:', e);
        }
    }
};
