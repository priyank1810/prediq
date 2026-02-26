const Watchlist = {
    init() {
        document.getElementById('addWatchlistForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.addSymbol();
        });
    },

    async load() {
        try {
            const items = await API.getWatchlistOverview();
            this.display(items);
        } catch (e) {
            console.error('Failed to load watchlist:', e);
        }
    },

    display(items) {
        const tbody = document.getElementById('watchlistTable');
        const empty = document.getElementById('emptyWatchlist');

        if (!items || items.length === 0) {
            tbody.innerHTML = '';
            empty.classList.remove('hidden');
            return;
        }

        empty.classList.add('hidden');
        tbody.innerHTML = items.map(item => {
            const changeColor = item.pct_change >= 0 ? 'text-green-400' : 'text-red-400';
            const sign = item.pct_change >= 0 ? '+' : '';
            const dirColor = item.signal_direction === 'BULLISH' ? 'text-green-400' :
                           (item.signal_direction === 'BEARISH' ? 'text-red-400' : 'text-yellow-400');
            const confColor = item.signal_confidence >= 60 ? 'text-white font-bold' : 'text-gray-400';
            const typeBadge = item.item_type === 'index'
                ? '<span class="text-[10px] px-1.5 py-0.5 rounded bg-purple-900 text-purple-300">INDEX</span>'
                : '<span class="text-[10px] px-1.5 py-0.5 rounded bg-blue-900 text-blue-300">STOCK</span>';

            return `
                <tr class="border-b border-gray-800 hover:bg-dark-700 transition">
                    <td class="px-4 py-3">
                        <span class="text-white font-medium cursor-pointer hover:text-accent-blue"
                              onclick="Search.select('${item.symbol}', '${item.symbol}')">${item.symbol}</span>
                    </td>
                    <td class="px-4 py-3">${typeBadge}</td>
                    <td class="px-4 py-3 text-right text-white">${item.ltp ? '₹' + item.ltp.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2}) : '-'}</td>
                    <td class="px-4 py-3 text-right ${changeColor}">${sign}${(item.pct_change || 0).toFixed(2)}%</td>
                    <td class="px-4 py-3 text-center ${dirColor} font-medium text-xs">${item.signal_direction || '-'}</td>
                    <td class="px-4 py-3 text-right ${confColor}">${item.signal_confidence != null ? item.signal_confidence.toFixed(1) + '%' : '-'}</td>
                    <td class="px-4 py-3 text-center">
                        <button onclick="Search.select('${item.symbol}', '${item.symbol}')"
                                class="text-accent-blue hover:text-blue-300 text-xs">Analyze</button>
                    </td>
                    <td class="px-4 py-3 text-right">
                        <button onclick="Watchlist.removeSymbol('${item.symbol}')"
                                class="text-red-400 hover:text-red-300 text-xs">Remove</button>
                    </td>
                </tr>
            `;
        }).join('');
    },

    async addSymbol() {
        const input = document.getElementById('watchlistSymbolInput');
        const symbol = input.value.trim().toUpperCase();
        if (!symbol) return;

        try {
            await API.addToWatchlist({ symbol, item_type: 'stock' });
            input.value = '';
            App.showToast(`${symbol} added to watchlist`, 'success');
            this.load();
        } catch (e) {
            App.showToast('Failed to add: ' + e.message, 'error');
        }
    },

    async removeSymbol(symbol) {
        try {
            await API.removeFromWatchlist(symbol);
            App.showToast(`${symbol} removed`, 'success');
            this.load();
        } catch (e) {
            App.showToast('Failed to remove: ' + e.message, 'error');
        }
    }
};
