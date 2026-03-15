const Journal = {
    _trades: [],

    init() {
        const form = document.getElementById('addTradeForm');
        if (form) {
            form.addEventListener('submit', (e) => {
                e.preventDefault();
                this.addTrade();
            });
        }
    },

    async load() {
        try {
            const [trades, stats] = await Promise.all([
                API.getTradeJournal(50),
                API.getTradeStats(),
            ]);
            this._trades = trades;
            this.renderTrades(trades);
            this.renderStats(stats);
        } catch (e) {
            console.error('Journal load failed:', e);
            const el = document.getElementById('journalTrades');
            if (el) el.innerHTML = '<div class="text-center py-4 text-red-400 text-sm">Failed to load trades.</div>';
        }
    },

    renderStats(stats) {
        if (!stats) return;
        const totalEl = document.getElementById('jStatTotal');
        const winRateEl = document.getElementById('jStatWinRate');
        const pnlEl = document.getElementById('jStatPnL');
        const avgPnlEl = document.getElementById('jStatAvgPnL');

        if (totalEl) totalEl.textContent = stats.total_trades;
        if (winRateEl) {
            winRateEl.textContent = stats.sell_trades > 0 ? stats.win_rate + '%' : '-';
            winRateEl.className = 'font-bold ' + (stats.win_rate >= 50 ? 'text-green-400' : stats.win_rate > 0 ? 'text-red-400' : 'text-white');
        }
        if (pnlEl) {
            const sign = stats.total_pnl >= 0 ? '+' : '';
            pnlEl.textContent = stats.sell_trades > 0 ? sign + '\u20b9' + stats.total_pnl.toLocaleString('en-IN', { minimumFractionDigits: 2 }) : '-';
            pnlEl.className = 'font-bold ' + (stats.total_pnl >= 0 ? 'text-green-400' : 'text-red-400');
        }
        if (avgPnlEl) {
            const sign = stats.avg_pnl_pct >= 0 ? '+' : '';
            avgPnlEl.textContent = stats.sell_trades > 0 ? sign + stats.avg_pnl_pct + '%' : '-';
            avgPnlEl.className = 'font-bold ' + (stats.avg_pnl_pct >= 0 ? 'text-green-400' : 'text-red-400');
        }
    },

    renderTrades(trades) {
        const container = document.getElementById('journalTrades');
        if (!container) return;

        if (!trades || trades.length === 0) {
            container.innerHTML = '<div class="text-center py-4 text-gray-500 text-sm">No trades logged yet. Use the form above to log your first trade.</div>';
            return;
        }

        container.innerHTML = trades.map(t => {
            const isBuy = t.action === 'buy';
            const actionBadge = isBuy
                ? '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-green-900/50 text-green-400 border border-green-800/50">BUY</span>'
                : '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-red-900/50 text-red-400 border border-red-800/50">SELL</span>';

            let pnlHtml = '';
            if (!isBuy && t.pnl != null) {
                const pnlSign = t.pnl >= 0 ? '+' : '';
                const pnlColor = t.pnl >= 0 ? 'text-green-400' : 'text-red-400';
                pnlHtml = `<span class="${pnlColor} text-xs font-medium">${pnlSign}\u20b9${t.pnl.toFixed(2)}</span>`;
                if (t.pnl_pct != null) {
                    pnlHtml += `<span class="${pnlColor} text-[10px] ml-1">(${pnlSign}${t.pnl_pct.toFixed(1)}%)</span>`;
                }
            }

            const tagsHtml = t.tags ? t.tags.split(',').map(tag =>
                `<span class="px-1.5 py-0.5 rounded text-[10px] bg-dark-600 text-gray-400">${tag.trim()}</span>`
            ).join(' ') : '';

            const signalHtml = t.signal_direction
                ? `<span class="text-[10px] text-gray-500">${t.signal_direction} ${t.signal_confidence ? t.signal_confidence.toFixed(0) + '%' : ''}</span>`
                : '';

            const date = t.created_at ? new Date(t.created_at).toLocaleDateString('en-IN', {
                timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short', year: '2-digit',
                hour: '2-digit', minute: '2-digit'
            }) : '';

            return `
                <div class="flex items-start gap-3 bg-dark-700 rounded-lg px-3 py-2 hover:bg-dark-600 transition">
                    <div class="flex-1 min-w-0">
                        <div class="flex items-center gap-2 flex-wrap">
                            <span class="text-white font-bold text-sm">${t.symbol}</span>
                            ${actionBadge}
                            <span class="text-gray-400 text-xs">\u20b9${t.price.toFixed(2)} x ${t.quantity}</span>
                            ${pnlHtml}
                            ${signalHtml}
                        </div>
                        ${t.notes ? `<div class="text-gray-400 text-xs mt-1 truncate">${t.notes}</div>` : ''}
                        <div class="flex items-center gap-2 mt-1 flex-wrap">
                            ${tagsHtml}
                            <span class="text-[10px] text-gray-600">${date}</span>
                        </div>
                    </div>
                    <button onclick="Journal.deleteTrade(${t.id})" class="text-gray-600 hover:text-red-400 text-xs transition flex-shrink-0" title="Delete trade">&times;</button>
                </div>
            `;
        }).join('');
    },

    async addTrade() {
        const symbolEl = document.getElementById('tradeSymbol');
        const actionEl = document.getElementById('tradeAction');
        const priceEl = document.getElementById('tradePrice');
        const qtyEl = document.getElementById('tradeQty');
        const pnlEl = document.getElementById('tradePnL');
        const tagsEl = document.getElementById('tradeTags');
        const notesEl = document.getElementById('tradeNotes');

        const symbol = (symbolEl.value || '').trim().toUpperCase();
        const action = actionEl.value;
        const price = parseFloat(priceEl.value);
        const quantity = parseInt(qtyEl.value, 10);

        if (!symbol || isNaN(price) || isNaN(quantity) || quantity < 1) {
            if (typeof App !== 'undefined') App.showToast('Please fill required fields', 'error');
            return;
        }

        const data = { symbol, action, price, quantity };
        if (notesEl.value.trim()) data.notes = notesEl.value.trim();
        if (tagsEl.value.trim()) data.tags = tagsEl.value.trim();
        if (pnlEl.value) data.pnl = parseFloat(pnlEl.value);

        // Auto-fill signal info if current stock has a signal displayed
        try {
            const sigDir = document.getElementById('signalDirBadge');
            const sigConf = document.getElementById('signalConfBadge');
            if (sigDir && sigDir.textContent.trim()) {
                data.signal_direction = sigDir.textContent.trim().replace(/\s*AI\s*$/, '');
            }
            if (sigConf && sigConf.textContent) {
                const confMatch = sigConf.textContent.match(/([\d.]+)%/);
                if (confMatch) data.signal_confidence = parseFloat(confMatch[1]);
            }
        } catch (e) { /* ignore */ }

        try {
            await API.createTrade(data);
            // Reset form
            symbolEl.value = '';
            priceEl.value = '';
            qtyEl.value = '';
            pnlEl.value = '';
            tagsEl.value = '';
            notesEl.value = '';
            if (typeof App !== 'undefined') App.showToast('Trade logged', 'success');
            this.load();
        } catch (e) {
            if (typeof App !== 'undefined') App.showToast('Failed to log trade: ' + e.message, 'error');
        }
    },

    async deleteTrade(id) {
        if (!confirm('Delete this trade?')) return;
        try {
            await API.deleteTrade(id);
            if (typeof App !== 'undefined') App.showToast('Trade deleted', 'success');
            this.load();
        } catch (e) {
            if (typeof App !== 'undefined') App.showToast('Failed to delete trade: ' + e.message, 'error');
        }
    },
};
