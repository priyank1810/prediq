const Portfolio = {
    init() {
        document.getElementById('addHoldingForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.addHolding(e.target);
        });
    },

    async load() {
        try {
            const [holdings, summary] = await Promise.all([
                API.getPortfolio(),
                API.getPortfolioSummary()
            ]);
            this._clearError();
            this.displayHoldings(holdings);
            this.displaySummary(summary);
        } catch (e) {
            console.error('Failed to load portfolio:', e);
            this._showError('Failed to load portfolio');
        }
    },

    _showError(message) {
        const tbody = document.getElementById('holdingsTable');
        const empty = document.getElementById('emptyPortfolio');
        if (empty) empty.classList.add('hidden');
        if (tbody) {
            tbody.innerHTML = `<tr><td colspan="9" class="text-center py-6">
                <div class="text-red-400 text-sm mb-2">${message}</div>
                <button onclick="Portfolio.load()" class="text-xs px-3 py-1 bg-dark-600 text-gray-300 rounded hover:bg-dark-700">Retry</button>
            </td></tr>`;
        }
    },

    _clearError() {},


    displaySummary(summary) {
        document.getElementById('totalInvested').textContent = `₹${summary.total_invested.toLocaleString('en-IN', { minimumFractionDigits: 2 })}`;
        document.getElementById('currentValue').textContent = `₹${summary.current_value.toLocaleString('en-IN', { minimumFractionDigits: 2 })}`;

        const pnlEl = document.getElementById('totalPnl');
        pnlEl.textContent = `${summary.total_pnl >= 0 ? '+' : ''}₹${summary.total_pnl.toLocaleString('en-IN', { minimumFractionDigits: 2 })} (${summary.total_pnl_pct.toFixed(2)}%)`;
        pnlEl.className = `text-xl font-bold mt-1 ${summary.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`;

        document.getElementById('holdingsCount').textContent = summary.holdings_count;
    },

    displayHoldings(holdings) {
        const tbody = document.getElementById('holdingsTable');
        const empty = document.getElementById('emptyPortfolio');

        if (holdings.length === 0) {
            tbody.innerHTML = '';
            empty.classList.remove('hidden');
            return;
        }

        empty.classList.add('hidden');
        tbody.innerHTML = holdings.map(h => {
            const invested = h.quantity * h.buy_price;
            const current = h.current_price ? h.quantity * h.current_price : 0;
            const pnl = h.pnl || 0;
            const pnlPct = h.pnl_pct || 0;
            const pnlColor = pnl >= 0 ? 'text-green-400' : 'text-red-400';

            return `
                <tr class="border-b border-gray-800 hover:bg-dark-700">
                    <td class="px-4 py-3">
                        <span class="text-white font-medium cursor-pointer hover:text-accent-blue"
                              onclick="Search.select('${h.symbol}', '${h.symbol}')">${h.symbol}</span>
                    </td>
                    <td class="px-4 py-3 text-right text-gray-300">${h.quantity}</td>
                    <td class="px-4 py-3 text-right text-gray-300">₹${h.buy_price.toFixed(2)}</td>
                    <td class="px-4 py-3 text-right text-white">${h.current_price ? '₹' + h.current_price.toFixed(2) : '-'}</td>
                    <td class="px-4 py-3 text-right text-gray-300">₹${invested.toFixed(2)}</td>
                    <td class="px-4 py-3 text-right text-white">₹${current.toFixed(2)}</td>
                    <td class="px-4 py-3 text-right ${pnlColor}">${pnl >= 0 ? '+' : ''}₹${pnl.toFixed(2)}</td>
                    <td class="px-4 py-3 text-right ${pnlColor}">${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%</td>
                    <td class="px-4 py-3 text-right">
                        <button onclick="Portfolio.deleteHolding(${h.id})"
                                class="text-red-400 hover:text-red-300 text-xs">Delete</button>
                    </td>
                </tr>
            `;
        }).join('');
    },

    async addHolding(form) {
        const data = {
            symbol: form.symbol.value.toUpperCase(),
            quantity: parseInt(form.quantity.value),
            buy_price: parseFloat(form.buy_price.value),
            buy_date: form.buy_date.value,
        };

        try {
            await API.addHolding(data);
            form.reset();
            App.showToast(`${data.symbol} added to portfolio`, 'success');
            this.load();
        } catch (e) {
            App.showToast('Failed to add holding: ' + e.message, 'error');
        }
    },

    async exportCSV() {
        try {
            await API.exportPortfolioCSV();
        } catch (e) {
            App.showToast('Failed to export CSV: ' + e.message, 'error');
        }
    },

    async exportHTML() {
        try {
            await API.exportPortfolioHTML();
        } catch (e) {
            App.showToast('Failed to export report: ' + e.message, 'error');
        }
    },

    async deleteHolding(id) {
        try {
            await API.deleteHolding(id);
            App.showToast('Holding removed', 'success');
            this.load();
        } catch (e) {
            App.showToast('Failed to delete: ' + e.message, 'error');
        }
    },

    // ------------------------------------------------------------------
    // Analytics
    // ------------------------------------------------------------------

    async loadAnalytics() {
        const container = document.getElementById('portfolioAnalytics');
        if (!container) return;

        container.innerHTML = '<div class="text-center py-6 text-gray-500 text-sm">Loading analytics...</div>';

        try {
            const data = await API.getPortfolioAnalytics();
            this.displayAnalytics(data);
        } catch (e) {
            console.error('Failed to load analytics:', e);
            container.innerHTML = `<div class="text-center py-6">
                <div class="text-red-400 text-sm mb-2">Failed to load analytics</div>
                <button onclick="Portfolio.loadAnalytics()" class="text-xs px-3 py-1 bg-dark-600 text-gray-300 rounded hover:bg-dark-700">Retry</button>
            </div>`;
        }
    },

    displayAnalytics(data) {
        const container = document.getElementById('portfolioAnalytics');
        if (!container) return;

        const fmt = (v, suffix = '%') => v != null ? v.toFixed(2) + suffix : '--';
        const color = (v) => v == null ? 'text-gray-400' : (v >= 0 ? 'text-green-400' : 'text-red-400');

        // Metric cards
        const cagrColor = color(data.cagr);
        const sharpeColor = data.sharpe_ratio != null ? (data.sharpe_ratio >= 1 ? 'text-green-400' : data.sharpe_ratio >= 0 ? 'text-yellow-400' : 'text-red-400') : 'text-gray-400';
        const ddColor = color(data.max_drawdown);

        // Sector allocation bar chart (horizontal)
        let sectorHtml = '';
        if (data.sector_allocation && data.sector_allocation.length > 0) {
            const sectorColors = [
                'bg-blue-500', 'bg-green-500', 'bg-yellow-500', 'bg-purple-500',
                'bg-pink-500', 'bg-cyan-500', 'bg-orange-500', 'bg-teal-500',
                'bg-indigo-500', 'bg-red-500', 'bg-lime-500', 'bg-amber-500',
            ];
            sectorHtml = data.sector_allocation.map((s, i) => {
                const barColor = sectorColors[i % sectorColors.length];
                return `
                    <div class="flex items-center gap-2 text-xs">
                        <span class="w-24 sm:w-28 text-gray-300 truncate">${s.sector}</span>
                        <div class="flex-1 bg-dark-700 rounded-full h-3 overflow-hidden">
                            <div class="${barColor} h-3 rounded-full transition-all" style="width: ${s.percentage}%"></div>
                        </div>
                        <span class="text-gray-400 w-14 text-right">${s.percentage.toFixed(1)}%</span>
                    </div>`;
            }).join('');
        } else {
            sectorHtml = '<div class="text-gray-500 text-xs text-center py-2">No sector data</div>';
        }

        container.innerHTML = `
            <!-- Analytics Metric Cards -->
            <div class="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
                <div class="bg-dark-800 rounded-lg p-3 sm:p-4">
                    <div class="text-xs text-gray-400 mb-1">CAGR</div>
                    <div class="text-xl font-bold ${cagrColor}">${fmt(data.cagr)}</div>
                    <div class="text-[10px] text-gray-500 mt-1">Compound Annual Growth Rate</div>
                </div>
                <div class="bg-dark-800 rounded-lg p-3 sm:p-4">
                    <div class="text-xs text-gray-400 mb-1">Sharpe Ratio</div>
                    <div class="text-xl font-bold ${sharpeColor}">${data.sharpe_ratio != null ? data.sharpe_ratio.toFixed(2) : '--'}</div>
                    <div class="text-[10px] text-gray-500 mt-1">Risk-adjusted return (Rf = 6%)</div>
                </div>
                <div class="bg-dark-800 rounded-lg p-3 sm:p-4">
                    <div class="text-xs text-gray-400 mb-1">Max Drawdown</div>
                    <div class="text-xl font-bold ${ddColor}">${fmt(data.max_drawdown)}</div>
                    <div class="text-[10px] text-gray-500 mt-1">Largest peak-to-trough decline</div>
                </div>
            </div>

            <!-- Sector Allocation -->
            <div class="bg-dark-800 rounded-lg p-3 sm:p-4">
                <h4 class="text-sm font-semibold text-white mb-3">Sector Allocation</h4>
                <div class="space-y-2">${sectorHtml}</div>
            </div>
        `;
    }
};
