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

    async deleteHolding(id) {
        try {
            await API.deleteHolding(id);
            App.showToast('Holding removed', 'success');
            this.load();
        } catch (e) {
            App.showToast('Failed to delete: ' + e.message, 'error');
        }
    }
};
