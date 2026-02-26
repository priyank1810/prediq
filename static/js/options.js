const Options = {
    currentSymbol: null,
    currentExpiry: null,
    isLoading: false,

    init() {
        const loadBtn = document.getElementById('optionLoadBtn');
        const symbolInput = document.getElementById('optionSymbolInput');
        const expirySelect = document.getElementById('optionExpirySelect');

        if (!loadBtn) return;

        loadBtn.addEventListener('click', () => {
            const sym = symbolInput.value.trim().toUpperCase();
            if (sym) this.load(sym);
        });

        symbolInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                const sym = symbolInput.value.trim().toUpperCase();
                if (sym) this.load(sym);
            }
        });

        expirySelect.addEventListener('change', () => {
            if (this.currentSymbol && expirySelect.value) {
                this.loadChain(this.currentSymbol, expirySelect.value);
            }
        });
    },

    _renderLoading() {
        const container = document.getElementById('optionChainContainer');
        container.innerHTML = `
            <div class="text-center py-12">
                <div class="animate-spin w-8 h-8 border-2 border-accent-blue border-t-transparent rounded-full mx-auto"></div>
                <p class="text-gray-400 text-sm mt-3">Fetching option chain from NSE...</p>
            </div>`;
    },

    _renderError(message) {
        const container = document.getElementById('optionChainContainer');
        const summary = document.getElementById('optionSummary');
        summary.classList.add('hidden');
        container.innerHTML = `
            <div class="text-center py-10 px-6">
                <div class="text-yellow-400 text-3xl mb-3">&#9888;</div>
                <p class="text-gray-300 text-sm mb-2">${message}</p>
                <p class="text-gray-600 text-xs mb-4">Tip: NSE option chain works best during market hours (Mon-Fri, 9:15 AM - 3:30 PM IST) from a non-cloud network.</p>
                <button onclick="Options.currentSymbol && Options.load(Options.currentSymbol)"
                    class="px-4 py-1.5 bg-accent-blue text-white text-sm rounded hover:bg-blue-600 transition font-medium">
                    Retry
                </button>
            </div>`;
    },

    async load(symbol) {
        if (this.isLoading) return;
        this.isLoading = true;
        this.currentSymbol = symbol;

        const summary = document.getElementById('optionSummary');
        summary.classList.add('hidden');
        this._renderLoading();

        try {
            const data = await API.getOptionChain(symbol);
            this.currentExpiry = data.expiry;
            this._populateExpiries(data.expiry_dates, data.expiry);
            this.renderChain(data);
        } catch (e) {
            this._renderError(e.message);
        } finally {
            this.isLoading = false;
        }
    },

    async loadChain(symbol, expiry) {
        if (this.isLoading) return;
        this.isLoading = true;
        this.currentExpiry = expiry;

        this._renderLoading();

        try {
            const data = await API.getOptionChain(symbol, expiry);
            this.renderChain(data);
        } catch (e) {
            this._renderError(e.message);
        } finally {
            this.isLoading = false;
        }
    },

    _populateExpiries(dates, selected) {
        const sel = document.getElementById('optionExpirySelect');
        sel.innerHTML = dates.map(d =>
            `<option value="${d}" ${d === selected ? 'selected' : ''}>${d}</option>`
        ).join('');
    },

    _formatNum(n) {
        if (n >= 10000000) return (n / 10000000).toFixed(2) + 'Cr';
        if (n >= 100000) return (n / 100000).toFixed(2) + 'L';
        if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
        return n.toLocaleString('en-IN');
    },

    renderChain(data) {
        const container = document.getElementById('optionChainContainer');
        const summary = document.getElementById('optionSummary');

        if (!data.data || data.data.length === 0) {
            container.innerHTML = '<div class="text-center py-8 text-gray-500">No option chain data available.</div>';
            summary.classList.add('hidden');
            return;
        }

        // Summary bar
        const pcrColor = data.pcr > 1 ? 'text-green-400' : (data.pcr < 0.7 ? 'text-red-400' : 'text-yellow-400');
        summary.classList.remove('hidden');
        summary.innerHTML = `
            <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3 sm:gap-4 text-sm">
                <div>
                    <span class="text-gray-400">Spot Price</span>
                    <div class="text-white font-bold text-lg">${data.spot_price.toLocaleString('en-IN', { style: 'currency', currency: 'INR' })}</div>
                </div>
                <div>
                    <span class="text-gray-400">PCR (OI)</span>
                    <div class="${pcrColor} font-bold text-lg">${data.pcr}</div>
                </div>
                <div>
                    <span class="text-gray-400">Max Pain</span>
                    <div class="text-white font-bold text-lg">${data.max_pain.toLocaleString('en-IN', { style: 'currency', currency: 'INR' })}</div>
                </div>
                <div>
                    <span class="text-gray-400">Total CE OI</span>
                    <div class="text-red-400 font-bold text-lg">${this._formatNum(data.total_ce_oi)}</div>
                </div>
                <div>
                    <span class="text-gray-400">Total PE OI</span>
                    <div class="text-green-400 font-bold text-lg">${this._formatNum(data.total_pe_oi)}</div>
                </div>
            </div>
        `;

        // Find max OI for color intensity
        const maxOi = Math.max(...data.data.map(r => Math.max(r.ce_oi, r.pe_oi)), 1);
        const spot = data.spot_price;
        const atm = data.atm_strike;

        // Build table
        const rows = data.data.map(r => {
            const isATM = r.strike_price === atm;
            const isITM_CE = r.strike_price < spot;
            const isITM_PE = r.strike_price > spot;

            // OI intensity (0-100%)
            const ceOiPct = Math.min((r.ce_oi / maxOi) * 100, 100);
            const peOiPct = Math.min((r.pe_oi / maxOi) * 100, 100);

            // CE background: ITM calls get subtle yellow tint
            const ceBg = isITM_CE ? 'bg-yellow-900/10' : '';
            const peBg = isITM_PE ? 'bg-yellow-900/10' : '';
            const atmBorder = isATM ? 'border-y-2 border-accent-blue' : 'border-b border-gray-800';

            const ceChgColor = r.ce_oi_change > 0 ? 'text-green-400' : (r.ce_oi_change < 0 ? 'text-red-400' : 'text-gray-500');
            const peChgColor = r.pe_oi_change > 0 ? 'text-green-400' : (r.pe_oi_change < 0 ? 'text-red-400' : 'text-gray-500');
            const ceLtpChgColor = r.ce_change > 0 ? 'text-green-400' : (r.ce_change < 0 ? 'text-red-400' : 'text-gray-400');
            const peLtpChgColor = r.pe_change > 0 ? 'text-green-400' : (r.pe_change < 0 ? 'text-red-400' : 'text-gray-400');

            return `
                <tr class="${atmBorder} hover:bg-dark-600 text-xs">
                    <td class="${ceBg} px-2 py-1.5 text-right">
                        <div class="relative">
                            <div class="absolute inset-0 bg-red-500/10 rounded" style="width:${ceOiPct}%"></div>
                            <span class="relative text-gray-200">${this._formatNum(r.ce_oi)}</span>
                        </div>
                    </td>
                    <td class="${ceBg} px-2 py-1.5 text-right ${ceChgColor}">${this._formatNum(r.ce_oi_change)}</td>
                    <td class="${ceBg} px-2 py-1.5 text-right text-gray-300">${this._formatNum(r.ce_volume)}</td>
                    <td class="${ceBg} px-2 py-1.5 text-right text-gray-400">${r.ce_iv || '-'}</td>
                    <td class="${ceBg} px-2 py-1.5 text-right text-white font-medium">${r.ce_ltp || '-'}</td>
                    <td class="${ceBg} px-2 py-1.5 text-right ${ceLtpChgColor}">${r.ce_change || '-'}</td>
                    <td class="px-3 py-1.5 text-center font-bold ${isATM ? 'text-accent-blue bg-accent-blue/10' : 'text-white bg-dark-700'}">${r.strike_price}</td>
                    <td class="${peBg} px-2 py-1.5 text-right ${peLtpChgColor}">${r.pe_change || '-'}</td>
                    <td class="${peBg} px-2 py-1.5 text-right text-white font-medium">${r.pe_ltp || '-'}</td>
                    <td class="${peBg} px-2 py-1.5 text-right text-gray-400">${r.pe_iv || '-'}</td>
                    <td class="${peBg} px-2 py-1.5 text-right text-gray-300">${this._formatNum(r.pe_volume)}</td>
                    <td class="${peBg} px-2 py-1.5 text-right ${peChgColor}">${this._formatNum(r.pe_oi_change)}</td>
                    <td class="${peBg} px-2 py-1.5 text-right">
                        <div class="relative">
                            <div class="absolute inset-0 bg-green-500/10 rounded" style="width:${peOiPct}%"></div>
                            <span class="relative text-gray-200">${this._formatNum(r.pe_oi)}</span>
                        </div>
                    </td>
                </tr>
            `;
        }).join('');

        container.innerHTML = `
            <div class="overflow-x-auto">
                <table class="w-full text-sm">
                    <thead>
                        <tr class="text-gray-400 border-b border-gray-700">
                            <th colspan="6" class="text-center py-2 text-red-400 font-semibold border-r border-gray-700">CALLS</th>
                            <th class="py-2 text-center text-white font-semibold">STRIKE</th>
                            <th colspan="6" class="text-center py-2 text-green-400 font-semibold border-l border-gray-700">PUTS</th>
                        </tr>
                        <tr class="text-gray-500 border-b border-gray-700 text-xs">
                            <th class="text-right px-2 py-1.5">OI</th>
                            <th class="text-right px-2 py-1.5">Chg OI</th>
                            <th class="text-right px-2 py-1.5">Volume</th>
                            <th class="text-right px-2 py-1.5">IV</th>
                            <th class="text-right px-2 py-1.5">LTP</th>
                            <th class="text-right px-2 py-1.5">Chg</th>
                            <th class="text-center px-2 py-1.5"></th>
                            <th class="text-right px-2 py-1.5">Chg</th>
                            <th class="text-right px-2 py-1.5">LTP</th>
                            <th class="text-right px-2 py-1.5">IV</th>
                            <th class="text-right px-2 py-1.5">Volume</th>
                            <th class="text-right px-2 py-1.5">Chg OI</th>
                            <th class="text-right px-2 py-1.5">OI</th>
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
        `;

        // Scroll to ATM strike
        if (atm) {
            setTimeout(() => {
                const atmRow = container.querySelector('.border-accent-blue');
                if (atmRow) atmRow.scrollIntoView({ block: 'center', behavior: 'smooth' });
            }, 100);
        }
    },
};
