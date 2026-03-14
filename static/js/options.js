/**
 * Options chain viewer module.
 * Displays calls/puts table with ITM/OTM highlighting, OI change coloring,
 * and max pain indicator.
 */
const Options = {
    _symbol: null,
    _chainData: null,
    _maxPainData: null,

    init() {
        const expirySelect = document.getElementById('optionsExpirySelect');
        if (expirySelect) {
            expirySelect.addEventListener('change', () => {
                if (this._symbol) {
                    this.loadChain(this._symbol, expirySelect.value);
                }
            });
        }
    },

    async loadChain(symbol, expiry) {
        if (!symbol) return;
        this._symbol = symbol;

        const loading = document.getElementById('optionsLoading');
        const tableDiv = document.getElementById('optionsChainTable');
        const emptyDiv = document.getElementById('optionsEmpty');

        // Show loading
        loading.classList.remove('hidden');
        tableDiv.innerHTML = '';
        emptyDiv.classList.add('hidden');

        try {
            const expiryParam = expiry ? `?expiry=${encodeURIComponent(expiry)}` : '';
            const [chain, maxPain] = await Promise.all([
                API.request(`/api/options/${encodeURIComponent(symbol)}/chain${expiryParam}`),
                API.request(`/api/options/${encodeURIComponent(symbol)}/maxpain${expiryParam}`),
            ]);

            this._chainData = chain;
            this._maxPainData = maxPain;

            loading.classList.add('hidden');

            if (!chain.available) {
                emptyDiv.classList.remove('hidden');
                this._clearSummary();
                return;
            }

            // Populate expiry dropdown (only on first load or symbol change)
            if (!expiry) {
                this._populateExpiries(chain.expiry_dates, chain.selected_expiry);
            }

            this._renderSummary(chain, maxPain);
            this._renderTable(chain, maxPain);

        } catch (e) {
            loading.classList.add('hidden');
            emptyDiv.classList.remove('hidden');
            console.error('Options chain load failed:', e);
        }
    },

    _populateExpiries(dates, selected) {
        const sel = document.getElementById('optionsExpirySelect');
        if (!sel || !dates || dates.length === 0) return;
        sel.innerHTML = dates.map(d =>
            `<option value="${d}" ${d === selected ? 'selected' : ''}>${d}</option>`
        ).join('');
    },

    _clearSummary() {
        document.getElementById('optTotalCallOI').textContent = '-';
        document.getElementById('optTotalPutOI').textContent = '-';
        document.getElementById('optPCR').textContent = '-';
        document.getElementById('optMaxPain').textContent = '-';
        const badge = document.getElementById('optionsMaxPainBadge');
        if (badge) badge.classList.add('hidden');
    },

    _renderSummary(chain, maxPain) {
        const fmt = (v) => {
            if (v >= 10000000) return (v / 10000000).toFixed(2) + ' Cr';
            if (v >= 100000) return (v / 100000).toFixed(2) + ' L';
            if (v >= 1000) return (v / 1000).toFixed(1) + ' K';
            return v.toLocaleString('en-IN');
        };

        const t = chain.totals;
        document.getElementById('optTotalCallOI').textContent = fmt(t.call_oi);
        document.getElementById('optTotalPutOI').textContent = fmt(t.put_oi);

        const pcrEl = document.getElementById('optPCR');
        pcrEl.textContent = t.pcr.toFixed(2);
        pcrEl.className = 'text-sm font-bold ' + (t.pcr > 1 ? 'text-green-400' : t.pcr < 0.7 ? 'text-red-400' : 'text-yellow-400');

        const mpEl = document.getElementById('optMaxPain');
        if (maxPain.available) {
            mpEl.textContent = '\u20b9' + maxPain.max_pain.toLocaleString('en-IN');
        } else {
            mpEl.textContent = '-';
        }

        // Max pain badge
        const badge = document.getElementById('optionsMaxPainBadge');
        if (badge && maxPain.available) {
            const diff = chain.underlying ? ((maxPain.max_pain - chain.underlying) / chain.underlying * 100).toFixed(2) : 0;
            const sign = diff >= 0 ? '+' : '';
            badge.textContent = `Max Pain: \u20b9${maxPain.max_pain.toLocaleString('en-IN')} (${sign}${diff}%)`;
            badge.classList.remove('hidden');
        }
    },

    _renderTable(chain, maxPain) {
        const tableDiv = document.getElementById('optionsChainTable');
        const underlying = chain.underlying || 0;
        const mp = maxPain.available ? maxPain.max_pain : null;

        // Build a map of strike -> {call, put}
        const strikeMap = {};
        chain.calls.forEach(c => {
            strikeMap[c.strike] = strikeMap[c.strike] || {};
            strikeMap[c.strike].call = c;
        });
        chain.puts.forEach(p => {
            strikeMap[p.strike] = strikeMap[p.strike] || {};
            strikeMap[p.strike].put = p;
        });

        const strikes = Object.keys(strikeMap).map(Number).sort((a, b) => a - b);

        // Find max OI for bar widths
        let maxOI = 1;
        strikes.forEach(s => {
            const row = strikeMap[s];
            if (row.call) maxOI = Math.max(maxOI, row.call.oi);
            if (row.put) maxOI = Math.max(maxOI, row.put.oi);
        });

        const fmtNum = (v, dec) => {
            if (v === 0 || v == null) return '-';
            return v.toLocaleString('en-IN', { maximumFractionDigits: dec || 0 });
        };

        const fmtOI = (v) => {
            if (v === 0 || v == null) return '-';
            if (v >= 10000000) return (v / 10000000).toFixed(1) + 'Cr';
            if (v >= 100000) return (v / 100000).toFixed(1) + 'L';
            if (v >= 1000) return (v / 1000).toFixed(1) + 'K';
            return v.toLocaleString('en-IN');
        };

        const oiChangeClass = (v) => {
            if (v > 0) return 'text-green-400';  // buildup
            if (v < 0) return 'text-red-400';    // unwinding
            return 'text-gray-500';
        };

        const oiBarStyle = (oi, side) => {
            const pct = Math.min(100, (oi / maxOI) * 100);
            const color = side === 'call' ? 'rgba(239,68,68,0.2)' : 'rgba(34,197,94,0.2)';
            const dir = side === 'call' ? 'left' : 'right';
            return `background: linear-gradient(to ${dir}, ${color} ${pct}%, transparent ${pct}%)`;
        };

        let html = `
        <table class="w-full text-[11px] sm:text-xs border-collapse">
            <thead>
                <tr class="text-gray-400 border-b border-gray-700">
                    <th colspan="6" class="text-center py-1.5 text-red-400 border-r border-gray-700">CALLS</th>
                    <th class="text-center py-1.5 text-white">STRIKE</th>
                    <th colspan="6" class="text-center py-1.5 text-green-400 border-l border-gray-700">PUTS</th>
                </tr>
                <tr class="text-gray-500 border-b border-gray-600 text-[10px]">
                    <th class="py-1 px-1 text-right">OI Chg</th>
                    <th class="py-1 px-1 text-right">OI</th>
                    <th class="py-1 px-1 text-right">Vol</th>
                    <th class="py-1 px-1 text-right">IV</th>
                    <th class="py-1 px-1 text-right">Chg</th>
                    <th class="py-1 px-1 text-right border-r border-gray-700">LTP</th>
                    <th class="py-1 px-1 text-center"></th>
                    <th class="py-1 px-1 text-left border-l border-gray-700">LTP</th>
                    <th class="py-1 px-1 text-left">Chg</th>
                    <th class="py-1 px-1 text-left">IV</th>
                    <th class="py-1 px-1 text-left">Vol</th>
                    <th class="py-1 px-1 text-left">OI</th>
                    <th class="py-1 px-1 text-left">OI Chg</th>
                </tr>
            </thead>
            <tbody>`;

        strikes.forEach(strike => {
            const row = strikeMap[strike];
            const c = row.call || {};
            const p = row.put || {};

            // ITM: calls where strike < underlying, puts where strike > underlying
            const callITM = underlying > 0 && strike < underlying;
            const putITM = underlying > 0 && strike > underlying;
            const isMaxPain = mp && strike === mp;
            const isATM = underlying > 0 && strikes.length > 1 &&
                Math.abs(strike - underlying) === Math.min(...strikes.map(s => Math.abs(s - underlying)));

            // Row classes
            let rowClass = 'border-b border-gray-800 hover:bg-dark-700 transition-colors';
            if (isMaxPain) rowClass += ' bg-purple-900/20 border-l-2 border-l-purple-500';
            else if (isATM) rowClass += ' bg-accent-blue/10 border-l-2 border-l-accent-blue';

            // Cell background for ITM
            const callBg = callITM ? 'bg-red-900/10' : '';
            const putBg = putITM ? 'bg-green-900/10' : '';

            html += `<tr class="${rowClass}">`;

            // CALL side (right-aligned)
            html += `<td class="py-1 px-1 text-right ${callBg} ${oiChangeClass(c.oi_change || 0)}">${fmtOI(c.oi_change)}</td>`;
            html += `<td class="py-1 px-1 text-right ${callBg}" style="${oiBarStyle(c.oi || 0, 'call')}">${fmtOI(c.oi)}</td>`;
            html += `<td class="py-1 px-1 text-right ${callBg} text-gray-400">${fmtOI(c.volume)}</td>`;
            html += `<td class="py-1 px-1 text-right ${callBg} text-gray-400">${c.iv ? c.iv.toFixed(1) : '-'}</td>`;
            html += `<td class="py-1 px-1 text-right ${callBg} ${(c.change || 0) >= 0 ? 'text-green-400' : 'text-red-400'}">${fmtNum(c.change, 2)}</td>`;
            html += `<td class="py-1 px-1 text-right ${callBg} text-white font-medium border-r border-gray-700">${fmtNum(c.ltp, 2)}</td>`;

            // STRIKE (center)
            let strikeClass = 'py-1 px-2 text-center font-bold text-white';
            if (isMaxPain) strikeClass += ' text-purple-300';
            else if (isATM) strikeClass += ' text-accent-blue';
            const mpIcon = isMaxPain ? ' <span class="text-purple-400 text-[9px]">MP</span>' : '';
            const atmIcon = isATM ? ' <span class="text-blue-400 text-[9px]">ATM</span>' : '';
            html += `<td class="${strikeClass}">${strike.toLocaleString('en-IN')}${mpIcon}${atmIcon}</td>`;

            // PUT side (left-aligned)
            html += `<td class="py-1 px-1 text-left ${putBg} text-white font-medium border-l border-gray-700">${fmtNum(p.ltp, 2)}</td>`;
            html += `<td class="py-1 px-1 text-left ${putBg} ${(p.change || 0) >= 0 ? 'text-green-400' : 'text-red-400'}">${fmtNum(p.change, 2)}</td>`;
            html += `<td class="py-1 px-1 text-left ${putBg} text-gray-400">${p.iv ? p.iv.toFixed(1) : '-'}</td>`;
            html += `<td class="py-1 px-1 text-left ${putBg} text-gray-400">${fmtOI(p.volume)}</td>`;
            html += `<td class="py-1 px-1 text-left ${putBg}" style="${oiBarStyle(p.oi || 0, 'put')}">${fmtOI(p.oi)}</td>`;
            html += `<td class="py-1 px-1 text-left ${putBg} ${oiChangeClass(p.oi_change || 0)}">${fmtOI(p.oi_change)}</td>`;

            html += '</tr>';
        });

        html += '</tbody></table>';

        // Legend
        html += `
        <div class="flex flex-wrap gap-3 mt-2 text-[10px] text-gray-500">
            <span class="flex items-center gap-1"><span class="w-2 h-2 rounded bg-red-900/40 inline-block"></span> Call ITM</span>
            <span class="flex items-center gap-1"><span class="w-2 h-2 rounded bg-green-900/40 inline-block"></span> Put ITM</span>
            <span class="flex items-center gap-1"><span class="w-2 h-2 rounded bg-purple-900/40 inline-block"></span> Max Pain</span>
            <span class="flex items-center gap-1"><span class="w-2 h-2 rounded bg-blue-900/40 inline-block"></span> ATM</span>
            <span class="flex items-center gap-1"><span class="text-green-400">+OI</span> = Buildup</span>
            <span class="flex items-center gap-1"><span class="text-red-400">-OI</span> = Unwinding</span>
        </div>`;

        tableDiv.innerHTML = html;

        // Scroll ATM row into view
        this._scrollToATM(tableDiv, underlying, strikes);
    },

    _scrollToATM(container, underlying, strikes) {
        if (!underlying || strikes.length === 0) return;
        // Find the closest strike to underlying
        let closestIdx = 0;
        let minDiff = Infinity;
        strikes.forEach((s, i) => {
            const diff = Math.abs(s - underlying);
            if (diff < minDiff) { minDiff = diff; closestIdx = i; }
        });

        // Scroll the table row into view
        const rows = container.querySelectorAll('tbody tr');
        if (rows[closestIdx]) {
            rows[closestIdx].scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    },
};
