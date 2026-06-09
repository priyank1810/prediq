// IPO tab: upcoming IPOs with AI Apply/Neutral/Avoid verdicts + performance scorecard.
window.IPO = {
    _loaded: false,
    _expanded: null,

    init() {
        // Nothing to wire up beyond the buttons in the template (inline onclick).
    },

    _esc(s) {
        if (s === null || s === undefined) return '';
        return String(s).replace(/[&<>"']/g, c => (
            { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
        ));
    },

    _verdictClass(v) {
        if (v === 'APPLY') return 'bg-green-900/40 text-green-400 border-green-700';
        if (v === 'AVOID') return 'bg-red-900/40 text-red-400 border-red-700';
        if (v === 'NEUTRAL') return 'bg-yellow-900/40 text-yellow-400 border-yellow-700';
        return 'bg-dark-700 text-gray-400 border-gray-700';
    },

    _fmtDate(d) {
        if (!d) return '—';
        try {
            return new Date(d).toLocaleDateString('en-IN', { day: '2-digit', month: 'short' });
        } catch (e) { return d; }
    },

    showUpcoming() {
        document.getElementById('ipoUpcoming').classList.remove('hidden');
        document.getElementById('ipoScorecard').classList.add('hidden');
        document.getElementById('ipoTabUpcoming').className = 'px-3 py-1.5 bg-accent-blue text-white text-xs rounded hover:bg-blue-600';
        document.getElementById('ipoTabScorecard').className = 'px-3 py-1.5 bg-dark-700 text-gray-300 text-xs rounded hover:bg-dark-600';
        if (!this._loaded) this.loadUpcoming();
    },

    showScorecard() {
        document.getElementById('ipoUpcoming').classList.add('hidden');
        document.getElementById('ipoScorecard').classList.remove('hidden');
        document.getElementById('ipoTabScorecard').className = 'px-3 py-1.5 bg-accent-blue text-white text-xs rounded hover:bg-blue-600';
        document.getElementById('ipoTabUpcoming').className = 'px-3 py-1.5 bg-dark-700 text-gray-300 text-xs rounded hover:bg-dark-600';
        this.loadScorecard();
    },

    async loadUpcoming() {
        const el = document.getElementById('ipoUpcoming');
        try {
            const res = await fetch('/api/ipo/upcoming');
            const data = await res.json();
            this._loaded = true;
            if (!Array.isArray(data) || data.length === 0) {
                el.innerHTML = '<div class="text-center py-8 text-gray-500">No upcoming IPOs right now.</div>';
                return;
            }
            el.innerHTML = '<div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">' +
                data.map(ipo => this._card(ipo)).join('') + '</div>';
        } catch (e) {
            el.innerHTML = '<div class="text-center py-8 text-red-400">Failed to load IPOs.</div>';
        }
    },

    _card(ipo) {
        const band = (ipo.price_band_low && ipo.price_band_high)
            ? `₹${ipo.price_band_low}–${ipo.price_band_high}` : '—';
        const lotCost = ipo.lot_cost ? `₹${Math.round(ipo.lot_cost).toLocaleString('en-IN')}` : '—';
        const verdict = ipo.verdict || 'PENDING';
        const conf = ipo.confidence != null ? `${ipo.confidence}%` : '';
        const key = ipo.symbol || ipo.id;
        return `
        <div class="bg-dark-800 rounded-lg p-3 border border-gray-800">
            <div class="flex items-start justify-between gap-2">
                <div class="min-w-0">
                    <div class="text-sm font-semibold text-white truncate">${this._esc(ipo.name)}</div>
                    <div class="text-[11px] text-gray-500">${this._esc(ipo.symbol || '')}</div>
                </div>
                <span class="shrink-0 text-[10px] font-bold px-2 py-0.5 rounded border ${this._verdictClass(verdict)}">${this._esc(verdict)} ${conf}</span>
            </div>
            <div class="grid grid-cols-2 gap-x-3 gap-y-1 mt-2 text-[11px] text-gray-400">
                <div>Open: <span class="text-gray-300">${this._fmtDate(ipo.open_date)}</span></div>
                <div>Close: <span class="text-gray-300">${this._fmtDate(ipo.close_date)}</span></div>
                <div>Band: <span class="text-gray-300">${band}</span></div>
                <div>1 lot: <span class="text-gray-300">${lotCost}</span></div>
                <div>GMP: <span class="text-gray-300">${ipo.gmp != null ? ipo.gmp + '%' : '—'}</span></div>
                <div>Subs: <span class="text-gray-300">${ipo.subscription_total != null ? ipo.subscription_total + 'x' : '—'}</span></div>
            </div>
            <button onclick="IPO.toggleDetail('${this._esc(key)}', this)" class="mt-2 text-[11px] text-accent-blue hover:underline">Why this verdict?</button>
            <div class="ipo-detail mt-2 hidden text-[11px]"></div>
        </div>`;
    },

    async toggleDetail(key, btn) {
        const box = btn.nextElementSibling;
        if (!box.classList.contains('hidden')) {
            box.classList.add('hidden');
            return;
        }
        box.classList.remove('hidden');
        box.innerHTML = '<div class="text-gray-500">Loading…</div>';
        try {
            const res = await fetch('/api/ipo/' + encodeURIComponent(key));
            if (!res.ok) { box.innerHTML = '<div class="text-gray-500">No analysis available.</div>'; return; }
            const d = await res.json();
            const reasons = (d.reasons || []).map(r => `<li class="text-gray-400">• ${this._esc(r)}</li>`).join('');
            const flags = (d.risk_flags || []).map(r => `<li class="text-amber-400">⚠ ${this._esc(r)}</li>`).join('');
            box.innerHTML = `
                <ul class="space-y-0.5">${reasons || '<li class="text-gray-500">No reasons recorded.</li>'}</ul>
                ${flags ? `<ul class="space-y-0.5 mt-1 border-t border-gray-800 pt-1">${flags}</ul>` : ''}`;
        } catch (e) {
            box.innerHTML = '<div class="text-red-400">Failed to load.</div>';
        }
    },

    async loadScorecard() {
        const el = document.getElementById('ipoScorecard');
        el.innerHTML = '<div class="text-center py-8 text-gray-500">Loading scorecard…</div>';
        try {
            const res = await fetch('/api/ipo/scorecard');
            const d = await res.json();
            const rows = (d.history || []).map(h => {
                const hit = h.hit === true ? '<span class="text-green-400">Hit</span>'
                    : h.hit === false ? '<span class="text-red-400">Miss</span>'
                    : '<span class="text-gray-500">—</span>';
                const gain = h.listing_gain_pct != null ? `${h.listing_gain_pct}%` : '—';
                const gc = h.listing_gain_pct > 0 ? 'text-green-400' : h.listing_gain_pct < 0 ? 'text-red-400' : 'text-gray-300';
                return `<tr class="border-t border-gray-800">
                    <td class="py-1.5 pr-2 text-gray-300">${this._esc(h.name)}</td>
                    <td class="py-1.5 pr-2"><span class="text-[10px] font-bold px-1.5 py-0.5 rounded border ${this._verdictClass(h.verdict)}">${this._esc(h.verdict)}</span></td>
                    <td class="py-1.5 pr-2 ${gc}">${gain}</td>
                    <td class="py-1.5">${hit}</td>
                </tr>`;
            }).join('');
            el.innerHTML = `
                <div class="bg-dark-800 rounded-lg p-3 sm:p-4 mb-3 flex flex-wrap gap-4">
                    <div><div class="text-2xl font-bold text-white">${d.accuracy_pct ?? 0}%</div><div class="text-[11px] text-gray-500">Accuracy</div></div>
                    <div><div class="text-2xl font-bold text-white">${d.hits ?? 0}/${d.total_graded ?? 0}</div><div class="text-[11px] text-gray-500">Hits / Graded</div></div>
                </div>
                ${(d.history && d.history.length) ? `
                <div class="bg-dark-800 rounded-lg p-3 sm:p-4 overflow-x-auto">
                    <table class="w-full text-xs"><thead><tr class="text-gray-500 text-left">
                        <th class="pb-1 pr-2 font-medium">IPO</th><th class="pb-1 pr-2 font-medium">Verdict</th>
                        <th class="pb-1 pr-2 font-medium">Listing gain</th><th class="pb-1 font-medium">Result</th>
                    </tr></thead><tbody>${rows}</tbody></table>
                </div>` : '<div class="text-center py-8 text-gray-500">No listed IPOs graded yet.</div>'}`;
        } catch (e) {
            el.innerHTML = '<div class="text-center py-8 text-red-400">Failed to load scorecard.</div>';
        }
    },
};
