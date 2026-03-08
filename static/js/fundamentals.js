const Fundamentals = {
    _currentSymbol: null,

    async load(symbol) {
        this._currentSymbol = symbol;
        const panel = document.getElementById('fundamentalsPanel');
        if (!panel) return;
        panel.classList.remove('hidden');
        panel.innerHTML = '<div class="text-center py-6"><div class="animate-spin w-6 h-6 border-2 border-accent-blue border-t-transparent rounded-full mx-auto"></div><p class="text-gray-500 text-xs mt-2">Loading fundamentals...</p></div>';

        try {
            const [data, news] = await Promise.all([
                API.getFundamentals(symbol).catch(() => null),
                API.getStockNews(symbol).catch(() => null),
            ]);
            if (symbol !== this._currentSymbol) return;
            this.render(data, news);
        } catch (e) {
            panel.innerHTML = '<div class="text-center py-4 text-gray-500 text-sm">Failed to load fundamentals</div>';
        }
    },

    render(data, news) {
        const panel = document.getElementById('fundamentalsPanel');
        if (!panel) return;
        if (!data || !data.symbol) {
            panel.innerHTML = '<div class="text-center py-4 text-gray-500 text-sm">No fundamental data available for this stock</div>';
            return;
        }

        panel.innerHTML = `
            ${this._renderKeyMetrics(data)}
            ${this._renderFinancials(data)}
            ${this._renderQuarterlyResults(data)}
            ${this._renderBalanceSheet(data)}
            ${this._renderNews(news)}
        `;
    },

    _renderKeyMetrics(d) {
        const fmt = (v, suffix) => {
            if (v == null || v === 0) return '<span class="text-gray-600">-</span>';
            return v.toFixed(2) + (suffix || '');
        };
        const fmtCr = (v) => {
            if (v == null) return '<span class="text-gray-600">-</span>';
            if (Math.abs(v) >= 10000000) return '₹' + (v / 10000000).toFixed(1) + ' Cr';
            if (Math.abs(v) >= 100000) return '₹' + (v / 100000).toFixed(1) + ' L';
            return '₹' + v.toLocaleString('en-IN');
        };
        const colorPct = (v) => {
            if (v == null || v === 0) return 'text-gray-500';
            return v > 0 ? 'text-green-400' : 'text-red-400';
        };

        const w52h = d['52_week_high'];
        const w52l = d['52_week_low'];

        return `
        <div class="bg-dark-800 rounded-lg p-3 sm:p-4 mb-3">
            <div class="flex items-center justify-between mb-3">
                <h4 class="text-sm font-semibold text-white">Key Metrics</h4>
                <span class="text-xs text-gray-500">${d.sector}${d.industry ? ' · ' + d.industry : ''}</span>
            </div>
            <div class="grid grid-cols-3 sm:grid-cols-5 gap-2 text-center">
                <div class="bg-dark-700 rounded p-2">
                    <div class="text-[10px] text-gray-500">P/E Ratio</div>
                    <div class="text-sm font-bold text-white">${fmt(d.pe)}</div>
                    ${d.forward_pe ? '<div class="text-[9px] text-gray-500">Fwd: ' + fmt(d.forward_pe) + '</div>' : ''}
                </div>
                <div class="bg-dark-700 rounded p-2">
                    <div class="text-[10px] text-gray-500">P/B Ratio</div>
                    <div class="text-sm font-bold text-white">${fmt(d.pb)}</div>
                </div>
                <div class="bg-dark-700 rounded p-2">
                    <div class="text-[10px] text-gray-500">ROE</div>
                    <div class="text-sm font-bold ${colorPct(d.roe)}">${fmt(d.roe, '%')}</div>
                </div>
                <div class="bg-dark-700 rounded p-2">
                    <div class="text-[10px] text-gray-500">D/E Ratio</div>
                    <div class="text-sm font-bold text-white">${fmt(d.de)}</div>
                </div>
                <div class="bg-dark-700 rounded p-2">
                    <div class="text-[10px] text-gray-500">Div Yield</div>
                    <div class="text-sm font-bold text-yellow-400">${fmt(d.div_yield, '%')}</div>
                </div>
                <div class="bg-dark-700 rounded p-2">
                    <div class="text-[10px] text-gray-500">Market Cap</div>
                    <div class="text-sm font-bold text-white">${fmtCr(d.market_cap)}</div>
                </div>
                <div class="bg-dark-700 rounded p-2">
                    <div class="text-[10px] text-gray-500">EPS (TTM)</div>
                    <div class="text-sm font-bold text-white">${d.eps_trailing != null ? '₹' + d.eps_trailing.toFixed(2) : '-'}</div>
                </div>
                <div class="bg-dark-700 rounded p-2">
                    <div class="text-[10px] text-gray-500">Book Value</div>
                    <div class="text-sm font-bold text-white">${d.book_value != null ? '₹' + d.book_value.toFixed(2) : '-'}</div>
                </div>
                <div class="bg-dark-700 rounded p-2">
                    <div class="text-[10px] text-gray-500">Beta</div>
                    <div class="text-sm font-bold text-white">${d.beta != null ? d.beta.toFixed(2) : '-'}</div>
                </div>
                <div class="bg-dark-700 rounded p-2">
                    <div class="text-[10px] text-gray-500">PEG Ratio</div>
                    <div class="text-sm font-bold text-white">${d.peg_ratio != null ? d.peg_ratio.toFixed(2) : '-'}</div>
                </div>
            </div>
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-2 text-center">
                <div class="bg-dark-700 rounded p-2">
                    <div class="text-[10px] text-gray-500">Profit Margin</div>
                    <div class="text-sm font-bold ${colorPct(d.profit_margin)}">${fmt(d.profit_margin, '%')}</div>
                </div>
                <div class="bg-dark-700 rounded p-2">
                    <div class="text-[10px] text-gray-500">Op. Margin</div>
                    <div class="text-sm font-bold ${colorPct(d.operating_margin)}">${fmt(d.operating_margin, '%')}</div>
                </div>
                <div class="bg-dark-700 rounded p-2">
                    <div class="text-[10px] text-gray-500">Rev Growth</div>
                    <div class="text-sm font-bold ${colorPct(d.rev_growth)}">${d.rev_growth ? (d.rev_growth > 0 ? '+' : '') + d.rev_growth.toFixed(1) + '%' : '-'}</div>
                </div>
                <div class="bg-dark-700 rounded p-2">
                    <div class="text-[10px] text-gray-500">Earn Growth</div>
                    <div class="text-sm font-bold ${colorPct(d.earn_growth)}">${d.earn_growth ? (d.earn_growth > 0 ? '+' : '') + d.earn_growth.toFixed(1) + '%' : '-'}</div>
                </div>
            </div>
            ${w52h != null && w52l != null ? this._render52WeekRange(w52l, w52h, d['50_day_avg'], d['200_day_avg']) : ''}
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-2 text-center">
                <div class="bg-dark-700 rounded p-2">
                    <div class="text-[10px] text-gray-500">Promoter Hold</div>
                    <div class="text-sm font-bold text-white">${fmt(d.promoter_holding, '%')}</div>
                </div>
                <div class="bg-dark-700 rounded p-2">
                    <div class="text-[10px] text-gray-500">Current Ratio</div>
                    <div class="text-sm font-bold text-white">${d.current_ratio != null ? d.current_ratio.toFixed(2) : '-'}</div>
                </div>
                <div class="bg-dark-700 rounded p-2">
                    <div class="text-[10px] text-gray-500">Free Cash Flow</div>
                    <div class="text-sm font-bold ${colorPct(d.free_cashflow)}">${fmtCr(d.free_cashflow)}</div>
                </div>
                <div class="bg-dark-700 rounded p-2">
                    <div class="text-[10px] text-gray-500">ROA</div>
                    <div class="text-sm font-bold ${colorPct(d.roa)}">${fmt(d.roa, '%')}</div>
                </div>
            </div>
        </div>`;
    },

    _render52WeekRange(low, high, avg50, avg200) {
        const range = high - low;
        const pctPos = range > 0 ? ((avg50 || low) - low) / range * 100 : 50;
        return `
        <div class="mt-3 px-2">
            <div class="flex justify-between text-[10px] text-gray-500 mb-1">
                <span>52W Low: ₹${low.toFixed(2)}</span>
                <span>52W High: ₹${high.toFixed(2)}</span>
            </div>
            <div class="relative h-1.5 bg-dark-600 rounded-full">
                <div class="absolute h-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 rounded-full" style="width:100%"></div>
                ${avg50 != null ? '<div class="absolute top-0 w-2 h-2 bg-blue-400 rounded-full -mt-0.5" style="left:' + Math.min(100, Math.max(0, pctPos)).toFixed(0) + '%" title="50d avg: ₹' + avg50.toFixed(2) + '"></div>' : ''}
            </div>
            <div class="flex justify-between text-[9px] text-gray-600 mt-1">
                ${avg50 != null ? '<span>50D: ₹' + avg50.toFixed(0) + '</span>' : '<span></span>'}
                ${avg200 != null ? '<span>200D: ₹' + avg200.toFixed(0) + '</span>' : '<span></span>'}
            </div>
        </div>`;
    },

    _renderFinancials(d) {
        const fmtCr = (v) => {
            if (v == null) return '-';
            const sign = v >= 0 ? '' : '-';
            const abs = Math.abs(v);
            if (abs >= 10000000) return sign + '₹' + (abs / 10000000).toFixed(1) + ' Cr';
            if (abs >= 100000) return sign + '₹' + (abs / 100000).toFixed(1) + ' L';
            return sign + '₹' + abs.toLocaleString('en-IN');
        };

        const annual = d.income_annual || [];
        const yearly = d.yearly_financials || [];
        if (annual.length === 0 && yearly.length === 0) return '';

        // Use income_annual if available, else yearly_financials
        let rows = '';
        if (annual.length > 0) {
            rows = annual.slice(0, 4).map(s => {
                const rev = s.revenue;
                const ni = s.net_income;
                const margin = (rev && ni) ? ((ni / rev) * 100).toFixed(1) + '%' : '-';
                const niColor = ni != null ? (ni >= 0 ? 'text-green-400' : 'text-red-400') : 'text-gray-500';
                return `<tr class="border-b border-gray-800">
                    <td class="py-1.5 px-2 text-xs text-gray-400">${s.date ? s.date.split('T')[0] : '-'}</td>
                    <td class="py-1.5 px-2 text-xs text-white text-right">${fmtCr(rev)}</td>
                    <td class="py-1.5 px-2 text-xs text-right ${niColor}">${fmtCr(ni)}</td>
                    <td class="py-1.5 px-2 text-xs text-right">${fmtCr(s.operating_income)}</td>
                    <td class="py-1.5 px-2 text-xs text-right text-gray-400">${margin}</td>
                </tr>`;
            }).join('');
        } else {
            rows = yearly.map(s => `<tr class="border-b border-gray-800">
                <td class="py-1.5 px-2 text-xs text-gray-400">${s.year || '-'}</td>
                <td class="py-1.5 px-2 text-xs text-white text-right">${fmtCr(s.revenue)}</td>
                <td class="py-1.5 px-2 text-xs text-right ${s.earnings >= 0 ? 'text-green-400' : 'text-red-400'}">${fmtCr(s.earnings)}</td>
                <td class="py-1.5 px-2 text-xs text-right text-gray-500">-</td>
                <td class="py-1.5 px-2 text-xs text-right text-gray-400">${(s.revenue && s.earnings) ? ((s.earnings / s.revenue) * 100).toFixed(1) + '%' : '-'}</td>
            </tr>`).join('');
        }

        return `
        <div class="bg-dark-800 rounded-lg p-3 sm:p-4 mb-3">
            <h4 class="text-sm font-semibold text-white mb-2">Profit & Loss (Annual)</h4>
            <div class="overflow-x-auto">
                <table class="w-full text-sm min-w-[400px]">
                    <thead><tr class="text-gray-400 border-b border-gray-700">
                        <th class="text-left px-2 py-1.5 text-xs">Period</th>
                        <th class="text-right px-2 py-1.5 text-xs">Revenue</th>
                        <th class="text-right px-2 py-1.5 text-xs">Net Profit</th>
                        <th class="text-right px-2 py-1.5 text-xs">Op. Income</th>
                        <th class="text-right px-2 py-1.5 text-xs">NPM</th>
                    </tr></thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
        </div>`;
    },

    _renderQuarterlyResults(d) {
        const quarterly = d.income_quarterly || [];
        const earnings = d.earnings_quarterly || [];
        if (quarterly.length === 0 && earnings.length === 0) return '';

        const fmtCr = (v) => {
            if (v == null) return '-';
            const sign = v >= 0 ? '' : '-';
            const abs = Math.abs(v);
            if (abs >= 10000000) return sign + '₹' + (abs / 10000000).toFixed(1) + ' Cr';
            if (abs >= 100000) return sign + '₹' + (abs / 100000).toFixed(1) + ' L';
            return sign + '₹' + abs.toLocaleString('en-IN');
        };

        let qRows = '';
        if (quarterly.length > 0) {
            qRows = quarterly.slice(0, 4).map(s => {
                const niColor = s.net_income != null ? (s.net_income >= 0 ? 'text-green-400' : 'text-red-400') : 'text-gray-500';
                return `<tr class="border-b border-gray-800">
                    <td class="py-1.5 px-2 text-xs text-gray-400">${s.date ? s.date.split('T')[0] : '-'}</td>
                    <td class="py-1.5 px-2 text-xs text-white text-right">${fmtCr(s.revenue)}</td>
                    <td class="py-1.5 px-2 text-xs text-right ${niColor}">${fmtCr(s.net_income)}</td>
                    <td class="py-1.5 px-2 text-xs text-right">${fmtCr(s.operating_income)}</td>
                </tr>`;
            }).join('');
        }

        let earningsRows = '';
        if (earnings.length > 0) {
            earningsRows = earnings.slice(0, 4).map(e => {
                const surprise = e.surprise_pct != null ? (e.surprise_pct * 100) : null;
                const surpriseColor = surprise != null ? (surprise >= 0 ? 'text-green-400' : 'text-red-400') : 'text-gray-500';
                return `<tr class="border-b border-gray-800">
                    <td class="py-1.5 px-2 text-xs text-gray-400">${e.date || '-'}</td>
                    <td class="py-1.5 px-2 text-xs text-white text-right">${e.actual_eps != null ? '₹' + e.actual_eps.toFixed(2) : '-'}</td>
                    <td class="py-1.5 px-2 text-xs text-gray-400 text-right">${e.estimate_eps != null ? '₹' + e.estimate_eps.toFixed(2) : '-'}</td>
                    <td class="py-1.5 px-2 text-xs text-right ${surpriseColor}">${surprise != null ? (surprise >= 0 ? '+' : '') + surprise.toFixed(1) + '%' : '-'}</td>
                </tr>`;
            }).join('');
        }

        return `
        <div class="bg-dark-800 rounded-lg p-3 sm:p-4 mb-3">
            <h4 class="text-sm font-semibold text-white mb-2">Quarterly Results</h4>
            ${qRows ? `<div class="overflow-x-auto mb-3">
                <table class="w-full text-sm min-w-[350px]">
                    <thead><tr class="text-gray-400 border-b border-gray-700">
                        <th class="text-left px-2 py-1.5 text-xs">Quarter</th>
                        <th class="text-right px-2 py-1.5 text-xs">Revenue</th>
                        <th class="text-right px-2 py-1.5 text-xs">Net Profit</th>
                        <th class="text-right px-2 py-1.5 text-xs">Op. Income</th>
                    </tr></thead>
                    <tbody>${qRows}</tbody>
                </table>
            </div>` : ''}
            ${earningsRows ? `<div class="overflow-x-auto">
                <h5 class="text-xs text-gray-400 mb-1">EPS History</h5>
                <table class="w-full text-sm min-w-[300px]">
                    <thead><tr class="text-gray-400 border-b border-gray-700">
                        <th class="text-left px-2 py-1.5 text-xs">Quarter</th>
                        <th class="text-right px-2 py-1.5 text-xs">Actual EPS</th>
                        <th class="text-right px-2 py-1.5 text-xs">Est. EPS</th>
                        <th class="text-right px-2 py-1.5 text-xs">Surprise</th>
                    </tr></thead>
                    <tbody>${earningsRows}</tbody>
                </table>
            </div>` : ''}
        </div>`;
    },

    _renderBalanceSheet(d) {
        const bs = d.balance_annual || [];
        if (bs.length === 0) return '';

        const fmtCr = (v) => {
            if (v == null) return '-';
            const sign = v >= 0 ? '' : '-';
            const abs = Math.abs(v);
            if (abs >= 10000000) return sign + '₹' + (abs / 10000000).toFixed(1) + ' Cr';
            if (abs >= 100000) return sign + '₹' + (abs / 100000).toFixed(1) + ' L';
            return sign + '₹' + abs.toLocaleString('en-IN');
        };

        const rows = bs.slice(0, 3).map(s => `<tr class="border-b border-gray-800">
            <td class="py-1.5 px-2 text-xs text-gray-400">${s.date ? s.date.split('T')[0] : '-'}</td>
            <td class="py-1.5 px-2 text-xs text-white text-right">${fmtCr(s.total_assets)}</td>
            <td class="py-1.5 px-2 text-xs text-red-400 text-right">${fmtCr(s.total_liabilities)}</td>
            <td class="py-1.5 px-2 text-xs text-green-400 text-right">${fmtCr(s.total_equity)}</td>
            <td class="py-1.5 px-2 text-xs text-right">${fmtCr(s.total_debt)}</td>
        </tr>`).join('');

        return `
        <div class="bg-dark-800 rounded-lg p-3 sm:p-4 mb-3">
            <h4 class="text-sm font-semibold text-white mb-2">Balance Sheet (Net Worth)</h4>
            <div class="overflow-x-auto">
                <table class="w-full text-sm min-w-[400px]">
                    <thead><tr class="text-gray-400 border-b border-gray-700">
                        <th class="text-left px-2 py-1.5 text-xs">Period</th>
                        <th class="text-right px-2 py-1.5 text-xs">Total Assets</th>
                        <th class="text-right px-2 py-1.5 text-xs">Liabilities</th>
                        <th class="text-right px-2 py-1.5 text-xs">Equity (Net Worth)</th>
                        <th class="text-right px-2 py-1.5 text-xs">Total Debt</th>
                    </tr></thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
        </div>`;
    },

    _renderNews(news) {
        if (!news || !news.headlines || news.headlines.length === 0) {
            return `<div class="bg-dark-800 rounded-lg p-3 sm:p-4 mb-3">
                <h4 class="text-sm font-semibold text-white mb-2">Latest News</h4>
                <div class="text-gray-500 text-sm py-2">No recent news</div>
            </div>`;
        }

        const sentColor = (s) => s === 'positive' ? 'text-green-400 bg-green-900' :
                                  s === 'negative' ? 'text-red-400 bg-red-900' : 'text-gray-400 bg-gray-800';

        const scoreBadge = news.score != null ?
            `<span class="text-xs px-2 py-0.5 rounded-full ${news.score >= 10 ? 'bg-green-900 text-green-400' : news.score <= -10 ? 'bg-red-900 text-red-400' : 'bg-gray-800 text-gray-400'}">
                Sentiment: ${news.score > 0 ? '+' : ''}${news.score.toFixed(0)}
            </span>` : '';

        const headlines = news.headlines.slice(0, 10).map(h => `
            <div class="flex items-start gap-2 py-1.5 border-b border-gray-800 last:border-0">
                <span class="text-[10px] px-1.5 py-0.5 rounded ${sentColor(h.sentiment)} whitespace-nowrap flex-shrink-0">${h.sentiment}</span>
                <div class="flex-1 min-w-0">
                    <a href="${h.link}" target="_blank" rel="noopener"
                       class="text-xs text-gray-300 hover:text-white line-clamp-2">${h.title}</a>
                    <div class="flex items-center gap-2 mt-0.5">
                        ${h.source ? '<span class="text-[9px] text-gray-600">' + h.source + '</span>' : ''}
                        ${h.published ? '<span class="text-[9px] text-gray-600">' + h.published.split(',')[0] + '</span>' : ''}
                    </div>
                </div>
            </div>
        `).join('');

        return `
        <div class="bg-dark-800 rounded-lg p-3 sm:p-4 mb-3">
            <div class="flex items-center justify-between mb-2">
                <h4 class="text-sm font-semibold text-white">Latest News</h4>
                <div class="flex items-center gap-2">
                    <span class="text-[10px] text-gray-500">${news.headline_count} articles</span>
                    ${scoreBadge}
                </div>
            </div>
            <div class="flex gap-3 mb-2">
                <span class="text-[10px] text-green-400">${news.positive_count || 0} positive</span>
                <span class="text-[10px] text-gray-400">${news.neutral_count || 0} neutral</span>
                <span class="text-[10px] text-red-400">${news.negative_count || 0} negative</span>
            </div>
            <div class="max-h-72 overflow-y-auto">${headlines}</div>
        </div>`;
    },
};
