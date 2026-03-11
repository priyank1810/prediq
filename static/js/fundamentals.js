const Fundamentals = {
    _currentSymbol: null,

    async load(symbol) {
        this._currentSymbol = symbol;
        const panel = document.getElementById('fundamentalsPanel');
        if (!panel) return;
        Shimmer.show('fundamentalsPanel', 'fundamentals');
        Shimmer.show('newsTabFundNews', 'news', 4);
        Shimmer.show('overviewKeyMetrics', 'keyMetrics');

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
        `;

        // Overview tab: key metrics summary
        const overviewMetrics = document.getElementById('overviewKeyMetrics');
        if (overviewMetrics) {
            overviewMetrics.innerHTML = this._renderKeyMetrics(data);
        }

        // News tab: fundamental-sourced news
        const newsTab = document.getElementById('newsTabFundNews');
        if (newsTab) {
            newsTab.innerHTML = this._renderNews(news);
        }
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

    // ── Shared chart helpers ──

    _fmtCr(v) {
        if (v == null) return '-';
        const sign = v >= 0 ? '' : '-';
        const abs = Math.abs(v);
        if (abs >= 10000000) return sign + '₹' + (abs / 10000000).toFixed(1) + 'Cr';
        if (abs >= 100000) return sign + '₹' + (abs / 100000).toFixed(1) + 'L';
        return sign + '₹' + abs.toLocaleString('en-IN');
    },

    _barChart(items, maxVal, colorPositive, colorNegative) {
        // items: [{label, value}]
        if (!maxVal || maxVal === 0) maxVal = 1;
        return items.map(item => {
            const pct = Math.min(100, (Math.abs(item.value || 0) / maxVal) * 100);
            const isNeg = (item.value || 0) < 0;
            const color = isNeg ? (colorNegative || 'bg-red-500') : (colorPositive || 'bg-blue-500');
            return `<div class="flex items-center gap-2 py-1">
                <div class="w-16 sm:w-20 text-[10px] text-gray-400 text-right flex-shrink-0 truncate">${item.label}</div>
                <div class="flex-1 h-5 bg-dark-600 rounded overflow-hidden relative">
                    <div class="${color} h-full rounded transition-all" style="width:${pct}%"></div>
                </div>
                <div class="w-16 sm:w-20 text-[10px] ${isNeg ? 'text-red-400' : 'text-gray-300'} text-right flex-shrink-0">${this._fmtCr(item.value)}</div>
            </div>`;
        }).join('');
    },

    _groupedBarChart(groups) {
        // groups: [{label, bars: [{value, color, name}]}]
        const allVals = groups.flatMap(g => g.bars.map(b => Math.abs(b.value || 0)));
        const maxVal = Math.max(...allVals, 1);

        return `<div class="space-y-3">
            ${groups.map(g => `
                <div>
                    <div class="text-[10px] text-gray-500 mb-1">${g.label}</div>
                    ${g.bars.map(b => {
                        const pct = Math.min(100, (Math.abs(b.value || 0) / maxVal) * 100);
                        const isNeg = (b.value || 0) < 0;
                        return `<div class="flex items-center gap-2 py-0.5">
                            <div class="w-14 sm:w-16 text-[9px] text-gray-500 text-right flex-shrink-0">${b.name}</div>
                            <div class="flex-1 h-4 bg-dark-600 rounded overflow-hidden">
                                <div class="${b.color} h-full rounded transition-all" style="width:${pct}%"></div>
                            </div>
                            <div class="w-16 sm:w-20 text-[10px] ${isNeg ? 'text-red-400' : 'text-gray-300'} text-right flex-shrink-0">${this._fmtCr(b.value)}</div>
                        </div>`;
                    }).join('')}
                </div>
            `).join('')}
        </div>`;
    },

    // ── P&L (Annual) ──

    _renderFinancials(d) {
        const annual = d.income_annual || [];
        const yearly = d.yearly_financials || [];
        if (annual.length === 0 && yearly.length === 0) return '';

        let items = [];
        if (annual.length > 0) {
            items = annual.slice(0, 5).reverse().map(s => ({
                label: s.date ? s.date.split('T')[0].slice(0, 7) : '-',
                revenue: s.revenue,
                profit: s.net_income,
                opIncome: s.operating_income,
            }));
        } else {
            items = yearly.map(s => ({
                label: s.year || '-',
                revenue: s.revenue,
                profit: s.earnings,
                opIncome: null,
            }));
        }

        const maxRev = Math.max(...items.map(i => Math.abs(i.revenue || 0)), 1);

        return `
        <div class="bg-dark-800 rounded-lg p-3 sm:p-4 mb-3">
            <h4 class="text-sm font-semibold text-white mb-1">Profit & Loss (Annual)</h4>
            <div class="flex gap-3 mb-2">
                <span class="flex items-center gap-1 text-[9px] text-gray-400"><span class="w-2 h-2 rounded bg-blue-500 inline-block"></span>Revenue</span>
                <span class="flex items-center gap-1 text-[9px] text-gray-400"><span class="w-2 h-2 rounded bg-green-500 inline-block"></span>Net Profit</span>
                ${items.some(i => i.opIncome != null) ? '<span class="flex items-center gap-1 text-[9px] text-gray-400"><span class="w-2 h-2 rounded bg-purple-500 inline-block"></span>Op. Income</span>' : ''}
            </div>
            ${this._groupedBarChart(items.map(i => ({
                label: i.label,
                bars: [
                    { value: i.revenue, color: 'bg-blue-500', name: 'Rev' },
                    { value: i.profit, color: i.profit >= 0 ? 'bg-green-500' : 'bg-red-500', name: 'Profit' },
                    ...(i.opIncome != null ? [{ value: i.opIncome, color: 'bg-purple-500', name: 'OpInc' }] : []),
                ],
            })))}
            ${items.length > 0 ? `<div class="flex justify-around mt-2 border-t border-gray-700 pt-2">
                ${items.map(i => {
                    const margin = (i.revenue && i.profit) ? ((i.profit / i.revenue) * 100).toFixed(1) : null;
                    return `<div class="text-center">
                        <div class="text-[9px] text-gray-500">${i.label}</div>
                        <div class="text-[10px] ${margin != null && margin >= 0 ? 'text-green-400' : 'text-red-400'}">${margin != null ? margin + '% NPM' : '-'}</div>
                    </div>`;
                }).join('')}
            </div>` : ''}
        </div>`;
    },

    // ── Quarterly Results ──

    _renderQuarterlyResults(d) {
        const quarterly = d.income_quarterly || [];
        const earnings = d.earnings_quarterly || [];
        if (quarterly.length === 0 && earnings.length === 0) return '';

        let qChart = '';
        if (quarterly.length > 0) {
            const items = quarterly.slice(0, 5).reverse();
            qChart = `
                <div class="mb-3">
                    <div class="flex gap-3 mb-2">
                        <span class="flex items-center gap-1 text-[9px] text-gray-400"><span class="w-2 h-2 rounded bg-blue-500 inline-block"></span>Revenue</span>
                        <span class="flex items-center gap-1 text-[9px] text-gray-400"><span class="w-2 h-2 rounded bg-green-500 inline-block"></span>Net Profit</span>
                    </div>
                    ${this._groupedBarChart(items.map(s => ({
                        label: s.date ? s.date.split('T')[0].slice(0, 7) : '-',
                        bars: [
                            { value: s.revenue, color: 'bg-blue-500', name: 'Rev' },
                            { value: s.net_income, color: s.net_income >= 0 ? 'bg-green-500' : 'bg-red-500', name: 'Profit' },
                        ],
                    })))}
                </div>`;
        }

        let epsChart = '';
        if (earnings.length > 0) {
            const items = earnings.slice(0, 5).reverse();
            const maxEps = Math.max(...items.map(e => Math.max(Math.abs(e.actual_eps || 0), Math.abs(e.estimate_eps || 0))), 0.01);

            epsChart = `
                <div class="border-t border-gray-700 pt-2">
                    <h5 class="text-xs text-gray-400 mb-2">EPS: Actual vs Estimate</h5>
                    <div class="flex gap-3 mb-2">
                        <span class="flex items-center gap-1 text-[9px] text-gray-400"><span class="w-2 h-2 rounded bg-cyan-500 inline-block"></span>Actual</span>
                        <span class="flex items-center gap-1 text-[9px] text-gray-400"><span class="w-2 h-2 rounded bg-gray-500 inline-block"></span>Estimate</span>
                    </div>
                    ${items.map(e => {
                        const actPct = Math.min(100, (Math.abs(e.actual_eps || 0) / maxEps) * 100);
                        const estPct = Math.min(100, (Math.abs(e.estimate_eps || 0) / maxEps) * 100);
                        const surprise = e.surprise_pct != null ? (e.surprise_pct * 100) : null;
                        const surpriseColor = surprise != null ? (surprise >= 0 ? 'text-green-400' : 'text-red-400') : '';
                        return `<div class="mb-2">
                            <div class="text-[10px] text-gray-500 mb-0.5">${e.date || '-'}</div>
                            <div class="flex items-center gap-2">
                                <div class="flex-1 relative h-8 bg-dark-600 rounded overflow-hidden">
                                    <div class="bg-gray-600/50 h-4 rounded-t" style="width:${estPct}%"></div>
                                    <div class="bg-cyan-500 h-4 rounded-b" style="width:${actPct}%"></div>
                                </div>
                                <div class="w-20 text-right flex-shrink-0">
                                    <div class="text-[10px] text-cyan-400">₹${e.actual_eps != null ? e.actual_eps.toFixed(2) : '-'}</div>
                                    ${surprise != null ? `<div class="text-[9px] ${surpriseColor}">${surprise >= 0 ? '+' : ''}${surprise.toFixed(1)}%</div>` : ''}
                                </div>
                            </div>
                        </div>`;
                    }).join('')}
                </div>`;
        }

        return `
        <div class="bg-dark-800 rounded-lg p-3 sm:p-4 mb-3">
            <h4 class="text-sm font-semibold text-white mb-1">Quarterly Results</h4>
            ${qChart}
            ${epsChart}
        </div>`;
    },

    // ── Balance Sheet ──

    _renderBalanceSheet(d) {
        const bs = d.balance_annual || [];
        if (bs.length === 0) return '';

        const items = bs.slice(0, 4).reverse();

        return `
        <div class="bg-dark-800 rounded-lg p-3 sm:p-4 mb-3">
            <h4 class="text-sm font-semibold text-white mb-1">Balance Sheet</h4>
            <div class="flex gap-3 mb-2">
                <span class="flex items-center gap-1 text-[9px] text-gray-400"><span class="w-2 h-2 rounded bg-blue-500 inline-block"></span>Assets</span>
                <span class="flex items-center gap-1 text-[9px] text-gray-400"><span class="w-2 h-2 rounded bg-red-500 inline-block"></span>Liabilities</span>
                <span class="flex items-center gap-1 text-[9px] text-gray-400"><span class="w-2 h-2 rounded bg-green-500 inline-block"></span>Equity</span>
                <span class="flex items-center gap-1 text-[9px] text-gray-400"><span class="w-2 h-2 rounded bg-yellow-500 inline-block"></span>Debt</span>
            </div>
            ${this._groupedBarChart(items.map(s => ({
                label: s.date ? s.date.split('T')[0].slice(0, 7) : '-',
                bars: [
                    { value: s.total_assets, color: 'bg-blue-500', name: 'Assets' },
                    { value: s.total_liabilities, color: 'bg-red-500', name: 'Liab' },
                    { value: s.total_equity, color: 'bg-green-500', name: 'Equity' },
                    { value: s.total_debt, color: 'bg-yellow-500', name: 'Debt' },
                ],
            })))}
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
