const Insights = {
    _seeded: false,
    _lastLoadTime: 0,

    init() {},

    async load() {
        await this.loadPortfolio();
        await this.loadTrackRecord();
    },

    async loadPortfolio() {
        if (this._portfolioLoadTime && Date.now() - this._portfolioLoadTime < 30000) return;
        this._portfolioLoadTime = Date.now();
        await this.loadVirtualPortfolio();
    },

    async loadTrackRecord() {
        if (this._trackRecordLoadTime && Date.now() - this._trackRecordLoadTime < 30000) return;
        this._trackRecordLoadTime = Date.now();
        await Promise.all([
            this.loadPredictionLeaderboard(),
            this.loadTradeTrackRecord(),
        ]);
    },


    // ─── Trade Prediction Track Record ──────────────────────────

    async loadVirtualPortfolio() {
        try {
            const resp = await fetch(`${API.baseUrl}/api/signals/stats/virtual-portfolio?capital=100000`);
            if (!resp.ok) return;
            const data = await resp.json();

            // Summary cards
            const cv = data.current_value || 10000;
            const pnl = data.total_pnl || 0;
            const pnlPct = data.total_pnl_pct || 0;
            const pnlColor = pnl >= 0 ? 'text-green-400' : 'text-red-400';
            const pnlSign = pnl >= 0 ? '+' : '';

            document.getElementById('vpCurrentValue').textContent = `₹${cv.toLocaleString('en-IN')}`;
            document.getElementById('vpCurrentValue').className = `text-lg font-bold ${pnlColor}`;

            const pnlEl = document.getElementById('vpTotalPnl');
            pnlEl.textContent = `${pnlSign}₹${Math.abs(pnl).toLocaleString('en-IN')}`;
            pnlEl.className = `text-lg font-bold ${pnlColor}`;

            const retEl = document.getElementById('vpReturnPct');
            retEl.textContent = `${pnlSign}${pnlPct}%`;
            retEl.className = `text-lg font-bold ${pnlColor}`;

            const wrEl = document.getElementById('vpWinRate');
            const wr = data.win_rate || 0;
            wrEl.textContent = `${wr}%`;
            wrEl.className = `text-lg font-bold ${wr >= 50 ? 'text-green-400' : wr > 0 ? 'text-red-400' : 'text-white'}`;

            document.getElementById('vpPerTrade').textContent = `₹${(data.per_trade_allocation || 2000).toLocaleString('en-IN')}`;
            document.getElementById('vpTotalTrades').textContent = `${data.total_trades || 0} trades | ${(data.open_positions || []).length} open | ${data.skipped_signals || 0} skipped`;

            // Open positions
            const openEl = document.getElementById('vpOpenPositions');
            const positions = data.open_positions || [];
            if (positions.length > 0) {
                openEl.innerHTML = `
                    <h4 class="text-xs font-medium text-white mb-1">Open Positions</h4>
                    <div class="flex flex-wrap gap-2">
                        ${positions.map(p => {
                            const confC = (p.confidence || 0) >= 70 ? 'text-green-400' : 'text-yellow-400';
                            return `<div class="bg-dark-700 rounded px-2.5 py-1.5 text-xs cursor-pointer hover:bg-dark-600" onclick="Search.select('${p.symbol}','')">
                                <span class="text-white font-medium">${p.symbol}</span>
                                <span class="text-gray-500 ml-1">${p.qty}×₹${p.entry.toFixed(0)}</span>
                                <span class="text-gray-500 ml-1">→ ₹${p.target ? p.target.toFixed(0) : '-'}</span>
                                <span class="${confC} ml-1">${(p.confidence || 0).toFixed(0)}%</span>
                            </div>`;
                        }).join('')}
                    </div>`;
            } else {
                openEl.innerHTML = '';
            }

            // Near-bullish suggestions
            this._loadNearBullish();

            // Equity curve chart
            this._renderEquityCurve(data.equity_curve || []);

            // Per-stock breakdown
            const stockEl = document.getElementById('vpStockBreakdown');
            const stocks = data.stock_summary || [];
            if (stocks.length > 0) {
                stockEl.innerHTML = `
                    <h4 class="text-xs font-medium text-white mb-1">P&L by Stock</h4>
                    <div class="grid grid-cols-2 sm:grid-cols-4 gap-2">
                        ${stocks.map(s => {
                            const wrColor = s.win_rate >= 60 ? 'text-green-400' : s.win_rate >= 40 ? 'text-yellow-400' : 'text-red-400';
                            return `<div class="bg-dark-700 rounded p-2 text-center cursor-pointer hover:bg-dark-600" onclick="Search.select('${s.symbol}','')">
                                <div class="text-xs text-white font-medium">${s.symbol}</div>
                                <div class="text-sm font-bold ${wrColor}">${s.win_rate}%</div>
                                <div class="text-[10px] text-gray-500">${s.trades} trades</div>
                            </div>`;
                        }).join('')}
                    </div>`;
            } else {
                stockEl.innerHTML = '';
            }

            // Daily P&L
            const dailyEl = document.getElementById('vpDailyPnl');
            const dailyData = data.daily_pnl || [];
            if (dailyData.length > 0) {
                dailyEl.innerHTML = `
                    <h4 class="text-xs font-medium text-white mb-1">Daily P&L</h4>
                    <div class="flex flex-wrap gap-1">
                        ${dailyData.slice(-20).map(d => {
                            const c = d.pnl >= 0 ? 'bg-green-900/50 text-green-400' : 'bg-red-900/50 text-red-400';
                            const s = d.pnl >= 0 ? '+' : '';
                            return `<div class="rounded px-2 py-1 text-[10px] ${c}">${d.date.slice(5)} ${s}₹${Math.abs(d.pnl).toFixed(0)}</div>`;
                        }).join('')}
                    </div>`;
            } else {
                dailyEl.innerHTML = '';
            }

            // Best / Worst trade
            const bwEl = document.getElementById('vpBestWorst');
            const best = data.best_trade;
            const worst = data.worst_trade;
            if (best || worst) {
                const renderTrade = (label, t, color) => {
                    if (!t) return '';
                    const s = t.pnl >= 0 ? '+' : '';
                    return `<div class="bg-dark-700 rounded p-2 flex-1">
                        <div class="text-[10px] text-gray-500 mb-0.5">${label}</div>
                        <div class="text-xs"><span class="text-white font-medium">${t.symbol}</span> <span class="${color} font-bold">${s}₹${Math.abs(t.pnl).toFixed(0)}</span> <span class="text-gray-500">(${s}${t.pnl_pct}%)</span></div>
                    </div>`;
                };
                bwEl.innerHTML = `<div class="flex gap-2">${renderTrade('Best Trade', best, 'text-green-400')}${renderTrade('Worst Trade', worst, 'text-red-400')}</div>`;
            } else {
                bwEl.innerHTML = '';
            }

            // All trades table
            const trades = data.recent_trades || [];
            const tbody = document.getElementById('vpTradesTable');
            document.getElementById('vpTradeCount').textContent = `${trades.length} trades`;

            if (trades.length === 0) {
                tbody.innerHTML = '<tr><td colspan="10" class="text-center py-4 text-gray-500">No trades yet. Virtual portfolio follows bullish signals from your watchlist.</td></tr>';
                return;
            }

            const tfShort = { intraday_10m: '10m', intraday_30m: '30m', short_15m: '15m', short_1h: '1h', short_4h: '4h' };

            tbody.innerHTML = trades.reverse().map(t => {
                const pColor = t.pnl >= 0 ? 'text-green-400' : 'text-red-400';
                const sign = t.pnl >= 0 ? '+' : '';
                const rowBg = t.status === 'target_hit' ? 'bg-green-900/10 border-l-2 border-l-green-500' :
                              t.status === 'sl_hit' ? 'bg-red-900/10 border-l-2 border-l-red-500' :
                              'border-l-2 border-l-gray-700';
                let badge = '';
                if (t.status === 'target_hit') badge = '<span class="px-1.5 py-0.5 rounded bg-green-900/50 text-green-400 text-[10px] font-medium">✓ Win</span>';
                else if (t.status === 'sl_hit') badge = '<span class="px-1.5 py-0.5 rounded bg-red-900/50 text-red-400 text-[10px] font-medium">✗ Loss</span>';
                else badge = `<span class="px-1.5 py-0.5 rounded bg-gray-800 ${t.pnl >= 0 ? 'text-green-400' : 'text-red-400'} text-[10px]">Expired</span>`;

                const confColor = (t.confidence || 0) >= 70 ? 'text-green-400' : (t.confidence || 0) >= 50 ? 'text-yellow-400' : 'text-gray-500';

                return `<tr class="${rowBg} hover:bg-dark-700/50">
                    <td class="px-2 py-1.5 text-gray-400">${t.date || '-'}</td>
                    <td class="px-2 py-1.5 text-white font-medium cursor-pointer" onclick="Search.select('${t.symbol}','')">${t.symbol}</td>
                    <td class="px-2 py-1.5 text-gray-400">${tfShort[t.timeframe] || t.timeframe}</td>
                    <td class="px-2 py-1.5 text-right ${confColor}">${(t.confidence || 0).toFixed(0)}%</td>
                    <td class="px-2 py-1.5 text-right text-gray-300">${t.qty}</td>
                    <td class="px-2 py-1.5 text-right text-gray-300">₹${t.entry.toFixed(2)}</td>
                    <td class="px-2 py-1.5 text-right text-gray-300">₹${t.exit_price.toFixed(2)}</td>
                    <td class="px-2 py-1.5 text-right text-gray-300">₹${t.invested.toFixed(0)}</td>
                    <td class="px-2 py-1.5 text-right ${pColor} font-bold">${sign}₹${Math.abs(t.pnl).toFixed(0)}</td>
                    <td class="px-2 py-1.5 text-center">${badge}</td>
                </tr>`;
            }).join('');
        } catch (e) {
            // Silently fail
        }
    },

    async _loadNearBullish() {
        const el = document.getElementById('vpNearBullish');
        if (!el) return;
        try {
            const resp = await fetch(`${API.baseUrl}/api/signals/stats/near-bullish`);
            if (!resp.ok) { el.innerHTML = ''; return; }
            const stocks = await resp.json();
            if (!stocks || stocks.length === 0) { el.innerHTML = ''; return; }

            el.innerHTML = `
                <h4 class="text-xs font-medium text-yellow-400 mb-1">Stocks To Watch (Near Bullish)</h4>
                <div class="flex flex-wrap gap-2">
                    ${stocks.map(s => {
                        const scoreColor = s.score > 5 ? 'text-green-400' : 'text-yellow-400';
                        return `<div class="bg-dark-700 border border-yellow-800/30 rounded px-2.5 py-1.5 text-xs cursor-pointer hover:bg-dark-600" onclick="Search.select('${s.symbol}','')">
                            <span class="text-white font-medium">${s.symbol}</span>
                            <span class="${scoreColor} ml-1">+${s.score.toFixed(1)}</span>
                            <span class="text-gray-500 ml-1 text-[10px]">${s.label}</span>
                        </div>`;
                    }).join('')}
                </div>
                <p class="text-[10px] text-gray-500 mt-1">These stocks have positive momentum but haven't crossed the bullish threshold yet.</p>
            `;
        } catch (e) {
            el.innerHTML = '';
        }
    },

    _renderEquityCurve(curve) {
        const container = document.getElementById('vpEquityChart');
        if (!container || !curve || curve.length < 2) {
            container.innerHTML = '<div class="text-center py-8 text-gray-500 text-xs">Not enough data for equity curve</div>';
            return;
        }

        try {
            // Clean up previous chart
            container.innerHTML = '';

            const chart = LightweightCharts.createChart(container, {
                width: container.clientWidth,
                height: 200,
                layout: { background: { color: '#1a1d29' }, textColor: '#9ca3af' },
                grid: { vertLines: { color: '#2d2d44' }, horzLines: { color: '#2d2d44' } },
                rightPriceScale: { borderColor: '#374151' },
                timeScale: { borderColor: '#374151' },
            });

            const lineSeries = chart.addLineSeries({
                color: '#2979ff',
                lineWidth: 2,
                priceFormat: { type: 'custom', formatter: (p) => '₹' + p.toFixed(0) },
            });

            // Convert dates to timestamps
            const data = curve.filter(p => p.date && p.date !== 'start').map(p => ({
                time: p.date,
                value: p.value,
            }));

            if (data.length > 0) {
                lineSeries.setData(data);
                chart.timeScale().fitContent();

                // Add baseline at initial capital
                const baseline = chart.addLineSeries({
                    color: '#374151', lineWidth: 1, lineStyle: 2,
                    priceLineVisible: false, lastValueVisible: false,
                });
                baseline.setData(data.map(d => ({ time: d.time, value: curve[0].value || 10000 })));
            }

            // Resize handler
            const ro = new ResizeObserver(() => {
                chart.applyOptions({ width: container.clientWidth });
            });
            ro.observe(container);
        } catch (e) {
            container.innerHTML = '<div class="text-center py-8 text-gray-500 text-xs">Chart unavailable</div>';
        }
    },

    async loadPredictionLeaderboard() {
        try {
            const resp = await fetch(`${API.baseUrl}/api/signals/stats/prediction-leaderboard`);
            if (!resp.ok) return;
            const data = await resp.json();

            // Model cards
            const modelsEl = document.getElementById('predLeaderboardModels');
            if (modelsEl && data.models) {
                modelsEl.innerHTML = data.models.map((m, i) => {
                    const isBest = i === 0;
                    const border = isBest ? 'border-green-600' : 'border-gray-700';
                    const mapeColor = m.avg_mape <= 3 ? 'text-green-400' : m.avg_mape <= 6 ? 'text-yellow-400' : 'text-red-400';
                    const wrColor = m.win_rate >= 60 ? 'text-green-400' : m.win_rate >= 40 ? 'text-yellow-400' : 'text-red-400';
                    return `<div class="bg-dark-700 border ${border} rounded-lg p-4">
                        <div class="flex items-center gap-2 mb-2">
                            <span class="text-white font-bold capitalize">${m.model}</span>
                            ${isBest ? '<span class="text-[10px] px-1.5 py-0.5 rounded bg-green-900 text-green-400 font-medium">BEST</span>' : ''}
                        </div>
                        <div class="grid grid-cols-3 gap-2 text-center">
                            <div><div class="text-[10px] text-gray-500">Avg MAPE</div><div class="text-sm font-bold ${mapeColor}">${m.avg_mape}%</div></div>
                            <div><div class="text-[10px] text-gray-500">Win Rate</div><div class="text-sm font-bold ${wrColor}">${m.win_rate}%</div></div>
                            <div><div class="text-[10px] text-gray-500">Predictions</div><div class="text-sm font-bold text-white">${m.total}</div></div>
                        </div>
                    </div>`;
                }).join('');
            }

            // By symbol
            const bySymEl = document.getElementById('predLeaderboardBySymbol');
            if (bySymEl && data.by_symbol) {
                bySymEl.innerHTML = data.by_symbol.slice(0, 15).map(s => {
                    const mapeColor = s.avg_mape <= 1 ? 'text-green-400' : s.avg_mape <= 3 ? 'text-yellow-400' : 'text-red-400';
                    return `<div class="flex items-center justify-between py-1 border-b border-gray-800">
                        <div class="flex items-center gap-2">
                            <span class="text-[9px] px-1 py-0.5 rounded bg-dark-600 text-gray-400 uppercase">${s.model}</span>
                            <span class="text-xs text-white cursor-pointer hover:text-accent-blue" onclick="Search.select('${s.symbol}','')">${s.symbol}</span>
                        </div>
                        <div class="flex items-center gap-2">
                            <span class="text-[10px] text-gray-500">${s.total}x</span>
                            <span class="text-xs font-bold ${mapeColor}">${s.avg_mape}%</span>
                        </div>
                    </div>`;
                }).join('') || '<div class="text-center py-2 text-gray-500 text-xs">No data</div>';
            }

            // By sector
            const bySectorEl = document.getElementById('predLeaderboardBySector');
            if (bySectorEl && data.by_sector) {
                bySectorEl.innerHTML = data.by_sector.slice(0, 10).map(s => {
                    const mapeColor = s.avg_mape <= 1 ? 'text-green-400' : s.avg_mape <= 3 ? 'text-yellow-400' : 'text-red-400';
                    return `<div class="flex items-center justify-between py-1 border-b border-gray-800">
                        <div class="flex items-center gap-2">
                            <span class="text-[9px] px-1 py-0.5 rounded bg-dark-600 text-gray-400 uppercase">${s.model}</span>
                            <span class="text-xs text-white">${s.sector}</span>
                        </div>
                        <div class="flex items-center gap-2">
                            <span class="text-[10px] text-gray-500">${s.total}x</span>
                            <span class="text-xs font-bold ${mapeColor}">${s.avg_mape}%</span>
                        </div>
                    </div>`;
                }).join('') || '<div class="text-center py-2 text-gray-500 text-xs">No data</div>';
            }
        } catch (e) {
            // Silently fail
        }
    },

    async loadTradeTrackRecord() {
        try {
            const resp = await fetch(`${API.baseUrl}/api/signals/stats/trades`);
            if (!resp.ok) return;
            const data = await resp.json();
            this.renderTradeTrackRecord(data);
        } catch (e) {
            // Silently fail if no trade data yet
        }
    },

    renderTradeTrackRecord(data) {
        if (!data || data.total === 0) {
            document.getElementById('tradeRecentTable').innerHTML =
                '<tr><td colspan="8" class="text-center py-4 text-gray-500">No trade predictions resolved yet. Predictions are tracked automatically from your watchlist.</td></tr>';
            return;
        }

        // Summary cards
        const winRateEl = document.getElementById('tradeWinRate');
        const wr = data.win_rate || 0;
        winRateEl.textContent = `${wr}%`;
        winRateEl.className = `text-xl font-bold ${wr >= 60 ? 'text-green-400' : wr >= 45 ? 'text-yellow-400' : 'text-red-400'}`;

        document.getElementById('tradeTargetHits').textContent = data.target_hit || 0;
        document.getElementById('tradeSLHits').textContent = data.sl_hit || 0;

        const avgPnlEl = document.getElementById('tradeAvgPnl');
        avgPnlEl.innerHTML = `<span class="text-green-400">+${data.avg_win_pct || 0}%</span> / <span class="text-red-400">${data.avg_loss_pct || 0}%</span>`;

        const openEl = document.getElementById('tradeTrackOpenCount');
        if (openEl) openEl.textContent = `${data.open_trades || 0} open predictions | ${data.total} resolved`;

        // By timeframe
        const tfEl = document.getElementById('tradeByTimeframe');
        const tfLabels = { intraday: 'Intraday', short_term: 'Short-term' };
        const bt = data.by_timeframe || {};

        tfEl.innerHTML = Object.entries(tfLabels).map(([key, label]) => {
            const tf = bt[key];
            if (!tf) return `<div class="bg-dark-700 rounded-lg p-3 text-center">
                <div class="text-xs text-gray-500 mb-1">${label}</div>
                <div class="text-sm text-gray-600">No data</div>
            </div>`;

            const wrColor = tf.win_rate >= 60 ? 'text-green-400' : tf.win_rate >= 45 ? 'text-yellow-400' : 'text-red-400';
            const borderColor = tf.win_rate >= 60 ? 'border-green-700/50' : tf.win_rate >= 45 ? 'border-yellow-700/50' : 'border-red-700/50';
            const bgColor = tf.win_rate >= 60 ? 'bg-green-900/10' : tf.win_rate >= 45 ? 'bg-yellow-900/10' : 'bg-red-900/10';
            return `<div class="rounded-lg p-3 text-center border ${borderColor} ${bgColor}">
                <div class="text-xs text-gray-500 mb-1">${label}</div>
                <div class="text-lg font-bold ${wrColor}">${tf.win_rate}%</div>
                <div class="text-[10px] mt-1"><span class="text-green-400">${tf.target_hit} wins</span> · <span class="text-red-400">${tf.sl_hit} losses</span> · <span class="text-gray-500">${tf.expired} expired</span></div>
                <div class="text-[10px] text-gray-600">${tf.total} trades</div>
            </div>`;
        }).join('');

        // Recent trades table
        const trades = data.recent_trades || [];
        const tbody = document.getElementById('tradeRecentTable');

        if (trades.length === 0) {
            tbody.innerHTML = '<tr><td colspan="9" class="text-center py-4 text-gray-500">No resolved trades yet</td></tr>';
            return;
        }

        tbody.innerHTML = trades.map(t => {
            const dirColor = t.direction === 'BULLISH' ? 'text-green-400' : 'text-red-400';
            const dirArrow = t.direction === 'BULLISH' ? '▲' : '▼';

            let statusBadge = '';
            if (t.status === 'target_hit') {
                statusBadge = '<span class="px-1.5 py-0.5 rounded bg-green-900 text-green-400">Target ✓</span>';
            } else if (t.status === 'sl_hit') {
                statusBadge = '<span class="px-1.5 py-0.5 rounded bg-red-900 text-red-400">SL Hit ✗</span>';
            } else {
                const expPnlColor = (t.outcome_pct || 0) >= 0 ? 'text-green-400' : 'text-red-400';
                statusBadge = `<span class="px-1.5 py-0.5 rounded bg-gray-800 ${expPnlColor}">Expired</span>`;
            }

            const pnlColor = (t.outcome_pct || 0) >= 0 ? 'text-green-400' : 'text-red-400';
            const pnlSign = (t.outcome_pct || 0) >= 0 ? '+' : '';

            const tfShort = { intraday_10m: '10m', intraday_15m: '15m', intraday_30m: '30m', short_1h: '1h', short_4h: '4h' };

            // Target label: "Re-entry" for bearish
            const targetLabel = t.direction === 'BEARISH' ? 'Re-entry' : 'Target';

            // Actual price color: green if moved in predicted direction
            const actualPrice = t.outcome_price;
            let actualColor = 'text-gray-300';
            if (actualPrice && t.entry) {
                if (t.direction === 'BULLISH') {
                    actualColor = actualPrice > t.entry ? 'text-green-400' : 'text-red-400';
                } else {
                    actualColor = actualPrice < t.entry ? 'text-green-400' : 'text-red-400';
                }
            }

            const rowBg = t.status === 'target_hit' ? 'bg-green-900/10 border-l-2 border-l-green-500' :
                          t.status === 'sl_hit' ? 'bg-red-900/10 border-l-2 border-l-red-500' :
                          (t.outcome_pct || 0) >= 0 ? 'border-l-2 border-l-green-800' : 'border-l-2 border-l-red-800';

            return `<tr class="${rowBg} hover:bg-dark-700/50">
                <td class="px-2 py-1.5 font-medium text-white cursor-pointer" onclick="Search.select('${t.symbol}','')">${t.symbol}</td>
                <td class="px-2 py-1.5 text-gray-400">${tfShort[t.timeframe] || t.timeframe}</td>
                <td class="px-2 py-1.5 text-center ${dirColor}">${dirArrow}</td>
                <td class="px-2 py-1.5 text-right text-gray-300">₹${t.entry ? t.entry.toFixed(2) : '-'}</td>
                <td class="px-2 py-1.5 text-right text-gray-300">₹${t.target ? t.target.toFixed(2) : '-'}</td>
                <td class="px-2 py-1.5 text-right text-gray-300">₹${t.stop_loss ? t.stop_loss.toFixed(2) : '-'}</td>
                <td class="px-2 py-1.5 text-right ${actualColor} font-medium">₹${actualPrice ? actualPrice.toFixed(2) : '-'}</td>
                <td class="px-2 py-1.5 text-center">
                    ${statusBadge}
                    ${t.target_progress != null && t.status !== 'target_hit' ? `<div class="mt-0.5 w-full bg-dark-600 rounded-full h-1"><div class="h-1 rounded-full ${t.target_progress >= 80 ? 'bg-yellow-400' : 'bg-gray-500'}" style="width:${Math.min(100, t.target_progress)}%"></div></div><div class="text-[8px] text-gray-500">${t.target_progress}% to target</div>` : ''}
                </td>
                <td class="px-2 py-1.5 text-right ${pnlColor} font-medium">${pnlSign}${(t.outcome_pct || 0).toFixed(2)}%</td>
            </tr>`;
        }).join('');
    }
};
