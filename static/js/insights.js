window.Insights = {
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

    async loadAIAnalysis() {
        if (this._analysisLoaded && Date.now() - this._analysisLoaded < 60000) return;
        this._analysisLoaded = Date.now();

        const container = document.getElementById('aiAnalysisContent');
        if (!container) return;

        try {
            const resp = await fetch(`${API.baseUrl}/api/signals/stats/ai-analysis`);
            if (!resp.ok) { container.innerHTML = '<div class="text-center py-8 text-gray-500">Failed to load analysis</div>'; return; }
            const d = await resp.json();
            if (!d || d.total === 0) { container.innerHTML = '<div class="text-center py-8 text-gray-500">No trade data yet</div>'; return; }

            const tfShort = { intraday_10m: '10m', intraday_15m: '15m', intraday_30m: '30m', short_1h: '1h', short_4h: '4h' };
            const wrColor = (wr) => wr >= 65 ? 'text-green-400' : wr >= 50 ? 'text-yellow-400' : 'text-red-400';
            const pnlColor = (p) => p >= 0 ? 'text-green-400' : 'text-red-400';
            const pnlSign = (p) => p >= 0 ? '+' : '';

            let html = `
            <!-- Overview -->
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
                <div class="bg-dark-800 rounded-lg p-3 text-center">
                    <div class="text-[10px] text-gray-500">Total Trades</div>
                    <div class="text-xl font-bold text-white">${d.total}</div>
                    <div class="text-[10px] text-gray-500">${d.open} open</div>
                </div>
                <div class="bg-dark-800 rounded-lg p-3 text-center">
                    <div class="text-[10px] text-gray-500">Win Rate</div>
                    <div class="text-xl font-bold ${wrColor(d.win_rate)}">${d.win_rate}%</div>
                </div>
                <div class="bg-dark-800 rounded-lg p-3 text-center">
                    <div class="text-[10px] text-gray-500">Avg Win / Loss</div>
                    <div class="text-sm font-bold"><span class="text-green-400">${pnlSign(d.avg_win)}${d.avg_win}%</span> / <span class="text-red-400">${d.avg_loss}%</span></div>
                </div>
                <div class="bg-dark-800 rounded-lg p-3 text-center">
                    <div class="text-[10px] text-gray-500">Prediction Error</div>
                    <div class="text-xl font-bold ${d.prediction_error.avg <= 2 ? 'text-green-400' : 'text-yellow-400'}">${d.prediction_error.avg || '-'}%</div>
                    <div class="text-[10px] text-gray-500">${d.prediction_error.within_1pct}% within 1%</div>
                </div>
            </div>

            <!-- Status Breakdown -->
            <div class="bg-dark-800 rounded-lg p-3 sm:p-4 mb-4">
                <h3 class="text-sm font-semibold text-white mb-3">Result Breakdown</h3>
                <div class="grid grid-cols-2 sm:grid-cols-4 gap-2">
                    ${Object.entries(d.status_counts || {}).map(([s, c]) => {
                        const colors = { target_hit: 'text-green-400 bg-green-900/20', sl_hit: 'text-red-400 bg-red-900/20', correct: 'text-green-400 bg-green-900/10', wrong: 'text-red-400 bg-red-900/10' };
                        return `<div class="rounded-lg p-2 text-center ${colors[s] || 'bg-dark-700'}">
                            <div class="text-lg font-bold">${c}</div>
                            <div class="text-[10px]">${s.replace('_', ' ')}</div>
                        </div>`;
                    }).join('')}
                </div>
            </div>

            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
                <!-- By Direction -->
                <div class="bg-dark-800 rounded-lg p-3 sm:p-4">
                    <h3 class="text-sm font-semibold text-white mb-3">By Direction</h3>
                    ${Object.entries(d.by_direction || {}).map(([dir, s]) => {
                        const c = dir === 'BULLISH' ? 'border-green-700/50 bg-green-900/10' : 'border-red-700/50 bg-red-900/10';
                        const arrow = dir === 'BULLISH' ? '▲' : '▼';
                        return `<div class="rounded-lg p-3 border ${c} mb-2">
                            <div class="flex justify-between items-center mb-1">
                                <span class="text-sm font-medium text-white">${arrow} ${dir}</span>
                                <span class="text-sm font-bold ${wrColor(s.win_rate)}">${s.win_rate}% WR</span>
                            </div>
                            <div class="text-xs text-gray-400">${s.total} trades | Avg P&L: <span class="${pnlColor(s.avg_pnl)}">${pnlSign(s.avg_pnl)}${s.avg_pnl}%</span> | Best: <span class="text-green-400">${pnlSign(s.best)}${s.best}%</span> | Worst: <span class="text-red-400">${s.worst}%</span></div>
                        </div>`;
                    }).join('')}
                </div>

                <!-- By Timeframe -->
                <div class="bg-dark-800 rounded-lg p-3 sm:p-4">
                    <h3 class="text-sm font-semibold text-white mb-3">By Timeframe</h3>
                    ${Object.entries(d.by_timeframe || {}).sort((a,b) => b[1].win_rate - a[1].win_rate).map(([tf, s]) => {
                        const border = s.win_rate >= 65 ? 'border-green-700/50 bg-green-900/10' : s.win_rate >= 50 ? 'border-yellow-700/50 bg-yellow-900/10' : 'border-red-700/50 bg-red-900/10';
                        return `<div class="rounded-lg p-2 border ${border} mb-1.5 flex justify-between items-center">
                            <div>
                                <span class="text-xs text-white font-medium">${tfShort[tf] || tf}</span>
                                <span class="text-[10px] text-gray-500 ml-2">${s.total} trades</span>
                            </div>
                            <div class="text-right">
                                <span class="text-sm font-bold ${wrColor(s.win_rate)}">${s.win_rate}%</span>
                                <span class="text-[10px] ${pnlColor(s.avg_pnl)} ml-1">${pnlSign(s.avg_pnl)}${s.avg_pnl}%</span>
                            </div>
                        </div>`;
                    }).join('')}
                </div>
            </div>

            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
                <!-- By Confidence -->
                <div class="bg-dark-800 rounded-lg p-3 sm:p-4">
                    <h3 class="text-sm font-semibold text-white mb-3">By Confidence Level</h3>
                    ${(d.by_confidence || []).map(b => {
                        const barWidth = Math.min(100, b.win_rate);
                        const barColor = b.win_rate >= 60 ? 'bg-green-500' : b.win_rate >= 45 ? 'bg-yellow-500' : 'bg-red-500';
                        return `<div class="mb-2">
                            <div class="flex justify-between text-xs mb-0.5">
                                <span class="text-gray-400">${b.range} <span class="text-gray-600">(${b.total})</span></span>
                                <span class="${wrColor(b.win_rate)} font-bold">${b.win_rate}%</span>
                            </div>
                            <div class="h-2 bg-dark-600 rounded-full"><div class="h-2 rounded-full ${barColor}" style="width:${barWidth}%"></div></div>
                        </div>`;
                    }).join('')}
                </div>

                <!-- By Time of Day -->
                <div class="bg-dark-800 rounded-lg p-3 sm:p-4">
                    <h3 class="text-sm font-semibold text-white mb-3">Best Time to Trade</h3>
                    ${(d.by_hour || []).map(h => {
                        const barWidth = Math.min(100, h.win_rate);
                        const barColor = h.win_rate >= 60 ? 'bg-green-500' : h.win_rate >= 45 ? 'bg-yellow-500' : 'bg-red-500';
                        const isBest = h.win_rate >= 65;
                        return `<div class="mb-2">
                            <div class="flex justify-between text-xs mb-0.5">
                                <span class="text-gray-400">${h.hour} <span class="text-gray-600">(${h.total})</span>${isBest ? ' <span class="text-green-400 text-[10px]">★ Best</span>' : ''}</span>
                                <span class="${wrColor(h.win_rate)} font-bold">${h.win_rate}%</span>
                            </div>
                            <div class="h-2 bg-dark-600 rounded-full"><div class="h-2 rounded-full ${barColor}" style="width:${barWidth}%"></div></div>
                        </div>`;
                    }).join('')}
                </div>
            </div>

            <!-- Top Stocks -->
            <div class="bg-dark-800 rounded-lg p-3 sm:p-4 mb-4">
                <h3 class="text-sm font-semibold text-white mb-3">Stock Performance</h3>
                <div class="overflow-x-auto">
                    <table class="w-full text-xs">
                        <thead><tr class="text-gray-500 border-b border-gray-700">
                            <th class="text-left px-2 py-1.5">Stock</th>
                            <th class="text-right px-2 py-1.5">Trades</th>
                            <th class="text-right px-2 py-1.5">Win Rate</th>
                            <th class="text-right px-2 py-1.5">Avg P&L</th>
                        </tr></thead>
                        <tbody>${(d.by_stock || []).map(s => {
                            const rowBg = s.win_rate >= 65 ? 'bg-green-900/5 border-l-2 border-l-green-600' : s.win_rate < 40 ? 'bg-red-900/5 border-l-2 border-l-red-600' : '';
                            return `<tr class="${rowBg} hover:bg-dark-700/50">
                                <td class="px-2 py-1.5 text-white font-medium cursor-pointer" onclick="Search.select('${s.symbol}','')">${s.symbol}</td>
                                <td class="px-2 py-1.5 text-right text-gray-300">${s.total}</td>
                                <td class="px-2 py-1.5 text-right ${wrColor(s.win_rate)} font-bold">${s.win_rate}%</td>
                                <td class="px-2 py-1.5 text-right ${pnlColor(s.avg_pnl)}">${pnlSign(s.avg_pnl)}${s.avg_pnl}%</td>
                            </tr>`;
                        }).join('')}</tbody>
                    </table>
                </div>
            </div>

            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
                <!-- Best Trades -->
                <div class="bg-dark-800 rounded-lg p-3 sm:p-4">
                    <h3 class="text-sm font-semibold text-green-400 mb-2">🏆 Best Trades</h3>
                    ${(d.best_trades || []).map(t => `<div class="flex justify-between py-1 border-b border-gray-800 text-xs">
                        <span class="text-white cursor-pointer" onclick="Search.select('${t.symbol}','')">${t.symbol} <span class="text-gray-500">${tfShort[t.timeframe] || t.timeframe}</span></span>
                        <span class="text-green-400 font-bold">+${t.pnl}%</span>
                    </div>`).join('')}
                </div>
                <!-- Worst Trades -->
                <div class="bg-dark-800 rounded-lg p-3 sm:p-4">
                    <h3 class="text-sm font-semibold text-red-400 mb-2">💀 Worst Trades</h3>
                    ${(d.worst_trades || []).map(t => `<div class="flex justify-between py-1 border-b border-gray-800 text-xs">
                        <span class="text-white cursor-pointer" onclick="Search.select('${t.symbol}','')">${t.symbol} <span class="text-gray-500">${tfShort[t.timeframe] || t.timeframe}</span></span>
                        <span class="text-red-400 font-bold">${t.pnl}%</span>
                    </div>`).join('')}
                </div>
            </div>

            <!-- Daily Trend -->
            <div class="bg-dark-800 rounded-lg p-3 sm:p-4 mb-4">
                <h3 class="text-sm font-semibold text-white mb-3">Daily P&L Trend</h3>
                <div class="flex flex-wrap gap-1">
                    ${(d.daily_trend || []).map(day => {
                        const c = day.pnl >= 0 ? 'bg-green-900/50 text-green-400 border-green-800/50' : 'bg-red-900/50 text-red-400 border-red-800/50';
                        return `<div class="rounded px-2 py-1 text-[10px] border ${c}">
                            <div class="font-medium">${day.date.slice(5)}</div>
                            <div>${pnlSign(day.pnl)}${day.pnl}%</div>
                            <div class="text-[8px] opacity-70">${day.trades}t ${day.win_rate}%</div>
                        </div>`;
                    }).join('')}
                </div>
            </div>`;

            // V1 vs V2 Model Accuracy section
            html += `<div id="modelAccuracySection" class="bg-dark-800 rounded-lg p-3 sm:p-4 mb-4">
                <h3 class="text-sm font-semibold text-white mb-2">V1 vs V2 Model Accuracy</h3>
                <div class="text-xs text-gray-500">Loading real accuracy data...</div>
            </div>`;

            container.innerHTML = html;

            // Load model accuracy comparison async
            this._loadModelAccuracy();
        } catch (e) {
            container.innerHTML = '<div class="text-center py-8 text-gray-500">Failed to load analysis</div>';
        }
    },

    async _loadModelAccuracy() {
        const section = document.getElementById('modelAccuracySection');
        if (!section) return;

        try {
            const resp = await fetch(`${API.baseUrl}/api/signals/stats/model-accuracy`);
            if (!resp.ok) { section.innerHTML = '<h3 class="text-sm font-semibold text-white mb-2">V1 vs V2 Model Accuracy</h3><div class="text-xs text-gray-500">Not available yet</div>'; return; }
            const d = await resp.json();

            if (!d || d.total_trades === 0) {
                section.innerHTML = `<h3 class="text-sm font-semibold text-white mb-2">V1 vs V2 Model Accuracy</h3>
                    <div class="text-xs text-gray-500">${d.message || 'No trades with both V1 and V2 data yet. Tracking started — check back after market hours.'}</div>`;
                return;
            }

            const s = d.summary;
            const v1Better = s.better_model === 'v1';
            const tfShort = { intraday_10m: '10m', intraday_15m: '15m', intraday_30m: '30m', short_1h: '1h', short_4h: '4h' };
            const errColor = (e) => e <= 1 ? 'text-green-400' : e <= 2 ? 'text-yellow-400' : 'text-red-400';
            const wrColor = (wr) => wr >= 65 ? 'text-green-400' : wr >= 50 ? 'text-yellow-400' : 'text-red-400';

            let html = `<h3 class="text-sm font-semibold text-white mb-3">V1 vs V2 Model Accuracy <span class="text-[10px] text-gray-500">(${d.total_trades} trades)</span></h3>`;

            // Summary cards: V1 vs V2
            html += `<div class="grid grid-cols-2 gap-3 mb-3">
                <div class="rounded-lg p-3 border ${v1Better ? 'border-green-600/50 bg-green-900/10' : 'border-gray-700 bg-dark-700'}">
                    <div class="flex items-center gap-2 mb-2">
                        <span class="text-sm font-bold text-white">V1</span>
                        <span class="text-[10px] px-1.5 py-0.5 rounded bg-gray-800 text-gray-400">Regression</span>
                        ${v1Better ? '<span class="text-[10px] px-1.5 py-0.5 rounded bg-green-900/50 text-green-400">Winner</span>' : ''}
                    </div>
                    <div class="text-xs text-gray-400">Avg Error: <span class="${errColor(s.v1_avg_error_pct)} font-bold">${s.v1_avg_error_pct}%</span></div>
                    <div class="text-xs text-gray-400">Direction: <span class="${wrColor(s.v1_direction_accuracy)} font-bold">${s.v1_direction_accuracy}%</span></div>
                    <div class="text-xs text-gray-400">Closer: <span class="text-white">${s.v1_closer_count}x</span></div>
                </div>
                <div class="rounded-lg p-3 border ${!v1Better ? 'border-purple-600/50 bg-purple-900/10' : 'border-gray-700 bg-dark-700'}">
                    <div class="flex items-center gap-2 mb-2">
                        <span class="text-sm font-bold text-white">V2</span>
                        <span class="text-[10px] px-1.5 py-0.5 rounded bg-purple-900/50 text-purple-400">Classifier</span>
                        ${!v1Better ? '<span class="text-[10px] px-1.5 py-0.5 rounded bg-green-900/50 text-green-400">Winner</span>' : ''}
                    </div>
                    <div class="text-xs text-gray-400">Avg Error: <span class="${errColor(s.v2_avg_error_pct)} font-bold">${s.v2_avg_error_pct}%</span></div>
                    <div class="text-xs text-gray-400">Direction: <span class="${wrColor(s.v2_direction_accuracy)} font-bold">${s.v2_direction_accuracy}%</span></div>
                    <div class="text-xs text-gray-400">Closer: <span class="text-white">${s.v2_closer_count}x</span></div>
                </div>
            </div>`;

            // By timeframe breakdown
            if (Object.keys(d.by_timeframe || {}).length > 0) {
                html += `<div class="mb-3"><div class="text-xs font-medium text-gray-400 mb-2">By Timeframe</div>`;
                for (const [tf, data] of Object.entries(d.by_timeframe).sort((a,b) => b[1].count - a[1].count)) {
                    const v1Wins = data.v1_avg_error < data.v2_avg_error;
                    html += `<div class="flex items-center justify-between py-1.5 border-b border-gray-800 text-xs">
                        <span class="text-white font-medium">${tfShort[tf] || tf} <span class="text-gray-600">(${data.count})</span></span>
                        <div class="flex gap-4">
                            <span class="${v1Wins ? 'text-green-400 font-bold' : 'text-gray-400'}">V1: ${data.v1_avg_error}% err / ${data.v1_direction_accuracy}% dir</span>
                            <span class="${!v1Wins ? 'text-purple-400 font-bold' : 'text-gray-400'}">V2: ${data.v2_avg_error}% err / ${data.v2_direction_accuracy}% dir</span>
                        </div>
                    </div>`;
                }
                html += `</div>`;
            }

            // Recent trades table
            if ((d.recent_trades || []).length > 0) {
                html += `<div class="text-xs font-medium text-gray-400 mb-2">Recent Trades — Side by Side</div>
                <div class="overflow-x-auto"><table class="w-full text-[10px]">
                    <thead><tr class="text-gray-500 border-b border-gray-700">
                        <th class="text-left px-1 py-1">Stock</th><th class="text-left px-1">TF</th>
                        <th class="text-right px-1">Entry</th><th class="text-right px-1">Actual</th>
                        <th class="text-right px-1">V1 Pred</th><th class="text-right px-1">V1 Err</th>
                        <th class="text-right px-1">V2 Pred</th><th class="text-right px-1">V2 Err</th>
                        <th class="text-center px-1">Winner</th>
                    </tr></thead><tbody>`;
                for (const t of d.recent_trades) {
                    const winnerColor = t.winner === 'v1' ? 'text-green-400' : 'text-purple-400';
                    html += `<tr class="border-b border-gray-800/50 hover:bg-dark-700/50">
                        <td class="px-1 py-1 text-white">${t.symbol}</td>
                        <td class="px-1 text-gray-400">${tfShort[t.timeframe] || t.timeframe}</td>
                        <td class="px-1 text-right text-gray-300">${t.entry_price}</td>
                        <td class="px-1 text-right text-white">${t.actual_price}</td>
                        <td class="px-1 text-right ${t.v1_direction_correct ? 'text-green-400' : 'text-red-400'}">${t.v1_predicted}</td>
                        <td class="px-1 text-right text-gray-300">${t.v1_error_pct}%</td>
                        <td class="px-1 text-right ${t.v2_direction_correct ? 'text-green-400' : 'text-red-400'}">${t.v2_predicted}</td>
                        <td class="px-1 text-right text-gray-300">${t.v2_error_pct}%</td>
                        <td class="px-1 text-center ${winnerColor} font-bold">${t.winner.toUpperCase()}</td>
                    </tr>`;
                }
                html += `</tbody></table></div>`;
            }

            section.innerHTML = html;
        } catch (e) {
            section.innerHTML = `<h3 class="text-sm font-semibold text-white mb-2">V1 vs V2 Model Accuracy</h3>
                <div class="text-xs text-gray-500">Loading failed — tracking in background</div>`;
        }
    },

    async compareModels() {
        const symbol = document.getElementById('modelCompSymbol')?.value?.toUpperCase();
        const el = document.getElementById('modelCompResult');
        if (!symbol || !el) return;

        el.innerHTML = '<div class="text-center py-4 text-gray-500">Computing predictions...</div>';

        try {
            const resp = await fetch(`${API.baseUrl}/api/signals/stats/model-comparison/${encodeURIComponent(symbol)}`);
            if (!resp.ok) {
                const err = await resp.json();
                el.innerHTML = `<div class="text-red-400 text-sm">${err.detail || 'Failed'}</div>`;
                return;
            }
            const d = await resp.json();

            const v1 = d.v1 || {};
            const v2 = d.v2 || {};
            const price = d.current_price;

            let html = `<div class="text-xs text-gray-400 mb-3">${d.symbol} — Current: ₹${price?.toFixed(2)}</div>`;

            html += `<div class="grid grid-cols-1 sm:grid-cols-2 gap-3">`;

            // V1 card
            const v1Err = v1.error;
            html += `<div class="bg-dark-700 rounded-lg p-3 border border-gray-700">
                <div class="flex items-center gap-2 mb-2">
                    <span class="text-sm font-bold text-white">V1 (Regression)</span>
                    <span class="text-[10px] px-1.5 py-0.5 rounded bg-gray-800 text-gray-400">Current</span>
                </div>
                ${v1Err ? `<div class="text-red-400 text-xs">${v1Err}</div>` : `
                    <div class="text-xs text-gray-400">Predicted: <span class="text-white font-medium">₹${v1.predictions?.toFixed(2) || '-'}</span></div>
                    <div class="text-xs text-gray-400">MAPE: <span class="text-white">${v1.mape?.toFixed(2) || '-'}%</span></div>
                    <div class="text-xs text-gray-400">Confidence: <span class="text-white">${v1.confidence?.toFixed(1) || '-'}%</span></div>
                    <div class="text-xs text-gray-400">Features: <span class="text-white">${v1.features || '-'}</span></div>
                `}
            </div>`;

            // V2 card
            const v2Err = v2.error;
            const dirColor = v2.direction === 'BULLISH' ? 'text-green-400' : v2.direction === 'BEARISH' ? 'text-red-400' : 'text-yellow-400';
            html += `<div class="bg-dark-700 rounded-lg p-3 border border-purple-700/50">
                <div class="flex items-center gap-2 mb-2">
                    <span class="text-sm font-bold text-white">V2 (Classifier)</span>
                    <span class="text-[10px] px-1.5 py-0.5 rounded bg-purple-900/50 text-purple-400">New</span>
                </div>
                ${v2Err ? `<div class="text-red-400 text-xs">${v2Err}</div>` : `
                    <div class="text-xs text-gray-400">Direction: <span class="${dirColor} font-bold">${v2.direction || '-'}</span></div>
                    <div class="text-xs text-gray-400">Probability: <span class="text-white font-medium">${((v2.probability || 0.5) * 100).toFixed(1)}% up</span></div>
                    <div class="text-xs text-gray-400">Predicted: <span class="text-white">₹${v2.predicted_price?.toFixed(2) || '-'}</span></div>
                    <div class="text-xs text-gray-400">Confidence: <span class="text-white">${v2.confidence_score?.toFixed(1) || '-'}%</span></div>
                    <div class="text-xs text-gray-400">Features: <span class="text-white">${v2.features_used || '-'}</span></div>
                    <div class="text-xs text-gray-400">Best iteration: <span class="text-white">${v2.best_iteration ?? '-'}</span></div>
                    ${v2.top_features?.length ? `<div class="mt-2 text-[10px] text-gray-500">Top features:</div>
                        ${v2.top_features.map(f => `<div class="flex justify-between text-[10px]"><span class="text-gray-400">${f.name}</span><span class="text-white">${f.importance}%</span></div>`).join('')}` : ''}
                `}
            </div>`;

            html += `</div>`;

            el.innerHTML = html;
        } catch (e) {
            el.innerHTML = `<div class="text-red-400 text-sm">Error: ${e.message}</div>`;
        }
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

            // Market closed warning
            if (data.total_trades === 0 && (data.open_positions || []).length === 0) {
                document.getElementById('vpTradeCount').textContent += ' | Market may be closed';
            }
            document.getElementById('vpTotalTrades').textContent = `${data.total_trades || 0} trades | ${(data.open_positions || []).length} open | ${data.skipped_signals || 0} skipped`;

            // Open positions
            const openEl = document.getElementById('vpOpenPositions');
            const positions = data.open_positions || [];
            if (positions.length > 0) {
                openEl.innerHTML = `
                    <h4 class="text-xs font-medium text-white mb-2">Open Positions (Live P&L)</h4>
                    <div class="space-y-1">
                        ${positions.map(p => {
                            const confC = (p.confidence || 0) >= 70 ? 'text-green-400' : 'text-yellow-400';
                            const pnlC = (p.live_pnl_pct || 0) >= 0 ? 'text-green-400' : 'text-red-400';
                            const pnlSign = (p.live_pnl_pct || 0) >= 0 ? '+' : '';
                            return `<div class="bg-dark-700 rounded-lg p-2 cursor-pointer hover:bg-dark-600 flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-3" onclick="Search.select('${p.symbol}','')">
                                <div class="flex items-center gap-2 min-w-[120px]">
                                    <span class="text-white font-medium text-sm">${p.symbol}</span>
                                    <span class="${confC} text-[10px]">${(p.confidence || 0).toFixed(0)}%</span>
                                </div>
                                <div class="flex items-center gap-3 text-xs">
                                    <span class="text-gray-500">${p.qty}×₹${p.entry.toFixed(0)}</span>
                                    <span class="text-gray-500">→ ₹${p.target ? p.target.toFixed(0) : '-'}</span>
                                    ${p.current_price ? `<span class="text-white">Now: ₹${p.current_price.toFixed(2)}</span>` : ''}
                                    ${p.live_pnl != null ? `<span class="${pnlC} font-bold">${pnlSign}₹${Math.abs(p.live_pnl).toFixed(0)} (${pnlSign}${p.live_pnl_pct}%)</span>` : ''}
                                </div>
                                <div class="text-[10px] text-gray-600 ml-auto whitespace-nowrap">${p.why_picked || ''} · ${p.scanned_at || ''}</div>
                            </div>`;
                        }).join('')}
                    </div>`;
            } else {
                openEl.innerHTML = '<div class="text-center py-3 text-gray-500 text-xs">No open bullish positions. All signals are bearish — portfolio is in cash.</div>';
            }

            // Scan status + near-bullish suggestions
            this._loadScanStatus();
            this._loadNearBullish();

            // Scan overview — all scanned stocks with portfolio pick status
            this._renderScanOverview(data.scan_overview || []);

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

            // Store trades for filtering
            this._vpAllTrades = data.recent_trades || [];
            this._filterPortfolioTrades();
        } catch (e) {}
    },

    _vpPage: 1,
    _vpPerPage: 15,

    _filterPortfolioTrades() {
        this._vpPage = 1;
        this._renderPaginatedTrades();
    },

    _vpChangePage(delta) {
        this._vpPage += delta;
        this._renderPaginatedTrades();
    },

    _getFilteredTrades() {
        const trades = this._vpAllTrades || [];
        const resultFilter = document.getElementById('vpFilterResult')?.value || '';
        const sortBy = document.getElementById('vpSortBy')?.value || 'date_desc';
        const symbolFilter = (document.getElementById('vpFilterSymbol')?.value || '').toUpperCase();
        const dateFilter = document.getElementById('vpFilterDate')?.value || '';

        const now = new Date();
        const istOffset = 5.5 * 60 * 60000;
        const istNow = new Date(now.getTime() + istOffset + now.getTimezoneOffset() * 60000);
        const todayStr = istNow.toISOString().slice(0, 10);
        const yesterday = new Date(istNow); yesterday.setDate(istNow.getDate() - 1);
        const yesterdayStr = yesterday.toISOString().slice(0, 10);
        const weekAgo = new Date(istNow); weekAgo.setDate(istNow.getDate() - istNow.getDay());
        const weekStr = weekAgo.toISOString().slice(0, 10);
        const monthStr = todayStr.slice(0, 7);

        let filtered = trades.filter(t => {
            if (resultFilter === 'win' && t.pnl < 0) return false;
            if (resultFilter === 'loss' && t.pnl >= 0) return false;
            if (symbolFilter && !t.symbol.includes(symbolFilter)) return false;
            if (dateFilter && t.date) {
                const d = t.date.slice(0, 10);
                if (dateFilter === 'today' && d !== todayStr) return false;
                if (dateFilter === 'yesterday' && d !== yesterdayStr) return false;
                if (dateFilter === 'week' && d < weekStr) return false;
                if (dateFilter === 'month' && !d.startsWith(monthStr)) return false;
            }
            return true;
        });

        if (sortBy === 'date_desc') filtered.sort((a, b) => (b.date || '').localeCompare(a.date || ''));
        else if (sortBy === 'date_asc') filtered.sort((a, b) => (a.date || '').localeCompare(b.date || ''));
        else if (sortBy === 'pnl_desc') filtered.sort((a, b) => (b.pnl || 0) - (a.pnl || 0));
        else if (sortBy === 'pnl_asc') filtered.sort((a, b) => (a.pnl || 0) - (b.pnl || 0));
        else if (sortBy === 'invested_desc') filtered.sort((a, b) => (b.invested || 0) - (a.invested || 0));

        return filtered;
    },

    _renderPaginatedTrades() {
        const tbody = document.getElementById('vpTradesTable');
        if (!tbody) return;

        const allTrades = this._vpAllTrades || [];
        const filtered = this._getFilteredTrades();
        const totalPages = Math.max(1, Math.ceil(filtered.length / this._vpPerPage));
        if (this._vpPage > totalPages) this._vpPage = totalPages;
        if (this._vpPage < 1) this._vpPage = 1;

        const start = (this._vpPage - 1) * this._vpPerPage;
        const page = filtered.slice(start, start + this._vpPerPage);

        document.getElementById('vpTradeCount').textContent = `${filtered.length} of ${allTrades.length} trades`;

        const prevBtn = document.getElementById('vpPagePrev');
        const nextBtn = document.getElementById('vpPageNext');
        const pageInfo = document.getElementById('vpPageInfo');
        if (prevBtn) prevBtn.disabled = this._vpPage <= 1;
        if (nextBtn) nextBtn.disabled = this._vpPage >= totalPages;
        if (pageInfo) pageInfo.textContent = filtered.length > 0 ? `Page ${this._vpPage} of ${totalPages}` : '';

        if (page.length === 0) {
            tbody.innerHTML = '<tr><td colspan="10" class="text-center py-4 text-gray-500">No trades match filters</td></tr>';
            return;
        }

        const tfShort = { intraday_10m: '10m', intraday_15m: '15m', intraday_30m: '30m', short_1h: '1h', short_4h: '4h' };

        tbody.innerHTML = page.map(t => {
                const pColor = t.pnl >= 0 ? 'text-green-400' : 'text-red-400';
                const sign = t.pnl >= 0 ? '+' : '';
                const rowBg = t.status === 'target_hit' ? 'bg-green-900/10 border-l-2 border-l-green-500' :
                              t.status === 'sl_hit' ? 'bg-red-900/10 border-l-2 border-l-red-500' :
                              'border-l-2 border-l-gray-700';
                let badge = '';
                if (t.status === 'target_hit') badge = '<span class="px-1.5 py-0.5 rounded bg-green-900/50 text-green-400 text-[10px] font-medium">✓ Win</span>';
                else if (t.status === 'sl_hit') badge = '<span class="px-1.5 py-0.5 rounded bg-red-900/50 text-red-400 text-[10px] font-medium">✗ Loss</span>';
                else if (t.status === 'correct') badge = '<span class="px-1.5 py-0.5 rounded bg-green-900/50 text-green-400 text-[10px] font-medium">✓ Correct</span>';
                else if (t.status === 'wrong') badge = '<span class="px-1.5 py-0.5 rounded bg-red-900/50 text-red-400 text-[10px] font-medium">✗ Wrong</span>';
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
    },

    async _loadScanStatus() {
        const el = document.getElementById('vpScanStatus');
        if (!el) return;
        try {
            const resp = await fetch(`${API.baseUrl}/api/signals/stats/scan-status`);
            if (!resp.ok) { el.innerHTML = ''; return; }
            const s = await resp.json();
            if (!s || s.message) {
                el.innerHTML = `<div class="bg-dark-700 rounded-lg p-2 text-xs text-gray-500">No scan data yet. Scans run during market hours.</div>`;
                return;
            }

            const time = s.timestamp ? new Date(s.timestamp).toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit' }) : '-';
            const scanLabel = { intraday: 'Intraday (15m/30m)', short: 'Short-term (1h/4h)', full: 'Full scan' };

            el.innerHTML = `
                <div class="bg-dark-700 rounded-lg p-3">
                    <div class="flex items-center justify-between mb-2">
                        <h4 class="text-xs font-medium text-white">Scan Status</h4>
                        <span class="text-[10px] text-gray-500">Last: ${time} IST</span>
                    </div>
                    <div class="grid grid-cols-2 sm:grid-cols-5 gap-2 text-center text-xs">
                        <div class="bg-dark-600 rounded p-1.5">
                            <div class="text-[10px] text-gray-500">Type</div>
                            <div class="text-white font-medium">${scanLabel[s.scan_type] || s.scan_type}</div>
                        </div>
                        <div class="bg-dark-600 rounded p-1.5">
                            <div class="text-[10px] text-gray-500">Stocks Scanned</div>
                            <div class="text-white font-bold">${s.total || 0}</div>
                        </div>
                        <div class="bg-dark-600 rounded p-1.5">
                            <div class="text-[10px] text-gray-500">Watchlist</div>
                            <div class="text-white">${s.watchlist || 0} <span class="text-green-400">(${s.watchlist_bullish || 0} bullish)</span></div>
                        </div>
                        <div class="bg-dark-600 rounded p-1.5">
                            <div class="text-[10px] text-gray-500">Popular Logged</div>
                            <div class="text-white">${s.popular_logged || 0} <span class="text-gray-500">/ ${s.popular_scanned || 0}</span></div>
                        </div>
                        <div class="bg-dark-600 rounded p-1.5">
                            <div class="text-[10px] text-gray-500">Near Bullish</div>
                            <div class="text-yellow-400 font-medium">${s.near_bullish || 0}</div>
                        </div>
                    </div>
                    <div class="mt-1 text-[10px] text-gray-600">
                        Intraday: every 10 min | Short-term: every 30 min | Threshold: ${s.popular_threshold || 60}% conf
                    </div>
                </div>
            `;
        } catch (e) {
            el.innerHTML = '';
        }
    },

    _renderScanOverview(stocks) {
        const el = document.getElementById('vpScanOverview');
        if (!el || !stocks || stocks.length === 0) { if (el) el.innerHTML = ''; return; }

        const tfShort = { intraday_10m: '10m', intraday_15m: '15m', intraday_30m: '30m', short_1h: '1h', short_4h: '4h' };

        el.innerHTML = `
            <h4 class="text-xs font-medium text-white mb-2">Scanned Stocks</h4>
            <div class="overflow-x-auto max-h-[300px] overflow-y-auto">
                <table class="w-full text-xs">
                    <thead class="sticky top-0 bg-dark-800">
                        <tr class="text-gray-500 border-b border-gray-700">
                            <th class="text-left px-2 py-1">Stock</th>
                            <th class="text-center px-2 py-1">Direction</th>
                            <th class="text-right px-2 py-1">Confidence</th>
                            <th class="text-left px-2 py-1">TF</th>
                            <th class="text-right px-2 py-1">Entry</th>
                            <th class="text-right px-2 py-1">Target</th>
                            <th class="text-center px-2 py-1">In Portfolio</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${stocks.map(s => {
                            const dirColor = s.direction === 'BULLISH' ? 'text-green-400' : 'text-red-400';
                            const arrow = s.direction === 'BULLISH' ? '▲' : '▼';
                            const confColor = s.confidence >= 70 ? 'text-green-400' : s.confidence >= 50 ? 'text-yellow-400' : 'text-gray-500';
                            const picked = s.picked_for_portfolio;
                            const rowBg = picked ? 'bg-green-900/10 border-l-2 border-l-green-500' : '';
                            const pickedBadge = picked
                                ? '<span class="px-1.5 py-0.5 rounded bg-green-900/50 text-green-400 text-[10px] font-medium">✓ Added</span>'
                                : '<span class="text-gray-600 text-[10px]">—</span>';
                            return `<tr class="${rowBg} hover:bg-dark-700/50">
                                <td class="px-2 py-1.5 text-white font-medium cursor-pointer" onclick="Search.select('${s.symbol}','')">${s.symbol}</td>
                                <td class="px-2 py-1.5 text-center ${dirColor}">${arrow} ${s.direction}</td>
                                <td class="px-2 py-1.5 text-right ${confColor} font-bold">${s.confidence.toFixed(1)}%</td>
                                <td class="px-2 py-1.5 text-gray-400">${tfShort[s.timeframe] || s.timeframe}</td>
                                <td class="px-2 py-1.5 text-right text-gray-300">₹${s.entry ? s.entry.toFixed(2) : '-'}</td>
                                <td class="px-2 py-1.5 text-right text-gray-300">₹${s.target ? s.target.toFixed(2) : '-'}</td>
                                <td class="px-2 py-1.5 text-center">${pickedBadge}</td>
                            </tr>`;
                        }).join('')}
                    </tbody>
                </table>
            </div>
        `;
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
                    const wrColor = s.win_rate >= 60 ? 'text-green-400' : s.win_rate >= 40 ? 'text-yellow-400' : 'text-red-400';
                    const dirColor = s.direction_accuracy >= 60 ? 'text-green-400' : s.direction_accuracy >= 40 ? 'text-yellow-400' : 'text-red-400';
                    return `<div class="flex items-center justify-between py-1.5 border-b border-gray-800">
                        <span class="text-xs text-white font-medium cursor-pointer hover:text-accent-blue" onclick="Search.select('${s.symbol}','')">${s.symbol}</span>
                        <div class="flex items-center gap-3 text-[10px]">
                            <span class="text-gray-500">${s.total} pred</span>
                            <span class="${wrColor} font-bold">${s.win_rate}% acc</span>
                            <span class="${dirColor}">${s.direction_accuracy}% dir</span>
                        </div>
                    </div>`;
                }).join('') || '<div class="text-center py-2 text-gray-500 text-xs">Not enough data yet (min 5 predictions per stock)</div>';
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

    _thPage: 1,

    async loadTradeTrackRecord() {
        try {
            const resp = await fetch(`${API.baseUrl}/api/signals/stats/trades`);
            if (!resp.ok) return;
            const data = await resp.json();
            this.renderTradeTrackRecord(data);
        } catch (e) {}

        // Load paginated history
        this._loadTradeHistory(1);
    },

    async _loadTradeHistory(page) {
        if (page === undefined) page = this._thPage || 1;
        this._thPage = page;

        const dir = document.getElementById('thFilterDir')?.value || '';
        const status = document.getElementById('thFilterStatus')?.value || '';
        const tf = document.getElementById('thFilterTF')?.value || '';
        const symbol = document.getElementById('thFilterSymbol')?.value || '';
        const date = document.getElementById('thFilterDate')?.value || '';
        const sort = document.getElementById('thSort')?.value || 'created_at';
        const conf = document.getElementById('thFilterConf')?.value || '40';
        const pnlFilter = document.getElementById('thFilterPnl')?.value || '';

        let url = `${API.baseUrl}/api/signals/stats/trades/history?page=${page}&per_page=20&sort_by=${sort}&sort_order=${sort === 'outcome_pct' ? 'desc' : 'desc'}`;
        if (dir) url += `&direction=${dir}`;
        if (status) url += `&status=${status}`;
        if (tf) url += `&timeframe=${tf}`;
        if (symbol) url += `&symbol=${symbol}`;
        if (date) url += `&date_from=${date}&date_to=${date}`;
        if (conf && conf !== '0') url += `&min_confidence=${conf}`;
        if (pnlFilter === 'profit') url += `&min_pnl=0.01`;
        if (pnlFilter === 'loss') url += `&max_pnl=-0.01`;

        try {
            const resp = await fetch(url);
            if (!resp.ok) return;
            const data = await resp.json();
            this._renderTradeHistory(data);
        } catch (e) {}
    },

    _renderTradeHistory(data) {
        const tbody = document.getElementById('tradeHistoryTable');
        const pageInfo = document.getElementById('thPageInfo');
        const pageNum = document.getElementById('thPageNum');
        const prevBtn = document.getElementById('thPrevBtn');
        const nextBtn = document.getElementById('thNextBtn');

        if (!tbody) return;

        pageInfo.textContent = `${data.total} trades | Page ${data.page}/${data.total_pages}`;
        pageNum.textContent = `${data.page} / ${data.total_pages}`;
        prevBtn.disabled = data.page <= 1;
        nextBtn.disabled = data.page >= data.total_pages;

        if (!data.trades || data.trades.length === 0) {
            tbody.innerHTML = '<tr><td colspan="11" class="text-center py-4 text-gray-500">No trades match filters</td></tr>';
            return;
        }

        const tfShort = { intraday_10m: '10m', intraday_15m: '15m', intraday_30m: '30m', short_1h: '1h', short_4h: '4h' };
        const _fmtDt = (d) => d ? new Date(d).toLocaleDateString('en-IN', { timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' }) : '-';

        tbody.innerHTML = data.trades.map(t => {
            const dirColor = t.direction === 'BULLISH' ? 'text-green-400' : 'text-red-400';
            const dirArrow = t.direction === 'BULLISH' ? '▲' : '▼';
            const pnlColor = (t.outcome_pct || 0) >= 0 ? 'text-green-400' : 'text-red-400';
            const pnlSign = (t.outcome_pct || 0) >= 0 ? '+' : '';
            const confColor = (t.confidence || 0) >= 60 ? 'text-green-400' : (t.confidence || 0) >= 40 ? 'text-yellow-400' : 'text-gray-500';

            let badge = '';
            if (t.status === 'target_hit') badge = '<span class="px-1 py-0.5 rounded bg-green-900 text-green-400 text-[10px]">Target ✓</span>';
            else if (t.status === 'sl_hit') badge = '<span class="px-1 py-0.5 rounded bg-red-900 text-red-400 text-[10px]">SL ✗</span>';
            else if (t.status === 'correct') badge = '<span class="px-1 py-0.5 rounded bg-green-900/50 text-green-400 text-[10px]">✓ Correct</span>';
            else if (t.status === 'wrong') badge = '<span class="px-1 py-0.5 rounded bg-red-900/50 text-red-400 text-[10px]">✗ Wrong</span>';

            const rowBg = (t.status === 'target_hit' || t.status === 'correct') ? 'bg-green-900/5 border-l-2 border-l-green-600' :
                          (t.status === 'sl_hit' || t.status === 'wrong') ? 'bg-red-900/5 border-l-2 border-l-red-600' : '';

            return `<tr class="${rowBg} hover:bg-dark-700/50">
                <td class="px-2 py-1.5 text-gray-400 text-[10px]">${_fmtDt(t.created_at)}</td>
                <td class="px-2 py-1.5 text-gray-400 text-[10px]">${_fmtDt(t.resolved_at)}</td>
                <td class="px-2 py-1.5 text-white font-medium cursor-pointer" onclick="Search.select('${t.symbol}','')">${t.symbol}</td>
                <td class="px-2 py-1.5 text-gray-400">${tfShort[t.timeframe] || t.timeframe} ${t.model_used === 'v2' ? '<span class="text-[8px] px-1 rounded bg-purple-900/50 text-purple-400">V2</span>' : ''}</td>
                <td class="px-2 py-1.5 text-center ${dirColor}">${dirArrow}</td>
                <td class="px-2 py-1.5 text-right ${confColor}">${(t.confidence || 0).toFixed(0)}%</td>
                <td class="px-2 py-1.5 text-right text-gray-300">₹${t.entry ? t.entry.toFixed(2) : '-'}</td>
                <td class="px-2 py-1.5 text-right text-gray-300">₹${t.target ? t.target.toFixed(2) : '-'}</td>
                <td class="px-2 py-1.5 text-right text-gray-300">₹${t.outcome_price ? t.outcome_price.toFixed(2) : '-'}${t.prediction_error != null ? '<br><span class=\"text-[8px] ' + (t.prediction_error <= 1 ? 'text-green-500' : 'text-red-500') + '\">' + t.prediction_error + '% err</span>' : ''}</td>
                <td class="px-2 py-1.5 text-center">${badge}</td>
                <td class="px-2 py-1.5 text-right ${pnlColor} font-bold">${pnlSign}${(t.outcome_pct || 0).toFixed(2)}%</td>
            </tr>`;
        }).join('');
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

        document.getElementById('tradeTargetHits').textContent = (data.target_hit || 0) + (data.correct || 0);
        document.getElementById('tradeSLHits').textContent = (data.sl_hit || 0) + (data.wrong || 0);

        const avgPnlEl = document.getElementById('tradeAvgPnl');
        avgPnlEl.innerHTML = `<span class="text-green-400">+${data.avg_win_pct || 0}%</span> / <span class="text-red-400">${data.avg_loss_pct || 0}%</span>`;

        const openEl = document.getElementById('tradeTrackOpenCount');
        if (openEl) {
            const dirAcc = data.direction_accuracy != null ? ` | Direction: ${data.direction_accuracy}%` : '';
            const predErr = data.avg_prediction_error != null ? ` | Pred Error: ${data.avg_prediction_error}%` : '';
            openEl.textContent = `${data.open_trades || 0} open | ${data.total} resolved${dirAcc}${predErr}`;
        }

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
                <div class="text-[10px] mt-1"><span class="text-green-400">${(tf.target_hit || 0) + (tf.correct || 0)} correct</span> · <span class="text-red-400">${(tf.sl_hit || 0) + (tf.wrong || 0)} wrong</span></div>
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
                if (t.status === 'correct') {
                    statusBadge = '<span class="px-1.5 py-0.5 rounded bg-green-900/50 text-green-400">✓ Correct</span>';
                } else if (t.status === 'wrong') {
                    statusBadge = '<span class="px-1.5 py-0.5 rounded bg-red-900/50 text-red-400">✗ Wrong</span>';
                } else {
                    statusBadge = `<span class="px-1.5 py-0.5 rounded bg-gray-800 ${expPnlColor}">Expired</span>`;
                }
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
                <td class="px-2 py-1.5 text-right ${actualColor} font-medium">
                    ₹${actualPrice ? actualPrice.toFixed(2) : '-'}
                    ${t.prediction_error != null ? `<div class="text-[8px] ${t.prediction_error <= 1 ? 'text-green-500' : t.prediction_error <= 3 ? 'text-yellow-500' : 'text-red-500'}">${t.prediction_error}% error</div>` : ''}
                </td>
                <td class="px-2 py-1.5 text-center">
                    ${statusBadge}
                    ${t.target_progress != null && t.status !== 'target_hit' ? `<div class="mt-0.5 w-full bg-dark-600 rounded-full h-1"><div class="h-1 rounded-full ${t.target_progress >= 80 ? 'bg-yellow-400' : 'bg-gray-500'}" style="width:${Math.min(100, t.target_progress)}%"></div></div><div class="text-[8px] text-gray-500">${t.target_progress}% to target</div>` : ''}
                </td>
                <td class="px-2 py-1.5 text-right ${pnlColor} font-medium">${pnlSign}${(t.outcome_pct || 0).toFixed(2)}%</td>
            </tr>`;
        }).join('');
    }
};
