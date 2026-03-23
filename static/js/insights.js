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

    async _seedSignals() {
        // Generate signals for a few popular stocks to populate insights
        const seeds = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'SBIN'];
        const container = document.getElementById('insightsHighConfidence');
        if (container) {
            container.innerHTML = '<div class="text-center py-6 text-gray-500 text-sm col-span-full">Generating initial signals for popular stocks...</div>';
        }
        // Fire all seed requests (each one logs a signal to the DB)
        await Promise.allSettled(seeds.map(s => API.getIntradaySignal(s)));
    },

    async loadHighConfidence() {
        try {
            const signals = await API.scanHighConfidence(60);
            const container = document.getElementById('insightsHighConfidence');
            if (!signals || signals.length === 0) {
                container.innerHTML = '<div class="text-center py-6 text-gray-500 text-sm col-span-3">No high-confidence signals right now. The background scanner will populate this data over time.</div>';
                return;
            }
            container.innerHTML = signals.map(s => {
                const isBull = s.direction === 'BULLISH';
                const borderColor = isBull ? 'border-green-500/40' : 'border-red-500/40';
                const glow = isBull ? 'shadow-green-900/20 shadow-lg' : 'shadow-red-900/20 shadow-lg';
                const color = isBull ? 'text-green-400' : 'text-red-400';
                const arrow = isBull ? '&#9650;' : '&#9660;';
                const time = s.created_at ? new Date(s.created_at + '+05:30').toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit' }) : '-';
                return `
                    <div class="bg-dark-700 border ${borderColor} ${glow} rounded-lg p-4 cursor-pointer hover:bg-dark-600 transition"
                         onclick="Search.select('${s.symbol}', '${s.symbol}')">
                        <div class="flex items-center justify-between mb-2">
                            <span class="text-white font-bold">${s.symbol}</span>
                            <span class="${color} text-xl">${arrow}</span>
                        </div>
                        <div class="${color} text-lg font-bold">${s.direction}</div>
                        <div class="flex items-center justify-between mt-2">
                            <span class="text-white text-sm font-bold">${s.confidence.toFixed(1)}%</span>
                            <span class="text-xs text-gray-500">${s.price_at_signal ? '₹' + s.price_at_signal.toFixed(2) : ''}</span>
                        </div>
                        <div class="text-xs text-gray-500 mt-1">${time}</div>
                    </div>
                `;
            }).join('');
        } catch (e) {
            console.error('Failed to load high confidence signals:', e);
        }
    },

    async loadAccuracy() {
        try {
            const stats = await API.getSignalAccuracy();
            const tbody = document.getElementById('accuracyTable');
            if (!stats || stats.length === 0) {
                tbody.innerHTML = '<tr><td colspan="4" class="text-center py-4 text-gray-500">No accuracy data yet. Signals need time to be verified.</td></tr>';
                return;
            }
            tbody.innerHTML = stats.map(s => {
                const accColor = s.accuracy >= 60 ? 'text-green-400' : (s.accuracy >= 40 ? 'text-yellow-400' : 'text-red-400');
                return `
                    <tr class="border-b border-gray-800 hover:bg-dark-700 cursor-pointer transition"
                        onclick="Search.select('${s.symbol}', '${s.symbol}')">
                        <td class="px-4 py-2 text-accent-blue font-medium">${s.symbol}</td>
                        <td class="px-4 py-2 text-right text-gray-300">${s.total}</td>
                        <td class="px-4 py-2 text-right text-gray-300">${s.correct}</td>
                        <td class="px-4 py-2 text-right ${accColor} font-bold">${s.accuracy.toFixed(1)}%</td>
                    </tr>
                `;
            }).join('');
        } catch (e) {
            console.error('Failed to load accuracy stats:', e);
        }
    },

    async loadRecentFeed() {
        try {
            const signals = await API.scanHighConfidence(0);
            const container = document.getElementById('insightsRecentFeed');
            if (!signals || signals.length === 0) {
                container.innerHTML = '<div class="text-center py-4 text-gray-500 text-sm">No recent signals. The background scanner runs every 2 minutes.</div>';
                return;
            }
            container.innerHTML = signals.slice(0, 30).map(s => {
                const color = s.direction === 'BULLISH' ? 'text-green-400' : (s.direction === 'BEARISH' ? 'text-red-400' : 'text-yellow-400');
                const isHigh = s.confidence >= 60;
                const highlight = isHigh ? 'border-l-2 border-accent-blue pl-3' : 'pl-4';
                const time = s.created_at ? new Date(s.created_at + '+05:30').toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit' }) : '-';
                const arrow = s.direction === 'BULLISH' ? '&#9650;' : (s.direction === 'BEARISH' ? '&#9660;' : '&#9654;');
                return `
                    <div class="flex items-center gap-3 py-2 ${highlight} cursor-pointer hover:bg-dark-700 rounded transition"
                         onclick="Search.select('${s.symbol}', '${s.symbol}')">
                        <span class="${color} text-sm">${arrow}</span>
                        <span class="text-sm text-white font-medium w-28 truncate">${s.symbol}</span>
                        <span class="text-xs ${color} font-medium w-16">${s.direction}</span>
                        <span class="text-xs text-gray-300 w-12 text-right">${s.confidence.toFixed(1)}%</span>
                        <span class="text-xs text-gray-500 w-16 text-right">${s.price_at_signal ? '₹' + s.price_at_signal.toFixed(0) : ''}</span>
                        <span class="text-xs text-gray-500 ml-auto">${time}</span>
                        ${isHigh ? '<span class="text-[10px] px-1.5 py-0.5 rounded bg-accent-blue text-white font-bold">HIGH</span>' : ''}
                    </div>
                `;
            }).join('');
        } catch (e) {
            console.error('Failed to load recent feed:', e);
        }
    },

    _renderStatRows(containerId, data, keyField, labelField) {
        const container = document.getElementById(containerId);
        if (!container) return;
        if (!data || data.length === 0) {
            container.innerHTML = '<div class="text-center py-2 text-gray-500 text-xs">No data yet</div>';
            return;
        }
        container.innerHTML = data.map(row => {
            const acc = row.accuracy || 0;
            const accColor = acc >= 60 ? 'text-green-400' : (acc >= 40 ? 'text-yellow-400' : 'text-red-400');
            return `
                <div class="flex items-center justify-between py-1">
                    <span class="text-xs text-gray-300">${row[labelField] || row[keyField] || '-'}</span>
                    <div class="flex items-center gap-2">
                        <span class="text-[10px] text-gray-500">${row.total || 0} signals</span>
                        <span class="text-xs font-bold ${accColor}">${acc.toFixed(1)}%</span>
                    </div>
                </div>
            `;
        }).join('');
    },

    async loadAccuracyBySector() {
        try {
            const data = await API.getStatsBySector();
            this._renderStatRows('accuracyBySector', data, 'sector', 'sector');
        } catch (e) {
            console.error('Failed to load accuracy by sector:', e);
        }
    },

    async loadAccuracyByHorizon() {
        try {
            const data = await API.getStatsByHorizon();
            const container = document.getElementById('accuracyByHorizon');
            if (!container) return;
            if (!data || data.length === 0) {
                container.innerHTML = '<div class="text-center py-2 text-gray-500 text-xs">No data yet</div>';
                return;
            }
            container.innerHTML = data.map(row => {
                const mape = row.avg_mape || 0;
                const mapeColor = mape <= 3 ? 'text-green-400' : (mape <= 7 ? 'text-yellow-400' : 'text-red-400');
                return `
                    <div class="flex items-center justify-between py-1">
                        <span class="text-xs text-gray-300">${row.horizon || '-'}</span>
                        <div class="flex items-center gap-2">
                            <span class="text-[10px] text-gray-500">${row.total || 0} predictions</span>
                            <span class="text-xs font-bold ${mapeColor}">${mape.toFixed(1)}% MAPE</span>
                        </div>
                    </div>
                `;
            }).join('');
        } catch (e) {
            console.error('Failed to load accuracy by horizon:', e);
        }
    },

    async loadAccuracyByRegime() {
        try {
            const data = await API.getStatsByRegime();
            this._renderStatRows('accuracyByRegime', data, 'regime', 'regime');
        } catch (e) {
            console.error('Failed to load accuracy by regime:', e);
        }
    },

    async loadBacktestPnL() {
        try {
            const data = await API.getBacktestPnL();
            if (!data) return;

            const tradesEl = document.getElementById('pnlTrades');
            const winEl = document.getElementById('pnlWinRate');
            const totalEl = document.getElementById('pnlTotal');
            const ddEl = document.getElementById('pnlDrawdown');

            if (tradesEl) tradesEl.textContent = data.trades ?? '-';
            if (winEl) {
                const wr = data.win_rate;
                winEl.textContent = wr != null ? wr.toFixed(1) + '%' : '-';
                winEl.className = `text-white font-bold ${wr >= 50 ? 'text-green-400' : 'text-red-400'}`;
            }
            if (totalEl) {
                const pnl = data.total_pnl_pct;
                totalEl.textContent = pnl != null ? (pnl >= 0 ? '+' : '') + pnl.toFixed(2) + '%' : '-';
                totalEl.className = `text-white font-bold ${pnl >= 0 ? 'text-green-400' : 'text-red-400'}`;
            }
            if (ddEl) {
                const dd = data.max_drawdown;
                ddEl.textContent = dd != null ? dd.toFixed(2) + '%' : '-';
                ddEl.className = 'text-red-400 font-bold';
            }
        } catch (e) {
            console.error('Failed to load backtest P&L:', e);
        }
    },

    async loadPredictionLeaderboard() {
        try {
            const data = await API.getPredictionLeaderboard();
            this._renderLeaderboardModels(data.models || []);
            this._renderLeaderboardBySymbol(data.by_symbol || []);
            this._renderLeaderboardBySector(data.by_sector || []);
        } catch (e) {
            console.error('Failed to load prediction leaderboard:', e);
        }
    },

    _renderLeaderboardModels(models) {
        const container = document.getElementById('predLeaderboardModels');
        if (!container) return;
        if (!models || models.length === 0) {
            container.innerHTML = '<div class="text-center py-4 text-gray-500 text-sm col-span-full">No prediction data yet. Run predictions on stocks to populate the leaderboard.</div>';
            return;
        }

        const nameMap = { ensemble: 'Ensemble', prophet: 'Prophet', xgboost: 'XGBoost' };
        const iconMap = { ensemble: '&#9733;', prophet: '&#9670;', xgboost: '&#9632;' };
        const colorMap = { ensemble: 'text-accent-blue', prophet: 'text-purple-400', xgboost: 'text-green-400' };
        const borderMap = { ensemble: 'border-accent-blue', prophet: 'border-purple-500', xgboost: 'border-green-500' };

        container.innerHTML = models.map((m, i) => {
            const name = nameMap[m.model] || m.model;
            const icon = iconMap[m.model] || '&#9679;';
            const color = colorMap[m.model] || 'text-gray-400';
            const border = borderMap[m.model] || 'border-gray-600';
            const mapeColor = m.avg_mape <= 3 ? 'text-green-400' : (m.avg_mape <= 7 ? 'text-yellow-400' : 'text-red-400');
            const rank = i === 0 ? '<span class="text-[10px] px-1.5 py-0.5 rounded-full bg-yellow-900 text-yellow-400 ml-1">BEST</span>' : '';

            return `
                <div class="bg-dark-700 border-l-4 ${border} rounded-lg p-3">
                    <div class="flex items-center gap-2 mb-2">
                        <span class="${color} text-lg">${icon}</span>
                        <span class="text-white font-bold text-sm">${name}</span>
                        ${rank}
                    </div>
                    <div class="grid grid-cols-3 gap-2 text-center">
                        <div>
                            <div class="text-[10px] text-gray-500">Avg MAPE</div>
                            <div class="text-sm font-bold ${mapeColor}">${m.avg_mape.toFixed(1)}%</div>
                        </div>
                        <div>
                            <div class="text-[10px] text-gray-500">Win Rate</div>
                            <div class="text-sm font-bold ${m.win_rate >= 50 ? 'text-green-400' : 'text-red-400'}">${m.win_rate.toFixed(0)}%</div>
                        </div>
                        <div>
                            <div class="text-[10px] text-gray-500">Predictions</div>
                            <div class="text-sm font-bold text-gray-300">${m.total}</div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    },

    _renderLeaderboardBySymbol(items) {
        const container = document.getElementById('predLeaderboardBySymbol');
        if (!container) return;
        if (!items || items.length === 0) {
            container.innerHTML = '<div class="text-center py-2 text-gray-500 text-xs">No data yet</div>';
            return;
        }

        const nameMap = { ensemble: 'ENS', prophet: 'PRO', xgboost: 'XGB' };
        const colorMap = { ensemble: 'bg-blue-900 text-blue-300', prophet: 'bg-purple-900 text-purple-300', xgboost: 'bg-green-900 text-green-300' };

        container.innerHTML = items.map(item => {
            const mapeColor = item.avg_mape <= 3 ? 'text-green-400' : (item.avg_mape <= 7 ? 'text-yellow-400' : 'text-red-400');
            const badge = nameMap[item.model] || item.model;
            const badgeColor = colorMap[item.model] || 'bg-gray-800 text-gray-300';
            return `
                <div class="flex items-center justify-between py-1">
                    <div class="flex items-center gap-1.5">
                        <span class="text-[9px] px-1 py-0.5 rounded ${badgeColor}">${badge}</span>
                        <span class="text-xs text-white cursor-pointer hover:text-accent-blue" onclick="Search.select('${item.symbol}', '${item.symbol}')">${item.symbol}</span>
                    </div>
                    <div class="flex items-center gap-2">
                        <span class="text-[10px] text-gray-500">${item.total}x</span>
                        <span class="text-xs font-bold ${mapeColor}">${item.avg_mape.toFixed(1)}%</span>
                    </div>
                </div>
            `;
        }).join('');
    },

    _renderLeaderboardBySector(items) {
        const container = document.getElementById('predLeaderboardBySector');
        if (!container) return;
        if (!items || items.length === 0) {
            container.innerHTML = '<div class="text-center py-2 text-gray-500 text-xs">No data yet</div>';
            return;
        }

        const nameMap = { ensemble: 'ENS', prophet: 'PRO', xgboost: 'XGB' };
        const colorMap = { ensemble: 'bg-blue-900 text-blue-300', prophet: 'bg-purple-900 text-purple-300', xgboost: 'bg-green-900 text-green-300' };

        container.innerHTML = items.map(item => {
            const mapeColor = item.avg_mape <= 3 ? 'text-green-400' : (item.avg_mape <= 7 ? 'text-yellow-400' : 'text-red-400');
            const badge = nameMap[item.model] || item.model;
            const badgeColor = colorMap[item.model] || 'bg-gray-800 text-gray-300';
            return `
                <div class="flex items-center justify-between py-1">
                    <div class="flex items-center gap-1.5">
                        <span class="text-[9px] px-1 py-0.5 rounded ${badgeColor}">${badge}</span>
                        <span class="text-xs text-gray-300">${item.sector}</span>
                    </div>
                    <div class="flex items-center gap-2">
                        <span class="text-[10px] text-gray-500">${item.total}x</span>
                        <span class="text-xs font-bold ${mapeColor}">${item.avg_mape.toFixed(1)}%</span>
                    </div>
                </div>
            `;
        }).join('');
    },

    async loadSmartAlerts() {
        try {
            const alerts = await API.getSmartAlerts();
            const container = document.getElementById('smartAlertsList2');
            if (!container) return;

            if (!alerts || alerts.length === 0) {
                container.innerHTML = '<div class="text-center py-2 text-gray-500 text-xs">No smart alerts configured</div>';
                return;
            }
            container.innerHTML = alerts.map(a => {
                const triggered = a.is_triggered;
                const statusColor = triggered ? 'text-green-400' : 'text-yellow-400';
                const statusBg = triggered ? 'bg-green-900/30' : 'bg-yellow-900/30';
                const created = a.created_at ? new Date(a.created_at).toLocaleDateString('en-IN') : '-';
                return `
                    <div class="flex items-center justify-between py-1.5 px-2 ${statusBg} rounded">
                        <div class="flex items-center gap-2">
                            <span class="${statusColor} text-[10px] font-bold">${triggered ? 'FIRED' : 'ACTIVE'}</span>
                            <span class="text-xs text-white">${a.alert_type}</span>
                            ${a.symbol ? `<span class="text-xs text-accent-blue">${a.symbol}</span>` : ''}
                        </div>
                        <div class="flex items-center gap-2">
                            <span class="text-[10px] text-gray-500">${created}</span>
                            <button onclick="Insights.deleteSmartAlert(${a.id})" class="text-red-400 hover:text-red-300 text-[10px]">x</button>
                        </div>
                    </div>
                `;
            }).join('');
        } catch (e) {
            console.error('Failed to load smart alerts:', e);
        }
    },

    async createSmartAlert() {
        const typeEl = document.getElementById('smartAlertType');
        const symbolEl = document.getElementById('smartAlertSymbol');
        const thresholdEl = document.getElementById('smartAlertThreshold');

        const data = {
            alert_type: typeEl.value,
            symbol: symbolEl.value.toUpperCase() || null,
            threshold: thresholdEl.value ? parseFloat(thresholdEl.value) : null,
        };

        try {
            await API.createSmartAlert(data);
            App.showToast('Smart alert created', 'success');
            symbolEl.value = '';
            thresholdEl.value = '';
            this.loadSmartAlerts();
        } catch (e) {
            App.showToast('Failed to create smart alert: ' + e.message, 'error');
        }
    },

    async deleteSmartAlert(id) {
        try {
            await API.deleteSmartAlert(id);
            App.showToast('Smart alert removed', 'success');
            this.loadSmartAlerts();
        } catch (e) {
            App.showToast('Failed to delete: ' + e.message, 'error');
        }
    },

    // ────────────────────────────────────────────────────────────
    // Visual Backtesting
    // ────────────────────────────────────────────────────────────
    _vbtCharts: {},
    _vbtLastResult: null,

    async loadBacktest() {
        const symbolEl = document.getElementById('vbtSymbol');
        const startEl = document.getElementById('vbtStartDate');
        const endEl = document.getElementById('vbtEndDate');
        const confEl = document.getElementById('vbtConfidence');
        const slEl = document.getElementById('vbtStopLoss');
        const tpEl = document.getElementById('vbtTakeProfit');
        const holdEl = document.getElementById('vbtHoldingDays');
        const stratEl = document.getElementById('vbtStrategy');
        const btn = document.getElementById('btnRunVisualBacktest');
        const loading = document.getElementById('vbtLoading');
        const results = document.getElementById('vbtResults');

        const symbol = (symbolEl.value || '').trim().toUpperCase();
        if (!symbol) {
            App.showToast('Enter a symbol to backtest', 'error');
            return;
        }

        loading.classList.remove('hidden');
        results.classList.add('hidden');
        btn.disabled = true;
        btn.textContent = 'Running...';

        try {
            const params = {
                symbol,
                strategy_params: {
                    signal_type: stratEl.value,
                    confidence_threshold: parseInt(confEl.value) || 0,
                    stop_loss_pct: parseFloat(slEl.value) || 5,
                    take_profit_pct: parseFloat(tpEl.value) || 10,
                    holding_period_days: parseInt(holdEl.value) || 0,
                },
                start_date: startEl.value || null,
                end_date: endEl.value || null,
            };

            const data = await API.runVisualBacktest(params);
            this._vbtLastResult = data;
            results.classList.remove('hidden');
            this._renderVbtMetrics(data.metrics);
            this._renderEquityCurve(data.equity_curve);
            this._renderDrawdownChart(data.drawdown_curve);
            this._renderMonthlyHeatmap(data.monthly_returns);
            this._renderTradeTable(data.trades);

            // Auto-run Monte Carlo
            this._runMonteCarloFromResult(data.equity_curve);
        } catch (e) {
            App.showToast('Visual backtest failed: ' + e.message, 'error');
        } finally {
            loading.classList.add('hidden');
            btn.disabled = false;
            btn.textContent = 'Run Backtest';
        }
    },

    _renderVbtMetrics(m) {
        const grid = document.getElementById('vbtMetricsGrid');
        if (!grid) return;

        const items = [
            { label: 'Total Return', value: (m.total_return >= 0 ? '+' : '') + m.total_return + '%', color: m.total_return >= 0 ? 'text-green-400' : 'text-red-400' },
            { label: 'CAGR', value: m.cagr + '%', color: m.cagr >= 0 ? 'text-green-400' : 'text-red-400' },
            { label: 'Sharpe Ratio', value: m.sharpe_ratio.toFixed(2), color: m.sharpe_ratio >= 1 ? 'text-green-400' : (m.sharpe_ratio >= 0 ? 'text-yellow-400' : 'text-red-400') },
            { label: 'Sortino Ratio', value: m.sortino_ratio.toFixed(2), color: m.sortino_ratio >= 1.5 ? 'text-green-400' : 'text-yellow-400' },
            { label: 'Calmar Ratio', value: m.calmar_ratio.toFixed(2), color: m.calmar_ratio >= 1 ? 'text-green-400' : 'text-yellow-400' },
            { label: 'Max Drawdown', value: '-' + m.max_drawdown + '%', color: 'text-red-400' },
            { label: 'Win Rate', value: m.win_rate + '%', color: m.win_rate >= 50 ? 'text-green-400' : 'text-red-400' },
            { label: 'Profit Factor', value: m.profit_factor === Infinity ? '---' : m.profit_factor.toFixed(2), color: m.profit_factor >= 1.5 ? 'text-green-400' : 'text-yellow-400' },
            { label: 'Avg Win', value: '+' + m.avg_win + '%', color: 'text-green-400' },
            { label: 'Avg Loss', value: '-' + m.avg_loss + '%', color: 'text-red-400' },
            { label: 'Total Trades', value: m.total_trades, color: 'text-white' },
            { label: 'Avg Holding', value: m.avg_holding_days + 'd', color: 'text-gray-300' },
        ];

        grid.innerHTML = items.map(it => `
            <div class="bg-dark-700 rounded-lg p-2 text-center">
                <div class="text-[10px] text-gray-500">${it.label}</div>
                <div class="text-sm font-bold ${it.color}">${it.value}</div>
            </div>
        `).join('');
    },

    _renderEquityCurve(curve) {
        const container = document.getElementById('vbtEquityChart');
        if (!container || !curve || curve.length === 0) return;
        container.innerHTML = '';

        if (this._vbtCharts.equity) {
            this._vbtCharts.equity.remove();
        }

        const chart = LightweightCharts.createChart(container, {
            width: container.clientWidth,
            height: 250,
            layout: { background: { color: '#1a1a2e' }, textColor: '#9ca3af' },
            grid: { vertLines: { color: '#2d2d44' }, horzLines: { color: '#2d2d44' } },
            timeScale: { borderColor: '#2d2d44' },
            rightPriceScale: { borderColor: '#2d2d44' },
        });
        this._vbtCharts.equity = chart;

        const series = chart.addLineSeries({
            color: '#3b82f6',
            lineWidth: 2,
            priceFormat: { type: 'custom', formatter: v => '₹' + v.toFixed(0) },
        });

        series.setData(curve.map(p => ({ time: p.date, value: p.equity })));
        chart.timeScale().fitContent();

        new ResizeObserver(() => {
            chart.applyOptions({ width: container.clientWidth });
        }).observe(container);
    },

    _renderDrawdownChart(curve) {
        const container = document.getElementById('vbtDrawdownChart');
        if (!container || !curve || curve.length === 0) return;
        container.innerHTML = '';

        if (this._vbtCharts.drawdown) {
            this._vbtCharts.drawdown.remove();
        }

        const chart = LightweightCharts.createChart(container, {
            width: container.clientWidth,
            height: 180,
            layout: { background: { color: '#1a1a2e' }, textColor: '#9ca3af' },
            grid: { vertLines: { color: '#2d2d44' }, horzLines: { color: '#2d2d44' } },
            timeScale: { borderColor: '#2d2d44' },
            rightPriceScale: { borderColor: '#2d2d44' },
        });
        this._vbtCharts.drawdown = chart;

        const series = chart.addAreaSeries({
            topColor: 'rgba(239, 68, 68, 0.4)',
            bottomColor: 'rgba(239, 68, 68, 0.05)',
            lineColor: '#ef4444',
            lineWidth: 1,
            priceFormat: { type: 'custom', formatter: v => '-' + v.toFixed(1) + '%' },
            invertFilledArea: true,
        });

        series.setData(curve.map(p => ({ time: p.date, value: p.drawdown_pct })));
        chart.timeScale().fitContent();

        new ResizeObserver(() => {
            chart.applyOptions({ width: container.clientWidth });
        }).observe(container);
    },

    _renderMonthlyHeatmap(monthly) {
        const container = document.getElementById('vbtMonthlyHeatmap');
        if (!container) return;
        if (!monthly || monthly.length === 0) {
            container.innerHTML = '<div class="text-center py-2 text-gray-500 text-xs">No monthly data</div>';
            return;
        }

        const years = {};
        for (const m of monthly) {
            const [year, month] = m.month.split('-');
            if (!years[year]) years[year] = {};
            years[year][parseInt(month)] = m.return_pct;
        }

        const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

        let html = '<table class="w-full text-[10px]"><thead><tr><th class="px-1 py-1 text-gray-500 text-left">Year</th>';
        for (const mn of monthNames) {
            html += `<th class="px-1 py-1 text-gray-500 text-center">${mn}</th>`;
        }
        html += '<th class="px-1 py-1 text-gray-500 text-center">Total</th></tr></thead><tbody>';

        for (const year of Object.keys(years).sort()) {
            html += `<tr><td class="px-1 py-1 text-gray-400 font-medium">${year}</td>`;
            let yearTotal = 0;
            for (let m = 1; m <= 12; m++) {
                const val = years[year][m];
                if (val !== undefined) {
                    yearTotal += val;
                    const bg = val > 5 ? 'bg-green-700' : val > 0 ? 'bg-green-900' : val > -5 ? 'bg-red-900' : 'bg-red-700';
                    const tc = val >= 0 ? 'text-green-300' : 'text-red-300';
                    html += `<td class="px-1 py-1 text-center rounded ${bg} ${tc}">${val > 0 ? '+' : ''}${val.toFixed(1)}</td>`;
                } else {
                    html += '<td class="px-1 py-1 text-center text-gray-700">-</td>';
                }
            }
            const ytBg = yearTotal > 0 ? 'bg-green-800' : 'bg-red-800';
            const ytC = yearTotal >= 0 ? 'text-green-300' : 'text-red-300';
            html += `<td class="px-1 py-1 text-center rounded font-bold ${ytBg} ${ytC}">${yearTotal > 0 ? '+' : ''}${yearTotal.toFixed(1)}</td>`;
            html += '</tr>';
        }
        html += '</tbody></table>';
        container.innerHTML = html;
    },

    _renderTradeTable(trades) {
        const container = document.getElementById('vbtTradeTable');
        if (!container) return;
        if (!trades || trades.length === 0) {
            container.innerHTML = '<div class="text-center py-2 text-gray-500 text-xs">No trades</div>';
            return;
        }

        let html = '<table class="w-full text-xs"><thead><tr class="text-gray-400 border-b border-gray-700">';
        html += '<th class="text-left px-2 py-1">Entry</th><th class="text-left px-2 py-1">Exit</th>';
        html += '<th class="text-right px-2 py-1">Entry ₹</th><th class="text-right px-2 py-1">Exit ₹</th>';
        html += '<th class="text-right px-2 py-1">P&amp;L %</th><th class="text-right px-2 py-1">Days</th>';
        html += '<th class="text-left px-2 py-1">Reason</th></tr></thead><tbody>';

        for (const t of trades.slice(-50)) {
            const color = t.pnl_pct >= 0 ? 'text-green-400' : 'text-red-400';
            html += `<tr class="border-b border-gray-800">
                <td class="px-2 py-1 text-gray-400">${t.entry_date}</td>
                <td class="px-2 py-1 text-gray-400">${t.exit_date}</td>
                <td class="px-2 py-1 text-right text-gray-300">${t.entry_price}</td>
                <td class="px-2 py-1 text-right text-gray-300">${t.exit_price}</td>
                <td class="px-2 py-1 text-right font-bold ${color}">${t.pnl_pct > 0 ? '+' : ''}${t.pnl_pct}%</td>
                <td class="px-2 py-1 text-right text-gray-400">${t.holding_days}</td>
                <td class="px-2 py-1 text-gray-500">${t.exit_reason || ''}</td>
            </tr>`;
        }
        html += '</tbody></table>';
        container.innerHTML = html;
    },

    async _runMonteCarloFromResult(equityCurve) {
        const container = document.getElementById('vbtMonteCarloChart');
        const statsEl = document.getElementById('vbtMonteCarloStats');
        if (!container) return;
        container.innerHTML = '<div class="text-center py-4 text-gray-500 text-xs">Running Monte Carlo...</div>';

        try {
            const mc = await API.runMonteCarlo({
                equity_curve: equityCurve,
                simulations: 1000,
            });
            this._renderMonteCarloChart(mc, equityCurve);
            if (statsEl) this._renderMonteCarloStats(mc.terminal_stats, statsEl);
        } catch (e) {
            container.innerHTML = '<div class="text-center py-4 text-red-400 text-xs">Monte Carlo failed: ' + e.message + '</div>';
        }
    },

    _renderMonteCarloChart(mc, originalCurve) {
        const container = document.getElementById('vbtMonteCarloChart');
        if (!container) return;
        container.innerHTML = '';

        if (this._vbtCharts.monteCarlo) {
            this._vbtCharts.monteCarlo.remove();
        }

        const chart = LightweightCharts.createChart(container, {
            width: container.clientWidth,
            height: 250,
            layout: { background: { color: '#1a1a2e' }, textColor: '#9ca3af' },
            grid: { vertLines: { color: '#2d2d44' }, horzLines: { color: '#2d2d44' } },
            timeScale: { borderColor: '#2d2d44' },
            rightPriceScale: { borderColor: '#2d2d44' },
        });
        this._vbtCharts.monteCarlo = chart;

        const bandConfigs = [
            { key: '5', color: 'rgba(239, 68, 68, 0.3)', label: 'P5' },
            { key: '25', color: 'rgba(251, 191, 36, 0.3)', label: 'P25' },
            { key: '50', color: 'rgba(156, 163, 175, 0.5)', label: 'P50' },
            { key: '75', color: 'rgba(52, 211, 153, 0.3)', label: 'P75' },
            { key: '95', color: 'rgba(59, 130, 246, 0.3)', label: 'P95' },
        ];

        for (const bc of bandConfigs) {
            const band = mc.bands[bc.key];
            if (!band) continue;
            const series = chart.addLineSeries({
                color: bc.color,
                lineWidth: 1,
                priceFormat: { type: 'custom', formatter: v => '₹' + v.toFixed(0) },
                crosshairMarkerVisible: false,
                lastValueVisible: false,
                priceLineVisible: false,
            });
            series.setData(band.map(p => ({ time: p.date, value: p.equity })));
        }

        const origSeries = chart.addLineSeries({
            color: '#3b82f6',
            lineWidth: 2,
            priceFormat: { type: 'custom', formatter: v => '₹' + v.toFixed(0) },
        });
        origSeries.setData(originalCurve.map(p => ({ time: p.date, value: p.equity })));

        chart.timeScale().fitContent();

        new ResizeObserver(() => {
            chart.applyOptions({ width: container.clientWidth });
        }).observe(container);
    },

    _renderMonteCarloStats(stats, el) {
        const probColor = stats.prob_profit >= 50 ? 'text-green-400' : 'text-red-400';
        el.innerHTML = `
            <div class="grid grid-cols-3 sm:grid-cols-6 gap-2 text-center">
                <div><div class="text-[10px] text-gray-500">Prob Profit</div><div class="text-xs font-bold ${probColor}">${stats.prob_profit}%</div></div>
                <div><div class="text-[10px] text-gray-500">Median</div><div class="text-xs font-bold text-gray-300">₹${(stats.median / 1000).toFixed(1)}K</div></div>
                <div><div class="text-[10px] text-gray-500">Mean</div><div class="text-xs font-bold text-gray-300">₹${(stats.mean / 1000).toFixed(1)}K</div></div>
                <div><div class="text-[10px] text-gray-500">P5 (Worst)</div><div class="text-xs font-bold text-red-400">₹${(stats.p5 / 1000).toFixed(1)}K</div></div>
                <div><div class="text-[10px] text-gray-500">P95 (Best)</div><div class="text-xs font-bold text-green-400">₹${(stats.p95 / 1000).toFixed(1)}K</div></div>
                <div><div class="text-[10px] text-gray-500">Std Dev</div><div class="text-xs font-bold text-gray-400">₹${(stats.std / 1000).toFixed(1)}K</div></div>
            </div>
        `;
    },

    async runSignalBacktest() {
        const symbolEl = document.getElementById('backtestSymbol');
        const daysEl = document.getElementById('backtestDays');
        const loading = document.getElementById('backtestSignalLoading');
        const results = document.getElementById('backtestSignalResults');
        const btn = document.getElementById('btnRunBacktest');

        const symbol = (symbolEl.value || '').trim().toUpperCase();
        if (!symbol) {
            App.showToast('Enter a symbol to backtest', 'error');
            return;
        }

        loading.classList.remove('hidden');
        results.classList.add('hidden');
        btn.disabled = true;

        try {
            const data = await API.getBacktestSignal(symbol, parseInt(daysEl.value));
            this.renderSignalBacktest(data);
        } catch (e) {
            App.showToast('Backtest failed: ' + e.message, 'error');
        } finally {
            loading.classList.add('hidden');
            btn.disabled = false;
        }
    },

    renderSignalBacktest(data) {
        const results = document.getElementById('backtestSignalResults');
        results.classList.remove('hidden');

        const ns = data.new_system;
        const os = data.old_system;

        // Accuracy numbers
        const newAccEl = document.getElementById('btNewAccuracy');
        const oldAccEl = document.getElementById('btOldAccuracy');
        const newColor = ns.accuracy >= 55 ? 'text-green-400' : (ns.accuracy >= 45 ? 'text-yellow-400' : 'text-red-400');
        const oldColor = os.accuracy >= 55 ? 'text-green-400' : (os.accuracy >= 45 ? 'text-yellow-400' : 'text-red-400');
        newAccEl.textContent = ns.accuracy.toFixed(1) + '%';
        newAccEl.className = `text-2xl font-bold ${newColor}`;
        oldAccEl.textContent = os.accuracy.toFixed(1) + '%';
        oldAccEl.className = `text-2xl font-bold ${oldColor}`;

        document.getElementById('btNewTotal').textContent = ns.total_days;
        document.getElementById('btNewDir').textContent = ns.directional_calls;
        document.getElementById('btNewNeutral').textContent = ns.neutral_calls;
        document.getElementById('btOldTotal').textContent = os.total_days;
        document.getElementById('btOldDir').textContent = os.directional_calls;
        document.getElementById('btOldNeutral').textContent = os.neutral_calls;

        // Delta badge
        const delta = ns.accuracy - os.accuracy;
        const deltaEl = document.getElementById('btDelta');
        if (delta > 0) {
            deltaEl.innerHTML = `<span class="px-2 py-0.5 rounded-full bg-green-900 text-green-400 font-bold">New system +${delta.toFixed(1)}% better</span>`;
        } else if (delta < 0) {
            deltaEl.innerHTML = `<span class="px-2 py-0.5 rounded-full bg-red-900 text-red-400 font-bold">Old system +${Math.abs(delta).toFixed(1)}% better</span>`;
        } else {
            deltaEl.innerHTML = `<span class="px-2 py-0.5 rounded-full bg-gray-800 text-gray-400">Both systems tied</span>`;
        }

        // Daily results table (new system)
        const tbody = document.getElementById('btDailyResults');
        if (ns.daily_results && ns.daily_results.length > 0) {
            tbody.innerHTML = ns.daily_results.map(r => {
                const predColor = r.predicted === 'UP' ? 'text-green-400' : (r.predicted === 'DOWN' ? 'text-red-400' : 'text-yellow-400');
                const actColor = r.actual === 'UP' ? 'text-green-400' : 'text-red-400';
                const icon = r.correct ? '<span class="text-green-400">&#10003;</span>' : '<span class="text-red-400">&#10007;</span>';
                return `<tr class="border-b border-gray-800">
                    <td class="px-2 py-1 text-gray-400">${r.date}</td>
                    <td class="px-2 py-1 text-center font-mono text-gray-300">${r.score}</td>
                    <td class="px-2 py-1 text-center ${predColor}">${r.predicted}</td>
                    <td class="px-2 py-1 text-center ${actColor}">${r.actual}</td>
                    <td class="px-2 py-1 text-center">${icon}</td>
                </tr>`;
            }).join('');
        } else {
            tbody.innerHTML = '<tr><td colspan="5" class="text-center py-3 text-gray-500">No results</td></tr>';
        }
    },

    // --- Correlation Analysis ---

    _corrPeriod: '6mo',

    setCorrPeriod(period) {
        this._corrPeriod = period;
        // Update button states
        document.querySelectorAll('.corr-period-btn').forEach(btn => {
            btn.classList.toggle('bg-accent-blue', btn.dataset.period === period);
            btn.classList.toggle('text-white', btn.dataset.period === period);
            btn.classList.toggle('bg-dark-700', btn.dataset.period !== period);
            btn.classList.toggle('text-gray-400', btn.dataset.period !== period);
        });
        this.loadCorrelation();
        // Reload stock correlations if a stock is currently viewed
        const stockCorrContainer = document.getElementById('stockCorrelations');
        if (stockCorrContainer && stockCorrContainer.dataset.symbol) {
            this.loadStockCorrelations(stockCorrContainer.dataset.symbol);
        }
    },

    async loadCorrelation() {
        const container = document.getElementById('sectorCorrHeatmap');
        if (!container) return;
        container.innerHTML = '<div class="text-center py-6 text-gray-500 text-sm">Loading sector correlations...</div>';
        try {
            const data = await API.getSectorCorrelation(this._corrPeriod);
            this.renderCorrelationHeatmap(data);
        } catch (e) {
            container.innerHTML = '<div class="text-center py-6 text-gray-500 text-sm">Failed to load sector correlations</div>';
            console.error('Failed to load sector correlation:', e);
        }
        // Auto-load stock correlations if a stock is currently selected
        if (typeof App !== 'undefined' && App.currentSymbol) {
            this.loadStockCorrelations(App.currentSymbol);
        }
    },

    _corrColor(val) {
        // Color scale: dark red (-1) -> white (0) -> dark green (+1)
        const clamped = Math.max(-1, Math.min(1, val));
        if (clamped >= 0) {
            // White to dark green
            const r = Math.round(255 * (1 - clamped));
            const g = Math.round(255 - (255 - 100) * clamped);
            const b = Math.round(255 * (1 - clamped));
            return `rgb(${r},${g},${b})`;
        } else {
            // White to dark red
            const abs = Math.abs(clamped);
            const r = Math.round(255 - (255 - 180) * abs);
            const g = Math.round(255 * (1 - abs));
            const b = Math.round(255 * (1 - abs));
            return `rgb(${r},${g},${b})`;
        }
    },

    _corrInterpretation(val) {
        const abs = Math.abs(val);
        if (abs >= 0.8) return 'Very strong';
        if (abs >= 0.6) return 'Strong';
        if (abs >= 0.4) return 'Moderate';
        if (abs >= 0.2) return 'Weak';
        return 'Very weak';
    },

    renderCorrelationHeatmap(data) {
        const container = document.getElementById('sectorCorrHeatmap');
        if (!container || !data || !data.symbols || !data.matrix) {
            if (container) container.innerHTML = '<div class="text-center py-6 text-gray-500 text-sm">No data available</div>';
            return;
        }

        const symbols = data.symbols;
        const matrix = data.matrix;
        const n = symbols.length;
        const cellSize = 52;
        const headerHeight = 80;
        const labelWidth = 70;

        let html = `<div class="overflow-x-auto"><div style="min-width:${labelWidth + n * cellSize}px">`;

        // Column headers (rotated 45 degrees)
        html += `<div style="display:flex;margin-left:${labelWidth}px;height:${headerHeight}px;align-items:flex-end">`;
        for (let j = 0; j < n; j++) {
            html += `<div style="width:${cellSize}px;text-align:center;position:relative;height:${headerHeight}px">
                <span style="position:absolute;bottom:4px;left:50%;transform:translateX(-50%) rotate(-45deg);transform-origin:center;white-space:nowrap;font-size:10px;color:#9ca3af">${symbols[j]}</span>
            </div>`;
        }
        html += '</div>';

        // Rows
        for (let i = 0; i < n; i++) {
            html += '<div style="display:flex;align-items:center">';
            // Row label
            html += `<div style="width:${labelWidth}px;text-align:right;padding-right:8px;font-size:10px;color:#9ca3af;flex-shrink:0">${symbols[i]}</div>`;
            // Cells
            for (let j = 0; j < n; j++) {
                const val = matrix[i][j];
                const bgColor = this._corrColor(val);
                const textColor = Math.abs(val) > 0.5 ? '#fff' : '#1f2937';
                const interp = this._corrInterpretation(val);
                const direction = val >= 0 ? 'positive' : 'negative';
                html += `<div
                    style="width:${cellSize}px;height:${cellSize}px;background:${bgColor};display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;color:${textColor};cursor:default;border:1px solid rgba(55,65,81,0.3);position:relative"
                    class="corr-cell"
                    title="${symbols[i]} vs ${symbols[j]}: ${val.toFixed(4)} (${interp} ${direction} correlation)"
                >${val.toFixed(2)}</div>`;
            }
            html += '</div>';
        }

        html += '</div></div>';
        container.innerHTML = html;
    },

    async loadStockCorrelations(symbol) {
        const container = document.getElementById('stockCorrelations');
        if (!container) return;
        if (!symbol) {
            container.innerHTML = '<div class="text-center py-4 text-gray-500 text-sm">Select a stock to see its correlations with NIFTY 50 stocks</div>';
            return;
        }
        container.dataset.symbol = symbol;
        container.innerHTML = '<div class="text-center py-4 text-gray-500 text-sm">Loading correlations...</div>';
        try {
            const data = await API.getStockCorrelations(symbol, 10, this._corrPeriod);
            this.renderStockCorrelations(data);
        } catch (e) {
            container.innerHTML = '<div class="text-center py-4 text-gray-500 text-sm">Failed to load correlations</div>';
            console.error('Failed to load stock correlations:', e);
        }
    },

    renderStockCorrelations(data) {
        const container = document.getElementById('stockCorrelations');
        if (!container || !data) return;

        const most = data.most_correlated || [];
        const least = data.least_correlated || [];

        let html = `<div class="mb-2 text-xs text-gray-400">Correlations for <span class="text-white font-bold">${data.symbol}</span> (${data.period})</div>`;

        // Most correlated
        if (most.length > 0) {
            html += '<div class="mb-3"><div class="text-[10px] text-gray-500 mb-1 font-semibold uppercase">Most Correlated (move together)</div>';
            most.forEach(item => {
                const pct = Math.round(Math.abs(item.correlation) * 100);
                const barColor = item.correlation >= 0.7 ? 'bg-green-500' : 'bg-green-700';
                html += `<div class="flex items-center gap-2 py-1">
                    <span class="text-xs text-white w-24 truncate cursor-pointer hover:text-accent-blue" onclick="Search.select('${item.symbol}','${item.symbol}')">${item.symbol}</span>
                    <div class="flex-1 bg-dark-700 rounded-full h-3 overflow-hidden">
                        <div class="${barColor} h-full rounded-full transition-all" style="width:${pct}%"></div>
                    </div>
                    <span class="text-xs text-green-400 w-12 text-right font-mono">${item.correlation.toFixed(2)}</span>
                </div>`;
            });
            html += '</div>';
        }

        // Least correlated
        if (least.length > 0) {
            html += '<div><div class="text-[10px] text-gray-500 mb-1 font-semibold uppercase">Least Correlated (diversification)</div>';
            least.forEach(item => {
                const pct = Math.round(Math.abs(item.correlation) * 100);
                const barColor = item.correlation < 0 ? 'bg-red-500' : 'bg-yellow-600';
                const textColor = item.correlation < 0 ? 'text-red-400' : 'text-yellow-400';
                html += `<div class="flex items-center gap-2 py-1">
                    <span class="text-xs text-white w-24 truncate cursor-pointer hover:text-accent-blue" onclick="Search.select('${item.symbol}','${item.symbol}')">${item.symbol}</span>
                    <div class="flex-1 bg-dark-700 rounded-full h-3 overflow-hidden">
                        <div class="${barColor} h-full rounded-full transition-all" style="width:${pct}%"></div>
                    </div>
                    <span class="text-xs ${textColor} w-12 text-right font-mono">${item.correlation.toFixed(2)}</span>
                </div>`;
            });
            html += '</div>';
        }

        container.innerHTML = html;
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
        const tfLabels = { intraday: 'Intraday', short_term: '1 Week', long_term: '3 Months' };
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

            const tfShort = { intraday_10m: '10m', intraday_30m: '30m', short_15m: '15m', short_1h: '1h', short_4h: '4h' };

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
