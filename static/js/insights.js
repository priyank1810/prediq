const Insights = {
    init() {},

    async load() {
        await Promise.all([
            this.loadHighConfidence(),
            this.loadAccuracy(),
            this.loadRecentFeed(),
            this.loadAccuracyBySector(),
            this.loadAccuracyByHorizon(),
            this.loadAccuracyByRegime(),
            this.loadBacktestPnL(),
            this.loadSmartAlerts(),
        ]);
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
                const time = s.created_at ? new Date(s.created_at).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' }) : '-';
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
                const time = s.created_at ? new Date(s.created_at).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' }) : '-';
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
            this._renderStatRows('accuracyByHorizon', data, 'horizon', 'horizon');
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

            if (tradesEl) tradesEl.textContent = data.total_trades ?? '-';
            if (winEl) {
                const wr = data.win_rate;
                winEl.textContent = wr != null ? wr.toFixed(1) + '%' : '-';
                winEl.className = `text-white font-bold ${wr >= 50 ? 'text-green-400' : 'text-red-400'}`;
            }
            if (totalEl) {
                const pnl = data.total_pnl;
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
    }
};
