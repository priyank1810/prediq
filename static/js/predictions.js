const Predictions = {
    currentHorizon: '1d',

    init() {
        document.querySelectorAll('.horizon-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.horizon-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.currentHorizon = btn.dataset.horizon;
                if (App.currentSymbol) this.loadPredictions(App.currentSymbol);
            });
        });
    },

    async loadPredictions(symbol) {
        const panel = document.getElementById('predictionPanel');
        const loading = document.getElementById('predictionLoading');
        const results = document.getElementById('predictionResults');

        panel.classList.remove('hidden');
        loading.classList.remove('hidden');
        results.classList.add('hidden');

        try {
            const data = await API.getPredictions(symbol, this.currentHorizon);
            this.displayResults(data);
        } catch (e) {
            loading.classList.add('hidden');
            App.showToast('Failed to generate predictions: ' + e.message, 'error');
        }
    },

    displayResults(data) {
        const loading = document.getElementById('predictionLoading');
        const results = document.getElementById('predictionResults');
        loading.classList.add('hidden');
        results.classList.remove('hidden');

        const horizonLabel = data.horizon_label || data.horizon;

        // LSTM
        if (data.lstm) {
            const lastPred = data.lstm.predictions[data.lstm.predictions.length - 1];
            document.getElementById('lstmPrediction').textContent = `₹${lastPred.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
            document.getElementById('lstmConfidence').textContent = `${data.lstm.confidence_score}% conf`;
            document.getElementById('lstmMape').textContent = `MAPE: ${data.lstm.mape}%`;
        } else {
            document.getElementById('lstmPrediction').textContent = data.lstm_error ? 'N/A' : '-';
            document.getElementById('lstmConfidence').textContent = '';
            document.getElementById('lstmMape').textContent = data.lstm_error || '';
        }

        // Prophet
        if (data.prophet) {
            const lastPred = data.prophet.predictions[data.prophet.predictions.length - 1];
            document.getElementById('prophetPrediction').textContent = `₹${lastPred.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
            const lower = data.prophet.confidence_lower[data.prophet.confidence_lower.length - 1];
            const upper = data.prophet.confidence_upper[data.prophet.confidence_upper.length - 1];
            document.getElementById('prophetConfidence').textContent = '80% band';
            document.getElementById('prophetBand').textContent = `₹${lower.toFixed(0)} - ₹${upper.toFixed(0)}`;
        } else {
            document.getElementById('prophetPrediction').textContent = data.prophet_error ? 'N/A' : '-';
            document.getElementById('prophetConfidence').textContent = '';
            document.getElementById('prophetBand').textContent = data.prophet_error || '';
        }

        // XGBoost
        if (data.xgboost) {
            const lastPred = data.xgboost.predictions[data.xgboost.predictions.length - 1];
            document.getElementById('xgbPrediction').textContent = `₹${lastPred.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
            document.getElementById('xgbConfidence').textContent = `${data.xgboost.confidence_score}% conf`;
            document.getElementById('xgbMape').textContent = `MAPE: ${data.xgboost.mape}%`;
        } else {
            document.getElementById('xgbPrediction').textContent = data.xgboost_error ? 'N/A' : '-';
            document.getElementById('xgbConfidence').textContent = '';
            document.getElementById('xgbMape').textContent = data.xgboost_error || '';
        }

        // Ensemble
        if (data.ensemble) {
            const lastPred = data.ensemble.predictions[data.ensemble.predictions.length - 1];
            document.getElementById('ensemblePrediction').textContent = `₹${lastPred.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
            const currentPrice = parseFloat(document.getElementById('stockPrice').textContent.replace('₹', '').replace(/,/g, ''));
            if (currentPrice) {
                const change = lastPred - currentPrice;
                const changePct = (change / currentPrice * 100);
                const changeEl = document.getElementById('ensembleChange');
                const sign = change >= 0 ? '+' : '';
                changeEl.textContent = `${sign}${change.toFixed(2)} (${changePct.toFixed(2)}%) in ${horizonLabel}`;
                changeEl.className = `text-xs mt-1 ${change >= 0 ? 'text-green-400' : 'text-red-400'}`;
            }
        }

        // Explanation panel
        this.renderExplanation(data.explanation);

        // SHAP drivers
        this.renderSHAPDrivers(data.shap_drivers);

        // Contribution breakdown
        this.renderContributionBreakdown(data.contribution_breakdown);

        // Regime badge
        this.renderRegime(data.regime);

        // Multi-step prediction table (show for horizons with multiple data points)
        const table = document.getElementById('predictionTable');
        const tbody = document.getElementById('predictionTableBody');
        if (data.ensemble && data.ensemble.predictions.length > 1) {
            table.classList.remove('hidden');
            tbody.innerHTML = data.ensemble.dates.map((date, i) => `
                <tr class="border-b border-gray-800">
                    <td class="py-2 text-gray-400">${date}</td>
                    <td class="py-2 text-right text-blue-400">${data.lstm && data.lstm.predictions[i] != null ? '₹' + data.lstm.predictions[i].toFixed(2) : '-'}</td>
                    <td class="py-2 text-right text-green-400">${data.prophet && data.prophet.predictions[i] != null ? '₹' + data.prophet.predictions[i].toFixed(2) : '-'}</td>
                    <td class="py-2 text-right text-purple-400">${data.xgboost && data.xgboost.predictions[i] != null ? '₹' + data.xgboost.predictions[i].toFixed(2) : '-'}</td>
                    <td class="py-2 text-right text-yellow-400">₹${data.ensemble.predictions[i].toFixed(2)}</td>
                </tr>
            `).join('');
        } else {
            table.classList.add('hidden');
        }
    },

    renderExplanation(explanation) {
        const panel = document.getElementById('predictionExplanation');
        if (!explanation) {
            panel.classList.add('hidden');
            return;
        }
        panel.classList.remove('hidden');

        // Direction badge
        const dirEl = document.getElementById('explDirection');
        const dir = explanation.direction || 'NEUTRAL';
        const colorMap = { BULLISH: 'text-green-400', BEARISH: 'text-red-400', NEUTRAL: 'text-gray-400' };
        dirEl.textContent = dir;
        dirEl.className = `font-bold ${colorMap[dir] || 'text-gray-400'}`;

        // Summary
        document.getElementById('explSummary').textContent = explanation.summary || '';

        // Key drivers
        const driversEl = document.getElementById('explDrivers');
        if (explanation.key_drivers && explanation.key_drivers.length) {
            driversEl.innerHTML = explanation.key_drivers.map(d => {
                const impactColor = d.impact === 'positive' ? 'border-green-600 bg-green-900/20'
                    : d.impact === 'negative' ? 'border-red-600 bg-red-900/20'
                    : 'border-gray-600 bg-gray-900/20';
                const icon = d.impact === 'positive' ? '&#9650;' : d.impact === 'negative' ? '&#9660;' : '&#9644;';
                const iconColor = d.impact === 'positive' ? 'text-green-400' : d.impact === 'negative' ? 'text-red-400' : 'text-gray-400';
                return `<div class="border ${impactColor} rounded-lg p-2.5 flex items-start gap-2">
                    <span class="${iconColor} text-sm mt-0.5">${icon}</span>
                    <div>
                        <div class="text-white text-sm font-medium">${d.factor}</div>
                        <div class="text-gray-400 text-xs">${d.detail}</div>
                    </div>
                </div>`;
            }).join('');
        } else {
            driversEl.innerHTML = '';
        }

        // Risk factors
        const risksEl = document.getElementById('explRisks');
        if (explanation.risk_factors && explanation.risk_factors.length) {
            risksEl.innerHTML = `<div class="text-xs text-gray-500 mb-1 font-medium">Risk Factors</div>` +
                explanation.risk_factors.map(r =>
                    `<div class="text-xs text-yellow-500 flex items-center gap-1 mb-1">
                        <span>&#9888;</span> ${r}
                    </div>`
                ).join('');
        } else {
            risksEl.innerHTML = '';
        }

        // Support/Resistance
        const srEl = document.getElementById('explSupportResistance');
        if (explanation.support_resistance) {
            const sr = explanation.support_resistance;
            srEl.textContent = `Support: ₹${sr.support?.toLocaleString('en-IN') || '-'}  |  Resistance: ₹${sr.resistance?.toLocaleString('en-IN') || '-'}`;
        } else {
            srEl.textContent = '';
        }
    },

    renderSHAPDrivers(drivers) {
        const panel = document.getElementById('shapDriversPanel');
        const list = document.getElementById('shapDriversList');
        if (!panel || !list) return;

        if (!drivers || drivers.length === 0) {
            panel.classList.add('hidden');
            return;
        }
        panel.classList.remove('hidden');

        list.innerHTML = drivers.map(d => {
            const isPos = d.direction === 'positive';
            const color = isPos ? 'text-green-400' : 'text-red-400';
            const bgColor = isPos ? 'bg-green-900/20 border-green-800' : 'bg-red-900/20 border-red-800';
            const arrow = isPos ? '&#9650;' : '&#9660;';
            const barWidth = Math.min(100, Math.abs(d.impact_value || 0) * 100);
            const barColor = isPos ? 'bg-green-500' : 'bg-red-500';
            return `
                <div class="flex items-center gap-3 ${bgColor} border rounded-lg p-2">
                    <span class="${color} text-sm">${arrow}</span>
                    <div class="flex-1">
                        <div class="text-white text-xs font-medium">${d.feature}</div>
                        <div class="mt-1 h-1.5 bg-dark-600 rounded-full overflow-hidden">
                            <div class="h-full ${barColor} rounded-full" style="width:${barWidth}%"></div>
                        </div>
                    </div>
                    <span class="${color} text-xs font-mono">${d.impact_value ? d.impact_value.toFixed(3) : ''}</span>
                </div>
            `;
        }).join('');
    },

    renderContributionBreakdown(breakdown) {
        const panel = document.getElementById('contributionPanel');
        if (!panel) return;

        if (!breakdown) {
            panel.classList.add('hidden');
            return;
        }
        panel.classList.remove('hidden');

        const tech = breakdown.technical || 0;
        const seasonal = breakdown.seasonal || 0;
        const fund = breakdown.fundamental || 0;
        const sent = breakdown.sentiment || 0;

        document.getElementById('contribTechnical').textContent = tech.toFixed(0) + '%';
        document.getElementById('contribSeasonal').textContent = seasonal.toFixed(0) + '%';
        document.getElementById('contribFundamental').textContent = fund.toFixed(0) + '%';
        document.getElementById('contribSentiment').textContent = sent.toFixed(0) + '%';

        document.getElementById('contribBarTech').style.width = tech + '%';
        document.getElementById('contribBarSeasonal').style.width = seasonal + '%';
        document.getElementById('contribBarFund').style.width = fund + '%';
        document.getElementById('contribBarSent').style.width = sent + '%';
    },

    renderRegime(regime) {
        const panel = document.getElementById('regimePanel');
        const badge = document.getElementById('regimeBadge');
        if (!panel || !badge) return;

        if (!regime) {
            panel.classList.add('hidden');
            return;
        }
        panel.classList.remove('hidden');

        const regimeColors = {
            'bull': 'bg-green-900 text-green-400',
            'bear': 'bg-red-900 text-red-400',
            'sideways': 'bg-yellow-900 text-yellow-400',
            'volatile': 'bg-purple-900 text-purple-400',
        };
        const label = (regime.regime || 'unknown').toUpperCase();
        const colorClass = regimeColors[regime.regime] || 'bg-gray-800 text-gray-400';
        const conf = regime.confidence ? ` (${(regime.confidence * 100).toFixed(0)}%)` : '';

        badge.className = `text-xs px-2 py-0.5 rounded-full font-medium ${colorClass}`;
        badge.textContent = label + conf;
    }
};
