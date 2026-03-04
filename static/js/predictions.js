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

        // Model label
        const modelLabelEl = document.getElementById('predModelLabel');
        if (modelLabelEl && data.models_used) {
            const nameMap = { prophet: 'Prophet', xgboost: 'XGBoost' };
            const names = data.models_used.map(m => nameMap[m] || m).join(' + ');
            modelLabelEl.textContent = `Powered by ${names}`;
        }

        // Confidence info (show confidence band if available)
        const confEl = document.getElementById('predConfidenceInfo');
        if (confEl) {
            const ens = data.ensemble;
            if (ens && ens.confidence_lower && ens.confidence_upper) {
                const lower = ens.confidence_lower[ens.confidence_lower.length - 1];
                const upper = ens.confidence_upper[ens.confidence_upper.length - 1];
                confEl.textContent = `80% confidence range: ₹${lower.toFixed(0)} – ₹${upper.toFixed(0)}`;
                confEl.classList.remove('hidden');
            } else if (ens && ens.confidence_score) {
                confEl.textContent = `Confidence: ${ens.confidence_score}%`;
                confEl.classList.remove('hidden');
            } else {
                confEl.classList.add('hidden');
            }
        }

        // Primary prediction (ensemble or single model)
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
                changeEl.className = `text-sm mt-1 ${change >= 0 ? 'text-green-400' : 'text-red-400'}`;

                // Update signal badge with AI prediction data
                const aiDir = changePct > 0.1 ? 'BULLISH' : (changePct < -0.1 ? 'BEARISH' : 'NEUTRAL');
                const aiConf = Math.min(99, Math.abs(changePct) * 10);
                App.displaySignalBadge({ direction: aiDir, confidence: aiConf, source: 'ai' });
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

        // Volume analysis
        this.renderVolumeAnalysis(data.volume_analysis);

        // Market context: sentiment + global data
        this.renderMarketContext(data);

        // Multi-step prediction table
        const table = document.getElementById('predictionTable');
        const tbody = document.getElementById('predictionTableBody');
        const thead = document.getElementById('predictionTableHead');
        if (data.ensemble && data.ensemble.predictions.length > 1) {
            table.classList.remove('hidden');

            if (thead) thead.innerHTML = `<tr>
                <th class="py-2 text-left text-gray-500 text-xs">Date</th>
                <th class="py-2 text-right text-yellow-400 text-xs">Predicted Price</th>
            </tr>`;

            tbody.innerHTML = data.ensemble.dates.map((date, i) => {
                return `<tr class="border-b border-gray-800">
                    <td class="py-2 text-gray-400">${date}</td>
                    <td class="py-2 text-right text-yellow-400">₹${data.ensemble.predictions[i].toFixed(2)}</td>
                </tr>`;
            }).join('');
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

    renderMarketContext(data) {
        const panel = document.getElementById('predMarketContext');
        if (!panel) return;

        const hasSentiment = data.sentiment && (data.sentiment.headline_count > 0 || data.sentiment.score !== 0);
        const hasGlobal = data.global_market && (data.global_market.markets?.length > 0 || data.global_market.headlines?.length > 0);

        if (!hasSentiment && !hasGlobal) {
            panel.classList.add('hidden');
            return;
        }
        panel.classList.remove('hidden');

        // Sentiment summary + headlines (merged: stock-specific + global)
        const summaryEl = document.getElementById('predSentimentSummary');
        const headlinesEl = document.getElementById('predSentimentHeadlines');
        if (data.sentiment) {
            const s = data.sentiment;
            const scoreColor = s.score > 10 ? 'text-green-400' : (s.score < -10 ? 'text-red-400' : 'text-gray-400');
            const magBadge = s.news_magnitude >= 60
                ? '<span class="text-[10px] px-1.5 py-0.5 rounded-full bg-red-900 text-red-400 ml-2">High Impact</span>'
                : s.news_magnitude >= 30
                ? '<span class="text-[10px] px-1.5 py-0.5 rounded-full bg-yellow-900 text-yellow-400 ml-2">Moderate</span>'
                : '';
            summaryEl.innerHTML = `Score: <span class="${scoreColor} font-medium">${s.score > 0 ? '+' : ''}${s.score.toFixed(0)}</span> | ${s.positive_count} positive, ${s.negative_count} negative, ${s.neutral_count} neutral${magBadge}`;

            if (s.headlines && s.headlines.length > 0) {
                headlinesEl.innerHTML = s.headlines.slice(0, 8).map(h => {
                    const c = h.sentiment === 'positive' ? 'text-green-400 bg-green-900' :
                              (h.sentiment === 'negative' ? 'text-red-400 bg-red-900' : 'text-gray-400 bg-gray-800');
                    const bigTag = h.big_event ? '<span class="text-yellow-400 ml-1">!</span>' : '';
                    const srcTag = h.source === 'global' ? '<span class="text-[9px] text-gray-600 ml-1">[global]</span>' : '';
                    return `<div class="flex items-start gap-2 py-1 border-b border-gray-800 last:border-0">
                        <span class="text-[10px] px-1.5 py-0.5 rounded ${c} whitespace-nowrap">${h.sentiment}</span>
                        <a href="${h.link || '#'}" target="_blank" rel="noopener" class="text-[11px] text-gray-300 hover:text-white line-clamp-2">${h.title}${bigTag}${srcTag}</a>
                    </div>`;
                }).join('');
            } else {
                headlinesEl.innerHTML = '<div class="text-gray-500 text-xs">No recent headlines</div>';
            }
        }

        // Global markets ticker + news
        const tickerEl = document.getElementById('predGlobalTicker');
        const newsEl = document.getElementById('predGlobalNews');
        if (data.global_market) {
            const gm = data.global_market;
            // Market ticker
            if (gm.markets && gm.markets.length > 0) {
                tickerEl.innerHTML = gm.markets.map(m => {
                    const color = m.change_pct > 0 ? 'text-green-400' : (m.change_pct < 0 ? 'text-red-400' : 'text-gray-400');
                    const sign = m.change_pct > 0 ? '+' : '';
                    const arrow = m.change_pct > 0 ? '&#9650;' : (m.change_pct < 0 ? '&#9660;' : '&#9654;');
                    return `<div class="flex items-center gap-1 px-2 py-1 bg-dark-600 rounded text-[10px]">
                        <span class="text-gray-400">${m.name}</span>
                        <span class="${color}">${arrow} ${sign}${m.change_pct}%</span>
                    </div>`;
                }).join('');
            }

            // Global news headlines
            const magnitude = gm.news_magnitude || 0;
            const magBadge = magnitude >= 60
                ? `<span class="text-[10px] px-2 py-0.5 rounded-full bg-red-900 text-red-400 mb-1 inline-block">High Impact (${magnitude})</span>`
                : magnitude >= 30
                ? `<span class="text-[10px] px-2 py-0.5 rounded-full bg-yellow-900 text-yellow-400 mb-1 inline-block">Moderate (${magnitude})</span>`
                : '';

            if (gm.headlines && gm.headlines.length > 0) {
                newsEl.innerHTML = magBadge + gm.headlines.slice(0, 5).map(h => {
                    const c = h.sentiment === 'positive' ? 'text-green-400 bg-green-900' :
                              (h.sentiment === 'negative' ? 'text-red-400 bg-red-900' : 'text-gray-400 bg-gray-800');
                    const bigTag = h.big_event ? '<span class="text-yellow-400 ml-1">!</span>' : '';
                    return `<div class="flex items-start gap-2 py-1 border-b border-gray-800 last:border-0">
                        <span class="text-[10px] px-1.5 py-0.5 rounded ${c} whitespace-nowrap">${h.sentiment}</span>
                        <span class="text-[11px] text-gray-300 line-clamp-2">${h.title}${bigTag}</span>
                    </div>`;
                }).join('');
            } else {
                newsEl.innerHTML = magBadge || '<div class="text-gray-500 text-xs">No global news</div>';
            }
        }

        // Market adjustment badge
        const adjEl = document.getElementById('predAdjustmentBadge');
        if (adjEl && data.ensemble && data.ensemble.market_adjustment) {
            const adj = data.ensemble.market_adjustment;
            adjEl.classList.remove('hidden');
            const sign = adj.adjustment_pct >= 0 ? '+' : '';
            const color = adj.adjustment_pct > 0 ? 'text-green-400' : (adj.adjustment_pct < 0 ? 'text-red-400' : 'text-gray-400');
            const ctxW = adj.context_weight ? (adj.context_weight * 100).toFixed(0) : '10';
            adjEl.innerHTML = `Market context adjustment: <span class="${color} font-medium">${sign}${adj.adjustment_pct.toFixed(2)}%</span> (context: ${ctxW}% weight, score: ${adj.blended_score || 0}, magnitude: ${adj.news_magnitude})`;
        } else if (adjEl) {
            adjEl.classList.add('hidden');
        }
    },

    renderVolumeAnalysis(vol) {
        const panel = document.getElementById('volumeAnalysisPanel');
        if (!panel) return;

        if (!vol || !vol.current_volume) {
            panel.classList.add('hidden');
            return;
        }
        panel.classList.remove('hidden');

        // Format volume in Indian notation (Cr / L)
        const fmtVol = (v) => {
            if (v >= 10000000) return (v / 10000000).toFixed(1) + 'Cr';
            if (v >= 100000) return (v / 100000).toFixed(1) + 'L';
            if (v >= 1000) return (v / 1000).toFixed(1) + 'K';
            return v.toLocaleString('en-IN');
        };

        document.getElementById('volCurrent').textContent = fmtVol(vol.current_volume);

        // Volume ratio with color
        const ratioEl = document.getElementById('volRatio');
        const ratio = vol.volume_ratio || 1;
        const ratioColor = ratio >= 1.2 ? 'text-green-400' : (ratio <= 0.8 ? 'text-red-400' : 'text-gray-400');
        ratioEl.textContent = ratio.toFixed(1) + 'x';
        ratioEl.className = `text-sm font-bold ${ratioColor}`;

        // Trend with arrow
        const trendEl = document.getElementById('volTrend');
        const trendMap = {
            'increasing': { arrow: '\u25B2', color: 'text-green-400' },
            'decreasing': { arrow: '\u25BC', color: 'text-red-400' },
            'stable': { arrow: '\u25B6', color: 'text-gray-400' },
        };
        const t = trendMap[vol.volume_trend] || trendMap['stable'];
        const pct = vol.volume_trend_pct != null ? ` ${vol.volume_trend_pct > 0 ? '+' : ''}${vol.volume_trend_pct}%` : '';
        trendEl.innerHTML = `<span class="${t.color}">${t.arrow} ${vol.volume_trend}${pct}</span>`;

        // Conviction badge
        const convEl = document.getElementById('volConviction');
        const convColors = {
            'high': 'bg-green-900 text-green-400 border-green-800',
            'moderate': 'bg-yellow-900 text-yellow-400 border-yellow-800',
            'low': 'bg-red-900 text-red-400 border-red-800',
        };
        const cc = convColors[vol.conviction] || convColors['moderate'];
        const supIcon = vol.supports_prediction ? '\u2713' : '\u26A0';
        convEl.innerHTML = `<span class="inline-block px-2 py-0.5 rounded-full border text-[10px] font-medium ${cc}">${supIcon} ${vol.conviction.toUpperCase()} conviction</span>` +
            (vol.conviction_detail ? `<div class="text-gray-500 mt-1">${vol.conviction_detail}</div>` : '');
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
