window.Signals = {
    intradayChart: null,
    isLoading: false,
    _intradayResizeHandler: null,
    _signalTimestamp: null,
    _ageTimer: null,

    init() {
        API.setSignalUpdateHandler((data) => {
            if (data.symbol === App.currentSymbol) {
                this.displaySignal(data);
                App._lastSignalData = data;
            }
        });
        API.setOIUpdateHandler((data) => {
            if (data.symbol === App.currentSymbol) {
                this.renderOIAnalysis({ oi_analysis: { ...data, available: true } });
            }
        });
        API.setMTFUpdateHandler((data) => {
            if (data.symbol === App.currentSymbol) {
                this.renderMTFConfluence(data);
            }
        });
    },

    async loadSignal(symbol) {
        if (this.isLoading) return;
        this.isLoading = true;

        // Show shimmer for MTF signals
        Shimmer.show('mtfSignalsGrid', 'mtfSignals', 3);

        try {
            // Load MTF signals (the main data source now) + AI summary + accuracy in parallel
            const [mtfData] = await Promise.all([
                API.getMultiTimeframeSignals(symbol),
                this.loadStockAccuracy(symbol),
            ]);

            if (mtfData) {
                this.displayMultiTimeframeSignals(mtfData);
                this.displayOverviewTargets(mtfData);
            }
        } catch (e) {
            // Silently handle — MTF panel stays empty
        } finally {
            this.isLoading = false;
        }
    },

    displaySignal(data) {
        const loadingEl = document.getElementById('signalLoading');
        const resultsEl = document.getElementById('signalResults');
        if (loadingEl) loadingEl.classList.add('hidden');
        if (resultsEl) resultsEl.classList.remove('hidden');

        // Populate Technical tab signal breakdown
        this._renderTechSignal(data);

        // Market closed notice
        const closedNotice = document.getElementById('marketClosedNotice');
        if (closedNotice) closedNotice.remove();
        if (data.market_closed) {
            const notice = document.createElement('div');
            notice.id = 'marketClosedNotice';
            notice.className = 'bg-yellow-900/30 border border-yellow-700/50 rounded-lg px-4 py-2 mb-4 text-yellow-400 text-xs text-center';
            notice.textContent = data.message || 'Market is closed. Showing last signal from market hours.';
            document.getElementById('signalResults').prepend(notice);
        }

        // Direction arrow
        const arrowEl = document.getElementById('signalArrow');
        const dirEl = document.getElementById('signalDirection');
        const confEl = document.getElementById('signalConfidence');

        if (data.direction === 'BULLISH') {
            arrowEl.innerHTML = '&#9650;';
            arrowEl.className = 'text-6xl text-green-400 signal-active';
            dirEl.textContent = 'BULLISH';
            dirEl.className = 'text-2xl font-bold text-green-400';
        } else if (data.direction === 'BEARISH') {
            arrowEl.innerHTML = '&#9660;';
            arrowEl.className = 'text-6xl text-red-400 signal-active';
            dirEl.textContent = 'BEARISH';
            dirEl.className = 'text-2xl font-bold text-red-400';
        } else {
            arrowEl.innerHTML = '&#9654;';
            arrowEl.className = 'text-6xl text-yellow-400';
            dirEl.textContent = 'NEUTRAL';
            dirEl.className = 'text-2xl font-bold text-yellow-400';
        }
        confEl.textContent = `${data.confidence}% confidence`;

        // Signal price
        const priceEl = document.getElementById('signalPrice');
        if (priceEl) {
            const candles = data.intraday_candles;
            const price = candles && candles.length > 0 ? candles[candles.length - 1].close : null;
            priceEl.textContent = price ? `₹${price.toFixed(2)}` : '';
        }

        // Score bars (weights are now dynamic)
        this.renderScoreBar('technicalBar', 'technicalScore', data.technical.score, data.technical.weight);
        this.renderScoreBar('sentimentBar', 'sentimentScore', data.sentiment.score, data.sentiment.weight);
        this.renderScoreBar('globalBar', 'globalScore', data.global_market.score, data.global_market.weight);

        // Update global weight label if it changed
        const gwLabel = document.getElementById('globalWeightLabel');
        if (gwLabel) {
            const pct = ((data.global_market.weight || 0) * 100).toFixed(0);
            gwLabel.textContent = `Global (${pct}%)`;
            if ((data.global_market.weight || 0) > 0.15) {
                gwLabel.className = 'text-xs text-yellow-400 font-medium';
            } else {
                gwLabel.className = 'text-xs text-gray-400';
            }
        }

        // Technical details
        const techDet = document.getElementById('technicalDetails');
        if (data.technical.details && Object.keys(data.technical.details).length > 0) {
            const d = data.technical.details;
            const maCrossColor = d.ma_cross === 'bullish' ? 'text-green-400' : (d.ma_cross === 'bearish' ? 'text-red-400' : 'text-gray-500');
            const volColor = d.volume_ratio > 1.5 ? 'text-yellow-400' : 'text-gray-500';
            // Candlestick pattern tags
            const candleTags = (d.candlestick_patterns || []).map(p => {
                const cls = p.type === 'bullish' ? 'bg-green-900 text-green-400' :
                            p.type === 'bearish' ? 'bg-red-900 text-red-400' : 'bg-gray-800 text-gray-400';
                return `<span class="text-[10px] px-1.5 py-0.5 rounded ${cls}">${p.name}</span>`;
            }).join(' ');

            techDet.innerHTML = `
                <div class="flex flex-wrap items-center gap-1 mb-1">
                    <span class="text-xs text-gray-500">RSI: ${d.rsi || '-'}</span>
                    <span class="mx-1 text-gray-600">|</span>
                    <span class="text-xs ${volColor}">Vol: ${d.volume_ratio ? d.volume_ratio.toFixed(1) + 'x' : '-'}</span>
                    <span class="mx-1 text-gray-600">|</span>
                    <span class="text-xs text-gray-500">BB: ${d.bb_position != null ? (d.bb_position * 100).toFixed(0) + '%' : '-'}</span>
                    <span class="mx-1 text-gray-600">|</span>
                    <span class="text-xs text-gray-500">VWAP: ₹${d.vwap || '-'}</span>
                    <span class="mx-1 text-gray-600">|</span>
                    <span class="text-xs ${maCrossColor}">MA 5/9: ${d.ma_cross || '-'}</span>
                </div>
                ${candleTags ? '<div class="flex flex-wrap gap-1 mt-1">' + candleTags + '</div>' : ''}
            `;
        }

        // Sector-relative strength badge
        this.renderSectorBadge(data.sector_strength);

        // News
        this.displayHeadlines(data.sentiment.headlines || []);

        // Global markets + global news
        this.displayGlobalMarkets(data.global_market.markets || []);
        this.displayGlobalNews(data.global_market.headlines || [], data.global_market.news_magnitude || 0);

        // Support/Resistance levels
        this.renderSRLevels(data.support_resistance);

        // Multi-Timeframe Confluence
        this.renderMTFConfluence(data.mtf_confluence);

        // OI Analysis
        this.renderOIAnalysis(data);

        // Adaptive weights badge
        this.renderAdaptiveBadge(data);

        // Stock learning insight badge
        this.renderLearningBadge(data.stock_learning);

        // Load AI summary and accuracy in background
        const summarySymbol = data.symbol || (typeof App !== 'undefined' ? App.currentSymbol : null);
        if (summarySymbol) {
            this.loadAISummary(summarySymbol);
            this.loadStockAccuracy(summarySymbol);
        }

        // Intraday chart
        if (data.intraday_candles && data.intraday_candles.length > 0) {
            this.renderIntradayChart(data.intraday_candles, data.support_resistance, data.technical && data.technical.raw);
        }

        // Timestamp — always display in IST regardless of browser timezone
        const tsEl = document.getElementById('signalTimestamp');
        if (tsEl && data.timestamp) {
            this._signalTimestamp = Date.now();
            const ts = new Date(data.timestamp);
            const timeStr = ts.toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', second: '2-digit' });
            tsEl.textContent = `Updated: ${timeStr} IST`;
            this._startAgeTimer();
        }

        // Market status badge
        const badge = document.getElementById('marketBadge');
        if (badge) {
            if (data.market_open) {
                badge.textContent = 'LIVE';
                badge.className = 'text-xs px-2 py-0.5 rounded-full bg-green-900 text-green-400';
            } else {
                badge.textContent = 'CLOSED';
                badge.className = 'text-xs px-2 py-0.5 rounded-full bg-red-900 text-red-400';
            }
        }
    },

    renderScoreBar(barId, scoreId, score, weight) {
        const bar = document.getElementById(barId);
        const scoreEl = document.getElementById(scoreId);
        if (!bar || !scoreEl) return;

        score = score || 0;
        weight = weight || 0;

        const absScore = Math.min(100, Math.abs(score));
        const color = score > 0 ? '#00c853' : (score < 0 ? '#ff1744' : '#6b7280');

        if (score >= 0) {
            bar.style.background = `linear-gradient(to right, #1e2a4a 50%, ${color} 50%, ${color} ${50 + absScore / 2}%, #1e2a4a ${50 + absScore / 2}%)`;
        } else {
            bar.style.background = `linear-gradient(to right, #1e2a4a ${50 - absScore / 2}%, ${color} ${50 - absScore / 2}%, ${color} 50%, #1e2a4a 50%)`;
        }

        const sign = score > 0 ? '+' : '';
        scoreEl.textContent = `${sign}${score.toFixed(1)} (w: ${(weight * 100).toFixed(0)}%)`;
        scoreEl.className = `text-sm font-mono ${score > 0 ? 'text-green-400' : (score < 0 ? 'text-red-400' : 'text-gray-400')}`;
    },

    displayHeadlines(headlines) {
        const el = document.getElementById('newsHeadlines');
        if (!el) return;
        if (headlines.length === 0) {
            el.innerHTML = '<div class="text-gray-500 text-sm py-2">No recent news found</div>';
            return;
        }
        el.innerHTML = headlines.slice(0, 8).map(h => {
            const c = h.sentiment === 'positive' ? 'text-green-400 bg-green-900' :
                      (h.sentiment === 'negative' ? 'text-red-400 bg-red-900' : 'text-gray-400 bg-gray-800');
            return `
                <div class="flex items-start gap-2 py-1.5 border-b border-gray-800 last:border-0">
                    <span class="text-xs px-1.5 py-0.5 rounded ${c} whitespace-nowrap">${h.sentiment}</span>
                    <a href="${h.link}" target="_blank" rel="noopener"
                       class="text-xs text-gray-300 hover:text-white line-clamp-2">${h.title}</a>
                </div>
            `;
        }).join('');
    },

    displayGlobalNews(headlines, magnitude) {
        const el = document.getElementById('globalNewsHeadlines');
        if (!el) return;
        if (!headlines || headlines.length === 0) {
            el.innerHTML = '<div class="text-gray-500 text-sm py-2">No global news</div>';
            return;
        }
        // Show magnitude badge if significant
        const magBadge = magnitude >= 60
            ? `<span class="text-xs px-2 py-0.5 rounded-full bg-red-900 text-red-400 mb-2 inline-block">High Impact (${magnitude})</span>`
            : magnitude >= 30
            ? `<span class="text-xs px-2 py-0.5 rounded-full bg-yellow-900 text-yellow-400 mb-2 inline-block">Moderate Impact (${magnitude})</span>`
            : '';

        el.innerHTML = magBadge + headlines.slice(0, 6).map(h => {
            const c = h.sentiment === 'positive' ? 'text-green-400 bg-green-900' :
                      (h.sentiment === 'negative' ? 'text-red-400 bg-red-900' : 'text-gray-400 bg-gray-800');
            const bigTag = h.big_event ? '<span class="text-xs text-yellow-400 ml-1">!</span>' : '';
            return `
                <div class="flex items-start gap-2 py-1.5 border-b border-gray-800 last:border-0">
                    <span class="text-xs px-1.5 py-0.5 rounded ${c} whitespace-nowrap">${h.sentiment}</span>
                    <a href="${h.link}" target="_blank" rel="noopener"
                       class="text-xs text-gray-300 hover:text-white line-clamp-2">${h.title}${bigTag}</a>
                </div>
            `;
        }).join('');
    },

    displayGlobalMarkets(markets) {
        const el = document.getElementById('globalTicker');
        if (!el) return;
        el.innerHTML = markets.map(m => {
            const color = m.change_pct > 0 ? 'text-green-400' : (m.change_pct < 0 ? 'text-red-400' : 'text-gray-400');
            const sign = m.change_pct > 0 ? '+' : '';
            const arrow = m.change_pct > 0 ? '&#9650;' : (m.change_pct < 0 ? '&#9660;' : '&#9654;');
            return `
                <div class="flex items-center gap-1.5 px-3 py-1.5 bg-dark-700 rounded">
                    <span class="text-xs text-gray-400">${m.name}</span>
                    <span class="text-xs ${color}">${arrow}</span>
                    <span class="text-xs ${color} font-mono">${sign}${m.change_pct}%</span>
                </div>
            `;
        }).join('');
    },

    renderSRLevels(sr) {
        const panel = document.getElementById('srLevelsPanel');
        const grid = document.getElementById('srLevelsGrid');
        const badge = document.getElementById('srProximityBadge');
        if (!panel || !grid) return;
        if (!sr || !sr.levels || Object.keys(sr.levels).length === 0) {
            panel.classList.add('hidden');
            return;
        }
        panel.classList.remove('hidden');
        const lvls = sr.levels;
        const items = [
            { label: 'S3', val: lvls.s3, cls: 'text-red-400' },
            { label: 'S2', val: lvls.s2, cls: 'text-red-400' },
            { label: 'S1', val: lvls.s1, cls: 'text-red-300' },
            { label: 'Pivot', val: lvls.pivot, cls: 'text-yellow-400' },
            { label: 'R1', val: lvls.r1, cls: 'text-green-300' },
            { label: 'R2', val: lvls.r2, cls: 'text-green-400' },
            { label: 'R3', val: lvls.r3, cls: 'text-green-400' },
        ];
        grid.innerHTML = items.map(i =>
            `<div class="bg-dark-700 rounded p-1.5">
                <div class="text-[10px] text-gray-500">${i.label}</div>
                <div class="text-xs font-mono ${i.cls}">₹${i.val != null ? i.val.toFixed(2) : '-'}</div>
            </div>`
        ).join('');

        if (badge) {
            const sig = sr.proximity_signal || 0;
            if (sig > 0) {
                badge.textContent = `Near level (+${sig})`;
                badge.className = 'text-[10px] px-2 py-0.5 rounded-full bg-green-900 text-green-400';
            } else if (sig < 0) {
                badge.textContent = `Near level (${sig})`;
                badge.className = 'text-[10px] px-2 py-0.5 rounded-full bg-red-900 text-red-400';
            } else {
                badge.textContent = 'No proximity';
                badge.className = 'text-[10px] px-2 py-0.5 rounded-full bg-gray-800 text-gray-500';
            }
        }
    },

    renderMTFConfluence(mtf) {
        const panel = document.getElementById('mtfPanel');
        const grid = document.getElementById('mtfGrid');
        const badge = document.getElementById('confluenceBadge');
        if (!panel || !grid) return;
        if (!mtf || !mtf.timeframes) {
            panel.classList.add('hidden');
            return;
        }
        panel.classList.remove('hidden');
        grid.innerHTML = mtf.timeframes.map(tf => {
            const dirCls = tf.direction === 'BULLISH' ? 'text-green-400' :
                           tf.direction === 'BEARISH' ? 'text-red-400' : 'text-yellow-400';
            const arrow = tf.direction === 'BULLISH' ? '&#9650;' :
                          tf.direction === 'BEARISH' ? '&#9660;' : '&#9654;';
            return `<div class="bg-dark-700 rounded p-2">
                <div class="text-[10px] text-gray-500">${tf.label}</div>
                <div class="text-sm ${dirCls} font-bold">${arrow} ${tf.direction}</div>
                <div class="text-[10px] text-gray-500">Score: ${tf.score != null ? tf.score.toFixed(1) : '-'}</div>
            </div>`;
        }).join('');

        if (badge) {
            const level = mtf.level || 'LOW';
            const cls = level === 'HIGH' ? 'bg-green-900 text-green-400' :
                        level === 'MEDIUM' ? 'bg-yellow-900 text-yellow-400' : 'bg-gray-800 text-gray-400';
            badge.textContent = `${level} (${mtf.agreement_count || 0}/3)`;
            badge.className = `text-[10px] px-2 py-0.5 rounded-full ${cls}`;
        }
    },

    renderOIAnalysis(data) {
        const oiSection = document.getElementById('oiScoreSection');
        const oiPanel = document.getElementById('oiPanel');
        const oiMetrics = document.getElementById('oiMetrics');

        if (!data.oi_analysis || !data.oi_analysis.available) {
            if (oiSection) oiSection.classList.add('hidden');
            if (oiPanel) oiPanel.classList.add('hidden');
            return;
        }
        const oi = data.oi_analysis;

        // Show OI score bar
        if (oiSection) {
            oiSection.classList.remove('hidden');
            this.renderScoreBar('oiBar', 'oiScore', oi.score, data.oi_analysis.weight || 0.10);
        }

        // Show OI metrics panel
        if (oiPanel && oiMetrics) {
            oiPanel.classList.remove('hidden');
            const pcrColor = oi.pcr > 1.0 ? 'text-green-400' : 'text-red-400';
            oiMetrics.innerHTML = `
                <div class="bg-dark-700 rounded p-1.5">
                    <div class="text-[10px] text-gray-500">PCR</div>
                    <div class="text-sm font-mono ${pcrColor}">${oi.pcr != null ? oi.pcr.toFixed(2) : '-'}</div>
                </div>
                <div class="bg-dark-700 rounded p-1.5">
                    <div class="text-[10px] text-gray-500">Max Pain</div>
                    <div class="text-sm font-mono text-yellow-400">₹${oi.max_pain || '-'}</div>
                </div>
                <div class="bg-dark-700 rounded p-1.5">
                    <div class="text-[10px] text-gray-500">Call OI Chg</div>
                    <div class="text-sm font-mono ${oi.call_oi_change > 0 ? 'text-red-400' : 'text-green-400'}">${oi.call_oi_change != null ? (oi.call_oi_change > 0 ? '+' : '') + oi.call_oi_change.toLocaleString() : '-'}</div>
                </div>
                <div class="bg-dark-700 rounded p-1.5">
                    <div class="text-[10px] text-gray-500">Put OI Chg</div>
                    <div class="text-sm font-mono ${oi.put_oi_change > 0 ? 'text-green-400' : 'text-red-400'}">${oi.put_oi_change != null ? (oi.put_oi_change > 0 ? '+' : '') + oi.put_oi_change.toLocaleString() : '-'}</div>
                </div>
            `;
        }
    },

    renderAdaptiveBadge(data) {
        // Update weight labels with actual values from signal
        const techLabel = document.querySelector('#signalResults .space-y-3 > div:nth-child(1) .text-xs.text-gray-400');
        const sentLabel = document.querySelector('#signalResults .space-y-3 > div:nth-child(2) .text-xs.text-gray-400');
        if (techLabel) {
            const pct = ((data.technical.weight || 0) * 100).toFixed(0);
            const adaptiveTag = data.adaptive_weights && data.adaptive_weights.adapted ? ' <span class="text-[10px] text-purple-400">[Adaptive]</span>' : '';
            techLabel.innerHTML = `Technical (${pct}%)${adaptiveTag}`;
        }
        if (sentLabel) {
            const pct = ((data.sentiment.weight || 0) * 100).toFixed(0);
            sentLabel.textContent = `Sentiment (${pct}%)`;
        }
    },

    renderLearningBadge(learning) {
        const el = document.getElementById('stockLearningBadge');
        if (!el) return;
        if (!learning || !learning.available) {
            el.classList.add('hidden');
            return;
        }
        el.classList.remove('hidden');

        const acc = learning.overall_accuracy || 0;
        const best = learning.best_timeframe || '30min';
        const trend = learning.trend || 'stable';
        const twa = learning.time_window_accuracy || {};

        // Best timeframe label
        const tfLabel = best === '30min' ? '30 Min' : '1 Hour';
        const bestAcc = twa[best] || acc;

        // Trend icon
        const trendIcon = trend === 'improving' ? '&#9650;' : trend === 'degrading' ? '&#9660;' : '&#9654;';
        const trendColor = trend === 'improving' ? 'text-green-400' : trend === 'degrading' ? 'text-red-400' : 'text-gray-400';

        // Accuracy color
        const accColor = acc >= 75 ? 'text-green-400' : acc >= 60 ? 'text-yellow-400' : 'text-red-400';

        el.innerHTML = `
            <div class="text-[10px] text-gray-500 mb-0.5">AI Learned</div>
            <div class="flex flex-wrap items-center gap-1">
                <span class="text-[10px] px-1.5 py-0.5 rounded bg-purple-900/50 text-purple-300">Best: ${tfLabel} (${bestAcc.toFixed(0)}%)</span>
                <span class="text-[10px] px-1.5 py-0.5 rounded bg-dark-600 ${accColor}">${acc.toFixed(0)}% acc</span>
                <span class="text-[10px] ${trendColor}">${trendIcon} ${trend}</span>
            </div>
        `;
    },

    renderSectorBadge(sector) {
        const el = document.getElementById('sectorStrengthBadge');
        if (!el) return;
        if (!sector || !sector.available || sector.relative_pct == null) {
            el.classList.add('hidden');
            return;
        }
        el.classList.remove('hidden');
        const rel = sector.relative_pct;
        const dir = rel >= 0 ? 'Outperforming' : 'Underperforming';
        const cls = rel >= 0 ? 'bg-green-900 text-green-400' : 'bg-red-900 text-red-400';
        const sign = rel >= 0 ? '+' : '';
        el.innerHTML = `<span class="text-[10px] px-2 py-0.5 rounded-full ${cls}">${dir} ${sector.sector} ${sign}${rel.toFixed(1)}%</span>`;
    },

    renderIntradayChart(candles, sr, techRaw) {
        const container = document.getElementById('intradayChart');
        if (!container) return;

        // Remove previous resize listener to prevent leaks
        if (this._intradayResizeHandler) {
            window.removeEventListener('resize', this._intradayResizeHandler);
            this._intradayResizeHandler = null;
        }

        if (this.intradayChart) {
            this.intradayChart.remove();
            this.intradayChart = null;
        }

        this.intradayChart = LightweightCharts.createChart(container, {
            width: container.clientWidth,
            height: 250,
            layout: { background: { color: '#1a1a2e' }, textColor: '#9ca3af' },
            grid: { vertLines: { color: '#1e2a4a' }, horzLines: { color: '#1e2a4a' } },
            rightPriceScale: { borderColor: '#374151' },
            timeScale: { borderColor: '#374151', timeVisible: true, secondsVisible: false },
        });

        const series = this.intradayChart.addCandlestickSeries({
            upColor: '#00c853', downColor: '#ff1744',
            borderUpColor: '#00c853', borderDownColor: '#ff1744',
            wickUpColor: '#00c853', wickDownColor: '#ff1744',
        });

        const chartData = candles.map(c => ({
            // Parse IST datetime string as UTC so LightweightCharts axis shows IST
            time: this._parseIST(c.time),
            open: c.open, high: c.high, low: c.low, close: c.close,
        }));

        series.setData(chartData);

        // Draw S/R price lines on chart
        if (sr && sr.levels && Object.keys(sr.levels).length > 0) {
            const lvls = sr.levels;
            const lines = [
                { price: lvls.s2, title: 'S2', color: '#ff5252' },
                { price: lvls.s1, title: 'S1', color: '#ff8a80' },
                { price: lvls.pivot, title: 'P', color: '#ffd600' },
                { price: lvls.r1, title: 'R1', color: '#69f0ae' },
                { price: lvls.r2, title: 'R2', color: '#00c853' },
            ];
            lines.forEach(l => {
                if (l.price != null) {
                    series.createPriceLine({
                        price: l.price,
                        color: l.color,
                        lineWidth: 1,
                        lineStyle: 2, // dashed
                        axisLabelVisible: true,
                        title: l.title,
                    });
                }
            });
        }

        // VWAP + band lines
        if (techRaw && techRaw.vwap && techRaw.datetimes) {
            const dts = techRaw.datetimes;
            const mkSeries = (values, color, style) => {
                const lineData = [];
                for (let i = 0; i < values.length && i < dts.length; i++) {
                    if (values[i] != null) {
                        lineData.push({ time: this._parseIST(dts[i]), value: values[i] });
                    }
                }
                if (lineData.length > 0) {
                    const ls = this.intradayChart.addLineSeries({
                        color, lineWidth: 1, lineStyle: style, priceLineVisible: false,
                        lastValueVisible: false, crosshairMarkerVisible: false,
                    });
                    ls.setData(lineData);
                }
            };
            mkSeries(techRaw.vwap, '#ffd600', 0);              // solid yellow
            mkSeries(techRaw.vwap_upper_1, 'rgba(255,214,0,0.5)', 2); // dashed
            mkSeries(techRaw.vwap_lower_1, 'rgba(255,214,0,0.5)', 2);
            mkSeries(techRaw.vwap_upper_2, 'rgba(255,214,0,0.25)', 2);
            mkSeries(techRaw.vwap_lower_2, 'rgba(255,214,0,0.25)', 2);
        }

        this.intradayChart.timeScale().fitContent();

        this._intradayResizeHandler = () => {
            if (this.intradayChart) {
                this.intradayChart.applyOptions({ width: container.clientWidth });
            }
        };
        window.addEventListener('resize', this._intradayResizeHandler);
    },

    _parseIST(dateStr) {
        // Parse "YYYY-MM-DD HH:MM" as UTC so LightweightCharts displays IST face value
        const [datePart, timePart] = dateStr.split(' ');
        const [y, m, d] = datePart.split('-').map(Number);
        const [h, min] = (timePart || '00:00').split(':').map(Number);
        return Math.floor(Date.UTC(y, m - 1, d, h, min) / 1000);
    },

    _startAgeTimer() {
        if (this._ageTimer) clearInterval(this._ageTimer);
        this._ageTimer = setInterval(() => {
            if (!this._signalTimestamp) return;
            const ageSec = Math.floor((Date.now() - this._signalTimestamp) / 1000);
            const badge = document.getElementById('marketBadge');
            if (!badge) return;
            if (ageSec > 300) {
                // Signal older than 5 min — mark stale
                badge.textContent = `${Math.floor(ageSec / 60)}m ago`;
                badge.className = 'text-xs px-2 py-0.5 rounded-full bg-yellow-900 text-yellow-400';
                // Auto-refresh if panel visible and stock selected
                if (App.currentSymbol && !this.isLoading) {
                    this.loadSignal(App.currentSymbol);
                }
            }
        }, 30000);
    },

    displayMultiTimeframeSignals(data) {
        const panel = document.getElementById('mtfSignalsPanel');
        const grid = document.getElementById('mtfSignalsGrid');
        if (!panel || !grid || !data) return;

        panel.classList.remove('hidden');

        const tsEl = document.getElementById('mtfSignalsTimestamp');
        if (tsEl && data.timestamp) {
            const ts = new Date(data.timestamp);
            tsEl.textContent = ts.toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit' }) + ' IST';
        }

        const icons = { BULLISH: '&#9650;', BEARISH: '&#9660;', NEUTRAL: '&#9654;' };
        const colors = {
            BULLISH: { text: 'text-green-400', bg: 'bg-green-900/30', border: 'border-green-700/50' },
            BEARISH: { text: 'text-red-400', bg: 'bg-red-900/30', border: 'border-red-700/50' },
            NEUTRAL: { text: 'text-yellow-400', bg: 'bg-yellow-900/30', border: 'border-yellow-700/50' },
        };

        // Build flat list of signals from nested structure
        const allSignals = [];
        const intraday = data.intraday || {};
        const shortTerm = data.short_term || {};

        // Support both old format (data.intraday is a signal) and new format (data.intraday is a group)
        if (intraday.direction) {
            // Old format
            allSignals.push(intraday);
            if (shortTerm.direction) allSignals.push(shortTerm);
            if (data.long_term && data.long_term.direction) allSignals.push(data.long_term);
        } else {
            // New format: grouped
            for (const key of ['2m', '10m', '30m']) {
                if (intraday[key]) allSignals.push({ ...intraday[key], _group: 'Intraday' });
            }
            for (const key of ['1h', '4h']) {
                if (shortTerm[key]) allSignals.push({ ...shortTerm[key], _group: 'Short-term' });
            }
        }

        // Add group headers
        let lastGroup = '';
        grid.innerHTML = allSignals.map(sig => {
            let groupHeader = '';
            if (sig._group && sig._group !== lastGroup) {
                lastGroup = sig._group;
                groupHeader = `<div class="col-span-full text-xs font-semibold text-gray-400 mt-2 mb-1 first:mt-0">${sig._group}</div>`;
            }

            const tf = sig._group ? 'grouped' : 'legacy';
            const _sig = sig;
            if (!sig) return '';
            const c = colors[sig.direction] || colors.NEUTRAL;
            const icon = icons[sig.direction] || icons.NEUTRAL;

            const levelRow = (label, value, cls) => {
                if (value == null) return '';
                return `<div class="flex justify-between items-center py-0.5">
                    <span class="text-[10px] text-gray-500">${label}</span>
                    <span class="text-xs font-mono ${cls}">₹${value.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</span>
                </div>`;
            };

            const rrBadge = sig.risk_reward != null
                ? `<span class="text-[10px] px-1.5 py-0.5 rounded-full bg-gray-800 text-gray-400">R:R 1:${sig.risk_reward.toFixed(1)}</span>`
                : '';

            const predPrice = '';

            const modelBadge = sig.weights && sig.weights.prediction > 0
                ? '<span class="text-[9px] px-1 py-0.5 rounded bg-purple-900/50 text-purple-300 ml-1">AI+Tech</span>'
                : '<span class="text-[9px] px-1 py-0.5 rounded bg-gray-800 text-gray-500 ml-1">Tech</span>';

            // Regime badge
            const regimeColors = {
                bull: 'bg-green-900/50 text-green-400',
                bear: 'bg-red-900/50 text-red-400',
                sideways: 'bg-yellow-900/50 text-yellow-400',
                volatile: 'bg-orange-900/50 text-orange-400',
            };
            const regimeBadge = sig.regime
                ? `<span class="text-[9px] px-1 py-0.5 rounded ${regimeColors[sig.regime] || 'bg-gray-800 text-gray-500'}">${sig.regime}</span>`
                : '';

            // Volume conviction badge
            const volColors = { high: 'text-green-400', moderate: 'text-gray-400', low: 'text-red-400' };
            const volBadge = sig.volume_conviction
                ? `<span class="text-[9px] px-1 py-0.5 rounded bg-gray-800 ${volColors[sig.volume_conviction] || 'text-gray-500'}">Vol:${sig.volume_conviction}</span>`
                : '';

            // Model confidence badge
            const modelConfBadge = sig.model_confidence != null
                ? `<span class="text-[9px] px-1 py-0.5 rounded bg-purple-900/30 text-purple-300">AI conf:${sig.model_confidence}%</span>`
                : '';

            // V1/V2 model version badge
            const modelVerBadge = sig.model_used === 'v2'
                ? '<span class="text-[9px] px-1 py-0.5 rounded bg-purple-900/50 text-purple-400 font-bold">V2</span>'
                : sig.model_used === 'v1'
                ? '<span class="text-[9px] px-1 py-0.5 rounded bg-gray-800 text-gray-400">V1</span>'
                : '';

            return `${groupHeader}<div class="border ${c.border} ${c.bg} rounded-lg p-3">
                <div class="flex items-center justify-between mb-2">
                    <span class="text-xs text-gray-400 font-medium">${sig.label}${modelBadge}</span>
                    <span class="text-xs font-bold ${c.text}">${icon} ${sig.direction}</span>
                </div>
                <div class="flex items-center gap-3">
                    <div class="text-center shrink-0">
                        <div class="text-lg font-bold ${c.text}">${sig.confidence}%</div>
                        <div class="text-[10px] text-gray-500">confidence</div>
                    </div>
                    <div class="border-l border-gray-700/50 pl-3 flex-1 space-y-0.5">
                        ${sig.direction === 'BEARISH'
                            ? `${levelRow('Exit Now', sig.entry, 'text-red-400 font-bold text-sm')}
                               ${levelRow('Re-enter At', sig.target, 'text-green-400 font-bold text-sm')}`
                            : `${levelRow('Entry', sig.entry, 'text-white font-bold text-sm')}
                               ${levelRow('Target', sig.target, 'text-green-400 font-bold text-sm')}
                               ${levelRow('Stop Loss', sig.stop_loss, 'text-red-400')}`
                        }
                    </div>
                </div>
                <div class="flex flex-wrap items-center gap-1 mt-2">
                    ${modelVerBadge}
                    ${rrBadge}
                    ${regimeBadge}
                    ${volBadge}
                    ${modelConfBadge}
                    ${sig.confidence_trend ? `<span class="text-[9px] px-1 py-0.5 rounded ${
                        sig.confidence_trend === 'rising' ? 'bg-green-900/50 text-green-400' :
                        sig.confidence_trend === 'falling' ? 'bg-red-900/50 text-red-400' :
                        'bg-gray-800 text-gray-500'
                    }">Conf:${sig.confidence_trend}</span>` : ''}
                </div>
                ${sig.reasoning ? `<div class="text-[10px] text-gray-500 mt-2 leading-relaxed">${sig.reasoning}</div>` : ''}
            </div>`;
        }).join('');
    },

    displaySignalHistory(history) {
        const tables = [
            document.getElementById('signalHistoryTable'),
            document.getElementById('techSignalHistoryTable'),
        ].filter(Boolean);
        if (tables.length === 0) return;
        if (!history || history.length === 0) {
            const empty = '<tr><td colspan="7" class="text-center text-gray-500 py-4">No signal history yet</td></tr>';
            tables.forEach(t => { t.innerHTML = empty; });
            return;
        }
        const checkIcon = (val) => val === true ? '<span class="text-green-400">&#10003;</span>' :
                           (val === false ? '<span class="text-red-400">&#10007;</span>' :
                           '<span class="text-gray-500">&#8987;</span>');
        const html = history.map(h => {
            const c = h.direction === 'BULLISH' ? 'text-green-400' :
                      (h.direction === 'BEARISH' ? 'text-red-400' : 'text-yellow-400');
            const time = h.created_at ? new Date(h.created_at).toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit' }) : '-';
            const signalPrice = h.price_at_signal ? '₹' + h.price_at_signal.toFixed(2) : '-';
            return `
                <tr class="border-b border-gray-800">
                    <td class="py-1.5 px-3 text-xs text-gray-400">${time}</td>
                    <td class="py-1.5 px-3 text-xs ${c} font-medium">${h.direction}</td>
                    <td class="py-1.5 px-3 text-xs text-gray-300 text-right">${h.confidence}%</td>
                    <td class="py-1.5 px-3 text-xs text-gray-300 text-right">${signalPrice}</td>
                    <td class="py-1.5 px-3 text-xs text-center">${checkIcon(h.was_correct)}</td>
                    <td class="py-1.5 px-3 text-xs text-center">${checkIcon(h.was_correct_30min)}</td>
                    <td class="py-1.5 px-3 text-xs text-center">${checkIcon(h.was_correct_1hr)}</td>
                </tr>
            `;
        }).join('');
        tables.forEach(t => { t.innerHTML = html; });
    },

    // ─── Technical Tab Signal Breakdown ──────────────────────────

    _renderTechSignal(data) {
        const panel = document.getElementById('techSignalPanel');
        if (!panel) return;

        // Direction
        const arrowEl = document.getElementById('techSignalArrow');
        const dirEl = document.getElementById('techSignalDir');
        const confEl = document.getElementById('techSignalConf');

        if (arrowEl && dirEl) {
            if (data.direction === 'BULLISH') {
                arrowEl.innerHTML = '&#9650;'; arrowEl.className = 'text-3xl text-green-400';
                dirEl.textContent = 'BULLISH'; dirEl.className = 'text-lg font-bold text-green-400';
            } else if (data.direction === 'BEARISH') {
                arrowEl.innerHTML = '&#9660;'; arrowEl.className = 'text-3xl text-red-400';
                dirEl.textContent = 'BEARISH'; dirEl.className = 'text-lg font-bold text-red-400';
            } else {
                arrowEl.innerHTML = '&#9654;'; arrowEl.className = 'text-3xl text-yellow-400';
                dirEl.textContent = 'NEUTRAL'; dirEl.className = 'text-lg font-bold text-yellow-400';
            }
        }
        if (confEl) confEl.textContent = `${data.confidence}% conf`;

        // Timestamp
        const tsEl = document.getElementById('techSignalTimestamp');
        if (tsEl && data.timestamp) {
            const ts = new Date(data.timestamp);
            tsEl.textContent = ts.toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit' }) + ' IST';
        }

        // Market badge
        const badge = document.getElementById('techMarketBadge');
        if (badge) {
            if (data.market_open) {
                badge.textContent = 'LIVE'; badge.className = 'text-xs px-2 py-0.5 rounded-full bg-green-900 text-green-400';
            } else {
                badge.textContent = 'CLOSED'; badge.className = 'text-xs px-2 py-0.5 rounded-full bg-red-900 text-red-400';
            }
        }

        // Score bars
        const renderBar = (barId, valId, score, weight) => {
            const bar = document.getElementById(barId);
            const val = document.getElementById(valId);
            if (!bar || !val) return;
            score = score || 0; weight = weight || 0;
            const abs = Math.min(100, Math.abs(score));
            const color = score > 0 ? '#00c853' : (score < 0 ? '#ff1744' : '#6b7280');
            if (score >= 0) {
                bar.style.background = `linear-gradient(to right, #1e2a4a 50%, ${color} 50%, ${color} ${50 + abs / 2}%, #1e2a4a ${50 + abs / 2}%)`;
            } else {
                bar.style.background = `linear-gradient(to right, #1e2a4a ${50 - abs / 2}%, ${color} ${50 - abs / 2}%, ${color} 50%, #1e2a4a 50%)`;
            }
            const sign = score > 0 ? '+' : '';
            val.textContent = `${sign}${score.toFixed(1)} (${(weight * 100).toFixed(0)}%)`;
            val.className = `text-xs font-mono ${score > 0 ? 'text-green-400' : (score < 0 ? 'text-red-400' : 'text-gray-400')}`;
        };

        renderBar('techScoreBar', 'techScoreVal', data.technical?.score, data.technical?.weight);
        renderBar('techSentBar', 'techSentVal', data.sentiment?.score, data.sentiment?.weight);
        renderBar('techGlobBar', 'techGlobVal', data.global_market?.score, data.global_market?.weight);

        // Technical indicator details (RSI, Vol, BB, VWAP, MA)
        const detailsEl = document.getElementById('techIndicatorDetails');
        if (detailsEl && data.technical?.details) {
            const d = data.technical.details;
            const maCrossColor = d.ma_cross === 'bullish' ? 'text-green-400' : (d.ma_cross === 'bearish' ? 'text-red-400' : 'text-gray-500');
            const volColor = d.volume_ratio > 1.5 ? 'text-yellow-400' : 'text-gray-500';
            detailsEl.innerHTML = `
                <div class="flex flex-wrap items-center gap-2 text-xs">
                    <span class="text-gray-500">RSI: <span class="text-white">${d.rsi || '-'}</span></span>
                    <span class="${volColor}">Vol: ${d.volume_ratio ? d.volume_ratio.toFixed(1) + 'x' : '-'}</span>
                    <span class="text-gray-500">BB: ${d.bb_position != null ? (d.bb_position * 100).toFixed(0) + '%' : '-'}</span>
                    <span class="text-gray-500">VWAP: ₹${d.vwap || '-'}</span>
                    <span class="${maCrossColor}">MA: ${d.ma_cross || '-'}</span>
                </div>
            `;
        }
    },

    // ─── Overview Targets & Timeframe ─────────────────────────────

    displayOverviewTargets(mtfData) {
        const targetPanel = document.getElementById('signalTargetInfo');
        const tfBadge = document.getElementById('signalBestTimeframe');
        if (!targetPanel || !mtfData) return;

        // Collect all signals from nested structure
        const signals = [];
        const intraday = mtfData.intraday || {};
        const shortTerm = mtfData.short_term || {};

        // Support old and new format
        if (intraday.direction) {
            signals.push({ key: 'intraday', label: 'Intraday', data: intraday });
            if (shortTerm.direction) signals.push({ key: 'short_term', label: 'Short-term', data: shortTerm });
            if (mtfData.long_term && mtfData.long_term.direction) signals.push({ key: 'long_term', label: 'Long-term', data: mtfData.long_term });
        } else {
            for (const [k, v] of Object.entries(intraday)) {
                if (v && v.direction) signals.push({ key: `intraday_${k}`, label: v.label || k, data: v });
            }
            for (const [k, v] of Object.entries(shortTerm)) {
                if (v && v.direction) signals.push({ key: `short_${k}`, label: v.label || k, data: v });
            }
        }

        // Filter non-neutral and sort by confidence
        const nonNeutral = signals.filter(s => s.data.direction !== 'NEUTRAL');
        nonNeutral.sort((a, b) => (b.data.confidence || 0) - (a.data.confidence || 0));

        const best = nonNeutral[0];
        if (!best) {
            targetPanel.classList.add('hidden');
            if (tfBadge) tfBadge.classList.add('hidden');
            return;
        }

        const sig = best.data;
        const currentPrice = mtfData.current_price;

        // AI Price
        const aiPriceEl = document.getElementById('signalAIPrice');
        if (sig.predicted_price) {
            const pctChange = currentPrice > 0 ? ((sig.predicted_price - currentPrice) / currentPrice * 100).toFixed(1) : 0;
            const sign = pctChange >= 0 ? '+' : '';
            const color = pctChange >= 0 ? 'text-green-400' : 'text-red-400';
            const verTag = sig.model_used === 'v2' ? '<span class="text-[9px] px-1 py-0.5 rounded bg-purple-900/50 text-purple-400 font-bold ml-1">V2</span>' : sig.model_used === 'v1' ? '<span class="text-[9px] px-1 py-0.5 rounded bg-gray-800 text-gray-400 ml-1">V1</span>' : '';
            aiPriceEl.innerHTML = `AI Target: <span class="${color}">₹${sig.predicted_price.toLocaleString('en-IN', {minimumFractionDigits: 2})}</span> <span class="text-xs ${color}">(${sign}${pctChange}%)</span>${verTag}`;
        } else {
            aiPriceEl.textContent = '';
        }

        // Target / Entry / SL
        const targetEl = document.getElementById('signalTarget');
        const slEl = document.getElementById('signalStopLoss');
        const rrEl = document.getElementById('signalRR');

        if (sig.direction === 'BULLISH') {
            targetEl.innerHTML = sig.entry ? `Entry: ₹${sig.entry.toFixed(2)} → Target: ₹${sig.target ? sig.target.toFixed(2) : '-'}` : '';
            slEl.innerHTML = sig.stop_loss ? `Stop Loss: ₹${sig.stop_loss.toFixed(2)}` : '';
        } else {
            targetEl.innerHTML = sig.target ? `Exit → Re-enter at ₹${sig.target.toFixed(2)}` : '';
            slEl.innerHTML = '';
        }
        rrEl.textContent = sig.risk_reward ? `R:R ${sig.risk_reward}` : '';

        targetPanel.classList.remove('hidden');

        // Timeframe direction summary
        if (tfBadge) {
            const tfSummary = signals.map(s => {
                const color = s.data.direction === 'BULLISH' ? 'text-green-400' : (s.data.direction === 'BEARISH' ? 'text-red-400' : 'text-gray-500');
                const arrow = s.data.direction === 'BULLISH' ? '▲' : (s.data.direction === 'BEARISH' ? '▼' : '▶');
                // Shorten label
                const short = (s.data.label || s.key).replace(/\s*\(.*\)/, '');
                return `<span class="${color} text-[10px]">${short} ${arrow}</span>`;
            }).filter(Boolean).join(' <span class="text-gray-700">|</span> ');

            tfBadge.innerHTML = `<div class="flex items-center justify-center gap-1 flex-wrap">${tfSummary}</div>`;
            tfBadge.classList.remove('hidden');
        }
    },

    // ─── AI Summary ────────────────────────────────────────────────

    async loadAISummary(symbol) {
        const panel = document.getElementById('aiSummaryPanel');
        if (!panel) return;

        try {
            const resp = await fetch(`${API.baseUrl}/api/stocks/ai/summary/${encodeURIComponent(symbol)}`);
            if (!resp.ok) { panel.classList.add('hidden'); return; }
            const data = await resp.json();
            panel.classList.remove('hidden');

            // Summary text (supports **bold** markdown)
            const textEl = document.getElementById('aiSummaryText');
            textEl.innerHTML = (data.summary || '').replace(/\*\*(.*?)\*\*/g, '<strong class="text-white">$1</strong>');

            // Also set one-liner in Technical tab overview
            const techLine = document.getElementById('techAISummaryLine');
            if (techLine) {
                // Strip markdown bold and truncate
                const plain = (data.summary || '').replace(/\*\*/g, '');
                techLine.textContent = plain.length > 120 ? plain.substring(0, 117) + '...' : plain;
            }

            // Factors
            const factorsEl = document.getElementById('aiFactors');
            const factors = data.factors || [];
            factorsEl.innerHTML = factors.map(f => {
                const icon = f.impact === 'positive' ? '&#9650;' : (f.impact === 'negative' ? '&#9660;' : '&#9654;');
                const color = f.impact === 'positive' ? 'text-green-400' : (f.impact === 'negative' ? 'text-red-400' : 'text-gray-400');
                const bg = f.impact === 'positive' ? 'border-green-800/50' : (f.impact === 'negative' ? 'border-red-800/50' : 'border-gray-700');
                return `
                    <div class="flex items-start gap-2 p-2 rounded border ${bg}">
                        <span class="${color} text-sm mt-0.5">${icon}</span>
                        <div>
                            <div class="text-xs font-medium text-white">${f.factor}</div>
                            <div class="text-xs text-gray-400">${f.detail}</div>
                        </div>
                    </div>
                `;
            }).join('');

            // Risk factors
            const riskEl = document.getElementById('aiRiskFactors');
            const risks = data.risk_factors || [];
            if (risks.length > 0) {
                riskEl.innerHTML = `
                    <div class="text-[10px] text-red-400 font-medium mb-1">Risk Factors</div>
                    ${risks.map(r => `<div class="text-xs text-gray-400 flex items-start gap-1"><span class="text-red-500 mt-0.5">&#9888;</span> ${r}</div>`).join('')}
                `;
            } else {
                riskEl.innerHTML = '';
            }

            // Earnings insight
            const earningsEl = document.getElementById('aiEarningsInsight');
            const earnings = data.earnings_analysis;
            if (earnings && earnings.available && earnings.insights && earnings.insights.length > 0) {
                const ratingColor = earnings.rating === 'positive' ? 'text-green-400' : (earnings.rating === 'negative' ? 'text-red-400' : 'text-yellow-400');
                earningsEl.innerHTML = `
                    <div class="text-[10px] text-purple-400 font-medium mb-1">Earnings Analysis</div>
                    <div class="text-xs text-gray-300">${earnings.insights.slice(0, 3).map(i => `<span class="${ratingColor}">&#8226;</span> ${i}`).join('<br>')}</div>
                `;
            } else {
                earningsEl.innerHTML = '';
            }
        } catch (e) {
            panel.classList.add('hidden');
        }
    },

    // ─── Stock Accuracy Summary ────────────────────────────────────

    async loadStockAccuracy(symbol) {
        const panel = document.getElementById('stockAccuracyPanel');
        if (!panel) return;

        try {
            // Fetch prediction accuracy and trade accuracy in parallel
            const [predResp, tradeResp] = await Promise.all([
                fetch(`${API.baseUrl}/api/signals/stats/prediction-leaderboard`).catch(() => null),
                fetch(`${API.baseUrl}/api/signals/stats/trades?symbol=${encodeURIComponent(symbol)}`).catch(() => null),
            ]);

            let hasPredData = false;
            let hasTradeData = false;

            // Prediction accuracy for this symbol
            if (predResp && predResp.ok) {
                const predData = await predResp.json();
                const bySymbol = (predData.by_symbol || []).filter(s => s.symbol === symbol);
                if (bySymbol.length > 0) {
                    // Pick best model for this symbol
                    const best = bySymbol.sort((a, b) => a.avg_mape - b.avg_mape)[0];
                    const accEl = document.getElementById('stockPredAccuracy');
                    const mapeEl = document.getElementById('stockPredMAPE');
                    const countEl = document.getElementById('stockPredCount');

                    const winRate = best.avg_mape <= 2 ? 100 : best.avg_mape <= 5 ? 90 : best.avg_mape <= 10 ? 70 : 50;
                    const accColor = winRate >= 80 ? 'text-green-400' : winRate >= 60 ? 'text-yellow-400' : 'text-red-400';
                    accEl.textContent = `${winRate}%`;
                    accEl.className = `text-lg font-bold ${accColor}`;
                    mapeEl.textContent = `${best.avg_mape.toFixed(1)}%`;
                    mapeEl.className = `text-lg font-bold ${best.avg_mape <= 3 ? 'text-green-400' : best.avg_mape <= 6 ? 'text-yellow-400' : 'text-red-400'}`;
                    countEl.textContent = `${best.total} predictions`;
                    hasPredData = true;
                }
            }

            // Trade signal accuracy for this symbol
            if (tradeResp && tradeResp.ok) {
                const tradeData = await tradeResp.json();
                if (tradeData.total > 0) {
                    const wrEl = document.getElementById('stockTradeWinRate');
                    const countEl = document.getElementById('stockTradeCount');
                    const pnlEl = document.getElementById('stockTradeAvgPnl');

                    const wr = tradeData.win_rate || 0;
                    const wrColor = wr >= 60 ? 'text-green-400' : wr >= 45 ? 'text-yellow-400' : 'text-red-400';
                    wrEl.textContent = `${wr}%`;
                    wrEl.className = `text-lg font-bold ${wrColor}`;
                    countEl.textContent = `${tradeData.total} trades`;

                    const avgPnl = tradeData.avg_win_pct || 0;
                    const avgLoss = tradeData.avg_loss_pct || 0;
                    pnlEl.innerHTML = `<span class="text-green-400 text-sm">+${avgPnl}%</span> <span class="text-gray-600">/</span> <span class="text-red-400 text-sm">${avgLoss}%</span>`;
                    hasTradeData = true;
                }
            }

            if (hasPredData || hasTradeData) {
                panel.classList.remove('hidden');
            } else {
                panel.classList.add('hidden');
            }
        } catch (e) {
            panel.classList.add('hidden');
        }
    },

    // ─── AI Learning Profile Tab ────────────────────────────────────

    async loadLearningProfile(symbol) {
        const loading = document.getElementById('aiLearningLoading');
        const empty = document.getElementById('aiLearningEmpty');
        const content = document.getElementById('aiLearningContent');
        if (!loading) return;

        loading.classList.remove('hidden');
        empty.classList.add('hidden');
        content.classList.add('hidden');

        try {
            const resp = await fetch(`${API.baseUrl}/api/signals/stats/learning/${encodeURIComponent(symbol)}`);
            if (!resp.ok) {
                loading.classList.add('hidden');
                empty.classList.remove('hidden');
                return;
            }
            const profile = await resp.json();
            loading.classList.add('hidden');
            content.classList.remove('hidden');
            this.renderLearningProfile(profile);
        } catch (e) {
            loading.classList.add('hidden');
            empty.classList.remove('hidden');
        }
    },

    renderLearningProfile(p) {
        const trendIcons = { improving: '&#9650; Up', degrading: '&#9660; Down', stable: '&#9654; Stable' };
        const trendColors = { improving: 'text-green-400', degrading: 'text-red-400', stable: 'text-yellow-400' };
        const summary = p.summary || {};
        const pred = p.predictions;
        const trades = p.trades;

        // Sample size
        const sizeEl = document.getElementById('aiLearningSampleSize');
        if (sizeEl) {
            const parts = [];
            if (summary.total_predictions) parts.push(`${summary.total_predictions} predictions`);
            if (summary.total_trades) parts.push(`${summary.total_trades} trades`);
            sizeEl.textContent = parts.length ? `Based on ${parts.join(' + ')}` : '';
        }

        // Summary cards
        const cardsEl = document.getElementById('aiLearningSummaryCards');
        const card = (label, value, color, sub) => `
            <div class="bg-dark-700 rounded-lg p-3 text-center">
                <div class="text-[10px] text-gray-500 mb-1">${label}</div>
                <div class="text-xl font-bold ${color}">${value}</div>
                ${sub ? `<div class="text-[10px] text-gray-500 mt-0.5">${sub}</div>` : ''}
            </div>`;

        let cards = '';
        if (summary.prediction_accuracy != null) {
            const acc = summary.prediction_accuracy;
            const accColor = acc >= 95 ? 'text-green-400' : acc >= 90 ? 'text-yellow-400' : 'text-red-400';
            cards += card('Prediction Accuracy', `${acc}%`, accColor, `MAPE: ${summary.prediction_mape || 0}%`);
        }
        if (summary.trade_win_rate != null) {
            const wr = summary.trade_win_rate;
            const wrColor = wr >= 60 ? 'text-green-400' : wr >= 45 ? 'text-yellow-400' : 'text-red-400';
            cards += card('Trade Win Rate', `${wr}%`, wrColor, `Avg P&L: ${summary.trade_avg_pnl || 0}%`);
        }
        if (summary.best_model) {
            cards += card('Best Model', summary.best_model, 'text-purple-400', '');
        }
        if (summary.overall_trend) {
            const t = summary.overall_trend;
            cards += card('Trend', trendIcons[t] || t, trendColors[t] || 'text-white', '');
        }
        cardsEl.innerHTML = cards || '<div class="col-span-full text-center text-gray-500 text-sm">No data yet</div>';

        // Prediction section
        const predSection = document.getElementById('aiPredictionSection');
        if (pred && predSection) {
            predSection.classList.remove('hidden');
            const modelCards = document.getElementById('aiPredModelCards');
            modelCards.innerHTML = Object.entries(pred.models || {}).map(([name, m]) => {
                const isBest = name === pred.best_model;
                const border = isBest ? 'border-purple-500' : 'border-gray-700';
                return `
                    <div class="rounded-lg p-2.5 border ${border} bg-dark-600">
                        <div class="flex items-center justify-between mb-1">
                            <span class="text-xs font-medium text-white capitalize">${name}</span>
                            ${isBest ? '<span class="text-[9px] px-1 py-0.5 rounded bg-purple-900/50 text-purple-300">Best</span>' : ''}
                        </div>
                        <div class="text-lg font-bold ${m.avg_mape <= 3 ? 'text-green-400' : m.avg_mape <= 6 ? 'text-yellow-400' : 'text-red-400'}">${m.avg_mape}% MAPE</div>
                        <div class="text-[10px] text-gray-500">${m.total} predictions | ${m.accuracy || m.accuracy_2pct || 0}% accurate</div>
                    </div>`;
            }).join('');

            const trendEl = document.getElementById('aiPredTrend');
            const t = pred.trend || 'stable';
            trendEl.innerHTML = `Trend: <span class="${trendColors[t] || 'text-white'}">${trendIcons[t] || t}</span> | Recent MAPE: ${pred.recent_mape || 0}%`;
        } else if (predSection) {
            predSection.classList.add('hidden');
        }

        // Trade section
        const tradeSection = document.getElementById('aiTradeSection');
        if (trades && tradeSection) {
            tradeSection.classList.remove('hidden');
            const statsEl = document.getElementById('aiTradeStats');
            const wr = trades.win_rate || 0;
            const wrColor = wr >= 60 ? 'text-green-400' : wr >= 45 ? 'text-yellow-400' : 'text-red-400';
            statsEl.innerHTML = `
                <div class="bg-dark-600 rounded p-2 text-center">
                    <div class="text-[10px] text-gray-500">Win Rate</div>
                    <div class="text-lg font-bold ${wrColor}">${wr}%</div>
                </div>
                <div class="bg-dark-600 rounded p-2 text-center">
                    <div class="text-[10px] text-gray-500">Target Hit</div>
                    <div class="text-lg font-bold text-green-400">${trades.target_hits || 0}</div>
                </div>
                <div class="bg-dark-600 rounded p-2 text-center">
                    <div class="text-[10px] text-gray-500">SL Hit</div>
                    <div class="text-lg font-bold text-red-400">${trades.sl_hits || 0}</div>
                </div>
                <div class="bg-dark-600 rounded p-2 text-center">
                    <div class="text-[10px] text-gray-500">Avg P&L</div>
                    <div class="text-sm font-bold"><span class="text-green-400">+${trades.avg_win || 0}%</span> / <span class="text-red-400">${trades.avg_loss || 0}%</span></div>
                </div>`;

            const tfEl = document.getElementById('aiTradeByTF');
            const tfData = trades.by_timeframe || {};
            tfEl.innerHTML = Object.entries(tfData).map(([tf, s]) => {
                const isBest = tf === trades.best_timeframe;
                const border = isBest ? 'border-purple-500 bg-purple-900/20' : 'border-gray-700 bg-dark-600';
                const twrColor = s.win_rate >= 60 ? 'text-green-400' : s.win_rate >= 45 ? 'text-yellow-400' : 'text-red-400';
                return `
                    <div class="rounded-lg p-2.5 text-center border ${border}">
                        <div class="text-xs text-gray-400 mb-1">${tf}</div>
                        <div class="text-lg font-bold ${twrColor}">${s.win_rate}%</div>
                        <div class="text-[10px] text-gray-500">${s.trades} trades</div>
                        ${isBest ? '<div class="text-[10px] text-purple-400">Best</div>' : ''}
                    </div>`;
            }).join('') || '<div class="col-span-full text-center text-gray-500 text-xs">No timeframe data yet</div>';
        } else if (tradeSection) {
            tradeSection.classList.add('hidden');
        }
    },
};
