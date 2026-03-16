const Signals = {
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

        const loading = document.getElementById('signalLoading');
        const results = document.getElementById('signalResults');

        loading.classList.remove('hidden');
        results.classList.add('hidden');

        // Show shimmer for MTF signals while loading
        Shimmer.show('mtfSignalsGrid', 'mtfSignals', 3);

        try {
            const [signal, history] = await Promise.all([
                API.getIntradaySignal(symbol),
                API.getSignalHistory(symbol, 10),
            ]);
            this.displaySignal(signal);
            App.displaySignalBadge(signal);
            App._lastSignalData = signal;
            this.displaySignalHistory(history);

            // Load multi-timeframe signals in background (non-blocking)
            API.getMultiTimeframeSignals(symbol).then(mtfData => {
                this.displayMultiTimeframeSignals(mtfData);
            }).catch(() => {});
        } catch (e) {
            loading.classList.add('hidden');
            results.classList.remove('hidden');
            results.innerHTML = `
                <div class="text-center py-8">
                    <div class="text-red-400 text-sm mb-3">Failed to load signal: ${e.message}</div>
                    <button onclick="Signals.loadSignal('${symbol}')"
                        class="text-xs px-4 py-2 bg-dark-600 text-gray-300 rounded hover:bg-dark-700 transition">
                        Retry
                    </button>
                </div>`;
        } finally {
            this.isLoading = false;
        }
    },

    displaySignal(data) {
        document.getElementById('signalLoading').classList.add('hidden');
        document.getElementById('signalResults').classList.remove('hidden');

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

        const timeframes = ['intraday', 'short_term', 'long_term'];
        const icons = { BULLISH: '&#9650;', BEARISH: '&#9660;', NEUTRAL: '&#9654;' };
        const colors = {
            BULLISH: { text: 'text-green-400', bg: 'bg-green-900/30', border: 'border-green-700/50' },
            BEARISH: { text: 'text-red-400', bg: 'bg-red-900/30', border: 'border-red-700/50' },
            NEUTRAL: { text: 'text-yellow-400', bg: 'bg-yellow-900/30', border: 'border-yellow-700/50' },
        };

        grid.innerHTML = timeframes.map(tf => {
            const sig = data[tf];
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

            const predPrice = sig.predicted_price != null
                ? `<div class="text-[10px] text-purple-400 mt-1">AI Price: ₹${sig.predicted_price.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>`
                : '';

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

            return `<div class="border ${c.border} ${c.bg} rounded-lg p-3">
                <div class="flex items-center justify-between mb-2">
                    <span class="text-xs text-gray-400 font-medium">${sig.label}${modelBadge}</span>
                    <span class="text-xs font-bold ${c.text}">${icon} ${sig.direction}</span>
                </div>
                <div class="text-center mb-2">
                    <div class="text-lg font-bold ${c.text}">${sig.confidence}%</div>
                    <div class="text-[10px] text-gray-500">confidence</div>
                    ${predPrice}
                </div>
                <div class="border-t border-gray-700/50 pt-2 space-y-0.5">
                    ${sig.direction === 'BEARISH'
                        ? `${levelRow('Exit', sig.entry, 'text-white')}
                           ${levelRow('Downside', sig.target, 'text-red-400')}
                           ${levelRow('Stop Loss', sig.stop_loss, 'text-yellow-400')}`
                        : `${levelRow('Entry', sig.entry, 'text-white')}
                           ${levelRow('Target', sig.target, 'text-green-400')}
                           ${levelRow('Stop Loss', sig.stop_loss, 'text-red-400')}`
                    }
                </div>
                <div class="flex flex-wrap items-center gap-1 mt-2">
                    ${rrBadge}
                    ${regimeBadge}
                    ${volBadge}
                    ${modelConfBadge}
                </div>
                ${sig.reasoning ? `<div class="text-[10px] text-gray-500 mt-2 leading-relaxed">${sig.reasoning}</div>` : ''}
            </div>`;
        }).join('');
    },

    displaySignalHistory(history) {
        const el = document.getElementById('signalHistoryTable');
        if (!el) return;
        if (!history || history.length === 0) {
            el.innerHTML = '<tr><td colspan="7" class="text-center text-gray-500 py-4">No signal history yet</td></tr>';
            return;
        }
        const checkIcon = (val) => val === true ? '<span class="text-green-400">&#10003;</span>' :
                           (val === false ? '<span class="text-red-400">&#10007;</span>' :
                           '<span class="text-gray-500">&#8987;</span>');
        el.innerHTML = history.map(h => {
            const c = h.direction === 'BULLISH' ? 'text-green-400' :
                      (h.direction === 'BEARISH' ? 'text-red-400' : 'text-yellow-400');
            const time = h.created_at ? new Date(h.created_at + '+05:30').toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit' }) : '-';
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
    },
};
