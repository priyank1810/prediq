const Signals = {
    intradayChart: null,
    isLoading: false,

    init() {
        const btn = document.getElementById('btnSignal');
        if (btn) {
            btn.addEventListener('click', () => {
                if (App.currentSymbol) this.loadSignal(App.currentSymbol);
            });
        }
        API.setSignalUpdateHandler((data) => {
            if (data.symbol === App.currentSymbol) {
                this.displaySignal(data);
            }
        });
    },

    async loadSignal(symbol) {
        if (this.isLoading) return;
        this.isLoading = true;

        const panel = document.getElementById('signalPanel');
        const loading = document.getElementById('signalLoading');
        const results = document.getElementById('signalResults');

        panel.classList.remove('hidden');
        loading.classList.remove('hidden');
        results.classList.add('hidden');

        try {
            const [signal, history] = await Promise.all([
                API.getIntradaySignal(symbol),
                API.getSignalHistory(symbol, 10),
            ]);
            this.displaySignal(signal);
            this.displaySignalHistory(history);
        } catch (e) {
            App.showToast('Failed to load signal: ' + e.message, 'error');
            loading.classList.add('hidden');
        } finally {
            this.isLoading = false;
        }
    },

    displaySignal(data) {
        document.getElementById('signalLoading').classList.add('hidden');
        document.getElementById('signalResults').classList.remove('hidden');

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

        // Score bars (weights are now dynamic)
        this.renderScoreBar('technicalBar', 'technicalScore', data.technical.score, data.technical.weight);
        this.renderScoreBar('sentimentBar', 'sentimentScore', data.sentiment.score, data.sentiment.weight);
        this.renderScoreBar('globalBar', 'globalScore', data.global_market.score, data.global_market.weight);

        // Update global weight label if it changed
        const gwLabel = document.getElementById('globalWeightLabel');
        if (gwLabel) {
            const pct = (data.global_market.weight * 100).toFixed(0);
            gwLabel.textContent = `Global (${pct}%)`;
            if (data.global_market.weight > 0.15) {
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
            techDet.innerHTML = `
                <span class="text-xs text-gray-500">RSI: ${d.rsi || '-'}</span>
                <span class="mx-1 text-gray-600">|</span>
                <span class="text-xs ${volColor}">Vol: ${d.volume_ratio ? d.volume_ratio.toFixed(1) + 'x' : '-'}</span>
                <span class="mx-1 text-gray-600">|</span>
                <span class="text-xs text-gray-500">BB: ${d.bb_position != null ? (d.bb_position * 100).toFixed(0) + '%' : '-'}</span>
                <span class="mx-1 text-gray-600">|</span>
                <span class="text-xs text-gray-500">VWAP: ₹${d.vwap || '-'}</span>
                <span class="mx-1 text-gray-600">|</span>
                <span class="text-xs ${maCrossColor}">MA 5/9: ${d.ma_cross || '-'}</span>
            `;
        }

        // News
        this.displayHeadlines(data.sentiment.headlines || []);

        // Global markets + global news
        this.displayGlobalMarkets(data.global_market.markets || []);
        this.displayGlobalNews(data.global_market.headlines || [], data.global_market.news_magnitude || 0);

        // Intraday chart
        if (data.intraday_candles && data.intraday_candles.length > 0) {
            this.renderIntradayChart(data.intraday_candles);
        }

        // Timestamp
        const tsEl = document.getElementById('signalTimestamp');
        if (tsEl) {
            const ts = new Date(data.timestamp);
            tsEl.textContent = `Updated: ${ts.toLocaleTimeString('en-IN')}`;
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

    renderIntradayChart(candles) {
        const container = document.getElementById('intradayChart');
        if (!container) return;
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
            time: Math.floor(new Date(c.time).getTime() / 1000),
            open: c.open, high: c.high, low: c.low, close: c.close,
        }));

        series.setData(chartData);
        this.intradayChart.timeScale().fitContent();

        window.addEventListener('resize', () => {
            if (this.intradayChart) {
                this.intradayChart.applyOptions({ width: container.clientWidth });
            }
        });
    },

    displaySignalHistory(history) {
        const el = document.getElementById('signalHistoryTable');
        if (!el) return;
        if (!history || history.length === 0) {
            el.innerHTML = '<tr><td colspan="5" class="text-center text-gray-500 py-4">No signal history yet</td></tr>';
            return;
        }
        el.innerHTML = history.map(h => {
            const c = h.direction === 'BULLISH' ? 'text-green-400' :
                      (h.direction === 'BEARISH' ? 'text-red-400' : 'text-yellow-400');
            const icon = h.was_correct === true ? '<span class="text-green-400">&#10003;</span>' :
                         (h.was_correct === false ? '<span class="text-red-400">&#10007;</span>' : '-');
            const time = h.created_at ? new Date(h.created_at).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' }) : '-';
            return `
                <tr class="border-b border-gray-800">
                    <td class="py-1.5 px-3 text-xs text-gray-400">${time}</td>
                    <td class="py-1.5 px-3 text-xs ${c} font-medium">${h.direction}</td>
                    <td class="py-1.5 px-3 text-xs text-gray-300 text-right">${h.confidence}%</td>
                    <td class="py-1.5 px-3 text-xs text-gray-300 text-right">${h.price_at_signal ? '₹' + h.price_at_signal.toFixed(2) : '-'}</td>
                    <td class="py-1.5 px-3 text-xs text-center">${icon}</td>
                </tr>
            `;
        }).join('');
    },
};
