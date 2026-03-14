class StockChart {
    constructor(containerId) {
        this.containerId = containerId;
        this.chart = null;
        this.candleSeries = null;
        this.lineSeries = null;
        this.volumeSeries = null;
        this.overlayLines = [];
        this.chartType = 'line';
        this.lastCandle = null;
        this.lastVolume = null;
        this._resizeHandler = null;
        // Throttle state for updateLastCandle
        this._throttleTimer = null;
        this._pendingPrice = null;
        this._pendingVolume = null;

        // Drawing tools state
        this._drawingMode = 'none'; // 'none' | 'horizontal' | 'trendline'
        this._drawnPriceLines = [];   // horizontal price lines
        this._drawnTrendlines = [];   // trendline series
        this._trendlineStart = null;  // first click point for trendline
        this._clickHandler = null;
    }

    init(timeVisible = false) {
        const container = document.getElementById(this.containerId);
        if (this.chart) { this.chart.remove(); }

        // Remove previous resize / orientation listeners to prevent leaks
        if (this._resizeHandler) {
            window.removeEventListener('resize', this._resizeHandler);
        }
        if (this._orientationHandler && screen.orientation) {
            screen.orientation.removeEventListener('change', this._orientationHandler);
            this._orientationHandler = null;
        }

        this.chart = LightweightCharts.createChart(container, {
            width: container.clientWidth,
            height: container.clientHeight,
            layout: {
                background: { color: '#1a1a2e' },
                textColor: '#9ca3af',
            },
            grid: {
                vertLines: { color: '#1e2a4a' },
                horzLines: { color: '#1e2a4a' },
            },
            crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
            rightPriceScale: { borderColor: '#374151' },
            timeScale: { borderColor: '#374151', timeVisible: timeVisible, secondsVisible: false },
        });

        this.candleSeries = this.chart.addCandlestickSeries({
            upColor: '#00c853', downColor: '#ff1744',
            borderUpColor: '#00c853', borderDownColor: '#ff1744',
            wickUpColor: '#00c853', wickDownColor: '#ff1744',
        });
        this.lineSeries = null;

        this.volumeSeries = this.chart.addHistogramSeries({
            priceFormat: { type: 'volume' },
            priceScaleId: 'volume',
        });
        this.chart.priceScale('volume').applyOptions({
            scaleMargins: { top: 0.8, bottom: 0 },
        });

        this._resizeHandler = () => {
            if (this.chart) {
                this.chart.applyOptions({
                    width: container.clientWidth,
                    height: container.clientHeight,
                });
            }
        };
        window.addEventListener('resize', this._resizeHandler);

        // Listen for orientation changes on mobile to resize chart height
        if (screen.orientation) {
            this._orientationHandler = () => {
                // Delay slightly so the container CSS has time to apply
                setTimeout(this._resizeHandler, 150);
            };
            screen.orientation.addEventListener('change', this._orientationHandler);
        }
    }

    setData(data) {
        if (!data || data.length === 0) return;

        const candleData = data.map(d => ({
            time: d.date,
            open: d.open,
            high: d.high,
            low: d.low,
            close: d.close,
        }));

        const volumeData = data.map(d => ({
            time: d.date,
            value: d.volume,
            color: d.close >= d.open ? 'rgba(0,200,83,0.3)' : 'rgba(255,23,68,0.3)',
        }));

        if (this.chartType === 'candlestick') {
            if (this.lineSeries) {
                this.chart.removeSeries(this.lineSeries);
                this.lineSeries = null;
            }
            if (!this.candleSeries) {
                this.candleSeries = this.chart.addCandlestickSeries({
                    upColor: '#00c853', downColor: '#ff1744',
                    borderUpColor: '#00c853', borderDownColor: '#ff1744',
                    wickUpColor: '#00c853', wickDownColor: '#ff1744',
                });
            }
            this.candleSeries.setData(candleData);
        } else {
            if (this.candleSeries) {
                this.chart.removeSeries(this.candleSeries);
                this.candleSeries = null;
            }
            if (!this.lineSeries) {
                this.lineSeries = this.chart.addLineSeries({ color: '#2979ff', lineWidth: 2 });
            }
            this.lineSeries.setData(data.map(d => ({ time: d.date, value: d.close })));
        }

        this.volumeSeries.setData(volumeData);
        this.chart.timeScale().fitContent();

        // Track last candle for real-time updates
        if (candleData.length > 0) {
            this.lastCandle = { ...candleData[candleData.length - 1] };
        }
    }

    setChartType(type) {
        this.chartType = type;
    }

    /**
     * Update the last candle in real-time with new price data.
     * Throttled to max 4 repaints/sec (250ms interval).
     */
    updateLastCandle(price, volume) {
        if (!this.lastCandle || !price) return;

        // Store latest values
        this._pendingPrice = price;
        this._pendingVolume = volume;

        // If no timer running, apply immediately and start throttle window
        if (!this._throttleTimer) {
            this._applyPendingUpdate();
            this._throttleTimer = setTimeout(() => {
                this._throttleTimer = null;
                // Apply any values that arrived during the throttle window
                if (this._pendingPrice !== null) {
                    this._applyPendingUpdate();
                }
            }, 250);
        }
    }

    _applyPendingUpdate() {
        const price = this._pendingPrice;
        const volume = this._pendingVolume;
        this._pendingPrice = null;
        this._pendingVolume = null;

        if (!this.lastCandle || !price) return;

        const updated = { ...this.lastCandle };
        updated.close = price;
        if (price > updated.high) updated.high = price;
        if (price < updated.low) updated.low = price;

        if (this.chartType === 'candlestick' && this.candleSeries) {
            this.candleSeries.update(updated);
        } else if (this.lineSeries) {
            this.lineSeries.update({ time: updated.time, value: price });
        }

        // Update volume bar
        if (this.volumeSeries && volume != null) {
            this.volumeSeries.update({
                time: updated.time,
                value: volume,
                color: updated.close >= updated.open ? 'rgba(0,200,83,0.3)' : 'rgba(255,23,68,0.3)',
            });
        }

        this.lastCandle = updated;
    }

    clearOverlays() {
        this.overlayLines.forEach(s => {
            try { this.chart.removeSeries(s); } catch (e) {}
        });
        this.overlayLines = [];
    }

    addLineOverlay(data, color, title = '') {
        const series = this.chart.addLineSeries({
            color: color,
            lineWidth: 1,
            title: title,
        });
        series.setData(data.filter(d => d.value !== null && d.value !== undefined));
        this.overlayLines.push(series);
        return series;
    }

    addPredictionMarkers(predictions, dates) {
        if (!this.candleSeries && !this.lineSeries) return;
        const series = this.candleSeries || this.lineSeries;
        const markers = predictions.map((price, i) => ({
            time: dates[i],
            position: 'aboveBar',
            color: '#ffd600',
            shape: 'circle',
            text: `$${price.toFixed(0)}`,
        }));
        series.setMarkers(markers);
    }

    // ---- Drawing Tools ----

    setDrawingMode(mode) {
        // Toggle off if already active
        if (this._drawingMode === mode) {
            this._drawingMode = 'none';
            this._trendlineStart = null;
            this._updateDrawingButtons();
            this._setCrosshairStyle(false);
            return;
        }
        this._drawingMode = mode;
        this._trendlineStart = null;
        this._updateDrawingButtons();
        this._setCrosshairStyle(mode !== 'none');
    }

    _setCrosshairStyle(drawing) {
        const container = document.getElementById(this.containerId);
        if (container) {
            container.style.cursor = drawing ? 'crosshair' : '';
        }
    }

    _updateDrawingButtons() {
        const hBtn = document.getElementById('btnDrawHLine');
        const tBtn = document.getElementById('btnDrawTrendline');
        if (hBtn) hBtn.classList.toggle('active', this._drawingMode === 'horizontal');
        if (tBtn) tBtn.classList.toggle('active', this._drawingMode === 'trendline');
    }

    initDrawingTools() {
        // Remove old DOM listeners to prevent duplicates on re-init
        if (this._drawBtnHandlers) {
            const { hBtn, tBtn, cBtn, hFn, tFn, cFn } = this._drawBtnHandlers;
            if (hBtn) hBtn.removeEventListener('click', hFn);
            if (tBtn) tBtn.removeEventListener('click', tFn);
            if (cBtn) cBtn.removeEventListener('click', cFn);
        }

        const hBtn = document.getElementById('btnDrawHLine');
        const tBtn = document.getElementById('btnDrawTrendline');
        const cBtn = document.getElementById('btnDrawClear');

        const hFn = () => this.setDrawingMode('horizontal');
        const tFn = () => this.setDrawingMode('trendline');
        const cFn = () => this.clearDrawings();

        if (hBtn) hBtn.addEventListener('click', hFn);
        if (tBtn) tBtn.addEventListener('click', tFn);
        if (cBtn) cBtn.addEventListener('click', cFn);

        this._drawBtnHandlers = { hBtn, tBtn, cBtn, hFn, tFn, cFn };

        // Reset drawing state
        this._drawingMode = 'none';
        this._drawnPriceLines = [];
        this._drawnTrendlines = [];
        this._trendlineStart = null;
        this._updateDrawingButtons();

        this._attachClickHandler();
    }

    _attachClickHandler() {
        if (!this.chart) return;

        // Remove previous handler if any
        if (this._clickHandler) {
            this.chart.unsubscribeClick(this._clickHandler);
        }

        this._clickHandler = (param) => {
            if (this._drawingMode === 'none') return;
            if (!param.point || !param.time) return;

            const activeSeries = this.candleSeries || this.lineSeries;
            if (!activeSeries) return;

            const price = activeSeries.coordinateToPrice(param.point.y);
            if (price == null || !isFinite(price)) return;

            if (this._drawingMode === 'horizontal') {
                this._addHorizontalLine(price);
                // Stay in horizontal mode for rapid placement
            } else if (this._drawingMode === 'trendline') {
                if (!this._trendlineStart) {
                    // First click: store start point
                    this._trendlineStart = { time: param.time, price: price };
                } else {
                    // Second click: draw the trendline
                    this._addTrendline(this._trendlineStart, { time: param.time, price: price });
                    this._trendlineStart = null;
                    // Exit trendline mode after drawing
                    this.setDrawingMode('none');
                }
            }
        };

        this.chart.subscribeClick(this._clickHandler);
    }

    _addHorizontalLine(price) {
        const activeSeries = this.candleSeries || this.lineSeries;
        if (!activeSeries) return;

        const priceLine = activeSeries.createPriceLine({
            price: price,
            color: '#ffd600',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            axisLabelVisible: true,
            title: '',
        });
        this._drawnPriceLines.push({ series: activeSeries, line: priceLine });
    }

    _addTrendline(start, end) {
        if (!this.chart) return;

        const trendSeries = this.chart.addLineSeries({
            color: '#00bcd4',
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Solid,
            crosshairMarkerVisible: false,
            lastValueVisible: false,
            priceLineVisible: false,
        });

        // Ensure points are in chronological order
        const points = [start, end].sort((a, b) => {
            if (a.time < b.time) return -1;
            if (a.time > b.time) return 1;
            return 0;
        });

        trendSeries.setData([
            { time: points[0].time, value: points[0].price },
            { time: points[1].time, value: points[1].price },
        ]);

        this._drawnTrendlines.push(trendSeries);
    }

    clearDrawings() {
        // Remove horizontal price lines
        this._drawnPriceLines.forEach(({ series, line }) => {
            try { series.removePriceLine(line); } catch (e) {}
        });
        this._drawnPriceLines = [];

        // Remove trendline series
        this._drawnTrendlines.forEach(s => {
            try { this.chart.removeSeries(s); } catch (e) {}
        });
        this._drawnTrendlines = [];

        // Reset drawing mode
        this._trendlineStart = null;
        this.setDrawingMode('none');
    }
}

// RSI & MACD sub-charts
class IndicatorChart {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.chart = null;
        this.options = options;
        this._resizeHandler = null;
    }

    init() {
        const container = document.getElementById(this.containerId);
        if (this.chart) { this.chart.remove(); }

        // Remove previous resize listener to prevent leaks
        if (this._resizeHandler) {
            window.removeEventListener('resize', this._resizeHandler);
        }

        this.chart = LightweightCharts.createChart(container, {
            width: container.clientWidth,
            height: container.clientHeight,
            layout: { background: { color: '#1a1a2e' }, textColor: '#9ca3af' },
            grid: { vertLines: { color: '#1e2a4a' }, horzLines: { color: '#1e2a4a' } },
            rightPriceScale: { borderColor: '#374151' },
            timeScale: { visible: false },
        });

        this._resizeHandler = () => {
            if (this.chart) {
                this.chart.applyOptions({
                    width: container.clientWidth,
                    height: container.clientHeight,
                });
            }
        };
        window.addEventListener('resize', this._resizeHandler);
    }

    setRSIData(data) {
        if (!this.chart) this.init();
        const series = this.chart.addLineSeries({ color: '#a78bfa', lineWidth: 1.5, title: 'RSI' });
        series.setData(data.filter(d => d.value !== null));

        // Overbought/oversold lines
        const line70 = this.chart.addLineSeries({ color: 'rgba(255,23,68,0.3)', lineWidth: 1, lineStyle: 2 });
        const line30 = this.chart.addLineSeries({ color: 'rgba(0,200,83,0.3)', lineWidth: 1, lineStyle: 2 });
        if (data.length > 0) {
            const times = data.filter(d => d.value !== null);
            line70.setData(times.map(d => ({ time: d.time, value: 70 })));
            line30.setData(times.map(d => ({ time: d.time, value: 30 })));
        }
        this.chart.timeScale().fitContent();
    }

    setMACDData(macdLine, signalLine, histogram) {
        if (!this.chart) this.init();
        const macd = this.chart.addLineSeries({ color: '#2979ff', lineWidth: 1.5, title: 'MACD' });
        const signal = this.chart.addLineSeries({ color: '#ff9800', lineWidth: 1.5, title: 'Signal' });
        const hist = this.chart.addHistogramSeries({ title: 'Histogram' });

        macd.setData(macdLine.filter(d => d.value !== null));
        signal.setData(signalLine.filter(d => d.value !== null));
        hist.setData(histogram.filter(d => d.value !== null).map(d => ({
            time: d.time, value: d.value,
            color: d.value >= 0 ? 'rgba(0,200,83,0.5)' : 'rgba(255,23,68,0.5)',
        })));
        this.chart.timeScale().fitContent();
    }
}
