class StockChart {
    constructor(containerId) {
        this.containerId = containerId;
        this.chart = null;
        this.candleSeries = null;
        this.lineSeries = null;
        this.volumeSeries = null;
        this.overlayLines = [];
        this.chartType = 'candlestick';
        this.lastCandle = null;
        this.lastVolume = null;
    }

    init(timeVisible = false) {
        const container = document.getElementById(this.containerId);
        if (this.chart) { this.chart.remove(); }
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

        window.addEventListener('resize', () => {
            if (this.chart) {
                this.chart.applyOptions({ width: container.clientWidth });
            }
        });
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
     * Called on every WebSocket price_update.
     */
    updateLastCandle(price, volume) {
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
}

// RSI & MACD sub-charts
class IndicatorChart {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.chart = null;
        this.options = options;
    }

    init() {
        const container = document.getElementById(this.containerId);
        if (this.chart) { this.chart.remove(); }
        this.chart = LightweightCharts.createChart(container, {
            width: container.clientWidth,
            height: container.clientHeight,
            layout: { background: { color: '#1a1a2e' }, textColor: '#9ca3af' },
            grid: { vertLines: { color: '#1e2a4a' }, horzLines: { color: '#1e2a4a' } },
            rightPriceScale: { borderColor: '#374151' },
            timeScale: { visible: false },
        });

        window.addEventListener('resize', () => {
            if (this.chart) {
                this.chart.applyOptions({ width: container.clientWidth });
            }
        });
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
