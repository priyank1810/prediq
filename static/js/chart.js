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

        // Mobile touch state
        this._isFullscreen = false;
        this._touchState = {
            lastTapTime: 0,
            longPressTimer: null,
            isLongPress: false,
            swipeStartX: 0,
            swipeStartY: 0,
            swipeStartTime: 0,
            pinchStartDist: 0,
            pinchStartScale: 1,
            isPanning: false,
            isSwiping: false,
        };
        this._activeIndicators = { sma: false, ema: false, bb: false, vol: true };
        this._rawData = null; // Store raw data for indicator recalculation
        this._mobileToolbarInited = false;
        this._crosshairHandler = null;
    }

    _isMobile() {
        return window.innerWidth < 640 || ('ontouchstart' in window);
    }

    _isLandscape() {
        return window.innerWidth > window.innerHeight;
    }

    init(timeVisible = false) {
        var container = document.getElementById(this.containerId);
        if (this.chart) { this.chart.remove(); }

        // Remove previous resize / orientation listeners to prevent leaks
        if (this._resizeHandler) {
            window.removeEventListener('resize', this._resizeHandler);
        }
        if (this._orientationHandler && screen.orientation) {
            screen.orientation.removeEventListener('change', this._orientationHandler);
            this._orientationHandler = null;
        }

        // Configure touch-friendly crosshair on mobile
        var crosshairConfig = { mode: LightweightCharts.CrosshairMode.Normal };
        if (this._isMobile()) {
            crosshairConfig = {
                mode: LightweightCharts.CrosshairMode.Magnet,
                vertLine: { labelVisible: true },
                horzLine: { labelVisible: true },
            };
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
            crosshair: crosshairConfig,
            rightPriceScale: { borderColor: '#374151' },
            timeScale: { borderColor: '#374151', timeVisible: timeVisible, secondsVisible: false },
            handleScroll: {
                mouseWheel: true,
                pressedMouseMove: true,
                horzTouchDrag: true,
                vertTouchDrag: false,
            },
            handleScale: {
                axisPressedMouseMove: true,
                mouseWheel: true,
                pinch: true,
            },
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

        // Debounced resize handler
        var self = this;
        var resizeDebounceTimer = null;
        this._resizeHandler = function() {
            if (resizeDebounceTimer) clearTimeout(resizeDebounceTimer);
            resizeDebounceTimer = setTimeout(function() {
                if (self.chart) {
                    self.chart.applyOptions({
                        width: container.clientWidth,
                        height: container.clientHeight,
                    });
                }
            }, 100);
        };
        window.addEventListener('resize', this._resizeHandler);

        // Listen for orientation changes on mobile to resize chart height
        if (screen.orientation) {
            this._orientationHandler = function() {
                // Delay slightly so the container CSS has time to apply
                setTimeout(function() {
                    self._resizeHandler();
                    // Auto-fullscreen on landscape for mobile
                    if (self._isMobile() && self._isLandscape() && !self._isFullscreen) {
                        self.enterFullscreen();
                    } else if (self._isMobile() && !self._isLandscape() && self._isFullscreen) {
                        self.exitFullscreen();
                    }
                }, 200);
            };
            screen.orientation.addEventListener('change', this._orientationHandler);
        }

        // Initialize mobile touch gestures
        this._initTouchGestures();
        // Initialize mobile toolbar
        this._initMobileToolbar();
        // Initialize OHLCV crosshair tooltip
        this._initCrosshairTooltip();
    }

    // ---- OHLCV Crosshair Tooltip ----

    _initCrosshairTooltip() {
        if (!this.chart) return;
        var self = this;

        // Unsubscribe previous handler
        if (this._crosshairHandler) {
            try { this.chart.unsubscribeCrosshairMove(this._crosshairHandler); } catch (e) {}
        }

        var tooltip = document.getElementById('chartOhlcvTooltip');
        if (!tooltip) return;

        this._crosshairHandler = function(param) {
            if (!param || !param.time || !param.point) {
                tooltip.classList.remove('visible');
                return;
            }

            var seriesData = null;
            var activeSeries = self.candleSeries || self.lineSeries;
            if (activeSeries && param.seriesData) {
                seriesData = param.seriesData.get(activeSeries);
            }

            if (!seriesData) {
                tooltip.classList.remove('visible');
                return;
            }

            // Build tooltip content
            var html = '';
            if (seriesData.open !== undefined) {
                // Candlestick data
                var color = seriesData.close >= seriesData.open ? '#00c853' : '#ff1744';
                html = '<div style="color:' + color + ';font-weight:600;">' +
                    'O: ' + seriesData.open.toFixed(2) +
                    '  H: ' + seriesData.high.toFixed(2) +
                    '  L: ' + seriesData.low.toFixed(2) +
                    '  C: ' + seriesData.close.toFixed(2) +
                    '</div>';
            } else if (seriesData.value !== undefined) {
                // Line data
                html = '<div>Price: ' + seriesData.value.toFixed(2) + '</div>';
            }

            // Volume from volume series
            if (self.volumeSeries && param.seriesData) {
                var volData = param.seriesData.get(self.volumeSeries);
                if (volData && volData.value !== undefined) {
                    var vol = volData.value;
                    var volStr = vol >= 10000000 ? (vol / 10000000).toFixed(2) + ' Cr' :
                                 vol >= 100000 ? (vol / 100000).toFixed(2) + ' L' :
                                 vol >= 1000 ? (vol / 1000).toFixed(1) + ' K' :
                                 vol.toString();
                    html += '<div style="color:#9ca3af;">Vol: ' + volStr + '</div>';
                }
            }

            if (html) {
                tooltip.innerHTML = html;
                tooltip.classList.add('visible');
            } else {
                tooltip.classList.remove('visible');
            }
        };

        this.chart.subscribeCrosshairMove(this._crosshairHandler);
    }

    // ---- Touch Gestures ----

    _initTouchGestures() {
        var container = document.getElementById(this.containerId);
        if (!container) return;

        // Remove old listeners if re-initing
        if (this._touchStartFn) {
            container.removeEventListener('touchstart', this._touchStartFn);
            container.removeEventListener('touchmove', this._touchMoveFn);
            container.removeEventListener('touchend', this._touchEndFn);
        }

        var self = this;
        var ts = this._touchState;

        this._touchStartFn = function(e) {
            var touches = e.touches;

            if (touches.length === 1) {
                var now = Date.now();
                var touch = touches[0];

                // Double-tap detection
                if (now - ts.lastTapTime < 300) {
                    // Double-tap: reset zoom
                    e.preventDefault();
                    if (self.chart) {
                        self.chart.timeScale().fitContent();
                    }
                    ts.lastTapTime = 0;
                    return;
                }
                ts.lastTapTime = now;

                // Long-press detection for crosshair tooltip
                ts.swipeStartX = touch.clientX;
                ts.swipeStartY = touch.clientY;
                ts.swipeStartTime = now;
                ts.isLongPress = false;
                ts.isSwiping = false;

                ts.longPressTimer = setTimeout(function() {
                    ts.isLongPress = true;
                    // Activate magnet crosshair on long-press
                    if (self.chart) {
                        self.chart.applyOptions({
                            crosshair: { mode: LightweightCharts.CrosshairMode.Magnet },
                        });
                    }
                }, 500);
            }
        };

        this._touchMoveFn = function(e) {
            var touches = e.touches;
            if (touches.length === 1) {
                var touch = touches[0];
                var dx = Math.abs(touch.clientX - ts.swipeStartX);
                var dy = Math.abs(touch.clientY - ts.swipeStartY);

                // If moved more than 10px, cancel long-press
                if (dx > 10 || dy > 10) {
                    if (ts.longPressTimer) {
                        clearTimeout(ts.longPressTimer);
                        ts.longPressTimer = null;
                    }
                }

                // Track if this looks like a horizontal swipe
                if (dx > 30 && dx > dy * 2) {
                    ts.isSwiping = true;
                }
            }
        };

        this._touchEndFn = function(e) {
            // Clear long-press timer
            if (ts.longPressTimer) {
                clearTimeout(ts.longPressTimer);
                ts.longPressTimer = null;
            }

            // Restore normal crosshair after long-press
            if (ts.isLongPress) {
                ts.isLongPress = false;
                if (self.chart && self._isMobile()) {
                    self.chart.applyOptions({
                        crosshair: { mode: LightweightCharts.CrosshairMode.Magnet },
                    });
                }
                // Hide tooltip
                var tooltip = document.getElementById('chartOhlcvTooltip');
                if (tooltip) tooltip.classList.remove('visible');
            }

            // Handle swipe left/right to switch stocks
            if (ts.isSwiping && e.changedTouches && e.changedTouches.length > 0) {
                var endX = e.changedTouches[0].clientX;
                var endTime = Date.now();
                var dx = endX - ts.swipeStartX;
                var elapsed = endTime - ts.swipeStartTime;

                // Fast horizontal swipe: at least 80px in < 400ms
                if (Math.abs(dx) > 80 && elapsed < 400) {
                    self._handleStockSwipe(dx > 0 ? 'right' : 'left');
                }
            }

            ts.isSwiping = false;
        };

        container.addEventListener('touchstart', this._touchStartFn, { passive: false });
        container.addEventListener('touchmove', this._touchMoveFn, { passive: true });
        container.addEventListener('touchend', this._touchEndFn, { passive: true });
    }

    _handleStockSwipe(direction) {
        // Get watchlist symbols
        var symbols = [];
        if (typeof Watchlist !== 'undefined' && Watchlist._items && Watchlist._items.length > 0) {
            symbols = Watchlist._items.map(function(i) { return { symbol: i.symbol, name: i.name || i.symbol }; });
        }
        if (symbols.length < 2) return;

        // Find current index
        var currentSymbol = (typeof App !== 'undefined' && App.currentSymbol) ? App.currentSymbol : null;
        if (!currentSymbol) return;

        var currentIdx = -1;
        for (var i = 0; i < symbols.length; i++) {
            if (symbols[i].symbol === currentSymbol) { currentIdx = i; break; }
        }
        if (currentIdx === -1) return;

        var newIdx;
        if (direction === 'left') {
            // Swipe left = next stock
            newIdx = (currentIdx + 1) % symbols.length;
        } else {
            // Swipe right = previous stock
            newIdx = (currentIdx - 1 + symbols.length) % symbols.length;
        }

        var newStock = symbols[newIdx];

        // Show visual swipe hint
        this._showSwipeHint(direction);

        // Load the new stock
        if (typeof App !== 'undefined' && App.loadStock) {
            App.loadStock(newStock.symbol, newStock.name);
        }
    }

    _showSwipeHint(direction) {
        var hintId = direction === 'left' ? 'chartSwipeRight' : 'chartSwipeLeft';
        var hint = document.getElementById(hintId);
        if (!hint) return;
        hint.classList.add('visible');
        setTimeout(function() { hint.classList.remove('visible'); }, 400);
    }

    // ---- Fullscreen ----

    enterFullscreen() {
        var wrapper = document.getElementById('chartWrapper');
        if (!wrapper || this._isFullscreen) return;

        wrapper.classList.add('chart-fullscreen', 'chart-fullscreen-enter');
        this._isFullscreen = true;

        var fsBtn = document.getElementById('btnChartFullscreen');
        if (fsBtn) fsBtn.innerHTML = '&#x2716;'; // X icon

        // Update chart size after CSS transition
        var self = this;
        setTimeout(function() {
            wrapper.classList.remove('chart-fullscreen-enter');
            self._resizeHandler();
        }, 300);
    }

    exitFullscreen() {
        var wrapper = document.getElementById('chartWrapper');
        if (!wrapper || !this._isFullscreen) return;

        wrapper.classList.add('chart-fullscreen-exit');
        var self = this;
        setTimeout(function() {
            wrapper.classList.remove('chart-fullscreen', 'chart-fullscreen-exit');
            self._isFullscreen = false;
            var fsBtn = document.getElementById('btnChartFullscreen');
            if (fsBtn) fsBtn.innerHTML = '&#x26F6;'; // Fullscreen icon
            self._resizeHandler();
        }, 250);
    }

    toggleFullscreen() {
        if (this._isFullscreen) {
            this.exitFullscreen();
        } else {
            this.enterFullscreen();
        }
    }

    // ---- Mobile Toolbar ----

    _initMobileToolbar() {
        if (this._mobileToolbarInited) return;
        this._mobileToolbarInited = true;

        var self = this;

        // Timeframe buttons
        var tfBtns = document.querySelectorAll('#mobileTimeframes .mobile-tf-btn');
        tfBtns.forEach(function(btn) {
            btn.addEventListener('click', function() {
                var period = btn.dataset.period;

                // Update active state on mobile toolbar
                tfBtns.forEach(function(b) { b.classList.remove('active'); });
                btn.classList.add('active');

                // Also sync with desktop period buttons
                var desktopBtns = document.querySelectorAll('.period-btn');
                desktopBtns.forEach(function(b) {
                    b.classList.toggle('active', b.dataset.period === period);
                });

                // Trigger period change via App
                if (typeof App !== 'undefined') {
                    App.currentPeriod = period;
                    if (App.currentSymbol) {
                        App.loadHistory(App.currentSymbol, period);
                    }
                }
            });
        });

        // Indicator toggle buttons
        var indBtns = document.querySelectorAll('#mobileIndicators .mobile-ind-btn');
        indBtns.forEach(function(btn) {
            btn.addEventListener('click', function() {
                var ind = btn.dataset.indicator;
                self._activeIndicators[ind] = !self._activeIndicators[ind];
                btn.classList.toggle('active', self._activeIndicators[ind]);
                self._applyMobileIndicators();
            });
        });

        // Set initial volume indicator as active
        var volBtn = document.querySelector('#mobileIndicators .mobile-ind-btn[data-indicator="vol"]');
        if (volBtn) volBtn.classList.add('active');

        // Fullscreen button
        var fsBtn = document.getElementById('btnChartFullscreen');
        if (fsBtn) {
            fsBtn.addEventListener('click', function() {
                self.toggleFullscreen();
            });
        }
    }

    _applyMobileIndicators() {
        // Clear existing overlays
        this.clearOverlays();

        if (!this._rawData || this._rawData.length === 0) return;

        var data = this._rawData;

        // SMA (20-period)
        if (this._activeIndicators.sma) {
            var smaData = this._calcSMA(data, 20);
            this.addLineOverlay(smaData, '#ff9800', 'SMA 20');
        }

        // EMA (20-period)
        if (this._activeIndicators.ema) {
            var emaData = this._calcEMA(data, 20);
            this.addLineOverlay(emaData, '#e040fb', 'EMA 20');
        }

        // Bollinger Bands (20, 2)
        if (this._activeIndicators.bb) {
            var bb = this._calcBB(data, 20, 2);
            this.addLineOverlay(bb.upper, 'rgba(41,121,255,0.5)', 'BB Upper');
            this.addLineOverlay(bb.lower, 'rgba(41,121,255,0.5)', 'BB Lower');
            this.addLineOverlay(bb.middle, 'rgba(41,121,255,0.3)', 'BB Mid');
        }

        // Volume visibility
        if (this.volumeSeries) {
            this.chart.priceScale('volume').applyOptions({
                scaleMargins: { top: this._activeIndicators.vol ? 0.8 : 1.0, bottom: 0 },
            });
        }
    }

    _calcSMA(data, period) {
        var result = [];
        for (var i = 0; i < data.length; i++) {
            if (i < period - 1) {
                result.push({ time: data[i].date, value: null });
            } else {
                var sum = 0;
                for (var j = i - period + 1; j <= i; j++) {
                    sum += data[j].close;
                }
                result.push({ time: data[i].date, value: sum / period });
            }
        }
        return result;
    }

    _calcEMA(data, period) {
        var result = [];
        var multiplier = 2 / (period + 1);
        var ema = null;

        for (var i = 0; i < data.length; i++) {
            if (i < period - 1) {
                result.push({ time: data[i].date, value: null });
            } else if (i === period - 1) {
                // First EMA = SMA
                var sum = 0;
                for (var j = 0; j < period; j++) sum += data[j].close;
                ema = sum / period;
                result.push({ time: data[i].date, value: ema });
            } else {
                ema = (data[i].close - ema) * multiplier + ema;
                result.push({ time: data[i].date, value: ema });
            }
        }
        return result;
    }

    _calcBB(data, period, stdDevMult) {
        var sma = this._calcSMA(data, period);
        var upper = [];
        var lower = [];
        var middle = [];

        for (var i = 0; i < data.length; i++) {
            if (sma[i].value === null) {
                upper.push({ time: data[i].date, value: null });
                lower.push({ time: data[i].date, value: null });
                middle.push({ time: data[i].date, value: null });
            } else {
                var sum = 0;
                for (var j = i - period + 1; j <= i; j++) {
                    var diff = data[j].close - sma[i].value;
                    sum += diff * diff;
                }
                var stdDev = Math.sqrt(sum / period);
                upper.push({ time: data[i].date, value: sma[i].value + stdDev * stdDevMult });
                lower.push({ time: data[i].date, value: sma[i].value - stdDev * stdDevMult });
                middle.push({ time: data[i].date, value: sma[i].value });
            }
        }
        return { upper: upper, lower: lower, middle: middle };
    }

    setData(data) {
        if (!data || data.length === 0) return;

        // Store raw data for indicator calculations
        this._rawData = data;

        var candleData = data.map(function(d) {
            return {
                time: d.date,
                open: d.open,
                high: d.high,
                low: d.low,
                close: d.close,
            };
        });

        var volumeData = data.map(function(d) {
            return {
                time: d.date,
                value: d.volume,
                color: d.close >= d.open ? 'rgba(0,200,83,0.3)' : 'rgba(255,23,68,0.3)',
            };
        });

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
            this.lineSeries.setData(data.map(function(d) { return { time: d.date, value: d.close }; }));
        }

        this.volumeSeries.setData(volumeData);
        this.chart.timeScale().fitContent();

        // Track last candle for real-time updates
        if (candleData.length > 0) {
            this.lastCandle = Object.assign({}, candleData[candleData.length - 1]);
        }

        // Re-apply mobile indicators if any are active
        this._applyMobileIndicators();
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
            var self = this;
            this._throttleTimer = setTimeout(function() {
                self._throttleTimer = null;
                // Apply any values that arrived during the throttle window
                if (self._pendingPrice !== null) {
                    self._applyPendingUpdate();
                }
            }, 250);
        }
    }

    _applyPendingUpdate() {
        var price = this._pendingPrice;
        var volume = this._pendingVolume;
        this._pendingPrice = null;
        this._pendingVolume = null;

        if (!this.lastCandle || !price) return;

        var updated = Object.assign({}, this.lastCandle);
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
        var self = this;
        this.overlayLines.forEach(function(s) {
            try { self.chart.removeSeries(s); } catch (e) {}
        });
        this.overlayLines = [];
    }

    addLineOverlay(data, color, title) {
        title = title || '';
        var series = this.chart.addLineSeries({
            color: color,
            lineWidth: 1,
            title: title,
        });
        series.setData(data.filter(function(d) { return d.value !== null && d.value !== undefined; }));
        this.overlayLines.push(series);
        return series;
    }

    addPredictionMarkers(predictions, dates) {
        if (!this.candleSeries && !this.lineSeries) return;
        var series = this.candleSeries || this.lineSeries;
        var markers = predictions.map(function(price, i) {
            return {
                time: dates[i],
                position: 'aboveBar',
                color: '#ffd600',
                shape: 'circle',
                text: '$' + price.toFixed(0),
            };
        });
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
        var container = document.getElementById(this.containerId);
        if (container) {
            container.style.cursor = drawing ? 'crosshair' : '';
        }
    }

    _updateDrawingButtons() {
        var hBtn = document.getElementById('btnDrawHLine');
        var tBtn = document.getElementById('btnDrawTrendline');
        if (hBtn) hBtn.classList.toggle('active', this._drawingMode === 'horizontal');
        if (tBtn) tBtn.classList.toggle('active', this._drawingMode === 'trendline');
    }

    initDrawingTools() {
        // Remove old DOM listeners to prevent duplicates on re-init
        if (this._drawBtnHandlers) {
            var old = this._drawBtnHandlers;
            if (old.hBtn) old.hBtn.removeEventListener('click', old.hFn);
            if (old.tBtn) old.tBtn.removeEventListener('click', old.tFn);
            if (old.cBtn) old.cBtn.removeEventListener('click', old.cFn);
        }

        var hBtn = document.getElementById('btnDrawHLine');
        var tBtn = document.getElementById('btnDrawTrendline');
        var cBtn = document.getElementById('btnDrawClear');

        var self = this;
        var hFn = function() { self.setDrawingMode('horizontal'); };
        var tFn = function() { self.setDrawingMode('trendline'); };
        var cFn = function() { self.clearDrawings(); };

        if (hBtn) hBtn.addEventListener('click', hFn);
        if (tBtn) tBtn.addEventListener('click', tFn);
        if (cBtn) cBtn.addEventListener('click', cFn);

        this._drawBtnHandlers = { hBtn: hBtn, tBtn: tBtn, cBtn: cBtn, hFn: hFn, tFn: tFn, cFn: cFn };

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

        var self = this;
        this._clickHandler = function(param) {
            if (self._drawingMode === 'none') return;
            if (!param.point || !param.time) return;

            var activeSeries = self.candleSeries || self.lineSeries;
            if (!activeSeries) return;

            var price = activeSeries.coordinateToPrice(param.point.y);
            if (price == null || !isFinite(price)) return;

            if (self._drawingMode === 'horizontal') {
                self._addHorizontalLine(price);
                // Stay in horizontal mode for rapid placement
            } else if (self._drawingMode === 'trendline') {
                if (!self._trendlineStart) {
                    // First click: store start point
                    self._trendlineStart = { time: param.time, price: price };
                } else {
                    // Second click: draw the trendline
                    self._addTrendline(self._trendlineStart, { time: param.time, price: price });
                    self._trendlineStart = null;
                    // Exit trendline mode after drawing
                    self.setDrawingMode('none');
                }
            }
        };

        this.chart.subscribeClick(this._clickHandler);
    }

    _addHorizontalLine(price) {
        var activeSeries = this.candleSeries || this.lineSeries;
        if (!activeSeries) return;

        var priceLine = activeSeries.createPriceLine({
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

        var trendSeries = this.chart.addLineSeries({
            color: '#00bcd4',
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Solid,
            crosshairMarkerVisible: false,
            lastValueVisible: false,
            priceLineVisible: false,
        });

        // Ensure points are in chronological order
        var points = [start, end].sort(function(a, b) {
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
        var self = this;
        // Remove horizontal price lines
        this._drawnPriceLines.forEach(function(item) {
            try { item.series.removePriceLine(item.line); } catch (e) {}
        });
        this._drawnPriceLines = [];

        // Remove trendline series
        this._drawnTrendlines.forEach(function(s) {
            try { self.chart.removeSeries(s); } catch (e) {}
        });
        this._drawnTrendlines = [];

        // Reset drawing mode
        this._trendlineStart = null;
        this.setDrawingMode('none');
    }
}

// RSI & MACD sub-charts
class IndicatorChart {
    constructor(containerId, options) {
        this.containerId = containerId;
        this.chart = null;
        this.options = options || {};
        this._resizeHandler = null;
    }

    init() {
        var container = document.getElementById(this.containerId);
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

        var self = this;
        var debounceTimer = null;
        this._resizeHandler = function() {
            if (debounceTimer) clearTimeout(debounceTimer);
            debounceTimer = setTimeout(function() {
                if (self.chart) {
                    self.chart.applyOptions({
                        width: container.clientWidth,
                        height: container.clientHeight,
                    });
                }
            }, 100);
        };
        window.addEventListener('resize', this._resizeHandler);
    }

    setRSIData(data) {
        if (!this.chart) this.init();
        var series = this.chart.addLineSeries({ color: '#a78bfa', lineWidth: 1.5, title: 'RSI' });
        series.setData(data.filter(function(d) { return d.value !== null; }));

        // Overbought/oversold lines
        var line70 = this.chart.addLineSeries({ color: 'rgba(255,23,68,0.3)', lineWidth: 1, lineStyle: 2 });
        var line30 = this.chart.addLineSeries({ color: 'rgba(0,200,83,0.3)', lineWidth: 1, lineStyle: 2 });
        if (data.length > 0) {
            var times = data.filter(function(d) { return d.value !== null; });
            line70.setData(times.map(function(d) { return { time: d.time, value: 70 }; }));
            line30.setData(times.map(function(d) { return { time: d.time, value: 30 }; }));
        }
        this.chart.timeScale().fitContent();
    }

    setMACDData(macdLine, signalLine, histogram) {
        if (!this.chart) this.init();
        var macd = this.chart.addLineSeries({ color: '#2979ff', lineWidth: 1.5, title: 'MACD' });
        var signal = this.chart.addLineSeries({ color: '#ff9800', lineWidth: 1.5, title: 'Signal' });
        var hist = this.chart.addHistogramSeries({ title: 'Histogram' });

        macd.setData(macdLine.filter(function(d) { return d.value !== null; }));
        signal.setData(signalLine.filter(function(d) { return d.value !== null; }));
        hist.setData(histogram.filter(function(d) { return d.value !== null; }).map(function(d) {
            return {
                time: d.time, value: d.value,
                color: d.value >= 0 ? 'rgba(0,200,83,0.5)' : 'rgba(255,23,68,0.5)',
            };
        }));
        this.chart.timeScale().fitContent();
    }
}
