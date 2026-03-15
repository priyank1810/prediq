/**
 * BrokerUI — Order placement panel controller.
 * Depends on: API (api.js), App (app.js for showToast and currentSymbol)
 */
const BrokerUI = {
    _txnType: 'BUY',
    _symbol: null,

    /** Initialise broker panel state and check broker status. */
    init() {
        this.checkStatus();
    },

    /** Update the panel when a new stock is selected. */
    setSymbol(symbol) {
        this._symbol = symbol;
        this.loadRecentOrders();
    },

    /** Toggle between BUY and SELL. */
    setTransactionType(type) {
        this._txnType = type;
        const buyBtn = document.getElementById('orderBuyBtn');
        const sellBtn = document.getElementById('orderSellBtn');
        const placeBtn = document.getElementById('orderPlaceBtn');
        if (!buyBtn) return;

        if (type === 'BUY') {
            buyBtn.className = 'flex-1 py-1.5 text-xs font-semibold rounded bg-green-600 text-white';
            sellBtn.className = 'flex-1 py-1.5 text-xs font-semibold rounded bg-dark-700 text-gray-400 hover:bg-dark-600';
            placeBtn.className = 'w-full py-2 text-xs font-bold rounded bg-green-600 hover:bg-green-500 text-white transition';
            placeBtn.textContent = 'PLACE BUY ORDER';
        } else {
            sellBtn.className = 'flex-1 py-1.5 text-xs font-semibold rounded bg-red-600 text-white';
            buyBtn.className = 'flex-1 py-1.5 text-xs font-semibold rounded bg-dark-700 text-gray-400 hover:bg-dark-600';
            placeBtn.className = 'w-full py-2 text-xs font-bold rounded bg-red-600 hover:bg-red-500 text-white transition';
            placeBtn.textContent = 'PLACE SELL ORDER';
        }
    },

    /** Show/hide price fields based on order type. */
    onOrderTypeChange() {
        const ot = document.getElementById('orderType').value;
        const priceRow = document.getElementById('orderPriceRow');
        const priceWrap = document.getElementById('orderPriceWrap');
        const triggerWrap = document.getElementById('orderTriggerWrap');
        if (!priceRow) return;

        if (ot === 'MARKET') {
            priceRow.classList.add('hidden');
        } else {
            priceRow.classList.remove('hidden');
            // Price visible for LIMIT and SL
            priceWrap.classList.toggle('hidden', ot === 'SL-M');
            // Trigger visible for SL and SL-M
            triggerWrap.classList.toggle('hidden', ot === 'LIMIT');
        }
    },

    /** Add to quantity input. */
    addQty(n) {
        const el = document.getElementById('orderQty');
        if (!el) return;
        el.value = Math.max(1, (parseInt(el.value) || 0) + n);
    },

    /** Set quantity to standard lot size (equity = 1, F&O varies). */
    setQtyLot() {
        const el = document.getElementById('orderQty');
        if (el) el.value = 25; // Default F&O lot; for equities user can adjust
    },

    /** Place the order. */
    async placeOrder() {
        const symbol = this._symbol || (typeof App !== 'undefined' ? App.currentSymbol : null);
        if (!symbol) {
            this._showStatus('Select a stock first', 'text-yellow-400');
            return;
        }

        const orderType = document.getElementById('orderType').value;
        const qty = parseInt(document.getElementById('orderQty').value) || 0;
        const price = parseFloat(document.getElementById('orderPrice').value) || null;
        const triggerPrice = parseFloat(document.getElementById('orderTriggerPrice').value) || null;
        const paperTrade = document.getElementById('orderPaperTrade').checked;

        if (qty <= 0) {
            this._showStatus('Quantity must be positive', 'text-yellow-400');
            return;
        }

        // Confirm live orders
        if (!paperTrade) {
            const msg = `Place LIVE ${this._txnType} order: ${qty} x ${symbol} (${orderType})?`;
            if (!confirm(msg)) return;
        }

        const btn = document.getElementById('orderPlaceBtn');
        const origText = btn.textContent;
        btn.disabled = true;
        btn.textContent = 'Placing...';

        try {
            const data = {
                symbol: symbol,
                exchange: 'NSE',
                order_type: orderType,
                transaction_type: this._txnType,
                quantity: qty,
                price: price,
                trigger_price: triggerPrice,
                paper_trade: paperTrade,
            };

            const result = await API.placeBrokerOrder(data);
            const label = paperTrade ? 'Paper' : 'Live';
            const statusColor = result.status === 'executed' || result.status === 'placed'
                ? 'text-green-400' : 'text-red-400';
            this._showStatus(
                `${label} ${this._txnType} ${qty} ${symbol} - ${result.status.toUpperCase()}`,
                statusColor
            );
            if (typeof App !== 'undefined' && App.showToast) {
                App.showToast(`Order ${result.status}: ${this._txnType} ${qty} ${symbol}`, 'success');
            }
            this.loadRecentOrders();
        } catch (e) {
            this._showStatus(e.message || 'Order failed', 'text-red-400');
        } finally {
            btn.disabled = false;
            btn.textContent = origText;
        }
    },

    /** Load recent orders into the list. */
    async loadRecentOrders() {
        const container = document.getElementById('orderRecentList');
        if (!container) return;

        try {
            const orders = await API.getBrokerRecent(5);
            if (!orders || orders.length === 0) {
                container.innerHTML = '<p class="text-xs text-gray-600 text-center py-2">No recent orders</p>';
                return;
            }

            container.innerHTML = orders.map(function(o) {
                var isBuy = o.transaction_type === 'BUY';
                var color = isBuy ? 'text-green-400' : 'text-red-400';
                var bgColor = isBuy ? 'border-green-900/30' : 'border-red-900/30';
                var statusBg = o.status === 'executed' ? 'bg-green-900/30 text-green-400'
                    : o.status === 'placed' ? 'bg-blue-900/30 text-blue-400'
                    : o.status === 'cancelled' ? 'bg-gray-700 text-gray-400'
                    : 'bg-red-900/30 text-red-400';
                var paper = o.paper_trade ? '<span class="text-[9px] text-yellow-500 ml-1">PAPER</span>' : '';
                var priceStr = o.price ? (' @ ' + o.price.toFixed(2)) : '';
                var time = o.created_at ? new Date(o.created_at).toLocaleTimeString('en-IN', {hour: '2-digit', minute: '2-digit'}) : '';
                var cancelBtn = (o.status === 'pending' || o.status === 'placed')
                    ? '<button class="text-[9px] text-gray-500 hover:text-red-400 ml-auto" onclick="BrokerUI.cancelOrder(' + o.id + ')">Cancel</button>'
                    : '';

                return '<div class="flex items-center gap-2 py-1 px-2 rounded border ' + bgColor + ' bg-dark-700/50 text-[11px]">'
                    + '<span class="font-semibold ' + color + '">' + o.transaction_type + '</span>'
                    + '<span class="text-white">' + o.quantity + ' ' + o.symbol + priceStr + '</span>'
                    + paper
                    + '<span class="' + statusBg + ' px-1 py-0.5 rounded text-[9px]">' + o.status + '</span>'
                    + '<span class="text-gray-600 text-[9px]">' + time + '</span>'
                    + cancelBtn
                    + '</div>';
            }).join('');
        } catch (e) {
            // Silently fail — user may not be logged in
        }
    },

    /** Cancel an order by ID. */
    async cancelOrder(orderId) {
        if (!confirm('Cancel this order?')) return;
        try {
            await API.cancelBrokerOrder(orderId);
            this.loadRecentOrders();
            if (typeof App !== 'undefined' && App.showToast) {
                App.showToast('Order cancelled', 'success');
            }
        } catch (e) {
            if (typeof App !== 'undefined' && App.showToast) {
                App.showToast(e.message || 'Cancel failed', 'error');
            }
        }
    },

    /** Check broker connection and update badge. */
    async checkStatus() {
        var badge = document.getElementById('orderBrokerBadge');
        if (!badge) return;
        try {
            var status = await API.getBrokerStatus();
            if (status.connected) {
                badge.textContent = 'Connected';
                badge.className = 'text-[10px] px-1.5 py-0.5 rounded bg-green-900/40 text-green-400';
            } else if (status.available) {
                badge.textContent = 'Available';
                badge.className = 'text-[10px] px-1.5 py-0.5 rounded bg-yellow-900/40 text-yellow-400';
            } else {
                badge.textContent = 'Disconnected';
                badge.className = 'text-[10px] px-1.5 py-0.5 rounded bg-dark-700 text-gray-500';
            }
        } catch (e) {
            // Not logged in or API error — leave as disconnected
        }
    },

    /** Show a transient status message. */
    _showStatus(msg, colorClass) {
        var el = document.getElementById('orderStatus');
        if (!el) return;
        el.textContent = msg;
        el.className = 'text-xs mt-2 text-center ' + (colorClass || 'text-gray-400');
        el.classList.remove('hidden');
        setTimeout(function() { el.classList.add('hidden'); }, 5000);
    }
};

// Auto-init when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Delay init slightly to let auth settle
    setTimeout(function() { BrokerUI.init(); }, 1000);
});
