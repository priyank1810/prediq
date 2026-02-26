const Notifications = {
    notifications: [],
    unreadCount: 0,

    init() {
        document.getElementById('notificationBell').addEventListener('click', (e) => {
            e.stopPropagation();
            const dropdown = document.getElementById('notificationDropdown');
            dropdown.classList.toggle('hidden');
            if (!dropdown.classList.contains('hidden')) {
                this.markAllRead();
            }
        });

        document.getElementById('clearNotifications').addEventListener('click', () => {
            this.notifications = [];
            this.unreadCount = 0;
            this.render();
        });

        document.addEventListener('click', (e) => {
            const bell = document.getElementById('notificationBell');
            const dropdown = document.getElementById('notificationDropdown');
            if (!bell.contains(e.target) && !dropdown.contains(e.target)) {
                dropdown.classList.add('hidden');
            }
        });
    },

    addNotification(data) {
        const notification = {
            id: Date.now(),
            symbol: data.symbol,
            direction: data.direction,
            confidence: data.confidence,
            price: data.price,
            timestamp: data.timestamp || new Date().toISOString(),
            read: false,
        };
        this.notifications.unshift(notification);
        if (this.notifications.length > 50) this.notifications.pop();
        this.unreadCount++;
        this.render();

        // Shake the bell
        const bell = document.getElementById('notificationBell');
        bell.classList.add('bell-shake');
        setTimeout(() => bell.classList.remove('bell-shake'), 500);

        // Show toast
        const type = data.direction === 'BULLISH' ? 'success' : 'error';
        App.showToast(
            `AI Signal: ${data.symbol} ${data.direction} (${data.confidence}% confidence)`,
            type
        );
    },

    markAllRead() {
        this.unreadCount = 0;
        this.notifications.forEach(n => n.read = true);
        this.updateBadge();
    },

    updateBadge() {
        const badge = document.getElementById('notificationCount');
        if (this.unreadCount > 0) {
            badge.textContent = this.unreadCount > 9 ? '9+' : this.unreadCount;
            badge.classList.remove('hidden');
        } else {
            badge.classList.add('hidden');
        }
    },

    render() {
        this.updateBadge();
        const list = document.getElementById('notificationList');
        if (this.notifications.length === 0) {
            list.innerHTML = '<div class="px-4 py-6 text-center text-gray-500 text-sm">No notifications yet</div>';
            return;
        }
        list.innerHTML = this.notifications.slice(0, 20).map(n => {
            const color = n.direction === 'BULLISH' ? 'text-green-400' : 'text-red-400';
            const bg = n.direction === 'BULLISH' ? 'bg-green-900/20' : 'bg-red-900/20';
            const arrow = n.direction === 'BULLISH' ? '&#9650;' : '&#9660;';
            const time = new Date(n.timestamp).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' });
            const unreadDot = !n.read ? '<div class="w-1.5 h-1.5 bg-accent-blue rounded-full"></div>' : '';
            const priceStr = n.price ? ` | ₹${n.price.toFixed(2)}` : '';
            return `
                <div class="px-4 py-3 hover:bg-dark-600 cursor-pointer flex items-center gap-3 ${bg}"
                     onclick="Search.select('${n.symbol}', '${n.symbol}')">
                    <span class="${color} text-lg">${arrow}</span>
                    <div class="flex-1 min-w-0">
                        <div class="text-sm text-white font-medium">${n.symbol}</div>
                        <div class="text-xs ${color}">${n.direction} - ${n.confidence}%${priceStr}</div>
                    </div>
                    <div class="flex items-center gap-2 flex-shrink-0">
                        <span class="text-xs text-gray-500">${time}</span>
                        ${unreadDot}
                    </div>
                </div>
            `;
        }).join('');
    }
};
