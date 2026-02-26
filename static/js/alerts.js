const Alerts = {
    init() {
        document.getElementById('createAlertForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.createAlert(e.target);
        });
    },

    async load() {
        try {
            const alerts = await API.getAlerts();
            this.displayAlerts(alerts);
        } catch (e) {
            console.error('Failed to load alerts:', e);
        }
    },

    displayAlerts(alerts) {
        const tbody = document.getElementById('alertsTable');
        const empty = document.getElementById('emptyAlerts');

        if (alerts.length === 0) {
            tbody.innerHTML = '';
            empty.classList.remove('hidden');
            return;
        }

        empty.classList.add('hidden');
        tbody.innerHTML = alerts.map(a => {
            const statusColor = a.is_triggered ? 'text-green-400' : 'text-yellow-400';
            const statusText = a.is_triggered ? 'Triggered' : 'Active';
            const created = new Date(a.created_at).toLocaleDateString('en-IN');

            return `
                <tr class="border-b border-gray-800 hover:bg-dark-700">
                    <td class="px-4 py-3">
                        <span class="text-white font-medium">${a.symbol}</span>
                    </td>
                    <td class="px-4 py-3 text-gray-300">
                        Price goes ${a.condition}
                    </td>
                    <td class="px-4 py-3 text-right text-white">₹${a.target_price.toFixed(2)}</td>
                    <td class="px-4 py-3">
                        <span class="${statusColor} text-xs font-medium px-2 py-0.5 rounded-full ${a.is_triggered ? 'bg-green-900' : 'bg-yellow-900'}">
                            ${statusText}
                        </span>
                    </td>
                    <td class="px-4 py-3 text-gray-400 text-xs">${created}</td>
                    <td class="px-4 py-3 text-right">
                        <button onclick="Alerts.deleteAlert(${a.id})"
                                class="text-red-400 hover:text-red-300 text-xs">Delete</button>
                    </td>
                </tr>
            `;
        }).join('');
    },

    async createAlert(form) {
        const data = {
            symbol: form.symbol.value.toUpperCase(),
            condition: form.condition.value,
            target_price: parseFloat(form.target_price.value),
        };

        try {
            await API.createAlert(data);
            form.reset();
            App.showToast(`Alert set for ${data.symbol}`, 'success');
            this.load();
        } catch (e) {
            App.showToast('Failed to create alert: ' + e.message, 'error');
        }
    },

    async deleteAlert(id) {
        try {
            await API.deleteAlert(id);
            App.showToast('Alert removed', 'success');
            this.load();
        } catch (e) {
            App.showToast('Failed to delete: ' + e.message, 'error');
        }
    },

    onAlertTriggered(data) {
        App.showToast(`Alert: ${data.symbol} is now ${data.condition} ₹${data.target_price}! Current: ₹${data.current_price}`, 'alert');
        this.load();
    }
};
