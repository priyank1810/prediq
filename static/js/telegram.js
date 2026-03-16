/**
 * Telegram Alerts settings panel controller.
 */
const Telegram = {
    _linked: false,

    init() {
        const btn = document.getElementById('telegramSettingsBtn');
        const modal = document.getElementById('telegramModal');
        const closeBtn = document.getElementById('telegramModalClose');

        if (!btn || !modal) return;

        btn.addEventListener('click', () => {
            modal.classList.remove('hidden');
            this.loadStatus();
        });

        closeBtn.addEventListener('click', () => modal.classList.add('hidden'));

        // Close on backdrop click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.classList.add('hidden');
        });

        // Link button
        document.getElementById('tgLinkBtn').addEventListener('click', () => this.link());

        // Save preferences
        document.getElementById('tgSavePrefsBtn').addEventListener('click', () => this.savePreferences());

        // Test button
        document.getElementById('tgTestBtn').addEventListener('click', () => this.sendTest());

        // Unlink button
        document.getElementById('tgUnlinkBtn').addEventListener('click', () => this.unlink());

        // Allow Enter key in chat ID input
        document.getElementById('tgChatIdInput').addEventListener('keydown', (e) => {
            if (e.key === 'Enter') { e.preventDefault(); this.link(); }
        });
    },

    _showMsg(text, isError) {
        const el = document.getElementById('tgMessage');
        el.textContent = text;
        el.className = 'mt-3 text-xs ' + (isError ? 'text-red-400' : 'text-green-400');
        el.classList.remove('hidden');
        setTimeout(() => el.classList.add('hidden'), 5000);
    },

    _setStatus(linked, chatId) {
        this._linked = linked;
        const dot = document.getElementById('tgStatusDot');
        const txt = document.getElementById('tgStatusText');
        const linkSection = document.getElementById('tgLinkSection');
        const prefsSection = document.getElementById('tgPrefsSection');

        if (linked) {
            dot.className = 'w-2.5 h-2.5 rounded-full bg-green-500';
            txt.textContent = 'Linked (Chat ID: ' + chatId + ')';
            txt.className = 'text-sm text-green-400';
            linkSection.classList.add('hidden');
            prefsSection.classList.remove('hidden');
        } else {
            dot.className = 'w-2.5 h-2.5 rounded-full bg-gray-500';
            txt.textContent = 'Not linked';
            txt.className = 'text-sm text-gray-400';
            linkSection.classList.remove('hidden');
            prefsSection.classList.add('hidden');
        }
    },

    _setCheckboxes(alertTypes) {
        const map = {
            signals: 'tgAlertSignals',
            price_alerts: 'tgAlertPriceAlerts',
            news: 'tgAlertNews',
            scanner: 'tgAlertScanner',
            predictions: 'tgAlertPredictions',
        };
        for (const [type, id] of Object.entries(map)) {
            const cb = document.getElementById(id);
            if (cb) cb.checked = alertTypes.includes(type);
        }
    },

    _getSelectedTypes() {
        const ids = ['tgAlertSignals', 'tgAlertPriceAlerts', 'tgAlertNews', 'tgAlertScanner', 'tgAlertPredictions'];
        const types = [];
        for (const id of ids) {
            const cb = document.getElementById(id);
            if (cb && cb.checked) types.push(cb.value);
        }
        return types;
    },

    async loadStatus() {
        try {
            const data = await API.getTelegramStatus();
            this._setStatus(data.linked, data.chat_id);
            if (data.linked && data.alert_types) {
                this._setCheckboxes(data.alert_types);
            }
        } catch (e) {
            // Not logged in or other error
            this._setStatus(false, null);
            if (e.message && e.message.includes('Not authenticated')) {
                this._showMsg('Please log in to use Telegram alerts.', true);
            }
        }
    },

    async link() {
        const chatId = document.getElementById('tgChatIdInput').value.trim();
        if (!chatId) {
            this._showMsg('Please enter a Chat ID.', true);
            return;
        }
        if (!/^-?\d{1,20}$/.test(chatId)) {
            this._showMsg('Invalid Chat ID format. It should be a number.', true);
            return;
        }

        try {
            const data = await API.linkTelegram(chatId);
            this._setStatus(true, data.chat_id);
            if (data.alert_types) this._setCheckboxes(data.alert_types);
            this._showMsg('Telegram linked successfully!', false);
            document.getElementById('tgChatIdInput').value = '';
        } catch (e) {
            this._showMsg(e.message || 'Failed to link Telegram.', true);
        }
    },

    async unlink() {
        if (!confirm('Unlink Telegram? You will stop receiving alerts.')) return;
        try {
            await API.unlinkTelegram();
            this._setStatus(false, null);
            this._showMsg('Telegram unlinked.', false);
        } catch (e) {
            this._showMsg(e.message || 'Failed to unlink.', true);
        }
    },

    async savePreferences() {
        const types = this._getSelectedTypes();
        try {
            await API.updateTelegramPreferences(types);
            this._showMsg('Preferences saved!', false);
        } catch (e) {
            this._showMsg(e.message || 'Failed to save preferences.', true);
        }
    },

    async sendTest() {
        try {
            await API.sendTelegramTest();
            this._showMsg('Test message sent! Check your Telegram.', false);
        } catch (e) {
            this._showMsg(e.message || 'Failed to send test message.', true);
        }
    },
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => Telegram.init());
