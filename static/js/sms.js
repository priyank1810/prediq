/**
 * SMS Alerts settings panel controller.
 */
const SMS = {
    _linked: false,
    _pendingPhone: null,

    init() {
        const btn = document.getElementById('smsSettingsBtn');
        const modal = document.getElementById('smsModal');
        const closeBtn = document.getElementById('smsModalClose');

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

        // Send OTP button
        document.getElementById('smsSendOtpBtn').addEventListener('click', () => this.sendOTP());

        // Verify OTP button
        document.getElementById('smsVerifyBtn').addEventListener('click', () => this.verifyOTP());

        // Save preferences
        document.getElementById('smsSavePrefsBtn').addEventListener('click', () => this.savePreferences());

        // Test button
        document.getElementById('smsTestBtn').addEventListener('click', () => this.sendTest());

        // Unlink button
        document.getElementById('smsUnlinkBtn').addEventListener('click', () => this.unlink());

        // Allow Enter key in phone input
        document.getElementById('smsPhoneInput').addEventListener('keydown', (e) => {
            if (e.key === 'Enter') { e.preventDefault(); this.sendOTP(); }
        });

        // Allow Enter key in OTP input
        document.getElementById('smsOtpInput').addEventListener('keydown', (e) => {
            if (e.key === 'Enter') { e.preventDefault(); this.verifyOTP(); }
        });
    },

    _showMsg(text, isError) {
        const el = document.getElementById('smsMessage');
        el.textContent = text;
        el.className = 'mt-3 text-xs ' + (isError ? 'text-red-400' : 'text-green-400');
        el.classList.remove('hidden');
        setTimeout(() => el.classList.add('hidden'), 5000);
    },

    _setStatus(linked, phone) {
        this._linked = linked;
        const dot = document.getElementById('smsStatusDot');
        const txt = document.getElementById('smsStatusText');
        const linkSection = document.getElementById('smsLinkSection');
        const prefsSection = document.getElementById('smsPrefsSection');
        const otpSection = document.getElementById('smsOtpSection');

        if (linked) {
            dot.className = 'w-2.5 h-2.5 rounded-full bg-green-500';
            txt.textContent = 'Linked (' + phone + ')';
            txt.className = 'text-sm text-green-400';
            linkSection.classList.add('hidden');
            otpSection.classList.add('hidden');
            prefsSection.classList.remove('hidden');
        } else {
            dot.className = 'w-2.5 h-2.5 rounded-full bg-gray-500';
            txt.textContent = 'Not linked';
            txt.className = 'text-sm text-gray-400';
            linkSection.classList.remove('hidden');
            otpSection.classList.add('hidden');
            prefsSection.classList.add('hidden');
        }
    },

    _setCheckboxes(alertTypes) {
        const map = {
            signals: 'smsAlertSignals',
            price_alerts: 'smsAlertPriceAlerts',
            news: 'smsAlertNews',
            scanner: 'smsAlertScanner',
        };
        for (const [type, id] of Object.entries(map)) {
            const cb = document.getElementById(id);
            if (cb) cb.checked = alertTypes.includes(type);
        }
    },

    _getSelectedTypes() {
        const ids = ['smsAlertSignals', 'smsAlertPriceAlerts', 'smsAlertNews', 'smsAlertScanner'];
        const types = [];
        for (const id of ids) {
            const cb = document.getElementById(id);
            if (cb && cb.checked) types.push(cb.value);
        }
        return types;
    },

    async loadStatus() {
        try {
            const data = await API.getSMSStatus();
            this._setStatus(data.linked, data.phone_number);
            if (data.linked && data.alert_types) {
                this._setCheckboxes(data.alert_types);
            }
        } catch (e) {
            this._setStatus(false, null);
            if (e.message && e.message.includes('Not authenticated')) {
                this._showMsg('Please log in to use SMS alerts.', true);
            }
        }
    },

    async sendOTP() {
        const phoneRaw = document.getElementById('smsPhoneInput').value.trim();
        const phone = phoneRaw.startsWith('+91') ? phoneRaw : '+91' + phoneRaw;

        if (!/^\+91\d{10}$/.test(phone)) {
            this._showMsg('Enter a valid 10-digit Indian mobile number.', true);
            return;
        }

        try {
            await API.linkSMS(phone);
            this._pendingPhone = phone;
            // Show OTP input section
            document.getElementById('smsLinkSection').classList.add('hidden');
            document.getElementById('smsOtpSection').classList.remove('hidden');
            this._showMsg('OTP sent to ' + phone + '. Check your phone.', false);
        } catch (e) {
            this._showMsg(e.message || 'Failed to send OTP.', true);
        }
    },

    async verifyOTP() {
        const otp = document.getElementById('smsOtpInput').value.trim();
        if (!otp || otp.length !== 6) {
            this._showMsg('Enter the 6-digit OTP.', true);
            return;
        }

        if (!this._pendingPhone) {
            this._showMsg('No phone number pending verification.', true);
            return;
        }

        try {
            const data = await API.verifySMSOTP(this._pendingPhone, otp);
            this._setStatus(true, data.phone_number);
            if (data.alert_types) this._setCheckboxes(data.alert_types);
            this._showMsg('Phone linked successfully!', false);
            this._pendingPhone = null;
            document.getElementById('smsPhoneInput').value = '';
            document.getElementById('smsOtpInput').value = '';
        } catch (e) {
            this._showMsg(e.message || 'OTP verification failed.', true);
        }
    },

    async unlink() {
        if (!confirm('Unlink SMS? You will stop receiving SMS alerts.')) return;
        try {
            await API.unlinkSMS();
            this._setStatus(false, null);
            this._showMsg('SMS unlinked.', false);
        } catch (e) {
            this._showMsg(e.message || 'Failed to unlink.', true);
        }
    },

    async savePreferences() {
        const types = this._getSelectedTypes();
        try {
            await API.updateSMSPreferences(types);
            this._showMsg('Preferences saved!', false);
        } catch (e) {
            this._showMsg(e.message || 'Failed to save preferences.', true);
        }
    },

    async sendTest() {
        try {
            await API.sendSMSTest();
            this._showMsg('Test SMS sent! Check your phone.', false);
        } catch (e) {
            this._showMsg(e.message || 'Failed to send test SMS.', true);
        }
    },
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => SMS.init());
