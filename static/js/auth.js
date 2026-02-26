const Auth = {
    token: null,
    refreshToken: null,
    user: null,
    _refreshing: false,

    init() {
        this.token = localStorage.getItem('access_token');
        this.refreshToken = localStorage.getItem('refresh_token');

        this.setupForms();
        this.setupUserMenu();

        if (this.token) {
            this.loadProfile();
        } else {
            this.showAuthModal();
        }
    },

    setupForms() {
        document.getElementById('loginForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = document.getElementById('loginEmail').value;
            const password = document.getElementById('loginPassword').value;
            const errEl = document.getElementById('loginError');
            errEl.classList.add('hidden');

            try {
                const formData = new URLSearchParams();
                formData.append('username', email);
                formData.append('password', password);

                const res = await fetch('/api/auth/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: formData,
                });

                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    throw new Error(err.detail || 'Login failed');
                }

                const data = await res.json();
                this.setTokens(data.access_token, data.refresh_token);
                await this.loadProfile();
                this.hideAuthModal();
            } catch (e) {
                errEl.textContent = e.message;
                errEl.classList.remove('hidden');
            }
        });

        document.getElementById('registerForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = document.getElementById('registerEmail').value;
            const password = document.getElementById('registerPassword').value;
            const confirm = document.getElementById('registerConfirm').value;
            const errEl = document.getElementById('registerError');
            errEl.classList.add('hidden');

            if (password !== confirm) {
                errEl.textContent = 'Passwords do not match';
                errEl.classList.remove('hidden');
                return;
            }

            try {
                const res = await fetch('/api/auth/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email, password }),
                });

                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    throw new Error(err.detail || 'Registration failed');
                }

                // Auto-login after register
                const formData = new URLSearchParams();
                formData.append('username', email);
                formData.append('password', password);

                const loginRes = await fetch('/api/auth/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: formData,
                });

                if (!loginRes.ok) throw new Error('Auto-login failed');

                const data = await loginRes.json();
                this.setTokens(data.access_token, data.refresh_token);
                await this.loadProfile();
                this.hideAuthModal();

                if (typeof App !== 'undefined' && App.showToast) {
                    App.showToast('Account created successfully!');
                }
            } catch (e) {
                errEl.textContent = e.message;
                errEl.classList.remove('hidden');
            }
        });
    },

    setupUserMenu() {
        const btn = document.getElementById('userMenuBtn');
        const dropdown = document.getElementById('userDropdown');

        if (btn) {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                dropdown.classList.toggle('hidden');
            });
        }

        document.addEventListener('click', () => {
            if (dropdown) dropdown.classList.add('hidden');
        });
    },

    setTokens(access, refresh) {
        this.token = access;
        this.refreshToken = refresh;
        localStorage.setItem('access_token', access);
        localStorage.setItem('refresh_token', refresh);
    },

    async loadProfile() {
        try {
            const res = await fetch('/api/auth/me', {
                headers: { 'Authorization': `Bearer ${this.token}` },
            });

            if (res.status === 401) {
                const refreshed = await this.tryRefresh();
                if (refreshed) return this.loadProfile();
                this.clearAuth();
                this.showAuthModal();
                return;
            }

            if (!res.ok) throw new Error('Failed to load profile');

            this.user = await res.json();
            this.updateUI();
        } catch (e) {
            this.clearAuth();
            this.showAuthModal();
        }
    },

    async tryRefresh() {
        if (this._refreshing || !this.refreshToken) return false;
        this._refreshing = true;

        try {
            const res = await fetch('/api/auth/refresh', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ refresh_token: this.refreshToken }),
            });

            if (!res.ok) return false;

            const data = await res.json();
            this.setTokens(data.access_token, data.refresh_token);
            return true;
        } catch {
            return false;
        } finally {
            this._refreshing = false;
        }
    },

    updateUI() {
        const modal = document.getElementById('authModal');
        const userMenu = document.getElementById('userMenu');
        const userEmail = document.getElementById('userEmail');
        const userRole = document.getElementById('userRole');
        const adminLink = document.getElementById('adminLink');

        if (this.user) {
            if (modal) modal.classList.add('hidden');
            if (userMenu) userMenu.classList.remove('hidden');
            if (userEmail) userEmail.textContent = this.user.email;
            if (userRole) userRole.textContent = this.user.role;
            if (adminLink) {
                if (this.user.role === 'admin') {
                    adminLink.classList.remove('hidden');
                } else {
                    adminLink.classList.add('hidden');
                }
            }
        } else {
            if (userMenu) userMenu.classList.add('hidden');
        }
    },

    getAuthHeaders() {
        return this.token ? { 'Authorization': `Bearer ${this.token}` } : {};
    },

    isLoggedIn() {
        return !!this.token && !!this.user;
    },

    logout() {
        this.clearAuth();
        this.showAuthModal();
        if (typeof App !== 'undefined' && App.showToast) {
            App.showToast('Logged out');
        }
    },

    clearAuth() {
        this.token = null;
        this.refreshToken = null;
        this.user = null;
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        this.updateUI();
    },

    showAuthModal() {
        const modal = document.getElementById('authModal');
        if (modal) modal.classList.remove('hidden');
        this.showTab('login');
    },

    hideAuthModal() {
        const modal = document.getElementById('authModal');
        if (modal) modal.classList.add('hidden');
    },

    showTab(tab) {
        const loginForm = document.getElementById('loginForm');
        const registerForm = document.getElementById('registerForm');
        const loginTab = document.getElementById('authTabLogin');
        const registerTab = document.getElementById('authTabRegister');

        if (tab === 'login') {
            loginForm.classList.remove('hidden');
            registerForm.classList.add('hidden');
            loginTab.classList.add('active');
            registerTab.classList.remove('active');
        } else {
            loginForm.classList.add('hidden');
            registerForm.classList.remove('hidden');
            loginTab.classList.remove('active');
            registerTab.classList.add('active');
        }
    },

    async showApiKey() {
        document.getElementById('userDropdown').classList.add('hidden');

        try {
            const res = await API.request('/api/auth/api-key', { method: 'POST' });
            const key = res.api_key;

            // Show in a simple prompt-style overlay
            const overlay = document.createElement('div');
            overlay.className = 'fixed inset-0 bg-black/60 z-[100] flex items-center justify-center';
            overlay.innerHTML = `
                <div class="bg-dark-800 rounded-xl p-6 w-96 border border-gray-700">
                    <h3 class="text-white font-bold mb-3">Your API Key</h3>
                    <p class="text-xs text-gray-400 mb-2">Use this in the <code class="text-accent-blue">X-API-Key</code> header for programmatic access.</p>
                    <input type="text" value="${key}" readonly
                        class="w-full bg-dark-700 border border-gray-600 rounded px-3 py-2 text-sm text-white font-mono select-all"
                        onclick="this.select()">
                    <button onclick="this.closest('.fixed').remove()"
                        class="mt-3 w-full px-4 py-2 bg-accent-blue text-white rounded hover:bg-blue-600 text-sm">Close</button>
                </div>
            `;
            document.body.appendChild(overlay);
        } catch (e) {
            if (typeof App !== 'undefined' && App.showToast) {
                App.showToast('Failed to generate API key: ' + e.message, 'error');
            }
        }
    },

    async showChangePassword() {
        document.getElementById('userDropdown').classList.add('hidden');

        const overlay = document.createElement('div');
        overlay.className = 'fixed inset-0 bg-black/60 z-[100] flex items-center justify-center';
        overlay.id = 'changePasswordOverlay';
        overlay.innerHTML = `
            <div class="bg-dark-800 rounded-xl p-6 w-96 border border-gray-700">
                <h3 class="text-white font-bold mb-4">Change Password</h3>
                <form id="changePasswordForm" class="space-y-3">
                    <input type="password" id="cpCurrent" placeholder="Current password" required
                        class="w-full bg-dark-700 border border-gray-600 rounded px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-accent-blue">
                    <input type="password" id="cpNew" placeholder="New password (min 8 chars)" required minlength="8"
                        class="w-full bg-dark-700 border border-gray-600 rounded px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-accent-blue">
                    <div id="cpError" class="text-red-400 text-sm hidden"></div>
                    <div class="flex gap-2">
                        <button type="submit" class="flex-1 px-4 py-2 bg-accent-blue text-white rounded hover:bg-blue-600 text-sm">Update</button>
                        <button type="button" onclick="document.getElementById('changePasswordOverlay').remove()"
                            class="flex-1 px-4 py-2 bg-dark-600 text-gray-300 rounded hover:bg-dark-700 text-sm">Cancel</button>
                    </div>
                </form>
            </div>
        `;
        document.body.appendChild(overlay);

        document.getElementById('changePasswordForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const errEl = document.getElementById('cpError');
            errEl.classList.add('hidden');

            try {
                await API.request('/api/auth/password', {
                    method: 'PUT',
                    body: JSON.stringify({
                        current_password: document.getElementById('cpCurrent').value,
                        new_password: document.getElementById('cpNew').value,
                    }),
                });
                overlay.remove();
                if (typeof App !== 'undefined' && App.showToast) {
                    App.showToast('Password updated!');
                }
            } catch (e) {
                errEl.textContent = e.message;
                errEl.classList.remove('hidden');
            }
        });
    },
};
