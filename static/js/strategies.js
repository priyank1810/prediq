const Strategies = {
    _strategies: [],
    _leaderboard: [],
    _sort: 'newest',

    init() {
        const form = document.getElementById('shareStrategyForm');
        if (form) {
            form.addEventListener('submit', (e) => {
                e.preventDefault();
                this.shareStrategy();
            });
        }
        const sortSelect = document.getElementById('strategySortSelect');
        if (sortSelect) {
            sortSelect.addEventListener('change', () => {
                this._sort = sortSelect.value;
                this.loadStrategies();
            });
        }
    },

    async load() {
        try {
            const [leaderboard, strategies] = await Promise.all([
                API.getStrategyLeaderboard().catch(() => []),
                API.getStrategies(this._sort).catch(() => []),
            ]);
            this._leaderboard = leaderboard;
            this._strategies = strategies;
            this.renderLeaderboard(leaderboard);
            this.renderStrategies(strategies);
        } catch (e) {
            console.error('Strategies load failed:', e);
        }
    },

    async loadStrategies() {
        try {
            const strategies = await API.getStrategies(this._sort);
            this._strategies = strategies;
            this.renderStrategies(strategies);
        } catch (e) {
            console.error('Strategies load failed:', e);
        }
    },

    renderLeaderboard(strategies) {
        const el = document.getElementById('strategyLeaderboard');
        if (!el) return;

        if (!strategies || strategies.length === 0) {
            el.innerHTML = '<div class="text-center py-4 text-gray-500 text-sm">No strategies with 10+ trades yet. Share yours to get on the leaderboard!</div>';
            return;
        }

        const rankBadges = ['text-yellow-400', 'text-gray-300', 'text-amber-600'];
        const rows = strategies.slice(0, 10).map((s, i) => {
            const rankColor = i < 3 ? rankBadges[i] : 'text-gray-500';
            const winColor = (s.win_rate || 0) >= 60 ? 'text-green-400' : (s.win_rate || 0) >= 45 ? 'text-yellow-400' : 'text-red-400';
            const retColor = (s.avg_return_pct || 0) >= 0 ? 'text-green-400' : 'text-red-400';
            return '<div class="flex items-center gap-3 py-2 px-3 bg-dark-700 rounded hover:bg-dark-600 transition">'
                + '<span class="' + rankColor + ' font-bold text-sm w-6 text-center">#' + (i + 1) + '</span>'
                + '<div class="flex-1 min-w-0">'
                + '<div class="text-white text-sm font-medium truncate">' + this._esc(s.name) + '</div>'
                + '<div class="text-gray-500 text-[10px] truncate">' + this._esc(s.symbols) + '</div>'
                + '</div>'
                + '<div class="text-center px-2"><div class="text-[10px] text-gray-500">Win Rate</div><div class="' + winColor + ' text-xs font-bold">' + (s.win_rate != null ? s.win_rate.toFixed(1) + '%' : '-') + '</div></div>'
                + '<div class="text-center px-2"><div class="text-[10px] text-gray-500">Avg Ret</div><div class="' + retColor + ' text-xs font-bold">' + (s.avg_return_pct != null ? (s.avg_return_pct >= 0 ? '+' : '') + s.avg_return_pct.toFixed(1) + '%' : '-') + '</div></div>'
                + '<div class="text-center px-2"><div class="text-[10px] text-gray-500">Sharpe</div><div class="text-white text-xs font-bold">' + (s.sharpe_ratio != null ? s.sharpe_ratio.toFixed(2) : '-') + '</div></div>'
                + '<div class="text-center px-2"><div class="text-[10px] text-gray-500">Followers</div><div class="text-accent-blue text-xs font-bold">' + (s.follower_count || 0) + '</div></div>'
                + '</div>';
        }).join('');

        el.innerHTML = rows;
    },

    renderStrategies(strategies) {
        const el = document.getElementById('strategyList');
        if (!el) return;

        if (!strategies || strategies.length === 0) {
            el.innerHTML = '<div class="text-center py-4 text-gray-500 text-sm col-span-full">No public strategies yet. Be the first to share!</div>';
            return;
        }

        const cards = strategies.map((s) => {
            const winColor = (s.win_rate || 0) >= 60 ? 'text-green-400' : (s.win_rate || 0) >= 45 ? 'text-yellow-400' : 'text-red-400';
            const symbols = (s.symbols || '').split(',').slice(0, 3).map(sym =>
                '<span class="bg-dark-600 text-gray-300 text-[10px] px-1.5 py-0.5 rounded">' + this._esc(sym.trim()) + '</span>'
            ).join(' ');
            const extraCount = (s.symbols || '').split(',').length - 3;
            const symbolsExtra = extraCount > 0 ? ' <span class="text-gray-500 text-[10px]">+' + extraCount + '</span>' : '';

            return '<div class="bg-dark-700 rounded-lg p-4 flex flex-col gap-2">'
                + '<div class="flex items-start justify-between gap-2">'
                + '<h4 class="text-white text-sm font-semibold truncate flex-1">' + this._esc(s.name) + '</h4>'
                + '<span class="bg-dark-600 text-gray-400 text-[10px] px-1.5 py-0.5 rounded shrink-0">' + this._esc(s.timeframe) + '</span>'
                + '</div>'
                + (s.description ? '<p class="text-gray-400 text-xs line-clamp-2">' + this._esc(s.description) + '</p>' : '')
                + '<div class="flex flex-wrap gap-1">' + symbols + symbolsExtra + '</div>'
                + '<div class="flex gap-3 text-[10px] mt-1">'
                + '<span class="text-gray-500">Trades: <span class="text-white">' + (s.total_trades || 0) + '</span></span>'
                + '<span class="text-gray-500">Win: <span class="' + winColor + '">' + (s.win_rate != null ? s.win_rate.toFixed(1) + '%' : '-') + '</span></span>'
                + '<span class="text-gray-500">Upvotes: <span class="text-white">' + (s.upvotes || 0) + '</span></span>'
                + '<span class="text-gray-500">Followers: <span class="text-accent-blue">' + (s.follower_count || 0) + '</span></span>'
                + '</div>'
                + '<div class="flex gap-2 mt-auto pt-2">'
                + '<button onclick="Strategies.upvote(' + s.id + ', this)" class="flex-1 px-2 py-1 bg-dark-600 hover:bg-dark-800 text-gray-300 text-xs rounded transition flex items-center justify-center gap-1">'
                + '<svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 15l7-7 7 7"/></svg>'
                + 'Upvote'
                + '</button>'
                + '<button onclick="Strategies.follow(' + s.id + ', this)" class="flex-1 px-2 py-1 bg-accent-blue hover:bg-blue-600 text-white text-xs rounded transition">'
                + 'Follow'
                + '</button>'
                + '</div>'
                + '</div>';
        }).join('');

        el.innerHTML = cards;
    },

    async shareStrategy() {
        const name = document.getElementById('stratName').value.trim();
        const description = document.getElementById('stratDescription').value.trim();
        const symbols = document.getElementById('stratSymbols').value.trim().toUpperCase();
        const timeframe = document.getElementById('stratTimeframe').value;
        const entryRules = document.getElementById('stratEntryRules').value.trim();
        const exitRules = document.getElementById('stratExitRules').value.trim();
        const isPublic = document.getElementById('stratPublic').checked;

        if (!name || !symbols || !entryRules || !exitRules) {
            if (typeof App !== 'undefined') App.showToast('Please fill in name, symbols, entry rules and exit rules.', 'error');
            return;
        }

        try {
            await API.createStrategy({
                name: name,
                description: description || null,
                symbols: symbols,
                timeframe: timeframe,
                entry_rules: entryRules,
                exit_rules: exitRules,
                is_public: isPublic,
            });
            if (typeof App !== 'undefined') App.showToast('Strategy shared successfully!', 'success');
            document.getElementById('shareStrategyForm').reset();
            document.getElementById('stratPublic').checked = true;
            this.load();
        } catch (e) {
            if (typeof App !== 'undefined') App.showToast('Failed to share strategy: ' + e.message, 'error');
        }
    },

    async upvote(id, btn) {
        try {
            const res = await API.upvoteStrategy(id);
            if (btn) {
                const parent = btn.closest('.bg-dark-700');
                if (parent) {
                    const upvoteSpan = parent.querySelector('.text-gray-500 span.text-white');
                    if (upvoteSpan) upvoteSpan.textContent = res.upvotes;
                }
            }
            if (typeof App !== 'undefined') App.showToast('Upvoted!', 'success');
        } catch (e) {
            if (typeof App !== 'undefined') App.showToast('Failed to upvote: ' + e.message, 'error');
        }
    },

    async follow(id, btn) {
        try {
            const res = await API.followStrategy(id);
            const msg = res.followed ? 'Following strategy!' : 'Unfollowed strategy.';
            if (btn) {
                btn.textContent = res.followed ? 'Unfollow' : 'Follow';
                btn.classList.toggle('bg-accent-blue', !res.followed);
                btn.classList.toggle('bg-red-600', res.followed);
            }
            if (typeof App !== 'undefined') App.showToast(msg, 'success');
        } catch (e) {
            if (typeof App !== 'undefined') App.showToast('Failed: ' + e.message, 'error');
        }
    },

    _esc(str) {
        if (!str) return '';
        const d = document.createElement('div');
        d.textContent = str;
        return d.innerHTML;
    },
};
