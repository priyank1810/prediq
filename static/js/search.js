const Search = {
    input: null,
    resultsDiv: null,
    debounceTimer: null,
    _abortController: null,
    _activeIndex: -1,

    init() {
        this.input = document.getElementById('searchInput');
        this.resultsDiv = document.getElementById('searchResults');

        this.input.addEventListener('input', () => {
            clearTimeout(this.debounceTimer);
            this.debounceTimer = setTimeout(() => this.search(), 300);
        });

        this.input.addEventListener('focus', () => {
            if (this.input.value.length === 0) this.search();
        });

        // Keyboard navigation for search dropdown
        this.input.addEventListener('keydown', (e) => {
            const items = this.resultsDiv.querySelectorAll('[role="option"]');
            if (!items.length || this.resultsDiv.classList.contains('hidden')) {
                if (e.key === 'Escape') {
                    this.resultsDiv.classList.add('hidden');
                    this.input.setAttribute('aria-expanded', 'false');
                }
                return;
            }

            if (e.key === 'ArrowDown') {
                e.preventDefault();
                this._activeIndex = Math.min(this._activeIndex + 1, items.length - 1);
                this._highlightItem(items);
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                this._activeIndex = Math.max(this._activeIndex - 1, 0);
                this._highlightItem(items);
            } else if (e.key === 'Enter') {
                e.preventDefault();
                if (this._activeIndex >= 0 && items[this._activeIndex]) {
                    items[this._activeIndex].click();
                }
            } else if (e.key === 'Escape') {
                this.resultsDiv.classList.add('hidden');
                this.input.setAttribute('aria-expanded', 'false');
                this._activeIndex = -1;
                this.input.setAttribute('aria-activedescendant', '');
            }
        });

        document.addEventListener('click', (e) => {
            if (!this.input.contains(e.target) && !this.resultsDiv.contains(e.target)) {
                this.resultsDiv.classList.add('hidden');
                this.input.setAttribute('aria-expanded', 'false');
            }
        });
    },

    _highlightItem(items) {
        items.forEach((item, i) => {
            if (i === this._activeIndex) {
                item.classList.add('bg-dark-600');
                item.setAttribute('aria-selected', 'true');
                this.input.setAttribute('aria-activedescendant', item.id);
                item.scrollIntoView({ block: 'nearest' });
            } else {
                item.classList.remove('bg-dark-600');
                item.setAttribute('aria-selected', 'false');
            }
        });
    },

    _isNLQuery(query) {
        if (!query || query.length < 4) return false;
        const nlKeywords = /\b(oversold|overbought|bullish|bearish|above|below|sma|top|gainer|loser|undervalued|growth|dividend|momentum|volume|spike|strong|weak|52\s*week|cheap|trending|breakout|banking|pharma|auto|metal|energy|fmcg|it\s|tech)\b/i;
        return nlKeywords.test(query) || (query.includes(' ') && query.split(' ').length >= 2 && !/^[A-Z0-9& ]+$/.test(query));
    },

    async search() {
        const query = this.input.value.trim();

        // Cancel any in-flight search request
        if (this._abortController) {
            this._abortController.abort();
        }
        this._abortController = new AbortController();

        try {
            // Detect natural language queries
            if (this._isNLQuery(query)) {
                const resp = await fetch(`${API.baseUrl}/api/stocks/ai/search?q=${encodeURIComponent(query)}`, {
                    signal: this._abortController.signal
                });
                if (resp.ok) {
                    const data = await resp.json();
                    this.showNLResults(data);
                    return;
                }
            }

            const results = await API.searchStocks(query, this._abortController.signal);
            this.showResults(results);
        } catch (e) {
            if (e.name === 'AbortError') return; // Superseded by newer search
            this.resultsDiv.classList.add('hidden');
            this.input.setAttribute('aria-expanded', 'false');
        }
    },

    showNLResults(data) {
        this._activeIndex = -1;
        this.input.setAttribute('aria-activedescendant', '');

        if (!data.results || data.results.length === 0) {
            const suggestions = data.suggestions || [];
            this.resultsDiv.innerHTML = `
                <div class="px-4 py-2 text-xs text-gray-400 border-b border-gray-700">
                    <span class="text-purple-400">AI Search:</span> ${data.interpreted_as || 'No results'}
                </div>
                ${suggestions.map(s => `<div class="px-4 py-1.5 text-xs text-gray-500 cursor-pointer hover:bg-dark-600" onclick="Search.input.value='${s.replace("Try: '","").replace("'","")}';Search.search()">${s}</div>`).join('')}
            `;
            this.resultsDiv.classList.remove('hidden');
            this.input.setAttribute('aria-expanded', 'true');
            return;
        }

        let html = `<div class="px-4 py-1.5 text-[10px] text-purple-400 border-b border-gray-700 bg-dark-800">
            AI: ${data.interpreted_as} (${data.result_count} found)
        </div>`;

        html += data.results.slice(0, 12).map((r, i) => {
            const changePct = r.change_pct != null ? r.change_pct.toFixed(1) : '0.0';
            const changeColor = r.change_pct > 0 ? 'text-green-400' : (r.change_pct < 0 ? 'text-red-400' : 'text-gray-400');
            const filters = (r.matched_filters || []).join(', ');
            return `
            <div id="search-option-${i}" role="option" aria-selected="false"
                 class="px-4 py-2 hover:bg-dark-600 cursor-pointer border-b border-gray-800"
                 onclick="Search.select('${r.symbol}', '')">
                <div class="flex justify-between items-center">
                    <span class="text-white font-medium text-sm">${r.symbol}</span>
                    <div class="flex items-center gap-2">
                        <span class="text-xs text-gray-400">${r.ltp ? '₹' + r.ltp.toFixed(2) : ''}</span>
                        <span class="text-xs ${changeColor}">${changePct}%</span>
                    </div>
                </div>
                ${filters ? `<div class="text-[10px] text-gray-500 mt-0.5">${filters}</div>` : ''}
            </div>`;
        }).join('');

        this.resultsDiv.innerHTML = html;
        this.resultsDiv.classList.remove('hidden');
        this.input.setAttribute('aria-expanded', 'true');
    },

    showResults(results) {
        this._activeIndex = -1;
        this.input.setAttribute('aria-activedescendant', '');
        if (results.length === 0) {
            this.resultsDiv.classList.add('hidden');
            this.input.setAttribute('aria-expanded', 'false');
            return;
        }
        this.resultsDiv.innerHTML = results.map((r, i) => {
            let badge = '';
            if (r.type === 'index') badge = '<span class="text-[10px] px-1.5 py-0.5 rounded bg-purple-900 text-purple-300 ml-2">INDEX</span>';
            else if (r.type === 'etf') badge = '<span class="text-[10px] px-1.5 py-0.5 rounded bg-teal-900 text-teal-300 ml-2">ETF</span>';
            return `
            <div id="search-option-${i}" role="option" aria-selected="false"
                 class="px-4 py-2 hover:bg-dark-600 cursor-pointer flex justify-between items-center"
                 onclick="Search.select('${r.symbol}', '${r.name.replace(/'/g, "\\'")}')">
                <span class="text-white font-medium text-sm">${r.symbol}${badge}</span>
                <span class="text-gray-500 text-xs truncate ml-4">${r.name}</span>
            </div>`;
        }).join('');
        this.resultsDiv.classList.remove('hidden');
        this.input.setAttribute('aria-expanded', 'true');
    },

    select(symbol, name) {
        this.input.value = symbol;
        this.resultsDiv.classList.add('hidden');
        this.input.setAttribute('aria-expanded', 'false');
        this.input.setAttribute('aria-activedescendant', '');
        this._activeIndex = -1;
        App.loadStock(symbol, name);
    }
};
