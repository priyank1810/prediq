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

    async search() {
        const query = this.input.value.trim();

        // Cancel any in-flight search request
        if (this._abortController) {
            this._abortController.abort();
        }
        this._abortController = new AbortController();

        try {
            const results = await API.searchStocks(query, this._abortController.signal);
            this.showResults(results);
        } catch (e) {
            if (e.name === 'AbortError') return; // Superseded by newer search
            this.resultsDiv.classList.add('hidden');
            this.input.setAttribute('aria-expanded', 'false');
        }
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
