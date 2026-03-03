const Search = {
    input: null,
    resultsDiv: null,
    debounceTimer: null,
    _abortController: null,

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

        document.addEventListener('click', (e) => {
            if (!this.input.contains(e.target) && !this.resultsDiv.contains(e.target)) {
                this.resultsDiv.classList.add('hidden');
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
        }
    },

    showResults(results) {
        if (results.length === 0) {
            this.resultsDiv.classList.add('hidden');
            return;
        }
        this.resultsDiv.innerHTML = results.map(r => {
            let badge = '';
            if (r.type === 'index') badge = '<span class="text-[10px] px-1.5 py-0.5 rounded bg-purple-900 text-purple-300 ml-2">INDEX</span>';
            else if (r.type === 'etf') badge = '<span class="text-[10px] px-1.5 py-0.5 rounded bg-teal-900 text-teal-300 ml-2">ETF</span>';
            return `
            <div class="px-4 py-2 hover:bg-dark-600 cursor-pointer flex justify-between items-center"
                 onclick="Search.select('${r.symbol}', '${r.name.replace(/'/g, "\\'")}')">
                <span class="text-white font-medium text-sm">${r.symbol}${badge}</span>
                <span class="text-gray-500 text-xs truncate ml-4">${r.name}</span>
            </div>`;
        }).join('');
        this.resultsDiv.classList.remove('hidden');
    },

    select(symbol, name) {
        this.input.value = symbol;
        this.resultsDiv.classList.add('hidden');
        App.loadStock(symbol, name);
    }
};
