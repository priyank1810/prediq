const Search = {
    input: null,
    resultsDiv: null,
    debounceTimer: null,

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
        try {
            const results = await API.searchStocks(query);
            this.showResults(results);
        } catch (e) {
            this.resultsDiv.classList.add('hidden');
        }
    },

    showResults(results) {
        if (results.length === 0) {
            this.resultsDiv.classList.add('hidden');
            return;
        }
        this.resultsDiv.innerHTML = results.map(r => {
            const badge = r.type === 'index'
                ? '<span class="text-[10px] px-1.5 py-0.5 rounded bg-purple-900 text-purple-300 ml-2">INDEX</span>'
                : '';
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
