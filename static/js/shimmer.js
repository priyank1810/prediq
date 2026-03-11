/**
 * Shimmer / Skeleton loading placeholders.
 * Call Shimmer.show(containerId, type) before async loads.
 * The shimmer is automatically replaced when real content is set via innerHTML.
 */
const Shimmer = {
    // Reusable shimmer line
    _line(w = 'w-full', h = 'h-3') {
        return `<div class="shimmer ${w} ${h} rounded"></div>`;
    },

    // Reusable shimmer block
    _block(w, h) {
        return `<div class="shimmer rounded" style="width:${w};height:${h}"></div>`;
    },

    /**
     * Show shimmer in a container.
     * @param {string} id - Element ID
     * @param {string} type - One of: ticker, grid, movers, card, signal, prediction, fundamentals, news, alerts, heatmap, mood, fiidii, mtfSignals
     * @param {number} count - Number of placeholder items
     */
    show(id, type, count) {
        const el = document.getElementById(id);
        if (!el) return;

        const generators = {
            ticker: (n) => Array(n).fill(0).map(() =>
                `<div class="flex-shrink-0 shimmer-card min-w-[130px] sm:min-w-[160px]">
                    <div class="shimmer h-3 w-16 rounded mb-2"></div>
                    <div class="shimmer h-4 w-20 rounded mb-1"></div>
                    <div class="shimmer h-3 w-12 rounded"></div>
                </div>`
            ).join(''),

            grid: (n) => Array(n).fill(0).map(() =>
                `<div class="shimmer-card">
                    <div class="flex items-center justify-between mb-1">
                        <div class="shimmer h-3 w-16 rounded"></div>
                        <div class="shimmer h-3 w-4 rounded"></div>
                    </div>
                    <div class="flex items-center justify-between mt-1">
                        <div class="shimmer h-3 w-14 rounded"></div>
                        <div class="shimmer h-3 w-10 rounded"></div>
                    </div>
                </div>`
            ).join(''),

            movers: (n) => Array(n).fill(0).map(() =>
                `<div class="shimmer-card flex items-center gap-3">
                    <div class="shimmer h-4 w-4 rounded-full"></div>
                    <div class="flex-1">
                        <div class="shimmer h-3 w-20 rounded mb-1"></div>
                        <div class="shimmer h-2.5 w-14 rounded"></div>
                    </div>
                    <div class="text-right">
                        <div class="shimmer h-3 w-12 rounded mb-1"></div>
                        <div class="shimmer h-2.5 w-10 rounded"></div>
                    </div>
                </div>`
            ).join(''),

            alerts: (n) => Array(n).fill(0).map(() =>
                `<div class="shimmer-card flex items-center gap-4">
                    <div class="shimmer h-6 w-6 rounded"></div>
                    <div class="flex-1">
                        <div class="shimmer h-3.5 w-24 rounded mb-1"></div>
                        <div class="shimmer h-2.5 w-16 rounded"></div>
                    </div>
                    <div class="text-right">
                        <div class="shimmer h-4 w-12 rounded mb-1"></div>
                        <div class="shimmer h-2.5 w-16 rounded"></div>
                    </div>
                </div>`
            ).join(''),

            heatmap: (n) => Array(n).fill(0).map(() =>
                `<div class="shimmer rounded h-10"></div>`
            ).join(''),

            mood: () =>
                `<div class="flex flex-col items-center gap-2">
                    <div class="shimmer h-10 w-16 rounded"></div>
                    <div class="shimmer h-3 w-20 rounded"></div>
                    <div class="shimmer h-2 w-full rounded mt-1"></div>
                </div>`,

            fiidii: () =>
                `<div class="space-y-2">
                    <div class="flex justify-between"><div class="shimmer h-3 w-8 rounded"></div><div class="shimmer h-3 w-16 rounded"></div></div>
                    <div class="flex justify-between"><div class="shimmer h-3 w-8 rounded"></div><div class="shimmer h-3 w-16 rounded"></div></div>
                    <div class="flex justify-between"><div class="shimmer h-3 w-10 rounded"></div><div class="shimmer h-3 w-16 rounded"></div></div>
                </div>`,

            mtfSignals: (n) => `<div class="grid grid-cols-1 sm:grid-cols-3 gap-3">${Array(n).fill(0).map(() =>
                `<div class="shimmer-card">
                    <div class="flex items-center justify-between mb-3">
                        <div class="shimmer h-3 w-20 rounded"></div>
                        <div class="shimmer h-3 w-16 rounded"></div>
                    </div>
                    <div class="flex flex-col items-center gap-1 mb-3">
                        <div class="shimmer h-8 w-14 rounded"></div>
                        <div class="shimmer h-2.5 w-16 rounded"></div>
                    </div>
                    <div class="border-t border-gray-800 pt-2 space-y-1.5">
                        <div class="flex justify-between"><div class="shimmer h-2.5 w-10 rounded"></div><div class="shimmer h-2.5 w-16 rounded"></div></div>
                        <div class="flex justify-between"><div class="shimmer h-2.5 w-10 rounded"></div><div class="shimmer h-2.5 w-16 rounded"></div></div>
                        <div class="flex justify-between"><div class="shimmer h-2.5 w-10 rounded"></div><div class="shimmer h-2.5 w-16 rounded"></div></div>
                    </div>
                </div>`
            ).join('')}</div>`,

            signal: () =>
                `<div class="space-y-3">
                    <div class="shimmer-card">
                        <div class="flex justify-between mb-2"><div class="shimmer h-3 w-20 rounded"></div><div class="shimmer h-3 w-24 rounded"></div></div>
                        <div class="shimmer h-2 w-full rounded mb-3"></div>
                        <div class="shimmer h-2 w-full rounded mb-3"></div>
                    </div>
                    <div class="shimmer-card">
                        <div class="shimmer h-3 w-24 rounded mb-2"></div>
                        <div class="shimmer h-2 w-full rounded mb-1"></div>
                        <div class="shimmer h-2 w-3/4 rounded"></div>
                    </div>
                </div>`,

            prediction: () =>
                `<div class="space-y-3">
                    <div class="shimmer-card text-center">
                        <div class="shimmer h-8 w-20 rounded mx-auto mb-2"></div>
                        <div class="shimmer h-3 w-32 rounded mx-auto mb-1"></div>
                        <div class="shimmer h-2.5 w-24 rounded mx-auto"></div>
                    </div>
                    <div class="shimmer-card">
                        <div class="shimmer h-3 w-28 rounded mb-2"></div>
                        <div class="grid grid-cols-3 gap-2">
                            <div class="shimmer h-10 rounded"></div>
                            <div class="shimmer h-10 rounded"></div>
                            <div class="shimmer h-10 rounded"></div>
                        </div>
                    </div>
                </div>`,

            fundamentals: () =>
                `<div class="space-y-3">
                    <div class="grid grid-cols-2 sm:grid-cols-4 gap-2">
                        ${Array(8).fill(0).map(() => `<div class="shimmer-card"><div class="shimmer h-2.5 w-12 rounded mb-1.5"></div><div class="shimmer h-4 w-16 rounded"></div></div>`).join('')}
                    </div>
                    <div class="shimmer-card">
                        <div class="shimmer h-3 w-32 rounded mb-2"></div>
                        <div class="shimmer h-2 w-full rounded mb-1"></div>
                        <div class="shimmer h-2 w-3/4 rounded"></div>
                    </div>
                </div>`,

            news: (n) => Array(n).fill(0).map(() =>
                `<div class="shimmer-card flex gap-3">
                    <div class="flex-1">
                        <div class="shimmer h-3 w-full rounded mb-1.5"></div>
                        <div class="shimmer h-3 w-3/4 rounded mb-1.5"></div>
                        <div class="shimmer h-2.5 w-20 rounded"></div>
                    </div>
                </div>`
            ).join(''),

            keyMetrics: () =>
                `<div class="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-3">
                    ${Array(4).fill(0).map(() => `<div class="shimmer-card"><div class="shimmer h-2.5 w-14 rounded mb-1.5"></div><div class="shimmer h-4 w-20 rounded"></div></div>`).join('')}
                </div>`,
        };

        const gen = generators[type];
        if (gen) {
            el.innerHTML = typeof gen === 'function' ? gen(count || 3) : gen;
        }
    },

    /**
     * Show shimmer for all market overview sections at once.
     */
    showMarketOverview() {
        this.show('indicesTicker', 'ticker', 6);
        this.show('indicesGrid', 'grid', 6);
        this.show('stocksGrid', 'grid', 10);
        this.show('gainersGrid', 'movers', 5);
        this.show('losersGrid', 'movers', 5);
        this.show('smartAlertsList', 'alerts', 3);
        this.show('sectorHeatmap', 'heatmap', 9);
    },

    /**
     * Show shimmer for stock detail sections.
     */
    showStockDetail() {
        this.show('mtfSignalsGrid', 'mtfSignals', 3);
        this.show('overviewKeyMetrics', 'keyMetrics');
    },
};
