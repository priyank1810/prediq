/**
 * Lazy module loader — loads JS files on demand and caches them.
 * Usage: await Lazy.load('predictions');
 */
const Lazy = {
    _loaded: {},
    _loading: {},

    // Module registry: name → {src, init}
    _modules: {
        predictions:  { src: '/static/js/predictions.js?v=40' },
        watchlist:    { src: '/static/js/watchlist.js?v=40' },
        insights:     { src: '/static/js/insights.js?v=40' },
        fundamentals: { src: '/static/js/fundamentals.js?v=40' },
        signals:      { src: '/static/js/signals.js?v=40' },
    },

    /**
     * Load a module by name. Returns a promise that resolves when the script
     * is loaded and executed. Safe to call multiple times — only loads once.
     */
    load(name) {
        if (this._loaded[name]) return Promise.resolve();
        if (this._loading[name]) return this._loading[name];

        const mod = this._modules[name];
        if (!mod) {
            console.warn(`Lazy: unknown module "${name}"`);
            return Promise.resolve();
        }

        this._loading[name] = new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = mod.src;
            script.onload = () => {
                this._loaded[name] = true;
                delete this._loading[name];
                resolve();
            };
            script.onerror = () => {
                delete this._loading[name];
                reject(new Error(`Failed to load module: ${name}`));
            };
            document.head.appendChild(script);
        });

        return this._loading[name];
    },

    // Map module names to their global variable names
    _globalNames: {
        predictions: 'Predictions', watchlist: 'Watchlist', insights: 'Insights',
        fundamentals: 'Fundamentals', signals: 'Signals',
    },

    /**
     * Get the global object for a module.
     * Uses direct variable lookup since const/let don't create window properties.
     */
    _getGlobal(name) {
        const globalName = this._globalNames[name] || (name.charAt(0).toUpperCase() + name.slice(1));
        // Try window first (var/function declarations), then eval for const/let
        if (window[globalName]) return window[globalName];
        try { return (0, eval)(globalName); } catch (e) { return null; }
    },

    /**
     * Load a module and call its init() if it exists.
     */
    async loadAndInit(name) {
        await this.load(name);
        const obj = this._getGlobal(name);
        if (obj && typeof obj.init === 'function' && !obj._lazyInited) {
            obj.init();
            obj._lazyInited = true;
        }
    },

    /**
     * Check if a module is already loaded.
     */
    isLoaded(name) {
        return !!this._loaded[name];
    },
};
