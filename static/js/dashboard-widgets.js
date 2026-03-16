/**
 * Dashboard Widgets — drag-and-drop rearrangeable dashboard panels
 * Uses native HTML5 Drag and Drop API (no external libraries)
 */
const DashboardWidgets = {
    _widgets: {
        'market-mood':    { id: 'market-mood',    title: 'Market Mood',           defaultOrder: 0 },
        'fii-dii':        { id: 'fii-dii',        title: 'FII/DII Flows',         defaultOrder: 1 },
        'sector-heatmap': { id: 'sector-heatmap', title: 'Sector Heatmap',        defaultOrder: 2 },
        'indices':        { id: 'indices',         title: 'Indices Ticker & Grid', defaultOrder: 3 },
        'stocks-grid':    { id: 'stocks-grid',     title: 'Stock Quotes Grid',     defaultOrder: 4 },
        'market-movers':  { id: 'market-movers',   title: 'Top Gainers / Losers',  defaultOrder: 5 },
        'earnings':       { id: 'earnings',        title: 'Earnings Calendar',     defaultOrder: 6 },
        'smart-alerts':   { id: 'smart-alerts',    title: 'AI Signals',            defaultOrder: 7 }
    },

    isEditMode: false,
    _draggedEl: null,
    _dropIndicator: null,
    _container: null,
    _initialized: false,

    init() {
        if (this._initialized) return;
        this._container = document.getElementById('marketOverview');
        if (!this._container) return;
        this._initialized = true;

        // Create a drop indicator element
        this._dropIndicator = document.createElement('div');
        this._dropIndicator.className = 'widget-drop-indicator';
        this._dropIndicator.style.display = 'none';

        this._wrapWidgets();

        var layout = this._loadLayout();
        if (layout) {
            this._applyLayout(layout);
        }

        this._setupDragDrop();
        this._setupButtons();
    },

    _loadLayout() {
        try {
            var raw = localStorage.getItem('dashboard_layout');
            if (!raw) return null;
            var parsed = JSON.parse(raw);
            if (parsed && Array.isArray(parsed.order)) return parsed;
        } catch (e) { /* corrupt data */ }
        return null;
    },

    _saveLayout() {
        var widgets = this._container.querySelectorAll('.dashboard-widget');
        var order = [];
        var hidden = {};
        widgets.forEach(function(w) {
            var wid = w.getAttribute('data-widget-id');
            order.push(wid);
            if (w.classList.contains('widget-hidden')) {
                hidden[wid] = true;
            }
        });
        localStorage.setItem('dashboard_layout', JSON.stringify({ order: order, hidden: hidden }));
    },

    _wrapWidgets() {
        var container = this._container;
        // Only select direct-child widget sections to avoid nesting issues
        var sections = container.querySelectorAll(':scope > [data-widget-id]');
        var self = this;

        sections.forEach(function(section) {
            var widgetId = section.getAttribute('data-widget-id');
            var meta = self._widgets[widgetId];
            if (!meta) return;

            // Add widget class and draggable attr directly on the section
            section.classList.add('dashboard-widget');
            section.setAttribute('draggable', 'false'); // only draggable in edit mode

            // Build drag handle
            var handle = document.createElement('button');
            handle.className = 'widget-drag-handle';
            handle.setAttribute('aria-label', 'Drag to reorder ' + meta.title);
            handle.setAttribute('title', 'Drag to reorder');
            handle.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><circle cx="8" cy="4" r="2"/><circle cx="16" cy="4" r="2"/><circle cx="8" cy="12" r="2"/><circle cx="16" cy="12" r="2"/><circle cx="8" cy="20" r="2"/><circle cx="16" cy="20" r="2"/></svg>';

            // Build visibility toggle
            var toggle = document.createElement('button');
            toggle.className = 'widget-visibility-toggle';
            toggle.setAttribute('aria-label', 'Toggle visibility of ' + meta.title);
            toggle.setAttribute('title', 'Toggle visibility');
            toggle.innerHTML = '<svg class="eye-open" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>' +
                '<svg class="eye-closed" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="display:none"><path d="M17.94 17.94A10.07 10.07 0 0112 20c-7 0-11-8-11-8a18.45 18.45 0 015.06-5.94M9.9 4.24A9.12 9.12 0 0112 4c7 0 11 8 11 8a18.5 18.5 0 01-2.16 3.19m-6.72-1.07a3 3 0 11-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>';

            toggle.addEventListener('click', function(e) {
                e.stopPropagation();
                self.toggleWidget(widgetId);
            });

            // Widget title label (shown when hidden)
            var label = document.createElement('div');
            label.className = 'widget-hidden-label';
            label.textContent = meta.title + ' (Hidden)';

            // Prepend controls into the section
            section.insertBefore(label, section.firstChild);
            section.insertBefore(toggle, section.firstChild);
            section.insertBefore(handle, section.firstChild);
        });
    },

    _setupDragDrop() {
        var self = this;
        var container = this._container;

        container.addEventListener('dragstart', function(e) {
            var widget = e.target.closest('.dashboard-widget');
            if (!widget || !self.isEditMode) { e.preventDefault(); return; }
            self._draggedEl = widget;
            widget.classList.add('dragging');
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/plain', widget.getAttribute('data-widget-id'));
        });

        container.addEventListener('dragover', function(e) {
            e.preventDefault();
            if (!self._draggedEl) return;
            e.dataTransfer.dropEffect = 'move';

            var target = self._getDropTarget(e);
            if (target && target !== self._draggedEl) {
                var rect = target.getBoundingClientRect();
                var midY = rect.top + rect.height / 2;
                var indicator = self._dropIndicator;

                if (indicator.parentNode !== container) {
                    container.appendChild(indicator);
                }
                indicator.style.display = 'block';

                if (e.clientY < midY) {
                    container.insertBefore(indicator, target);
                } else {
                    container.insertBefore(indicator, target.nextSibling);
                }
            }
        });

        container.addEventListener('dragleave', function(e) {
            if (!container.contains(e.relatedTarget)) {
                self._dropIndicator.style.display = 'none';
            }
        });

        container.addEventListener('drop', function(e) {
            e.preventDefault();
            if (!self._draggedEl) return;

            var indicator = self._dropIndicator;
            if (indicator.parentNode === container && indicator.style.display !== 'none') {
                container.insertBefore(self._draggedEl, indicator);
            }
            indicator.style.display = 'none';
            self._draggedEl.classList.remove('dragging');
            self._draggedEl = null;
            self._saveLayout();
        });

        container.addEventListener('dragend', function() {
            if (self._draggedEl) {
                self._draggedEl.classList.remove('dragging');
                self._draggedEl = null;
            }
            self._dropIndicator.style.display = 'none';
        });
    },

    _getDropTarget(e) {
        var widgets = this._container.querySelectorAll('.dashboard-widget');
        var closest = null;
        var closestDist = Infinity;
        widgets.forEach(function(w) {
            if (w.classList.contains('dragging')) return;
            var rect = w.getBoundingClientRect();
            var centerY = rect.top + rect.height / 2;
            var dist = Math.abs(e.clientY - centerY);
            if (dist < closestDist) {
                closestDist = dist;
                closest = w;
            }
        });
        return closest;
    },

    _applyLayout(layout) {
        var container = this._container;
        var order = layout.order || [];
        var hidden = layout.hidden || {};

        // Reorder DOM elements
        var widgetMap = {};
        container.querySelectorAll('.dashboard-widget').forEach(function(w) {
            widgetMap[w.getAttribute('data-widget-id')] = w;
        });

        // Append in saved order (after the controls div which stays first)
        order.forEach(function(wid) {
            var el = widgetMap[wid];
            if (el) {
                container.appendChild(el);
                if (hidden[wid]) {
                    el.classList.add('widget-hidden');
                    var eyeOpen = el.querySelector('.eye-open');
                    var eyeClosed = el.querySelector('.eye-closed');
                    if (eyeOpen) eyeOpen.style.display = 'none';
                    if (eyeClosed) eyeClosed.style.display = '';
                }
            }
        });

        // Append any widgets not in saved layout (new ones added later)
        container.querySelectorAll('.dashboard-widget').forEach(function(w) {
            var wid = w.getAttribute('data-widget-id');
            if (order.indexOf(wid) === -1) {
                container.appendChild(w);
            }
        });
    },

    toggleWidget(widgetId) {
        var widget = this._container.querySelector('.dashboard-widget[data-widget-id="' + widgetId + '"]');
        if (!widget) return;

        var isHidden = widget.classList.toggle('widget-hidden');
        var eyeOpen = widget.querySelector('.eye-open');
        var eyeClosed = widget.querySelector('.eye-closed');
        if (isHidden) {
            if (eyeOpen) eyeOpen.style.display = 'none';
            if (eyeClosed) eyeClosed.style.display = '';
        } else {
            if (eyeOpen) eyeOpen.style.display = '';
            if (eyeClosed) eyeClosed.style.display = 'none';
        }
        this._saveLayout();
    },

    resetLayout() {
        localStorage.removeItem('dashboard_layout');
        // Remove hidden class and restore eye icons
        this._container.querySelectorAll('.dashboard-widget').forEach(function(w) {
            w.classList.remove('widget-hidden');
            var eyeOpen = w.querySelector('.eye-open');
            var eyeClosed = w.querySelector('.eye-closed');
            if (eyeOpen) eyeOpen.style.display = '';
            if (eyeClosed) eyeClosed.style.display = 'none';
        });

        // Sort by default order
        var self = this;
        var widgets = Array.from(this._container.querySelectorAll('.dashboard-widget'));
        widgets.sort(function(a, b) {
            var aId = a.getAttribute('data-widget-id');
            var bId = b.getAttribute('data-widget-id');
            var aOrder = (self._widgets[aId] || {}).defaultOrder || 0;
            var bOrder = (self._widgets[bId] || {}).defaultOrder || 0;
            return aOrder - bOrder;
        });
        widgets.forEach(function(w) {
            self._container.appendChild(w);
        });
    },

    toggleEditMode() {
        this.isEditMode = !this.isEditMode;
        var overview = this._container;
        var btn = document.getElementById('btnEditLayout');
        var resetBtn = document.getElementById('btnResetLayout');

        if (this.isEditMode) {
            overview.classList.add('dashboard-edit-mode');
            if (btn) {
                btn.textContent = 'Done';
                btn.classList.add('bg-accent-blue', 'text-white');
                btn.classList.remove('bg-dark-700', 'text-gray-400');
            }
            if (resetBtn) resetBtn.classList.remove('hidden');
            // Enable draggable on all widgets
            overview.querySelectorAll('.dashboard-widget').forEach(function(w) {
                w.setAttribute('draggable', 'true');
            });
        } else {
            overview.classList.remove('dashboard-edit-mode');
            if (btn) {
                btn.textContent = 'Edit Layout';
                btn.classList.remove('bg-accent-blue', 'text-white');
                btn.classList.add('bg-dark-700', 'text-gray-400');
            }
            if (resetBtn) resetBtn.classList.add('hidden');
            // Disable draggable
            overview.querySelectorAll('.dashboard-widget').forEach(function(w) {
                w.setAttribute('draggable', 'false');
            });
        }
    },

    _setupButtons() {
        var self = this;
        var editBtn = document.getElementById('btnEditLayout');
        var resetBtn = document.getElementById('btnResetLayout');
        if (editBtn) {
            editBtn.addEventListener('click', function() { self.toggleEditMode(); });
        }
        if (resetBtn) {
            resetBtn.addEventListener('click', function() { self.resetLayout(); });
        }
    }
};
