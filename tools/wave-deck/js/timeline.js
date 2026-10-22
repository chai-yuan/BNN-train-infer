'use strict';

// Compressed handshake timeline. Each row shows handshake density, not a
// per-cycle waveform, so long simulations stay readable.
(function (root) {
  const G = root.WD.gfx;
  const C = G.C;
  const ROWS = [
    { key: 'task', label: 'TASK', color: C.BRASS },
    { key: 'cmd', label: 'CMD', color: C.BRASS },
    { key: 'act', label: 'ACT', color: C.GREEN },
    { key: 'wgt', label: 'WGT', color: C.BLUE },
    { key: 'res', label: 'RES', color: C.CREAM },
    { key: 'done', label: 'DONE', color: C.AMBER },
    { key: 'busy', label: 'BUSY', color: C.GREEN_D },
  ];

  function niceStep(span, target = 6) {
    const raw = span / target;
    const power = 10 ** Math.floor(Math.log10(raw));
    for (const m of [1, 2, 5, 10]) {
      if (raw <= power * m) return power * m;
    }
    return power * 10;
  }

  function fmtCycle(n) {
    if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + 'k';
    return String(n);
  }

  function statusColor(task) {
    if (task.status === 'ok' && task.verified) return C.GREEN;
    if (task.status === 'ok') return C.AMBER;
    if (task.status === 'unfinished') return C.MUTED;
    return C.RUST;
  }

  class TimelineView {
    constructor(canvas, callbacks = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.callbacks = callbacks;
      this.cssW = 0;
      this.cssH = 0;
      this.dpr = 1;
      this.trace = null;
      this.focus = null;
      this.viewStart = 0;
      this.viewEnd = 1;
      this.current = 0;
      this.base = document.createElement('canvas');
      this.baseCtx = this.base.getContext('2d');
      this.baseKey = '';
      this.geom = null;
      this.hoverX = null;
      this.dragging = false;
      this.bindEvents();
    }

    bindEvents() {
      const c = this.canvas;
      c.addEventListener('pointerdown', (e) => {
        if (!this.trace) return;
        const x = this.localX(e);
        if (x < 56 || x > this.cssW - 10) return;
        this.dragging = true;
        c.setPointerCapture(e.pointerId);
        this.seekFromX(x);
      });
      c.addEventListener('pointermove', (e) => {
        this.hoverX = this.localX(e);
        if (this.dragging) this.seekFromX(this.hoverX);
        else this.callbacks.onHover && this.callbacks.onHover();
      });
      const end = (e) => {
        if (!this.dragging) return;
        this.dragging = false;
        if (c.hasPointerCapture(e.pointerId)) c.releasePointerCapture(e.pointerId);
      };
      c.addEventListener('pointerup', end);
      c.addEventListener('pointercancel', end);
      c.addEventListener('pointerleave', () => {
        if (!this.dragging) this.hoverX = null;
      });
      c.addEventListener('wheel', (e) => {
        if (!this.trace) return;
        e.preventDefault();
        const rect = c.getBoundingClientRect();
        const x = Math.max(56, Math.min(this.cssW - 10, e.clientX - rect.left));
        const cycle = this.cycleAtX(x);
        const totalStart = 0;
        const totalEnd = Math.max(1, this.trace.summary.cycleMax);
        const full = totalEnd - totalStart;
        const span = this.viewEnd - this.viewStart;
        const factor = Math.exp(e.deltaY * 0.0014);
        let nextSpan = span * factor;
        nextSpan = Math.max(Math.min(8, full), Math.min(full, nextSpan));
        const ratio = (cycle - this.viewStart) / span;
        let start = cycle - ratio * nextSpan;
        let endCycle = start + nextSpan;
        if (start < totalStart) { start = totalStart; endCycle = start + nextSpan; }
        if (endCycle > totalEnd) { endCycle = totalEnd; start = endCycle - nextSpan; }
        if (endCycle - start < 1) return;
        if (this.callbacks.onView) this.callbacks.onView(start, endCycle);
      }, { passive: false });
      c.addEventListener('dblclick', () => {
        if (this.callbacks.onFit) this.callbacks.onFit();
      });
    }

    localX(e) {
      return e.clientX - this.canvas.getBoundingClientRect().left;
    }

    cycleAtX(x) {
      const g = this.geom;
      if (!g) return this.viewStart;
      const t = Math.max(0, Math.min(1, (x - g.plotX) / g.plotW));
      return this.viewStart + t * (this.viewEnd - this.viewStart);
    }

    seekFromX(x) {
      const cycle = Math.round(this.cycleAtX(x));
      if (this.callbacks.onSeek) this.callbacks.onSeek(cycle);
    }

    resize() {
      const rect = this.canvas.parentElement.getBoundingClientRect();
      const w = Math.max(320, Math.floor(rect.width));
      const h = Math.max(120, Math.floor(rect.height));
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      if (w === this.cssW && h === this.cssH && dpr === this.dpr) return;
      this.cssW = w;
      this.cssH = h;
      this.dpr = dpr;
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.base.width = this.canvas.width;
      this.base.height = this.canvas.height;
      this.baseCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.baseKey = '';
    }

    reset(trace) {
      this.trace = trace || null;
      this.focus = null;
      this.current = 0;
      this.hoverX = null;
      this.baseKey = '';
    }

    setView(start, end) {
      this.viewStart = start;
      this.viewEnd = Math.max(start + 1, end);
      this.baseKey = '';
    }

    setFocus(task) {
      if (this.focus === task) return;
      this.focus = task;
      this.baseKey = '';
    }

    setCurrent(cycle) {
      this.current = cycle;
    }

    invalidate() {
      this.baseKey = '';
    }

    draw() {
      this.resize();
      const key = [
        this.cssW, this.cssH, this.dpr,
        this.trace ? this.trace.header.scopePath : '-',
        this.focus ? this.focus.id : '-',
        this.viewStart.toFixed(4), this.viewEnd.toFixed(4),
      ].join('|');
      if (key !== this.baseKey) {
        this.baseKey = key;
        this.renderBase();
      }

      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.cssW, this.cssH);
      ctx.drawImage(this.base, 0, 0, this.cssW, this.cssH);
      if (!this.trace) return;

      const g = this.geom;
      if (!g) return;
      this.drawPlayhead(ctx, g);
      this.drawHover(ctx, g);
    }

    renderBase() {
      const ctx = this.baseCtx;
      const w = this.cssW;
      const h = this.cssH;
      G.fill(ctx, 0, 0, w, h, C.WELL);
      G.speckle(ctx, 0, 0, w, h, C.GRID_DOT, 29, 7, 3);
      G.hline(ctx, 0, 0, w, C.METAL_D);
      G.hline(ctx, 0, h - 1, w, C.METAL_D);

      const left = 52;
      const top = 20;
      const plotX = left;
      const plotW = Math.max(10, w - left - 12);
      const plotH = Math.max(40, h - top - 8);
      const rowH = Math.max(12, Math.floor(plotH / ROWS.length));
      const rows = {};
      for (let i = 0; i < ROWS.length; i++) {
        rows[ROWS[i].key] = top + i * rowH;
      }
      this.geom = { plotX, plotW, top, plotH, rowH, rows, left };
      this.drawRuler(ctx, plotX, plotW, top);
      if (!this.trace) {
        G.text(ctx, '等待波形', w / 2, h / 2 - 6, 12, C.DIM, { align: 'center' });
        return;
      }

      const vStart = this.viewStart;
      const vEnd = this.viewEnd;
      const span = Math.max(1, vEnd - vStart);
      const xOf = (cycle) => plotX + ((cycle - vStart) / span) * plotW;

      if (this.focus) {
        const x0 = Math.max(plotX, xOf(this.focus.cmdCycle));
        const x1 = Math.min(plotX + plotW, xOf(this.focus.endCycle + 1));
        if (x1 > x0) {
          ctx.save();
          ctx.globalAlpha = 0.08;
          G.fill(ctx, x0, top, x1 - x0, plotH, C.BRASS_L);
          ctx.restore();
          G.vline(ctx, Math.round(x0), top, plotH, C.BRASS_D);
          G.vline(ctx, Math.round(x1), top, plotH, C.BRASS_D);
        }
      }

      for (let i = 0; i < ROWS.length; i++) {
        const r = ROWS[i];
        const y = rows[r.key];
        if (i % 2 === 0) {
          ctx.save();
          ctx.globalAlpha = 0.85;
          G.fill(ctx, plotX, y, plotW, rowH, C.PANEL_D);
          ctx.restore();
        }
        G.hline(ctx, plotX, y + rowH - 1, plotW, C.METAL_D);
        G.text(ctx, r.label, 10, y + Math.max(2, (rowH - 9) / 2), 9, C.MUTED, { tracking: 1 });
      }

      this.drawTasks(ctx, plotX, plotW, rows.task, rowH, xOf);
      this.drawDensities(ctx, plotX, plotW, rows, rowH, xOf);
      this.drawBusy(ctx, plotX, plotW, rows.busy, rowH, xOf);
      this.drawMarkers(ctx, plotX, plotW, top, plotH, xOf);
    }

    drawRuler(ctx, x, w, y) {
      const span = this.viewEnd - this.viewStart;
      if (!this.trace) {
        G.hline(ctx, x, y - 3, w, C.METAL_D);
        return;
      }
      const step = Math.max(1, niceStep(span, 6));
      const first = Math.ceil(this.viewStart / step) * step;
      G.hline(ctx, x, y - 3, w, C.METAL_D);
      for (let c = first; c <= this.viewEnd; c += step) {
        const px = Math.round(x + ((c - this.viewStart) / span) * w);
        G.vline(ctx, px, y - 7, 3, C.DIM);
        G.vline(ctx, px, y, 3, C.FAINT);
        G.text(ctx, fmtCycle(c), Math.min(px + 3, x + w - 20), 2, 9, C.CREAM_D);
      }
    }

    drawTasks(ctx, x, w, y, rowH, xOf) {
      const barY = y + 4;
      const barH = Math.max(3, Math.min(6, rowH - 6));
      for (const task of this.trace.tasks) {
        if (task.endCycle < this.viewStart || task.cmdCycle > this.viewEnd) continue;
        const x0 = Math.max(x, xOf(task.cmdCycle));
        const x1 = Math.min(x + w, xOf(task.endCycle + 1));
        if (x1 <= x0) continue;
        const active = this.focus && this.focus.id === task.id;
        const color = statusColor(task);
        ctx.save();
        ctx.globalAlpha = active ? 1 : 0.42;
        G.fill(ctx, Math.round(x0), barY + 1, Math.max(1, Math.round(x1 - x0)), barH, color);
        if (active) G.fill(ctx, Math.round(x0), barY, Math.max(1, Math.round(x1 - x0)), 1, C.CREAM);
        ctx.restore();
        if (x1 - x0 > 46) {
          G.text(ctx, '#' + String(task.id + 1).padStart(2, '0'), Math.round(x0) + 4, barY - 1, 8, C.BLACK, { weight: 'bold' });
        }
      }
    }

    drawDensities(ctx, x, w, rows, rowH, xOf) {
      const events = this.focus ? this.focus.events : this.trace.events;
      const bins = Math.max(1, Math.round(w));
      const counts = {
        cmd: new Uint16Array(bins),
        act: new Uint16Array(bins),
        wgt: new Uint16Array(bins),
        res: new Uint16Array(bins),
        done: new Uint16Array(bins),
      };
      const span = this.viewEnd - this.viewStart;
      const vStart = this.viewStart;
      const vEnd = this.viewEnd;

      for (const ev of events) {
        if (ev.cycle < vStart || ev.cycle > vEnd) continue;
        const b = Math.max(0, Math.min(bins - 1, Math.floor(((ev.cycle - vStart) / span) * bins)));
        if (ev.type === 'cmd') counts.cmd[b]++;
        else if (ev.type === 'res') counts.res[b]++;
        else if (ev.type === 'done') counts.done[b]++;
        else if (ev.type === 'in') {
          if (ev.actH) counts.act[b]++;
          if (ev.wgtH) counts.wgt[b]++;
        }
      }

      const rowsMap = { cmd: 'cmd', act: 'act', wgt: 'wgt', res: 'res', done: 'done' };
      for (const key of Object.keys(rowsMap)) {
        const y = rows[key];
        const arr = counts[key];
        const color = ROWS.find((r) => r.key === key).color;
        const maxH = Math.max(2, rowH - 5);
        for (let i = 0; i < bins; i++) {
          const n = arr[i];
          if (!n) continue;
          const bh = n <= 1 ? 3 : Math.min(maxH, 3 + Math.floor(Math.log2(n + 1) * 2));
          G.fill(ctx, Math.max(x, x + i - 1), y + rowH - 2 - bh, 2, bh, color);
        }
      }
    }

    drawBusy(ctx, x, w, y, rowH, xOf) {
      const color = C.GREEN_D;
      for (const run of this.trace.busyRuns) {
        if (run.end < this.viewStart || run.start > this.viewEnd) continue;
        const x0 = Math.max(x, Math.round(xOf(run.start)));
        const x1 = Math.min(x + w, Math.round(xOf(run.end + 1)));
        if (x1 <= x0) continue;
        G.fill(ctx, x0, y + Math.floor(rowH / 2) - 1, x1 - x0, 3, color);
        if (x1 - x0 > 3) G.fill(ctx, x0, y + Math.floor(rowH / 2) - 1, x1 - x0, 1, C.GREEN_L);
      }
    }

    drawMarkers(ctx, x, w, top, h, xOf) {
      for (const m of this.trace.markers) {
        if (m.cycle < this.viewStart || m.cycle > this.viewEnd) continue;
        if (m.type === 'reset_release') continue;
        const px = Math.round(xOf(m.cycle));
        const color = m.type === 'abort' ? C.RUST_L : C.RUST_D;
        for (let yy = top; yy < top + h; yy += 6) G.vline(ctx, px, yy, 3, color);
        G.text(ctx, m.type === 'abort' ? 'ABORT' : 'RST', px + 3, top + 2, 8, color, { tracking: 1 });
      }
    }

    drawPlayhead(ctx, g) {
      if (this.current < this.viewStart || this.current > this.viewEnd) return;
      const x = Math.round(g.plotX + ((this.current - this.viewStart) / (this.viewEnd - this.viewStart)) * g.plotW);
      G.vline(ctx, x, g.top - 3, g.plotH, C.RUST_L);
      G.fill(ctx, x - 3, g.top - 8, 7, 4, C.RUST);
      G.fill(ctx, x - 1, g.top - 4, 3, 3, C.RUST_L);
      const label = 'C ' + Math.round(this.current);
      const tw = G.textW(label, 9, 0.5);
      const lx = Math.max(g.plotX, Math.min(x + 5, g.plotX + g.plotW - tw - 6));
      G.fill(ctx, lx - 3, g.top - 19, tw + 6, 13, C.BLACK);
      G.outline(ctx, lx - 3, g.top - 19, tw + 6, 13, C.RUST_D);
      G.text(ctx, label, lx, g.top - 17, 9, C.WELL);
    }

    drawHover(ctx, g) {
      if (this.hoverX === null || this.dragging) return;
      const x = this.hoverX;
      if (x < g.plotX || x > g.plotX + g.plotW) return;
      const cycle = Math.round(this.cycleAtX(x));
      G.vline(ctx, Math.round(x), g.top, this.cssH - g.top - 8, C.FAINT);
      const label = 'C ' + cycle;
      const tw = G.textW(label, 8);
      const lx = Math.min(x + 4, g.plotX + g.plotW - tw - 4);
      G.fill(ctx, lx, this.cssH - 16, tw + 5, 12, C.PANEL_D);
      G.text(ctx, label, lx + 2, this.cssH - 14, 8, C.DIM);
    }
  }

  root.WD.TimelineView = TimelineView;
})(typeof window !== 'undefined' ? window : globalThis);
