'use strict';

// Pixel drawing kit for the two canvases. Coordinates are CSS pixels; the
// caller applies the device pixel ratio transform.
(function (root) {
  const C = {
    BLACK: '#142019',
    INK: '#ffffff',
    CABINET: '#e2ebe4',
    WELL: '#f4f9f5',
    METAL_D: '#a9bbae',
    METAL: '#8b9e91',
    METAL_L: '#cfe0d3',
    BRASS_D: '#6f9a82',
    BRASS: '#257247',
    BRASS_L: '#1a5638',
    RUST: '#b3543f',
    RUST_D: '#7d3528',
    RUST_L: '#9d402f',
    CREAM: '#102018',
    CREAM_D: '#2e4638',
    MUTED: '#405648',
    DIM: '#50675a',
    FAINT: '#8fa396',
    PANEL: '#ffffff',
    PANEL_HI: '#ffffff',
    PANEL_D: '#e9f1ea',
    PAPER: '#f6faf5',
    TEXT_INK: '#ffffff',
    GREEN: '#2f8257',
    GREEN_D: '#578f70',
    GREEN_L: '#1a5e3f',
    AMBER: '#76883a',
    AMBER_D: '#94a36a',
    AMBER_L: '#55662a',
    RED: '#b3543f',
    RED_D: '#8a3c2e',
    BLUE: '#3f7d71',
    BLUE_D: '#92aaa2',
    GRID: '#e0ebe2',
    GRID_DOT: '#c4d6c9',
  };

  const measureCanvas = typeof document !== 'undefined' ? document.createElement('canvas') : null;
  const measureCtx = measureCanvas ? measureCanvas.getContext('2d') : null;
  const widthCache = new Map();

  function font(size, weight) {
    return (weight ? weight + ' ' : '') + size + 'px FusionPixel, ui-monospace, "DejaVu Sans Mono", monospace';
  }

  function charWidth(ch, size, weight) {
    const key = size + '|' + (weight || '') + '|' + ch;
    let w = widthCache.get(key);
    if (w === undefined) {
      if (!measureCtx) return Math.round(size * 0.6);
      measureCtx.font = font(size, weight);
      w = measureCtx.measureText(ch).width;
      widthCache.set(key, w);
    }
    return w;
  }

  function clearMetrics() {
    widthCache.clear();
  }

  function textW(s, size, tracking = 0, weight) {
    s = String(s);
    let w = 0;
    let n = 0;
    for (const ch of s) {
      w += charWidth(ch, size, weight);
      n++;
    }
    return Math.ceil(w + Math.max(0, n - 1) * tracking);
  }

  function setFont(ctx, size, weight) {
    ctx.font = font(size, weight);
    ctx.textBaseline = 'top';
    ctx.textAlign = 'left';
  }

  function text(ctx, s, x, y, size, color, opt = {}) {
    s = String(s);
    if (!s) return 0;
    const tracking = opt.tracking || 0;
    const weight = opt.weight;
    let w = textW(s, size, tracking, weight);
    if (opt.maxWidth && w > opt.maxWidth) {
      s = fit(ctx, s, opt.maxWidth, size, tracking, weight);
      w = textW(s, size, tracking, weight);
    }
    if (opt.align === 'center') x = Math.round(x - w / 2);
    else if (opt.align === 'right') x = Math.round(x - w);
    else x = Math.round(x);
    y = Math.round(y);
    ctx.save();
    setFont(ctx, size, weight);
    if (opt.alpha !== undefined) ctx.globalAlpha *= opt.alpha;
    if (opt.shadow) {
      ctx.fillStyle = opt.shadow;
      let sx = x;
      for (const ch of s) {
        ctx.fillText(ch, sx, y + 1);
        sx += charWidth(ch, size, weight) + tracking;
      }
    }
    ctx.fillStyle = color;
    let cx = x;
    for (const ch of s) {
      ctx.fillText(ch, cx, y);
      cx += charWidth(ch, size, weight) + tracking;
    }
    ctx.restore();
    return w;
  }

  function fit(ctx, s, maxWidth, size, tracking = 0, weight) {
    s = String(s);
    if (textW(s, size, tracking, weight) <= maxWidth) return s;
    const dots = '...';
    let out = '';
    let w = 0;
    const limit = maxWidth - textW(dots, size, tracking, weight);
    for (const ch of s) {
      const cw = charWidth(ch, size, weight) + tracking;
      if (w + cw > limit) break;
      out += ch;
      w += cw;
    }
    return out + dots;
  }

  function fill(ctx, x, y, w, h, color) {
    if (w <= 0 || h <= 0) return;
    ctx.fillStyle = color;
    ctx.fillRect(Math.round(x), Math.round(y), Math.round(w), Math.round(h));
  }

  function hline(ctx, x, y, w, color) {
    fill(ctx, x, y, w, 1, color);
  }

  function vline(ctx, x, y, h, color) {
    fill(ctx, x, y, 1, h, color);
  }

  function outline(ctx, x, y, w, h, color) {
    hline(ctx, x, y, w, color);
    hline(ctx, x, y + h - 1, w, color);
    vline(ctx, x, y, h, color);
    vline(ctx, x + w - 1, y, h, color);
  }

  function dashedOutline(ctx, x, y, w, h, color, on = 4, off = 4) {
    for (let i = 0; i < w; i += on + off) {
      hline(ctx, x + i, y, Math.min(on, w - i), color);
      hline(ctx, x + i, y + h - 1, Math.min(on, w - i), color);
    }
    for (let j = 0; j < h; j += on + off) {
      vline(ctx, x, y + j, Math.min(on, h - j), color);
      vline(ctx, x + w - 1, y + j, Math.min(on, h - j), color);
    }
  }

  function dot(ctx, x, y, color) {
    fill(ctx, x, y, 1, 1, color);
  }

  function line(ctx, x0, y0, x1, y1, color) {
    x0 = Math.round(x0);
    y0 = Math.round(y0);
    x1 = Math.round(x1);
    y1 = Math.round(y1);
    const dx = Math.abs(x1 - x0);
    const dy = -Math.abs(y1 - y0);
    const sx = x0 < x1 ? 1 : -1;
    const sy = y0 < y1 ? 1 : -1;
    let err = dx + dy;
    ctx.fillStyle = color;
    for (;;) {
      ctx.fillRect(x0, y0, 1, 1);
      if (x0 === x1 && y0 === y1) break;
      const e2 = 2 * err;
      if (e2 >= dy) { err += dy; x0 += sx; }
      if (e2 <= dx) { err += dx; y0 += sy; }
    }
  }

  function arrowRight(ctx, x0, x1, y, color, thickness = 1) {
    if (x1 < x0) [x0, x1] = [x1, x0];
    fill(ctx, x0, y, x1 - x0 - 4, thickness, color);
    for (let i = 0; i < 4; i++) {
      fill(ctx, x1 - 4 + i, y - (3 - i), 1, (3 - i) * 2 + 1, color);
    }
  }

  function arrowLeft(ctx, x0, x1, y, color, thickness = 1) {
    if (x1 < x0) [x0, x1] = [x1, x0];
    fill(ctx, x0 + 4, y, x1 - x0 - 4, thickness, color);
    for (let i = 0; i < 4; i++) {
      fill(ctx, x0 + i, y - (3 - i), 1, (3 - i) * 2 + 1, color);
    }
  }

  function screw(ctx, x, y) {
    fill(ctx, x, y, 4, 4, C.BLACK);
    fill(ctx, x + 1, y + 1, 2, 2, C.METAL_L);
    dot(ctx, x + 2, y + 2, C.METAL_D);
  }

  function panelInner(ctx, x, y, w, h) {
    fill(ctx, x, y, w, h, C.PANEL_D);
    fill(ctx, x + 1, y + 1, w - 2, h - 2, C.INK);
    hline(ctx, x + 1, y + 1, w - 2, C.PANEL_HI);
    vline(ctx, x + 1, y + 1, h - 2, C.PANEL_HI);
    outline(ctx, x, y, w, h, C.METAL_D);
  }

  function screen(ctx, x, y, w, h) {
    fill(ctx, x, y, w, h, C.WELL);
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.clip();
    ctx.fillStyle = C.GRID_DOT;
    for (let gy = y + 12; gy < y + h; gy += 16) {
      for (let gx = x + 12; gx < x + w; gx += 16) {
        ctx.fillRect(gx, gy, 1, 1);
      }
    }
    ctx.restore();
    outline(ctx, x, y, w, h, C.METAL_D);
  }

  function speckle(ctx, x, y, w, h, color, step, seed, every = 7) {
    let s = seed || 1;
    ctx.fillStyle = color;
    for (let yy = 0; yy < h; yy += step) {
      for (let xx = ((yy / step) | 0) % 2; xx < w; xx += step) {
        s = (s * 1664525 + 1013904223) >>> 0;
        if ((s >>> 28) < every) ctx.fillRect(x + xx, y + yy, 1, 1);
      }
    }
  }

  function led(ctx, x, y, color, phase = 0) {
    const lit = Math.sin(performance.now() / 420 + phase) > -0.2;
    fill(ctx, x - 1, y - 1, 6, 6, C.BLACK);
    fill(ctx, x, y, 4, 4, lit ? color : C.PANEL_HI);
    if (lit) dot(ctx, x + 1, y + 1, C.PANEL_HI);
  }

  function chip(ctx, x, y, label, bg, fg, opt = {}) {
    const size = opt.size || 10;
    const pad = opt.pad === undefined ? 5 : opt.pad;
    const tracking = opt.tracking === undefined ? 1 : opt.tracking;
    const w = textW(label, size, tracking) + pad * 2;
    const h = size + 7;
    fill(ctx, x, y, w, h, C.BLACK);
    outline(ctx, x, y, w, h, bg);
    fill(ctx, x + 1, y + 1, w - 2, h - 2, bg);
    text(ctx, label, x + pad, y + 3, size, fg, { tracking });
    return { w, h };
  }

  function emptyPlate(ctx, cx, cy, label, sub, phase = 0) {
    const w = Math.max(200, textW(label, 14) + 42);
    const h = 46;
    const x = Math.round(cx - w / 2);
    const y = Math.round(cy - h / 2);
    fill(ctx, x, y, w, h, C.WELL);
    dashedOutline(ctx, x, y, w, h, C.FAINT, 4, 4);
    const lit = Math.sin(performance.now() / 760 + phase) > 0.45;
    fill(ctx, x + 14, y + 20, 4, 4, C.BLACK);
    fill(ctx, x + 15, y + 21, 2, 2, lit ? C.AMBER : C.AMBER_D);
    text(ctx, label, x + 26, y + 9, 14, C.CREAM);
    if (sub) text(ctx, sub, x + 26, y + 27, 11, C.DIM);
  }

  function progress(ctx, x, y, w, h, ratio, color, bg = C.PANEL_D) {
    fill(ctx, x, y, w, h, C.BLACK);
    fill(ctx, x + 1, y + 1, w - 2, h - 2, bg);
    const inner = Math.max(0, w - 2) * Math.max(0, Math.min(1, ratio || 0));
    fill(ctx, x + 1, y + 1, inner, h - 2, color);
    if (inner > 0) fill(ctx, x + 1, y + 1, inner, 1, C.CREAM_D);
  }

  root.WD = root.WD || {};
  root.WD.gfx = {
    C,
    font,
    clearMetrics,
    textW,
    text,
    fit,
    fill,
    hline,
    vline,
    outline,
    dashedOutline,
    dot,
    line,
    arrowRight,
    arrowLeft,
    screw,
    panelInner,
    screen,
    speckle,
    led,
    chip,
    emptyPlate,
    progress,
  };
})(typeof window !== 'undefined' ? window : globalThis);
