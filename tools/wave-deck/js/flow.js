'use strict';

// Transaction flow canvas: four top-level channels, command tasks and the
// packets that were actually transferred by successful handshakes.
(function (root) {
  const G = root.WD.gfx;
  const C = G.C;
  const FLIGHT_MS = 620;
  const LANE_COLORS = {
    cmd: C.BRASS,
    act: C.GREEN,
    wgt: C.BLUE,
    res: C.CREAM,
    done: C.AMBER,
  };

  function fmt(n) {
    if (n === null || n === undefined) return 'x';
    return String(n);
  }

  function statusMeta(task, cycle) {
    if (!task) return { text: 'IDLE', color: C.DIM, led: C.GREEN_D };
    if (cycle < task.cmdCycle) return { text: 'IDLE', color: C.DIM, led: C.GREEN_D };
    if (cycle < task.endCycle) {
      if (task.abortCycle !== null && task.abortCycle !== undefined && cycle >= task.abortCycle) {
        return { text: 'ABORT', color: C.RUST_L, led: C.RED };
      }
      return { text: 'RUN', color: C.AMBER, led: C.AMBER };
    }
    if (task.status === 'ok' && task.verified) return { text: 'DONE', color: C.GREEN_L, led: C.GREEN };
    if (task.status === 'ok') return { text: 'CHECK', color: C.AMBER, led: C.AMBER };
    return { text: task.statusText || 'ERR', color: C.RUST_L, led: C.RED };
  }

  function eventLine(trace, ev) {
    if (!ev) return '等待握手事件 / NO HANDSHAKE';
    const t = ev.taskId === null ? 'T--' : 'T' + String(ev.taskId + 1).padStart(2, '0');
    if (ev.type === 'cmd') {
      return `${t} CMD  K=${fmt(ev.k)}  G=${fmt(ev.groups)}  N=${fmt(ev.n)}  ${ev.legal ? 'LEGAL' : 'ILLEGAL'}`;
    }
    if (ev.type === 'in') {
      const bits = ev.actIdx === null || ev.actIdx < 0 ? 'x' : (trace.actVectors[ev.actIdx] || 'x').slice(0, 12);
      return `${t} IN  CH${ev.channel === null ? '-' : ev.channel}  BEAT ${ev.beat === null ? '-' : ev.beat + 1}/${ev.beatsPerChannel || '?'}` +
        `  A=${bits}  W=${ev.weight === null ? 'x' : '0x' + Number(ev.weight).toString(16).toUpperCase()}` +
        `  AL${ev.actLast ? '1' : '0'} WL${ev.weightLast ? '1' : '0'}`;
    }
    if (ev.type === 'res') {
      return `${t} RES  CH${ev.channel === null ? '-' : ev.channel}  G${fmt(ev.group)}  ${fmt(ev.data)} (0x${ev.dataHex})` +
        `  CL${ev.chanLast ? '1' : '0'} L${ev.last ? '1' : '0'}`;
    }
    return `${t} DONE  STATUS ${fmt(ev.status)}  ${ev.statusText || ''}`;
  }

  class FlowView {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.cssW = 0;
      this.cssH = 0;
      this.dpr = 1;
      this.flights = [];
      this.spawnCursor = 0;
      this.flightTask = undefined;
      this.lastCycle = null;
      this.geom = null;
    }

    resize() {
      const rect = this.canvas.parentElement.getBoundingClientRect();
      const w = Math.max(320, Math.floor(rect.width));
      const h = Math.max(220, Math.floor(rect.height));
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      if (w === this.cssW && h === this.cssH && dpr === this.dpr) return;
      this.cssW = w;
      this.cssH = h;
      this.dpr = dpr;
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    reset(trace) {
      this.flights = [];
      this.spawnCursor = 0;
      this.flightTask = undefined;
      this.lastCycle = null;
      this.trace = trace || null;
    }

    spawn(ev, now) {
      if (!ev) return;
      const add = (lane, label, color, offset = 0) => {
        this.flights.push({ lane, label, color, born: now, offset });
      };
      if (ev.type === 'cmd') add('cmd', 'C', LANE_COLORS.cmd);
      else if (ev.type === 'res') add('res', 'R', LANE_COLORS.res);
      else if (ev.type === 'done') add('done', 'D', ev.status === 0 ? LANE_COLORS.done : C.RED);
      else if (ev.type === 'in') {
        if (ev.actH) add('act', 'A', LANE_COLORS.act, -1);
        if (ev.wgtH) add('wgt', 'W', LANE_COLORS.wgt, 1);
      }
    }

    collectFlights(task, currentCycle, playing, now) {
      if (!task) {
        this.flights = [];
        return;
      }
      const taskKey = task.id;
      if (this.flightTask !== taskKey) {
        this.flightTask = taskKey;
        this.flights = [];
        this.spawnCursor = this.lowerBound(task.events, currentCycle, true);
        this.lastCycle = currentCycle;
      }
      if (this.lastCycle !== null && currentCycle < this.lastCycle) {
        this.flights = [];
        this.spawnCursor = this.lowerBound(task.events, currentCycle, true);
      }
      if (!playing) {
        if (this.lastCycle !== currentCycle) {
          this.flights = [];
          this.spawnCursor = this.lowerBound(task.events, currentCycle, true);
        }
        this.lastCycle = currentCycle;
        return;
      }
      this.lastCycle = currentCycle;

      const due = [];
      while (this.spawnCursor < task.events.length && due.length < 8) {
        const ev = task.events[this.spawnCursor];
        if (ev.cycle > currentCycle) break;
        due.push(ev);
        this.spawnCursor++;
      }
      // If playback jumped over a dense region, skip the backlog instead of
      // spawning a wall of packets.
      while (this.spawnCursor < task.events.length && task.events[this.spawnCursor].cycle <= currentCycle) {
        this.spawnCursor++;
      }
      for (const ev of due) this.spawn(ev, now);
      if (this.flights.length > 36) this.flights.splice(0, this.flights.length - 36);
    }

    lowerBound(events, cycle, inclusive) {
      let lo = 0;
      let hi = events.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        const c = events[mid].cycle;
        if (c < cycle || (!inclusive && c === cycle)) lo = mid + 1;
        else hi = mid;
      }
      return lo;
    }

    draw(state, now) {
      this.resize();
      const ctx = this.ctx;
      const w = this.cssW;
      const h = this.cssH;
      const trace = state.trace;
      G.fill(ctx, 0, 0, w, h, C.WELL);
      G.speckle(ctx, 0, 0, w, h, C.GRID_DOT, 23, 19, 3);

      if (!trace) {
        this.drawEmpty(ctx, w, h);
        return;
      }

      const task = state.focusTask;
      const cycle = state.currentCycle;
      this.collectFlights(task, cycle, state.playing, now);
      this.flights = this.flights.filter((f) => now - f.born < FLIGHT_MS);

      const coreW = Math.round(Math.max(196, Math.min(286, w * 0.32)));
      const coreLeft = Math.round((w - coreW) / 2);
      const topRoom = 38;
      const bottomRoom = 40;
      const availH = Math.max(190, h - topRoom - bottomRoom);
      const coreH = Math.min(300, availH);
      const coreTop = Math.round(topRoom + (availH - coreH) / 2);
      const driverW = Math.round(Math.max(88, Math.min(126, w * 0.15)));
      const sinkW = driverW;
      const driverLeft = 18;
      const sinkLeft = w - 18 - sinkW;
      const cmdY = coreTop + 26;
      const actY = coreTop + Math.round(coreH * 0.36);
      const wgtY = actY + 20;
      const resY = coreTop + coreH - Math.min(coreH * 0.32, 94);
      const doneY = resY + 28;
      const lane = {
        cmd: { y: cmdY, x0: driverLeft + driverW + 8, x1: coreLeft - 8, dir: 1 },
        act: { y: actY, x0: driverLeft + driverW + 8, x1: coreLeft - 8, dir: 1 },
        wgt: { y: wgtY, x0: driverLeft + driverW + 8, x1: coreLeft - 8, dir: 1 },
        res: { y: resY, x0: coreLeft + coreW + 8, x1: sinkLeft - 8, dir: -1 },
        done: { y: doneY, x0: coreLeft + coreW + 8, x1: sinkLeft - 8, dir: -1 },
      };
      this.geom = { coreLeft, coreTop, coreW, coreH, lane, driverW, sinkW, driverLeft, sinkLeft };

      this.drawLegend(ctx, w);
      this.drawDriver(ctx, driverLeft, cmdY - 22, driverW, wgtY - cmdY + 44, state);
      this.drawSink(ctx, sinkLeft, resY - 22, sinkW, doneY - resY + 44, state);

      for (const [name, L] of Object.entries(lane)) {
        const color = name === 'act' ? C.GREEN : name === 'wgt' ? C.BLUE : name === 'res' ? C.CREAM_D : name === 'done' ? C.AMBER : C.BRASS;
        if (L.dir > 0) G.arrowRight(ctx, L.x0, L.x1, L.y, color, 2);
        else G.arrowLeft(ctx, L.x0, L.x1, L.y, color, 2);
      }
      this.drawLaneLabels(ctx, lane, state);
      this.drawFlights(ctx, lane, now);
      this.drawCore(ctx, coreLeft, coreTop, coreW, coreH, state);
      this.drawMonitor(ctx, 18, h - 29, w - 36, 21, state);
    }

    drawEmpty(ctx, w, h) {
      const cx = w / 2;
      const cy = h / 2;
      G.emptyPlate(ctx, cx, cy - 8, '等待波形', '拖入 Verilator VCD 开始接驳', 0.2);
      G.text(ctx, '只保留 bnn_core_top 顶层端口 · 只显示握手成功的数据移动', cx, cy + 36, 11, C.DIM, { align: 'center' });
    }

    drawLegend(ctx, w) {
      let x = w - 18;
      const items = [
        ['DONE', C.AMBER],
        ['RES', C.CREAM],
        ['WGT', C.BLUE],
        ['ACT', C.GREEN],
        ['CMD', C.BRASS],
      ];
      for (const [label, color] of items) {
        const tw = G.textW(label, 9, 1);
        G.fill(ctx, x - 6, 13, 4, 4, color);
        G.text(ctx, label, x - tw, 9, 9, C.MUTED, { tracking: 1 });
        x -= tw + 18;
      }
    }

    drawDriver(ctx, x, y, w, h, state) {
      G.panelInner(ctx, x, y, w, h);
      G.screw(ctx, x + 4, y + 4);
      G.screw(ctx, x + w - 8, y + h - 8);
      G.text(ctx, 'DRIVER', x + 10, y + 10, 12, C.CREAM, { tracking: 1 });
      G.text(ctx, '激励源', x + 10, y + 25, 10, C.MUTED);
      const s = state.trace.summary;
      const task = state.focusTask;
      const lines = task
        ? [['CMD', task.cmdCycle <= state.currentCycle ? 1 : 0], ['IN', state.counts ? state.counts.input : 0]]
        : [['CMD', s.cmdEvents], ['IN', s.inEvents]];
      let yy = y + h - 28;
      for (const [label, value] of lines) {
        G.text(ctx, label, x + 10, yy, 9, C.DIM, { tracking: 1 });
        G.text(ctx, String(value), x + w - 10, yy - 1, 11, C.BRASS_L, { align: 'right' });
        yy += 14;
      }
    }

    drawSink(ctx, x, y, w, h, state) {
      G.panelInner(ctx, x, y, w, h);
      G.screw(ctx, x + 4, y + 4);
      G.screw(ctx, x + w - 8, y + h - 8);
      G.text(ctx, 'SINK', x + 10, y + 10, 12, C.CREAM, { tracking: 1 });
      G.text(ctx, '接收端', x + 10, y + 25, 10, C.MUTED);
      const s = state.trace.summary;
      const task = state.focusTask;
      const lines = task
        ? [['RES', state.counts ? state.counts.res : 0], ['DONE', state.counts ? state.counts.done : 0]]
        : [['RES', s.resEvents], ['DONE', s.doneEvents]];
      let yy = y + h - 28;
      for (const [label, value] of lines) {
        G.text(ctx, label, x + 10, yy, 9, C.DIM, { tracking: 1 });
        G.text(ctx, String(value), x + w - 10, yy - 1, 11, C.CREAM, { align: 'right' });
        yy += 14;
      }
    }

    drawLaneLabels(ctx, lane, state) {
      const task = state.focusTask;
      const counts = state.counts;
      const rows = [
        ['cmd', 'CMD', C.BRASS, counts ? (task.cmdCycle <= state.currentCycle ? 1 : 0) : null],
        ['act', 'ACT', C.GREEN, counts ? counts.act : null],
        ['wgt', 'WGT', C.RUST_L, counts ? counts.wgt : null],
        ['res', 'RES', C.CREAM, counts ? counts.res : null],
        ['done', 'DONE', C.AMBER, counts ? counts.done : null],
      ];
      for (const [key, label, color, count] of rows) {
        const L = lane[key];
        G.text(ctx, label, L.x0 + 2, L.y - 12, 10, color, { tracking: 1 });
        if (count !== null) {
          G.text(ctx, String(count), L.x1 - 3, L.y - 12, 10, C.MUTED, { align: 'right' });
        }
      }
    }

    drawFlights(ctx, lane, now) {
      for (const f of this.flights) {
        const L = lane[f.lane];
        if (!L) continue;
        const t = Math.min(1, (now - f.born) / FLIGHT_MS);
        const eased = 1 - (1 - t) * (1 - t);
        const x = L.x0 + (L.x1 - L.x0) * eased;
        const y = L.y - 3 + f.offset;
        const fade = t > 0.82 ? 1 - (t - 0.82) / 0.18 : 1;
        ctx.save();
        ctx.globalAlpha = Math.max(0, fade);
        for (let i = 1; i <= 3; i++) {
          const trailX = x - L.dir * i * 6;
          G.fill(ctx, trailX - 1, L.y - 1, 2, 2, i === 1 ? f.color : C.FAINT);
        }
        G.fill(ctx, x - 5, y - 3, 10, 7, C.BLACK);
        G.fill(ctx, x - 4, y - 2, 8, 5, f.color);
        G.fill(ctx, x - 2, y - 1, 2, 1, C.PANEL_HI);
        G.text(ctx, f.label, x + 8, y - 4, 8, f.color);
        ctx.restore();
      }
    }

    drawCore(ctx, x, y, w, h, state) {
      const trace = state.trace;
      const task = state.focusTask;
      const status = statusMeta(task, state.currentCycle);
      G.panelInner(ctx, x, y, w, h);
      G.outline(ctx, x, y, w, h, C.BRASS_D);
      G.screw(ctx, x + 5, y + 5);
      G.screw(ctx, x + w - 9, y + 5);
      G.screw(ctx, x + 5, y + h - 9);
      G.screw(ctx, x + w - 9, y + h - 9);

      G.text(ctx, 'BNN CORE TOP', x + w / 2, y + 10, 14, C.CREAM, { align: 'center', tracking: 1 });
      const scope = trace.header.scopePath;
      G.text(ctx, scope, x + w / 2, y + 27, 9, C.MUTED, { align: 'center', maxWidth: w - 24 });

      G.led(ctx, x + 12, y + 46, status.led, task ? task.id : 0);
      G.text(ctx, status.text, x + 21, y + 41, 10, status.color, { tracking: 1 });
      if (task) {
        G.text(ctx, '#' + String(task.id + 1).padStart(2, '0'), x + w - 12, y + 41, 12, C.BRASS_L, { align: 'right' });
      }
      G.hline(ctx, x + 10, y + 58, w - 20, C.METAL_D);

      let yy = y + 66;
      if (!task) {
        G.text(ctx, 'OVERVIEW', x + 12, yy, 10, C.MUTED, { tracking: 1 });
        const s = trace.summary;
        const rows = [
          ['COMMANDS', s.tasks],
          ['ACT BEATS', s.actHandshakes],
          ['RES BEATS', s.resEvents],
          ['DONE', s.doneEvents],
        ];
        yy += 16;
        for (const [label, val] of rows) {
          G.text(ctx, label, x + 12, yy, 9, C.DIM, { tracking: 1 });
          G.text(ctx, String(val), x + w - 12, yy - 1, 11, C.CREAM, { align: 'right' });
          yy += 15;
        }
      } else {
        const c = state.counts || { input: 0, res: 0, done: 0, channels: 0 };
        G.text(ctx, 'CMD  K ' + fmt(task.k) + '  G ' + fmt(task.groups) + '  N ' + fmt(task.n), x + 12, yy, 10, C.CREAM_D, { maxWidth: w - 24 });
        yy += 14;
        G.text(ctx, 'BLK ' + fmt(task.beatsPerChannel) + ' · GRP ' + fmt(task.groupBeats) + ' · T ' + fmt(task.duration || 0) + 'c', x + 12, yy, 9, C.DIM, { tracking: 0 });
        yy += 18;

        G.text(ctx, 'IN', x + 12, yy, 9, C.GREEN, { tracking: 1 });
        G.text(ctx, String(c.input) + '/' + fmt(task.expectedInput), x + w - 12, yy, 9, C.MUTED, { align: 'right' });
        G.progress(ctx, x + 12, yy + 13, w - 24, 6, task.expectedInput ? c.input / task.expectedInput : 0, C.GREEN);
        yy += 26;

        G.text(ctx, 'RES', x + 12, yy, 9, C.CREAM, { tracking: 1 });
        G.text(ctx, String(c.res) + '/' + fmt(task.expectedResults), x + w - 12, yy, 9, C.MUTED, { align: 'right' });
        G.progress(ctx, x + 12, yy + 13, w - 24, 6, task.expectedResults ? c.res / task.expectedResults : 0, C.CREAM_D);
        yy += 26;

        G.text(ctx, 'CH ' + fmt(c.channels) + '/' + fmt(task.n) + '  GROUP ' + fmt(task.groups), x + 12, yy, 9, C.DIM);
        if (task.issues.length) {
          G.text(ctx, '! ' + task.issues.length + ' CHECK', x + w - 12, yy, 9, C.RUST_L, { align: 'right' });
        }
      }
      G.hline(ctx, x + 10, y + h - 28, w - 20, C.METAL_D);
      G.text(ctx, 'TOP-LEVEL ONLY', x + w / 2, y + h - 19, 8, C.FAINT, { align: 'center', tracking: 1 });
    }

    drawMonitor(ctx, x, y, w, h, state) {
      G.panelInner(ctx, x, y, w, h);
      G.fill(ctx, x + 3, y + 3, 18, h - 6, C.PANEL_D);
      G.text(ctx, 'MON', x + 6, y + 6, 8, C.AMBER, { tracking: 1 });
      const ev = state.selectedEvent;
      G.text(ctx, eventLine(state.trace, ev), x + 27, y + 5, 9, ev ? C.CREAM_D : C.DIM, {
        maxWidth: w - 34,
      });
    }
  }

  root.WD.FlowView = FlowView;
})(typeof window !== 'undefined' ? window : globalThis);
