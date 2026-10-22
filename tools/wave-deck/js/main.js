'use strict';

(function () {
  const WD = window.WD;
  const { FlowView, TimelineView } = WD;
  const { decodeActVector, actBitsLabel } = WD;
  const C = WD.gfx.C;
  const $ = (id) => document.getElementById(id);
  const ROW_H = 26;
  const SPEEDS = [0.5, 1, 2, 4];
  const TYPE_LABEL = { cmd: 'CMD', in: 'IN', res: 'RES', done: 'DONE' };
  const TYPE_CLASS = { cmd: 'ev-cmd', in: 'ev-in', res: 'ev-res', done: 'ev-done' };
  const STATUS_SHORT = {
    ok: 'OK', cfg: 'CFG', aborted: 'ABRT', protocol: 'PROTO',
    reset: 'RST', superseded: 'SUP', unfinished: 'OPEN', running: 'RUN', unknown: '?',
  };

  const S = {
    trace: null,
    file: null,
    selectedTaskIndex: -1,
    currentCycle: 0,
    playing: false,
    speedIndex: 1,
    viewStart: 0,
    viewEnd: 1,
    filter: 'all',
    tab: 'events',
    follow: true,
    selectedEvent: null,
    visibleEvents: [],
    focusTask: null,
    counts: null,
    loading: false,
    lastSync: 0,
    programmaticScroll: false,
  };

  const flow = new FlowView($('flowCanvas'));
  const timeline = new TimelineView($('timelineCanvas'), {
    onSeek: (cycle) => seekCycle(cycle, { user: true }),
    onView: (start, end) => {
      setPlaying(false);
      setView(start, end);
    },
    onFit: fitView,
  });

  let bootHidden = false;
  const hideBoot = () => {
    if (bootHidden) return;
    bootHidden = true;
    const boot = $('boot');
    boot.classList.add('done');
    setTimeout(() => boot.remove(), 500);
  };

  function formatBytes(bytes) {
    if (!bytes && bytes !== 0) return '--';
    if (bytes >= 1024 * 1024 * 1024) return (bytes / (1024 ** 3)).toFixed(2) + ' GB';
    if (bytes >= 1024 * 1024) return (bytes / (1024 ** 2)).toFixed(1) + ' MB';
    if (bytes >= 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return bytes + ' B';
  }

  function formatSeconds(sec) {
    if (sec === null || sec === undefined || !Number.isFinite(sec)) return '--';
    if (sec >= 1) return sec.toFixed(2) + ' s';
    if (sec >= 1e-3) return (sec * 1e3).toFixed(sec >= 0.1 ? 1 : 2) + ' ms';
    if (sec >= 1e-6) return (sec * 1e6).toFixed(sec >= 1e-4 ? 0 : 2) + ' µs';
    if (sec >= 1e-9) return (sec * 1e9).toFixed(sec >= 1e-7 ? 0 : 2) + ' ns';
    return (sec * 1e12).toFixed(0) + ' ps';
  }

  function formatTickTime(ticks) {
    if (ticks === null || ticks === undefined || !S.trace) return '--';
    const sec = ticks * (S.trace.header.timescale.seconds || 0);
    return formatSeconds(sec);
  }

  function fmt(n) {
    if (n === null || n === undefined) return 'x';
    return String(n);
  }

  function pad2(n) {
    return String(n).padStart(2, '0');
  }

  function colorForStatus(task) {
    if (!task) return C.DIM;
    if (task.status === 'ok' && task.verified) return C.GREEN;
    if (task.status === 'ok') return C.AMBER;
    if (task.status === 'running' || task.status === 'unfinished') return C.MUTED;
    return C.RUST;
  }

  function statusClass(task) {
    if (!task) return 'st-open';
    if (task.status === 'ok' && task.verified) return 'st-ok';
    if (task.status === 'ok') return 'st-warn';
    if (task.status === 'running' || task.status === 'unfinished') return 'st-open';
    return 'st-err';
  }

  function taskLabel(task) {
    return 'T' + pad2(task.id + 1);
  }

  function updateSummary() {
    const t = S.trace;
    if (!t) return;
    const s = t.summary;
    $('kpiTasks').textContent = String(s.tasks);
    $('kpiTasksSub').textContent = s.tasks ? `${s.ok} 成功 · ${s.tasks - s.ok} 异常` : '无命令';
    $('kpiSuccess').textContent = s.tasks ? `${s.ok}/${s.tasks}` : '--';
    $('kpiSuccessSub').textContent = s.tasks ? Math.round((s.ok / s.tasks) * 100) + '%' : '--';
    $('kpiCycles').textContent = String(s.edges);
    $('kpiCyclesSub').textContent = formatSeconds(s.traceSeconds);
    $('kpiIn').textContent = String(s.inEvents);
    $('kpiInSub').textContent = `ACT ${s.actHandshakes} · WGT ${s.weightHandshakes}`;
    $('kpiRes').textContent = String(s.resEvents);
    $('kpiResSub').textContent = `DONE ${s.doneEvents}`;
    $('kpiBusy').textContent = Math.round(s.busyRatio * 100) + '%';
    $('kpiBusySub').textContent = `${s.busyCycles}/${s.nonResetCycles} 周期`;

    $('taskCount').textContent = String(s.tasks);
    $('statusText').textContent = `已载入 ${t.file ? t.file.name : '波形'} · ${s.edges} 个上升沿 · ${s.tasks} 个命令`;
    $('clockInfo').textContent = t.summary.clockPeriodSec
      ? `CLK ${formatSeconds(t.summary.clockPeriodSec)} · ${formatSeconds(t.summary.traceSeconds)}`
      : `${s.edges} EDGE`;
  }

  function updateHeader(file) {
    const t = S.trace;
    $('fileName').textContent = file ? file.name : '未载入波形';
    $('fileSize').textContent = file ? formatBytes(file.size) : '';
    $('scopeName').textContent = t
      ? `${t.header.scopePath} · ${t.header.timescale.text || 'timescale?'} · ${t.header.varCount} vars`
      : '拖入 Verilator VCD · 自动锁定 bnn_core_top';
    $('linkLed').classList.toggle('on', !!t);
    $('linkLed').classList.toggle('err', false);
  }

  function eventShort(ev) {
    if (!ev) return '';
    if (ev.type === 'cmd') return `K ${fmt(ev.k)} · G ${fmt(ev.groups)} · N ${fmt(ev.n)}`;
    if (ev.type === 'in') {
      return `CH${fmt(ev.channel)} · ${fmt(ev.beat === null ? null : ev.beat + 1)}/${fmt(ev.beatsPerChannel)} · A${ev.actLast ? 'L' : '-'} W${ev.weightLast ? 'L' : '-'}`;
    }
    if (ev.type === 'res') {
      return `CH${fmt(ev.channel)} · G${fmt(ev.group)} · ${fmt(ev.data)}${ev.chanLast ? ' · CL' : ''}${ev.last ? ' · LAST' : ''}`;
    }
    return `${ev.statusText || 'DONE'} · ${fmt(ev.status)}`;
  }

  function eventRowSummary(ev) {
    if (!ev) return '';
    if (ev.type === 'cmd') return `K=${fmt(ev.k)} G=${fmt(ev.groups)} N=${fmt(ev.n)}`;
    if (ev.type === 'in') return `CH${fmt(ev.channel)} beat ${fmt(ev.beat === null ? null : ev.beat + 1)}/${fmt(ev.beatsPerChannel)}  A${ev.actLast ? 'L' : '-'} W${ev.weightLast ? 'L' : '-'}`;
    if (ev.type === 'res') return `CH${fmt(ev.channel)} G${fmt(ev.group)} = ${fmt(ev.data)}${ev.chanLast ? '  chan_last' : ''}${ev.last ? '  last' : ''}`;
    return `status ${fmt(ev.status)} · ${ev.statusText || ''}`;
  }

  function focusTaskAt(cycle) {
    const t = S.trace;
    if (!t) return null;
    if (S.selectedTaskIndex >= 0) return t.tasks[S.selectedTaskIndex] || null;
    const tasks = t.tasks;
    let lo = 0;
    let hi = tasks.length - 1;
    let found = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (tasks[mid].cmdCycle <= cycle) {
        found = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    if (found < 0) return null;
    const task = tasks[found];
    if (cycle <= task.endCycle) return task;
    return null;
  }

  function countsAt(task, cycle) {
    if (!task) return null;
    const counts = { input: 0, act: 0, wgt: 0, res: 0, done: 0, channels: 0 };
    if (cycle < task.cmdCycle) return counts;
    if (cycle >= task.endCycle) {
      return {
        input: task.inputBeats,
        act: task.inputBeats,
        wgt: task.inputBeats,
        res: task.resultBeats,
        done: task.doneCycle !== null && cycle >= task.doneCycle ? 1 : 0,
        channels: task.inputChannels,
        cmd: 1,
      };
    }
    let lo = 0;
    let hi = task.events.length - 1;
    let idx = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (task.events[mid].cycle <= cycle) {
        idx = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    if (idx >= 0) {
      const ev = task.events[idx];
      counts.input = ev.inSeq || 0;
      counts.act = ev.inSeq || 0;
      counts.wgt = ev.inSeq || 0;
      counts.res = ev.resSeq || 0;
    }
    counts.channels = task.beatsPerChannel ? Math.min(task.n || 0, Math.floor(counts.input / task.beatsPerChannel)) : 0;
    counts.done = task.doneCycle !== null && cycle >= task.doneCycle ? 1 : 0;
    counts.cmd = 1;
    return counts;
  }

  function rebuildVisibleEvents(resetScroll = false) {
    const t = S.trace;
    if (!t) {
      S.visibleEvents = [];
    } else {
      const base = S.selectedTaskIndex >= 0 ? t.tasks[S.selectedTaskIndex].events : t.events;
      S.visibleEvents = S.filter === 'all' ? base : base.filter((ev) => ev.type === S.filter);
    }
    $('eventSpacer').style.height = (S.visibleEvents.length * ROW_H) + 'px';
    $('eventRows').style.height = (S.visibleEvents.length * ROW_H) + 'px';
    $('eventCount').textContent = `${S.visibleEvents.length} EVENTS` +
      (t && t.summary.droppedEvents ? ` · 省略 ${t.summary.droppedEvents}` : '');
    if (resetScroll) $('eventScroll').scrollTop = 0;
    renderEventRows();
  }

  function renderEventRows() {
    const scroll = $('eventScroll');
    const total = S.visibleEvents.length;
    const viewport = scroll.clientHeight || 260;
    let start = Math.max(0, Math.floor(scroll.scrollTop / ROW_H) - 4);
    let end = Math.min(total, start + Math.ceil(viewport / ROW_H) + 8);
    if (start > total) start = Math.max(0, total - 1);
    const rows = $('eventRows');
    const frag = document.createDocumentFragment();
    for (let i = start; i < end; i++) {
      const ev = S.visibleEvents[i];
      const row = document.createElement('button');
      row.type = 'button';
      row.className = 'event-row' + (ev === S.selectedEvent ? ' active' : '');
      row.dataset.index = String(i);
      row.style.top = (i * ROW_H) + 'px';
      const type = document.createElement('span');
      type.className = 'event-type ' + (TYPE_CLASS[ev.type] || '') + (ev.type === 'done' && ev.status !== 0 ? ' ev-err' : '');
      type.textContent = TYPE_LABEL[ev.type] || '?';
      const main = document.createElement('span');
      main.className = 'event-main';
      main.textContent = eventRowSummary(ev);
      const cyc = document.createElement('span');
      cyc.className = 'event-cycle';
      cyc.textContent = String(ev.cycle);
      row.append(type, main, cyc);
      frag.append(row);
    }
    rows.replaceChildren(frag);
  }

  function highlightEvent() {
    const rows = $('eventRows').querySelectorAll('.event-row');
    for (const row of rows) {
      const idx = Number(row.dataset.index);
      row.classList.toggle('active', S.visibleEvents[idx] === S.selectedEvent);
    }
  }

  function scrollToSelected() {
    if (!S.follow || S.tab !== 'events' || !S.selectedEvent) return;
    const idx = S.visibleEvents.indexOf(S.selectedEvent);
    if (idx < 0) return;
    const scroll = $('eventScroll');
    const top = idx * ROW_H;
    const viewTop = scroll.scrollTop;
    const viewBottom = viewTop + scroll.clientHeight;
    if (top < viewTop + 26 || top + ROW_H > viewBottom - 26) {
      S.programmaticScroll = true;
      scroll.scrollTop = Math.max(0, top - Math.floor(scroll.clientHeight / 2));
      renderEventRows();
      setTimeout(() => { S.programmaticScroll = false; }, 0);
    }
  }

  function pickVisibleEvent(cycle) {
    if (!S.visibleEvents.length) return null;
    let lo = 0;
    let hi = S.visibleEvents.length - 1;
    let idx = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (S.visibleEvents[mid].cycle <= cycle) {
        idx = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return idx >= 0 ? S.visibleEvents[idx] : null;
  }

  function syncSelection(force = false) {
    if (!S.trace) return;
    const cycle = Math.floor(S.currentCycle);
    const ev = pickVisibleEvent(cycle);
    if (ev === S.selectedEvent && !force) return;
    S.selectedEvent = ev;
    renderDetail();
    highlightEvent();
    scrollToSelected();
  }

  function setCycle(cycle, opts = {}) {
    if (!S.trace) return;
    if (opts.user) setPlaying(false);
    const max = S.trace.summary.cycleMax;
    S.currentCycle = Math.max(0, Math.min(max, cycle));
    const current = Math.floor(S.currentCycle);
    timeline.setCurrent(current);
    if (opts.event) {
      S.selectedEvent = opts.event;
      renderDetail();
      highlightEvent();
      scrollToSelected();
    } else {
      syncSelection(true);
    }
    updateReadouts();
  }

  function seekCycle(cycle, opts = {}) {
    setCycle(cycle, { user: opts.user !== false });
  }

  function seekEvent(ev) {
    if (!ev) return;
    setPlaying(false);
    S.currentCycle = ev.cycle;
    timeline.setCurrent(ev.cycle);
    S.selectedEvent = ev;
    updateReadouts();
    renderDetail();
    highlightEvent();
    scrollToSelected();
  }

  function updateReadouts() {
    if (!S.trace) return;
    const c = Math.floor(S.currentCycle);
    const focus = S.selectedTaskIndex >= 0 ? S.trace.tasks[S.selectedTaskIndex] : focusTaskAt(c);
    $('cycleReadout').textContent = `CYCLE ${c} / ${S.trace.summary.cycleMax}`;
    $('flowCycles').textContent = `C ${c}`;
    $('flowFocus').textContent = focus
      ? `${taskLabel(focus)} · K${fmt(focus.k)} G${fmt(focus.groups)} N${fmt(focus.n)}`
      : '全部任务 / IDLE';
    const ev = S.selectedEvent;
    $('detailCycle').textContent = ev ? `C${ev.cycle} · ${formatTickTime(ev.time)}` : '--';
  }

  function setPlaying(on) {
    if (!S.trace) on = false;
    if (on && S.currentCycle >= S.trace.summary.cycleMax) {
      S.currentCycle = S.viewStart;
      timeline.setCurrent(S.currentCycle);
      syncSelection(true);
    }
    S.playing = on;
    $('playBtn').textContent = on ? 'PAUSE' : 'PLAY';
    $('playBtn').classList.toggle('active', on);
  }

  function togglePlay() {
    if (!S.trace) return;
    setPlaying(!S.playing);
  }

  function stepEvent(dir) {
    if (!S.trace || !S.visibleEvents.length) return;
    setPlaying(false);
    let idx = S.selectedEvent ? S.visibleEvents.indexOf(S.selectedEvent) : -1;
    if (idx < 0) {
      let lo = 0;
      let hi = S.visibleEvents.length - 1;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if (S.visibleEvents[mid].cycle <= S.currentCycle) {
          idx = mid;
          lo = mid + 1;
        } else {
          hi = mid - 1;
        }
      }
    }
    idx = Math.max(0, Math.min(S.visibleEvents.length - 1, idx + dir));
    seekEvent(S.visibleEvents[idx]);
  }

  function firstLastEvent(last) {
    if (!S.visibleEvents.length) return;
    seekEvent(last ? S.visibleEvents[S.visibleEvents.length - 1] : S.visibleEvents[0]);
  }

  function setView(start, end) {
    if (!S.trace) return;
    const max = S.trace.summary.cycleMax;
    let a = Math.max(0, Math.min(max, start));
    let b = Math.max(a + 1, Math.min(Math.max(1, max), end));
    if (b <= a) b = a + 1;
    S.viewStart = a;
    S.viewEnd = b;
    timeline.setView(a, b);
  }

  function fitView() {
    if (!S.trace) return;
    if (S.selectedTaskIndex >= 0) {
      const t = S.trace.tasks[S.selectedTaskIndex];
      const pad = Math.max(2, Math.round((t.endCycle - t.cmdCycle + 1) * 0.06));
      setView(t.cmdCycle - pad, t.endCycle + pad + 1);
    } else {
      setView(0, Math.max(1, S.trace.summary.cycleMax));
    }
  }

  function baseCyclesPerSecond() {
    const span = Math.max(1, S.viewEnd - S.viewStart);
    return Math.max(8, Math.min(20000, span / 10));
  }

  function updatePlaybackUI(now) {
    const c = Math.floor(S.currentCycle);
    $('cycleReadout').textContent = `CYCLE ${c} / ${S.trace.summary.cycleMax}`;
    $('flowCycles').textContent = `C ${c}`;
    const focus = S.selectedTaskIndex >= 0
      ? S.trace.tasks[S.selectedTaskIndex]
      : focusTaskAt(Math.floor(S.currentCycle));
    $('flowFocus').textContent = focus
      ? `${taskLabel(focus)} · K${fmt(focus.k)} G${fmt(focus.groups)} N${fmt(focus.n)}`
      : '全部任务 / IDLE';
    if (now - S.lastSync > 55) {
      S.lastSync = now;
      syncSelection(false);
    }
  }

  function frame(now) {
    if (!S.lastFrame) S.lastFrame = now;
    const dt = Math.min(0.12, Math.max(0, (now - S.lastFrame) / 1000));
    S.lastFrame = now;

    if (S.playing && S.trace) {
      const cps = baseCyclesPerSecond() * SPEEDS[S.speedIndex];
      S.currentCycle += cps * dt;
      const max = S.trace.summary.cycleMax;
      if (S.currentCycle >= max) {
        S.currentCycle = max;
        setPlaying(false);
      } else if (S.selectedTaskIndex >= 0 && S.currentCycle >= S.viewEnd) {
        S.currentCycle = S.viewEnd;
        setPlaying(false);
      } else if (S.currentCycle > S.viewEnd && S.selectedTaskIndex < 0 && S.viewEnd < max) {
        const span = S.viewEnd - S.viewStart;
        setView(S.currentCycle, Math.min(max, S.currentCycle + span));
      }
      updatePlaybackUI(now);
    }

    if (S.trace) {
      timeline.setCurrent(Math.floor(S.currentCycle));
      const focus = focusTaskAt(Math.floor(S.currentCycle));
      S.focusTask = focus;
      S.counts = countsAt(focus, Math.floor(S.currentCycle));
      $('detailCycle').textContent = S.selectedEvent
        ? `C${S.selectedEvent.cycle} · ${formatTickTime(S.selectedEvent.time)}`
        : '--';
    }

    flow.draw({
      trace: S.trace,
      focusTask: S.focusTask,
      counts: S.counts,
      currentCycle: S.currentCycle,
      playing: S.playing,
      selectedEvent: S.selectedEvent,
    }, now);
    timeline.draw();
    requestAnimationFrame(frame);
  }

  function selectTask(index) {
    if (!S.trace) return;
    S.selectedTaskIndex = index;
    setPlaying(false);
    if (index >= 0) {
      const t = S.trace.tasks[index];
      const pad = Math.max(2, Math.round((t.endCycle - t.cmdCycle + 1) * 0.06));
      setView(t.cmdCycle - pad, t.endCycle + pad + 1);
      timeline.setFocus(t);
      S.currentCycle = t.cmdCycle;
    } else {
      setView(0, Math.max(1, S.trace.summary.cycleMax));
      timeline.setFocus(null);
      S.currentCycle = S.trace.tasks.length ? S.trace.tasks[0].cmdCycle : 0;
    }
    timeline.setCurrent(S.currentCycle);
    updateTaskSelection();
    rebuildVisibleEvents(true);
    syncSelection(true);
    renderStats();
    updateReadouts();
  }

  function updateTaskSelection() {
    const cards = $('taskList').querySelectorAll('.task-card');
    for (const card of cards) {
      card.classList.toggle('active', Number(card.dataset.task) === S.selectedTaskIndex);
    }
    const active = $('taskList').querySelector('.task-card.active');
    if (active) active.scrollIntoView({ block: 'nearest' });
  }

  function renderTasks() {
    const list = $('taskList');
    if (!S.trace) {
      list.replaceChildren();
      return;
    }
    const frag = document.createDocumentFragment();
    const all = document.createElement('button');
    all.type = 'button';
    all.className = 'task-card all' + (S.selectedTaskIndex < 0 ? ' active' : '');
    all.dataset.task = '-1';
    all.innerHTML = '<div class="task-top"><span class="task-id">全部任务</span><span class="task-status">ALL</span></div>' +
      `<div class="task-meta"><span>${S.trace.summary.tasks} 个命令</span><span>${S.trace.summary.tasks ? S.trace.summary.ok + ' OK' : '--'}</span></div>`;
    frag.append(all);

    for (const task of S.trace.tasks) {
      const card = document.createElement('button');
      card.type = 'button';
      card.className = 'task-card ' + statusClass(task) + (S.selectedTaskIndex === task.id ? ' active' : '');
      card.dataset.task = String(task.id);
      card.style.setProperty('--status', colorForStatus(task));
      const short = task.status === 'ok' && !task.verified ? 'CHECK' : (STATUS_SHORT[task.status] || '?');
      const params = `K ${fmt(task.k)} · G ${fmt(task.groups)} · N ${fmt(task.n)}`;
      const issues = task.issues.length ? `<span class="task-warn">! ${task.issues.length}</span>` : '<span></span>';
      card.innerHTML =
        `<div class="task-top"><span class="task-id">#${pad2(task.id + 1)}</span><span class="task-status">${short}</span></div>` +
        `<div class="task-params">${params}</div>` +
        `<div class="task-meta"><span>C ${task.cmdCycle}–${task.endCycle} · ${task.duration}c</span>${issues}</div>` +
        `<div class="task-meta"><span>IN ${task.inputBeats}</span><span>RES ${task.resultBeats}</span><span>${task.statusText}</span></div>`;
      card.title = `${task.statusText} · 输入 ${task.inputBeats}/${fmt(task.expectedInput)} · 结果 ${task.resultBeats}/${fmt(task.expectedResults)}`;
      frag.append(card);
    }
    list.replaceChildren(frag);
    const active = list.querySelector('.task-card.active');
    if (active) active.scrollIntoView({ block: 'nearest' });
  }

  function statRow(label, value, cls = '') {
    const row = document.createElement('div');
    row.className = 'stat-row ' + cls;
    const a = document.createElement('span');
    a.textContent = label;
    const b = document.createElement('b');
    b.textContent = value;
    row.append(a, b);
    return row;
  }

  function statSection(title) {
    const sec = document.createElement('div');
    sec.className = 'stat-section';
    const h = document.createElement('div');
    h.className = 'stat-title';
    h.textContent = title;
    sec.append(h);
    return sec;
  }

  function renderStats() {
    const pane = $('statsPane');
    if (!S.trace) {
      pane.replaceChildren();
      return;
    }
    const t = S.trace;
    const s = t.summary;
    const frag = document.createDocumentFragment();

    const global = statSection('全波形 / GLOBAL');
    const grid = document.createElement('div');
    grid.className = 'stat-grid';
    grid.append(
      statRow('上升沿', String(s.edges)),
      statRow('非复位周期', String(s.nonResetCycles)),
      statRow('忙碌周期', `${s.busyCycles} (${Math.round(s.busyRatio * 100)}%)`),
      statRow('任务总数', String(s.tasks)),
      statRow('成功 / 异常', `${s.ok} / ${s.tasks - s.ok}`),
      statRow('CMD 握手', String(s.cmdEvents)),
      statRow('输入握手', `${s.inEvents} (A${s.actHandshakes}/W${s.weightHandshakes})`),
      statRow('结果握手', String(s.resEvents)),
      statRow('完成握手', String(s.doneEvents)),
      statRow('CMD 停顿', String(s.stalls.cmd)),
      statRow('ACT/WGT 停顿', `${s.stalls.act}/${s.stalls.weight}`),
      statRow('RES/DONE 停顿', `${s.stalls.res}/${s.stalls.done}`)
    );
    if (s.clockPeriodSec) grid.append(statRow('时钟周期', formatSeconds(s.clockPeriodSec)));
    if (s.traceSeconds) grid.append(statRow('波形时长', formatSeconds(s.traceSeconds)));
    global.append(grid);
    frag.append(global);

    if (S.selectedTaskIndex >= 0) {
      const task = t.tasks[S.selectedTaskIndex];
      const sec = statSection(`任务 #${pad2(task.id + 1)} / ${task.statusText}`);
      const g = document.createElement('div');
      g.className = 'stat-grid';
      g.append(
        statRow('状态', `${task.statusText} (${fmt(task.statusCode)})`),
        statRow('合法配置', task.legal ? 'YES' : 'NO', task.legal ? '' : 'bad'),
        statRow('K / groups / N', `${fmt(task.k)} / ${fmt(task.groups)} / ${fmt(task.n)}`),
        statRow('每通道拍数', fmt(task.beatsPerChannel)),
        statRow('周期范围', `${task.cmdCycle} → ${task.endCycle} (${task.duration}c)`),
        statRow('输入握手', `${task.inputBeats} / ${fmt(task.expectedInput)}`),
        statRow('结果握手', `${task.resultBeats} / ${fmt(task.expectedResults)}`),
        statRow('首输入偏移', task.firstInputOffset === null ? '--' : task.firstInputOffset + 'c'),
        statRow('首结果偏移', task.firstResultOffset === null ? '--' : task.firstResultOffset + 'c'),
        statRow('结果排空滞后', task.drainLag === null ? '--' : task.drainLag + 'c'),
        statRow('输入密度', task.inDensity.toFixed(2) + ' 拍/周期'),
        statRow('结果密度', task.resDensity.toFixed(2) + ' 个/周期'),
        statRow('输入通道数', `${task.inputChannels} / ${fmt(task.n)}`),
        statRow('结果通道数', `${task.resultChannels} / ${fmt(task.n)}`),
        statRow('忙 / 完成等待', `${task.busyCycles}c / ${task.doneWaitCycles}c`),
        statRow('停顿 ACT/WGT/RES', `${task.stalls.act}/${task.stalls.weight}/${task.stalls.res}`)
      );
      sec.append(g);
      if (task.issues.length) {
        const warn = document.createElement('div');
        warn.className = 'stat-issues';
        for (const issue of task.issues) {
          warn.append(warningItem(issue.text, issue.cycle));
        }
        sec.append(warn);
      }
      frag.append(sec);
    }

    const wsec = statSection('协议提示 / WARNINGS');
    if (!t.warnings.length) {
      const ok = document.createElement('div');
      ok.className = 'stat-none';
      ok.textContent = '未发现顶层协议异常';
      wsec.append(ok);
    } else {
      const list = document.createElement('div');
      list.className = 'stat-issues';
      for (const w of t.warnings) {
        list.append(warningItem(w.text, w.cycle));
      }
      if (t.warningOverflow) {
        const more = document.createElement('div');
        more.className = 'stat-none';
        more.textContent = '提示过多，已截断显示';
        list.append(more);
      }
      wsec.append(list);
    }
    frag.append(wsec);
    pane.replaceChildren(frag);
  }

  function warningItem(text, cycle) {
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'warn-item';
    const label = document.createElement('span');
    label.textContent = text;
    const c = document.createElement('span');
    c.className = 'warn-cycle';
    c.textContent = cycle === null || cycle === undefined ? '--' : 'C' + cycle;
    item.append(label, c);
    if (cycle !== null && cycle !== undefined) {
      item.title = '跳转到该周期';
      item.addEventListener('click', () => {
        seekCycle(cycle, { user: true });
        switchTab('events');
      });
    } else {
      item.disabled = true;
    }
    return item;
  }

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function kvRow(label, value, cls = '') {
    const row = el('div', 'kv ' + cls);
    row.append(el('span', '', label), el('b', '', value));
    return row;
  }

  function laneGrid(caption, note, values, mode) {
    const block = el('div', 'lane-block');
    const head = el('div', 'lane-head');
    head.append(el('span', '', caption), el('span', '', note));
    block.append(head);
    const grid = el('div', 'lane-grid');
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      let text;
      let cls = 'lane-cell';
      let title;
      if (mode === 'act') {
        if (v === null) {
          text = 'xx';
          cls += ' unknown';
          title = `lane ${i}: x`;
        } else {
          text = (v & 255).toString(16).padStart(2, '0').toUpperCase();
          cls += v > 0 ? ' pos' : v < 0 ? ' neg' : ' zero';
          title = `lane ${i}: ${v}`;
        }
      } else {
        if (v === null) {
          text = 'x';
          cls += ' unknown';
          title = `weight lane ${i}: x`;
        } else {
          text = v ? '+' : '-';
          cls += v ? ' wpos' : ' wneg';
          title = `weight lane ${i}: bit ${v} → ${v ? '+1' : '-1'}`;
        }
      }
      const cell = el('span', cls, text);
      cell.title = title;
      grid.append(cell);
    }
    block.append(grid);
    return block;
  }

  function bitsHex(bits, width) {
    if (!bits || !/^[01]+$/.test(bits)) return 'x';
    const value = BigInt('0b' + bits);
    const digits = Math.max(1, Math.ceil(width / 4));
    let h = value.toString(16).toUpperCase();
    if (h.length < digits) h = h.padStart(digits, '0');
    return h;
  }

  function weightBits(value, width) {
    const out = [];
    const big = BigInt(value === null || value === undefined ? 0 : value);
    for (let i = 0; i < width; i++) {
      out.push(value === null || value === undefined ? null : Number((big >> BigInt(i)) & 1n));
    }
    return out;
  }

  function renderDetail() {
    const body = $('detailBody');
    const ev = S.selectedEvent;
    if (!ev) {
      body.replaceChildren(el('div', 'detail-empty', S.trace ? '当前周期之前没有匹配的握手事件' : '尚未选择握手事件'));
      return;
    }
    const trace = S.trace;
    const frag = document.createDocumentFragment();
    const head = el('div', 'ev-title-line');
    const badge = el('span', 'ev-type-badge ' + (TYPE_CLASS[ev.type] || '') + (ev.type === 'done' && ev.status !== 0 ? ' ev-err' : ''), TYPE_LABEL[ev.type] || '?');
    const title = el('span', 'ev-title', ev.type === 'cmd' ? '命令握手' : ev.type === 'in' ? '输入握手 ACT + WGT' : ev.type === 'res' ? '结果握手' : '完成握手');
    const cyc = el('span', 'ev-time', `C${ev.cycle} · ${formatTickTime(ev.time)}`);
    head.append(badge, title, cyc);
    frag.append(head);

    if (ev.taskId !== null && ev.taskId !== undefined) {
      const task = trace.tasks[ev.taskId];
      frag.append(el('div', 'detail-task', `${taskLabel(task)} · ${task.statusText} · K ${fmt(task.k)} / G ${fmt(task.groups)} / N ${fmt(task.n)}`));
    } else {
      frag.append(el('div', 'detail-task warn', '任务关联缺失：该握手没有对应的 CMD'));
    }

    if (ev.type === 'cmd') {
      const grid = el('div', 'kv-grid');
      grid.append(
        kvRow('K_BLOCKS', fmt(ev.params.kBlocks)),
        kvRow('GROUP_DIV', fmt(ev.params.groupDiv)),
        kvRow('OUT_CHANS', fmt(ev.params.outChans)),
        kvRow('K (elements)', fmt(ev.k)),
        kvRow('GROUPS', fmt(ev.groups)),
        kvRow('GROUP SIZE', fmt(ev.params.groupDiv === null ? null : ev.params.groupDiv * 32) + ' elem'),
        kvRow('EXPECTED INPUT', fmt(ev.k !== null && ev.n !== null ? ev.params.kBlocks * ev.n : null)),
        kvRow('EXPECTED RES', fmt(ev.groups !== null && ev.n !== null ? ev.groups * ev.n : null)),
        kvRow('CONFIG', ev.legal ? 'LEGAL' : 'ILLEGAL', ev.legal ? 'good' : 'bad')
      );
      frag.append(grid);
      frag.append(el('div', 'detail-note', '命令握手后进入 RUN；尺寸只在握手时采样。'));
    } else if (ev.type === 'in') {
      const grid = el('div', 'kv-grid');
      grid.append(
        kvRow('CHANNEL', fmt(ev.channel)),
        kvRow('BEAT', `${fmt(ev.beat === null ? null : ev.beat + 1)} / ${fmt(ev.beatsPerChannel)}`),
        kvRow('ACT LAST', ev.actLast ? '1' : '0', ev.actLast ? 'good' : ''),
        kvRow('WEIGHT LAST', ev.weightLast ? '1' : '0', ev.weightLast ? 'good' : ''),
        kvRow('ACT HANDSHAKE', ev.actH ? 'YES' : 'NO', ev.actH ? '' : 'bad'),
        kvRow('WGT HANDSHAKE', ev.wgtH ? 'YES' : 'NO', ev.wgtH ? '' : 'bad'),
        kvRow('TASK BEAT', `${fmt(ev.inSeq)} / ${fmt(ev.taskId !== null ? trace.tasks[ev.taskId].expectedInput : null)}`),
        kvRow('WEIGHT HEX', ev.weight === null || ev.weight === undefined ? 'x' : '0x' + Number(ev.weight).toString(16).toUpperCase().padStart(Math.ceil(trace.weightWidth / 4), '0'))
      );
      frag.append(grid);

      const act = decodeActVector(trace, ev.actIdx);
      const lanes = act || new Array(Math.ceil(trace.actWidth / 8)).fill(null);
      frag.append(laneGrid('ACT LANE · INT8', 'LANE 0 → ' + (lanes.length - 1) + ' · HEX', lanes, 'act'));
      const bits = actBitsLabel(trace, ev.actIdx);
      frag.append(el('div', 'hexline', bitsHex(bits, trace.actWidth)));

      const weights = weightBits(ev.weight, trace.weightWidth);
      frag.append(laneGrid('WEIGHT LANE', 'BIT 0 → ' + (weights.length - 1) + ' · 1=+1 / 0=-1', weights, 'wgt'));
      const flags = el('div', 'flag-row');
      for (const [label, on] of [['act_last', ev.actLast], ['weight_last', ev.weightLast], ['act_hs', ev.actH], ['wgt_hs', ev.wgtH]]) {
        flags.append(el('span', 'flag' + (on ? ' on' : ''), label));
      }
      frag.append(flags);
    } else if (ev.type === 'res') {
      const big = el('div', 'result-big', fmt(ev.data));
      const sub = el('div', 'result-sub', ev.data === null
        ? 'payload x'
        : `0x${ev.dataHex} · signed ${trace.resWidth}`);
      frag.append(big, sub);
      const grid = el('div', 'kv-grid');
      grid.append(
        kvRow('CHANNEL', fmt(ev.channel)),
        kvRow('GROUP IDX', fmt(ev.group)),
        kvRow('RESULT SEQ', `${fmt(ev.resSeq)} / ${fmt(ev.taskId !== null ? trace.tasks[ev.taskId].expectedResults : null)}`),
        kvRow('RES_CHAN_LAST', ev.chanLast ? '1' : '0', ev.chanLast ? 'good' : ''),
        kvRow('RES_LAST', ev.last ? '1' : '0', ev.last ? 'good' : '')
      );
      frag.append(grid);
      const flags = el('div', 'flag-row');
      for (const [label, on] of [['res_chan_last', ev.chanLast], ['res_last', ev.last]]) {
        flags.append(el('span', 'flag' + (on ? ' on' : ''), label));
      }
      frag.append(flags);
    } else if (ev.type === 'done') {
      const grid = el('div', 'kv-grid');
      const task = ev.taskId !== null ? trace.tasks[ev.taskId] : null;
      grid.append(
        kvRow('DONE_STATUS', fmt(ev.status), ev.status === 0 ? 'good' : 'bad'),
        kvRow('STATUS TEXT', ev.statusText || '未知'),
        kvRow('INPUT BEATS', `${fmt(task ? task.inputBeats : ev.inSeq)} / ${fmt(task ? task.expectedInput : null)}`),
        kvRow('RESULT BEATS', `${fmt(task ? task.resultBeats : ev.resSeq)} / ${fmt(task ? task.expectedResults : null)}`),
        kvRow('VERIFIED', task ? (task.verified ? 'YES' : 'NO') : '--', task && task.verified ? 'good' : 'bad'),
        kvRow('TASK CYCLES', task ? `${task.cmdCycle} → ${task.endCycle} (${task.duration}c)` : '--')
      );
      frag.append(grid);
      if (task && task.issues.length) {
        const box = el('div', 'detail-note warn');
        box.textContent = task.issues.map((x) => x.text).join(' / ');
        frag.append(box);
      } else {
        frag.append(el('div', 'detail-note', ev.status === 0 ? '完整任务已完成，输入与结果计数在该视图中校验。' : '任务未成功完成，部分结果可能已被外部接收。'));
      }
    }
    body.replaceChildren(frag);
  }

  function switchTab(tab) {
    S.tab = tab;
    $('tabEvents').classList.toggle('active', tab === 'events');
    $('tabStats').classList.toggle('active', tab === 'stats');
    $('tabEvents').setAttribute('aria-selected', tab === 'events' ? 'true' : 'false');
    $('tabStats').setAttribute('aria-selected', tab === 'stats' ? 'true' : 'false');
    $('eventsPane').hidden = tab !== 'events';
    $('statsPane').hidden = tab !== 'stats';
    if (tab === 'stats') renderStats();
    else renderEventRows();
  }

  function onFilter(filter) {
    S.filter = filter;
    for (const btn of document.querySelectorAll('#eventFilters .filter')) {
      btn.classList.toggle('active', btn.dataset.filter === filter);
    }
    rebuildVisibleEvents(true);
    syncSelection(true);
  }

  function showToast(text, kind = '') {
    const toast = $('toast');
    toast.textContent = text;
    toast.className = 'toast show ' + kind;
    clearTimeout(showToast.timer);
    showToast.timer = setTimeout(() => toast.classList.remove('show'), 4200);
  }

  function setLoadingUI(visible, file, pct = 0, tip = '扫描头部...') {
    $('loadOverlay').classList.toggle('hidden', !visible);
    if (file) $('loadFile').textContent = file.name;
    $('loadBar').style.transform = `scaleX(${Math.max(0.01, Math.min(1, pct))})`;
    $('loadPct').textContent = Math.round(pct * 100) + '%';
    $('loadTip').textContent = tip;
  }

  async function loadFile(file) {
    if (!file || S.loading) return;
    if (!/\.vcd$/i.test(file.name)) {
      showToast('只支持 VCD 文件；FST 需要先转换为 VCD', 'err');
      return;
    }
    S.loading = true;
    const started = performance.now();
    setLoadingUI(true, file, 0, '扫描头部...');
    $('loadSub').textContent = '正在接驳 VCD';
    $('linkLed').classList.remove('on');
    try {
      const trace = await WD.analyzeVcd(file, {
        file: file,
        onProgress: (p) => {
          const pct = p.total ? p.loaded / p.total : 0;
          const tip = p.phase === 'start' ? '扫描头部...' : p.phase === 'done' ? '收起探针...' : `扫描 ${formatBytes(p.loaded)}`;
          setLoadingUI(true, file, pct, tip);
        },
      });
      const wait = Math.max(0, 260 - (performance.now() - started));
      if (wait) await new Promise((r) => setTimeout(r, wait));
      setLoadingUI(false);
      applyTrace(trace, file);
      showToast(`接驳完成 · ${trace.summary.tasks} 个任务 · ${trace.summary.edges} cycles`, 'ok');
    } catch (err) {
      console.error(err);
      setLoadingUI(false);
      $('linkLed').classList.toggle('on', !!S.trace);
      showToast('解析失败：' + (err && err.message ? err.message : String(err)), 'err');
    } finally {
      S.loading = false;
      $('fileInput').value = '';
    }
  }

  async function loadSample() {
    if (S.loading) return;
    try {
      setLoadingUI(true, { name: 'demo.vcd' }, 0, '获取内置示例...');
      const resp = await fetch('samples/demo.vcd');
      if (!resp.ok) throw new Error('找不到 samples/demo.vcd');
      const blob = await resp.blob();
      const file = new File([blob], 'demo.vcd', { type: 'text/plain' });
      setLoadingUI(false);
      await loadFile(file);
    } catch (err) {
      setLoadingUI(false);
      if (location.protocol === 'file:') {
        showToast('本地直接打开时无法读取示例；请用 HTTP 服务打开，或直接拖入 VCD', 'err');
      } else {
        showToast('示例载入失败：' + (err.message || err), 'err');
      }
    }
  }

  function applyTrace(trace, file) {
    S.trace = trace;
    S.file = file;
    S.selectedTaskIndex = -1;
    S.filter = 'all';
    S.follow = true;
    S.playing = false;
    S.tab = 'events';
    S.selectedEvent = null;
    S.visibleEvents = [];
    $('emptyState').classList.add('hidden');
    $('followBtn').textContent = '跟随 ON';
    $('followBtn').classList.add('active');
    for (const btn of document.querySelectorAll('#eventFilters .filter')) {
      btn.classList.toggle('active', btn.dataset.filter === 'all');
    }
    switchTab('events');
    updateHeader(file);
    updateSummary();
    setView(0, Math.max(1, trace.summary.cycleMax));
    const start = trace.tasks.length ? trace.tasks[0].cmdCycle : (trace.events[0] ? trace.events[0].cycle : 0);
    S.currentCycle = start;
    flow.reset(trace);
    timeline.reset(trace);
    timeline.setView(S.viewStart, S.viewEnd);
    timeline.setFocus(null);
    timeline.setCurrent(start);
    renderTasks();
    rebuildVisibleEvents(true);
    renderStats();
    syncSelection(true);
    updateReadouts();
    setPlaying(false);
  }

  function bindUI() {
    $('loadBtn').addEventListener('click', () => $('fileInput').click());
    $('welcomeLoad').addEventListener('click', () => $('fileInput').click());
    $('sampleBtn').addEventListener('click', loadSample);
    $('welcomeSample').addEventListener('click', loadSample);
    $('fileInput').addEventListener('change', (e) => {
      const file = e.target.files && e.target.files[0];
      if (file) loadFile(file);
    });

    $('taskList').addEventListener('click', (e) => {
      const card = e.target.closest('.task-card');
      if (!card) return;
      selectTask(Number(card.dataset.task));
    });

    $('playBtn').addEventListener('click', togglePlay);
    $('prevBtn').addEventListener('click', () => stepEvent(-1));
    $('nextBtn').addEventListener('click', () => stepEvent(1));
    $('startBtn').addEventListener('click', () => firstLastEvent(false));
    $('endBtn').addEventListener('click', () => firstLastEvent(true));
    $('fitBtn').addEventListener('click', fitView);
    $('speedBtn').addEventListener('click', () => {
      S.speedIndex = (S.speedIndex + 1) % SPEEDS.length;
      $('speedBtn').textContent = SPEEDS[S.speedIndex] + '×';
    });

    $('tabEvents').addEventListener('click', () => switchTab('events'));
    $('tabStats').addEventListener('click', () => switchTab('stats'));
    $('eventFilters').addEventListener('click', (e) => {
      const btn = e.target.closest('.filter');
      if (btn) onFilter(btn.dataset.filter);
    });
    $('followBtn').addEventListener('click', () => {
      S.follow = !S.follow;
      $('followBtn').textContent = S.follow ? '跟随 ON' : '跟随 OFF';
      $('followBtn').classList.toggle('active', S.follow);
      if (S.follow) scrollToSelected();
    });

    $('eventScroll').addEventListener('scroll', () => {
      if (!S.programmaticScroll && S.follow) {
        S.follow = false;
        $('followBtn').textContent = '跟随 OFF';
        $('followBtn').classList.remove('active');
      }
      renderEventRows();
    });
    $('eventRows').addEventListener('click', (e) => {
      const row = e.target.closest('.event-row');
      if (!row) return;
      const ev = S.visibleEvents[Number(row.dataset.index)];
      if (ev) seekEvent(ev);
    });

    window.addEventListener('keydown', (e) => {
      const tag = (e.target && e.target.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
      if (tag === 'button' && e.code === 'Space') return;
      if (e.code === 'Space') {
        e.preventDefault();
        togglePlay();
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        stepEvent(-1);
      } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        stepEvent(1);
      } else if (e.code === 'Home') {
        e.preventDefault();
        firstLastEvent(false);
      } else if (e.code === 'End') {
        e.preventDefault();
        firstLastEvent(true);
      }
    });

    let dragDepth = 0;
    window.addEventListener('dragenter', (e) => {
      if (!e.dataTransfer || !Array.from(e.dataTransfer.types || []).includes('Files')) return;
      e.preventDefault();
      dragDepth++;
      $('dropOverlay').classList.remove('hidden');
    });
    window.addEventListener('dragover', (e) => {
      if (e.dataTransfer && Array.from(e.dataTransfer.types || []).includes('Files')) e.preventDefault();
    });
    window.addEventListener('dragleave', (e) => {
      if (!e.dataTransfer) return;
      dragDepth = Math.max(0, dragDepth - 1);
      if (dragDepth === 0) $('dropOverlay').classList.add('hidden');
    });
    window.addEventListener('drop', (e) => {
      if (!e.dataTransfer || !e.dataTransfer.files.length) return;
      e.preventDefault();
      dragDepth = 0;
      $('dropOverlay').classList.add('hidden');
      loadFile(e.dataTransfer.files[0]);
    });

    const ro = new ResizeObserver(() => {
      flow.resize();
      timeline.resize();
    });
    ro.observe(document.querySelector('.flow-panel'));
    ro.observe(document.querySelector('.timeline-panel'));
  }

  function init() {
    bindUI();
    updateHeader(null);
    requestAnimationFrame(frame);
    if (document.fonts && document.fonts.load) {
      document.fonts.load('12px FusionPixel').catch(() => {}).finally(() => setTimeout(hideBoot, 420));
      document.fonts.ready.then(() => {
        WD.gfx.clearMetrics();
        timeline.invalidate();
      }).catch(() => {});
    } else {
      setTimeout(hideBoot, 520);
    }
    setTimeout(hideBoot, 1800);
  }

  init();
})();
