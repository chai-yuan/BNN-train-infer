'use strict';

// Rebuilds transaction-level data flow from the top-level handshake ports.
// No internal signal or per-cycle value is retained.
(function (root, factory) {
  const api = factory(typeof module === 'object' && module.exports ? require('./vcd.js') : (root && root.WD));
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) {
    root.WD = root.WD || {};
    Object.assign(root.WD, api);
  }
})(typeof window !== 'undefined' ? window : null, function (Vcd) {
  const STATUS = {
    0: { key: 'ok', zh: '成功', short: 'OK' },
  };

  function unsigned(raw, width) {
    if (raw === undefined || raw === null) return null;
    if (raw[0] === 'r' || raw[0] === 'R') {
      const n = Number(raw.slice(1));
      return Number.isFinite(n) ? Math.trunc(n) : null;
    }
    let bits = raw;
    if (bits[0] === 'b' || bits[0] === 'B') bits = bits.slice(1);
    if (!/^[01]+$/.test(bits)) return null;
    const value = parseInt(bits, 2);
    if (width && width < 32 && bits.length < width) return value;
    return value;
  }

  function signed(raw, width) {
    const value = unsigned(raw, width);
    if (value === null || !width || width > 52) return value;
    const limit = 2 ** (width - 1);
    return value >= limit ? value - 2 ** width : value;
  }

  function hex(value, width) {
    if (value === null || value === undefined) return 'x';
    const w = width || 32;
    const digits = Math.max(1, Math.ceil(w / 4));
    let h = (BigInt(value) & ((1n << BigInt(w)) - 1n)).toString(16).toUpperCase();
    if (h.length < digits) h = h.padStart(digits, '0');
    return h;
  }

  function fmtNum(n) {
    if (n === null || n === undefined) return 'x';
    return String(n);
  }

  class HandshakeAnalyzer {
    constructor(options = {}) {
      this.options = options;
      this.maxEvents = options.maxEvents || 500000;
      this.trace = null;
      this.header = null;
      this.ix = {};
      this.widths = {};
      this.activeTask = null;
      this.prevReset = null;
      this.pendingStalls = new Map();
      this.busy = false;
      this.busyStart = null;
      this.prevEdgeTime = null;
      this.actMap = new Map();
      this.warnKeys = new Set();
      this.values = [];
    }

    init(header) {
      this.header = header;
      const ix = {};
      const widths = {};
      for (const [name, meta] of Object.entries(header.fields)) {
        ix[name] = meta.index;
        widths[name] = meta.width;
      }
      this.ix = ix;
      this.widths = widths;
      this.trace = {
        file: this.options.file || null,
        header,
        tasks: [],
        events: [],
        markers: [],
        warnings: [],
        warningOverflow: false,
        actVectors: [],
        busyRuns: [],
        summary: {
          edges: 0,
          cycleMax: -1,
          clockPeriodTicks: null,
          firstEdgeTime: null,
          lastEdgeTime: null,
          tasks: 0,
          ok: 0,
          failed: 0,
          canceled: 0,
          completed: 0,
          unfinished: 0,
          cmdEvents: 0,
          inEvents: 0,
          inHandshakes: 0,
          actHandshakes: 0,
          weightHandshakes: 0,
          resEvents: 0,
          doneEvents: 0,
          inputBeats: 0,
          resultBeats: 0,
          busyCycles: 0,
          resetCycles: 0,
          droppedEvents: 0,
          stalls: { cmd: 0, in: 0, act: 0, weight: 0, res: 0, done: 0 },
        },
        completionSource: 'res_valid && res_ready && res_last',
        actWidth: widths.act_data || 256,
        weightWidth: widths.weight_data || 32,
        resWidth: widths.res_data || 32,
      };
      if (header.missingSignals.length) {
        this.addWarning(
          '顶层缺少信号，部分视图将不可用: ' + header.missingSignals.join(', '),
          null,
          'missing'
        );
      }
    }

    addWarning(text, cycle, key) {
      if (key) {
        if (this.warnKeys.has(key)) return;
        this.warnKeys.add(key);
      }
      if (this.trace.warnings.length >= 200) {
        this.trace.warningOverflow = true;
        return;
      }
      this.trace.warnings.push({ text, cycle, taskId: null });
    }

    taskIssue(task, text, cycle) {
      if (!task) return;
      const key = 't' + task.id + ':' + text;
      if (this.warnKeys.has(key)) return;
      this.warnKeys.add(key);
      task.issues.push({ text, cycle });
      if (this.trace.warnings.length < 200) {
        this.trace.warnings.push({ text: '任务 #' + String(task.id + 1).padStart(2, '0') + ' ' + text, cycle, taskId: task.id });
      } else {
        this.trace.warningOverflow = true;
      }
    }

    marker(type, cycle, time, label) {
      this.trace.markers.push({ type, cycle, time, label });
    }

    canKeep(type) {
      return type === 'cmd' || type === 'done' || this.trace.events.length < this.maxEvents;
    }

    pushEvent(event, task) {
      this.trace.events.push(event);
      if (task) task.events.push(event);
    }

    dropEvent() {
      this.trace.summary.droppedEvents++;
      if (this.activeTask) this.activeTask.droppedEvents++;
    }

    internAct(raw) {
      if (raw === undefined || raw === null) return null;
      let bits = raw;
      if (bits[0] === 'b' || bits[0] === 'B') bits = bits.slice(1);
      if (!/^[01]+$/.test(bits)) return -1;
      let index = this.actMap.get(bits);
      if (index === undefined) {
        index = this.trace.actVectors.length;
        this.trace.actVectors.push(bits);
        this.actMap.set(bits, index);
      }
      return index;
    }

    checkStability(cycle, raw) {
      const channels = {
        cmd: ['cmd_k_blocks', 'cmd_group_div', 'cmd_out_chans'],
        in: ['act_data', 'weight_data'],
        res: ['res_data', 'res_last', 'res_chan_last', 'res_group_idx'],
      };
      for (const [channel, payload] of Object.entries(channels)) {
        // Normalize short/zero-padded VCD binary encodings before comparing.
        const snapshot = [channel + '_valid', ...payload].map((name) => {
          const value = raw(name);
          return value === undefined ? null : value.padStart(this.widths[name], /^[xz]/i.test(value) ? value[0] : '0');
        });
        const previous = this.pendingStalls.get(channel);
        if (previous && snapshot.some((value, i) => value !== previous[i])) {
          const text = channel + ' 反压期间 valid/负载未保持稳定';
          if (this.activeTask) this.taskIssue(this.activeTask, text, cycle);
          else this.addWarning(text, cycle, 'stability-' + channel);
        }
        if (raw(channel + '_valid') === '1' && raw(channel + '_ready') !== '1') {
          this.pendingStalls.set(channel, snapshot);
        } else {
          this.pendingStalls.delete(channel);
        }
      }
    }

    onEdge(cycle, time, values) {
      const s = this.trace.summary;
      this.values = values;
      s.edges = cycle + 1;
      s.cycleMax = cycle;
      if (s.firstEdgeTime === null) s.firstEdgeTime = time;
      s.lastEdgeTime = time;
      if (this.prevEdgeTime !== null && s.clockPeriodTicks === null) {
        const delta = time - this.prevEdgeTime;
        if (delta > 0) s.clockPeriodTicks = delta;
      }
      this.prevEdgeTime = time;

      const v = values;
      const ix = this.ix;
      const raw = (name) => (ix[name] === undefined ? undefined : v[ix[name]]);
      const one = (name) => raw(name) === '1';

      const reset = one('reset');
      if (reset) {
        s.resetCycles++;
        this.pendingStalls.clear();
        if (this.prevReset !== true) {
          this.marker('reset', cycle, time, '复位');
          if (this.activeTask) {
            const task = this.activeTask;
            this.taskIssue(task, '任务被复位取消（无完成通知）', cycle);
            this.closeTask(task, cycle - 1, 'reset');
          }
        }
        if (this.busy) {
          this.trace.busyRuns.push({ start: this.busyStart, end: cycle - 1, reset: true });
          this.busy = false;
          this.busyStart = null;
        }
        this.prevReset = true;
        return;
      }
      if (this.prevReset === true) this.marker('reset_release', cycle, time, '复位释放');
      this.prevReset = false;

      this.checkStability(cycle, raw);

      const busy = one('status_busy');
      if (busy) {
        s.busyCycles++;
        if (!this.busy) {
          this.busy = true;
          this.busyStart = cycle;
        }
        if (this.activeTask) this.activeTask.busyCycles++;
      } else if (this.busy) {
        this.trace.busyRuns.push({ start: this.busyStart, end: cycle - 1, reset: false });
        this.busy = false;
        this.busyStart = null;
      }

      if (one('cmd_valid') && !one('cmd_ready')) s.stalls.cmd++;
      if (one('in_valid') && !one('in_ready')) {
        s.stalls.in++;
        s.stalls.act++;
        s.stalls.weight++;
        if (this.activeTask) {
          this.activeTask.stalls.in++;
          this.activeTask.stalls.act++;
          this.activeTask.stalls.weight++;
        }
      }
      if (one('res_valid') && !one('res_ready')) {
        s.stalls.res++;
        if (this.activeTask) {
          this.activeTask.stalls.res++;
          if (one('res_last')) this.activeTask.doneWaitCycles++;
        }
      }

      const cmdH = one('cmd_valid') && one('cmd_ready');
      const inH = one('in_valid') && one('in_ready');
      const resH = one('res_valid') && one('res_ready');
      // A new command is forbidden on the final-result edge, even though that
      // edge closes the old task. Attribute its result to the old task first.
      const overlappingCommand = cmdH && !!this.activeTask;
      if (overlappingCommand) {
        this.taskIssue(this.activeTask, '新命令在上一任务完成前或末结果同沿被接收', cycle);
      }
      if (cmdH && !overlappingCommand) this.handleCommand(cycle, time, raw);
      if (inH) {
        if (this.activeTask && this.activeTask.cmdCycle === cycle) {
          this.taskIssue(this.activeTask, '首对输入不得与命令同沿接收', cycle);
        }
        this.handleInput(cycle, time, raw);
      }
      if (resH) this.handleResult(cycle, time, raw);
      if (resH && one('res_last')) this.handleDone(cycle, time);
      if (overlappingCommand) this.handleCommand(cycle, time, raw);
    }

    handleCommand(cycle, time, raw) {
      const s = this.trace.summary;
      s.cmdEvents++;
      if (this.activeTask) {
        const old = this.activeTask;
        this.taskIssue(old, '新命令在上一任务完成前被接收', cycle);
        this.closeTask(old, cycle - 1, 'superseded');
      }

      const kBlocks = unsigned(raw('cmd_k_blocks'), this.widths.cmd_k_blocks);
      const groupDiv = unsigned(raw('cmd_group_div'), this.widths.cmd_group_div);
      const outChans = unsigned(raw('cmd_out_chans'), this.widths.cmd_out_chans);
      const k = kBlocks === null ? null : kBlocks * 32;
      const groups = groupDiv ? Math.floor((kBlocks || 0) / groupDiv) : null;
      const n = outChans === null ? null : outChans;
      const legal = !!(kBlocks > 0 && groupDiv > 0 && outChans > 0 &&
        kBlocks % groupDiv === 0 && (kBlocks / groupDiv) <= 256);

      const task = {
        id: this.trace.tasks.length,
        index: this.trace.tasks.length,
        cmdCycle: cycle,
        cmdTime: time,
        endCycle: null,
        doneCycle: null,
        completionSource: null,
        status: 'running',
        statusCode: null,
        statusText: '运行中',
        doneStatusRaw: null,
        params: { kBlocks, groupDiv, outChans },
        k,
        beatsPerChannel: kBlocks,
        groupBeats: groupDiv,
        groupSize: groupDiv === null ? null : groupDiv * 32,
        groups,
        n,
        legal,
        expectedInput: kBlocks !== null && n !== null ? kBlocks * n : null,
        expectedResults: groups !== null && n !== null ? groups * n : null,
        inputBeats: 0,
        resultBeats: 0,
        inputChannels: 0,
        resultChannels: 0,
        resChannel: 0,
        resGroupInChan: 0,
        firstInputCycle: null,
        lastInputCycle: null,
        firstResultCycle: null,
        lastResultCycle: null,
        busyCycles: 0,
        doneWaitCycles: 0,
        droppedEvents: 0,
        stalls: { in: 0, act: 0, weight: 0, res: 0 },
        issues: [],
        events: [],
        verified: false,
      };
      this.trace.tasks.push(task);
      s.tasks = this.trace.tasks.length;
      this.activeTask = task;

      const event = {
        type: 'cmd',
        cycle,
        time,
        taskId: task.id,
        inSeq: 0,
        resSeq: 0,
        params: { kBlocks, groupDiv, outChans },
        k,
        groups,
        n,
        legal,
        inChannel: null,
        resChannel: null,
      };
      this.pushEvent(event, task);
      if (!legal) {
        this.taskIssue(task, '命令尺寸不满足顶层合法条件', cycle);
      }
    }

    handleInput(cycle, time, raw) {
      const s = this.trace.summary;
      s.inEvents++;
      s.inHandshakes++;
      s.actHandshakes++;
      s.weightHandshakes++;
      const task = this.activeTask;
      let channelLast = null;
      let channel = null;
      let beat = null;

      if (!task) {
        this.addWarning('握手事件不属于任何命令任务（输入）', cycle, 'orphan-in');
      } else {
        const before = task.inputBeats;
        task.inputBeats++;
        s.inputBeats++;
        if (task.firstInputCycle === null) task.firstInputCycle = cycle;
        task.lastInputCycle = cycle;
        if (task.beatsPerChannel > 0) {
          channel = Math.floor(before / task.beatsPerChannel);
          beat = before % task.beatsPerChannel;
          task.inputChannels = Math.max(task.inputChannels, channel + 1);
          channelLast = beat === task.beatsPerChannel - 1;
          if (task.legal && task.expectedInput !== null && task.inputBeats > task.expectedInput) {
            this.taskIssue(task, '输入握手数超过 N×k_blocks', cycle);
          }
        }
        if (task.inputBeats === 1) task.firstInputCycle = cycle;
      }

      if (!this.canKeep('in')) {
        this.dropEvent();
        return;
      }
      const event = {
        type: 'in',
        cycle,
        time,
        taskId: task ? task.id : null,
        inSeq: task ? task.inputBeats : null,
        resSeq: task ? task.resultBeats : null,
        channel,
        beat,
        k: task ? task.k : null,
        beatsPerChannel: task ? task.beatsPerChannel : null,
        inH: true,
        actH: true,
        wgtH: true,
        channelLast,
        actLast: channelLast,
        weightLast: channelLast,
        boundarySource: 'cmd_k_blocks + input count',
        actIdx: this.internAct(raw('act_data')),
        weight: unsigned(raw('weight_data'), this.widths.weight_data),
      };
      this.pushEvent(event, task);
    }

    handleResult(cycle, time, raw) {
      const s = this.trace.summary;
      s.resEvents++;
      const task = this.activeTask;
      const groupIdx = unsigned(raw('res_group_idx'), this.widths.res_group_idx);
      const data = signed(raw('res_data'), this.widths.res_data);
      const chanLast = raw('res_chan_last') === '1';
      const last = raw('res_last') === '1';
      let channel = null;

      if (!task) {
        this.addWarning('握手事件不属于任何命令任务（结果）', cycle, 'orphan-res');
      } else {
        channel = task.resChannel;
        task.resultBeats++;
        s.resultBeats++;
        if (task.firstResultCycle === null) task.firstResultCycle = cycle;
        task.lastResultCycle = cycle;
        if (groupIdx !== null && groupIdx !== task.resGroupInChan) {
          this.taskIssue(task, '结果组号不连续（期望 ' + task.resGroupInChan + '，实际 ' + groupIdx + '）', cycle);
        }
        if (task.groups !== null && task.groups > 0) {
          const expectedChanLast = task.resGroupInChan === task.groups - 1;
          if (chanLast !== expectedChanLast) {
            this.taskIssue(task, 'res_chan_last 与组边界不一致（通道 ' + channel + '，组 ' + groupIdx + '）', cycle);
          }
        }
        if (task.expectedResults !== null && last !== (task.resultBeats === task.expectedResults)) {
          this.taskIssue(task, 'res_last 与任务末结果边界不一致', cycle);
        }
        if (last && !chanLast) this.taskIssue(task, 'res_last 置位但 res_chan_last 为 0', cycle);
        if (task.expectedResults !== null && task.resultBeats > task.expectedResults) {
          this.taskIssue(task, '结果握手数超过 N×groups', cycle);
        }
        if (chanLast) {
          task.resultChannels++;
          task.resChannel++;
          task.resGroupInChan = 0;
        } else {
          task.resGroupInChan++;
        }
      }

      if (!this.canKeep('res')) {
        this.dropEvent();
        return;
      }
      const event = {
        type: 'res',
        cycle,
        time,
        taskId: task ? task.id : null,
        inSeq: task ? task.inputBeats : null,
        resSeq: task ? task.resultBeats : null,
        channel,
        group: groupIdx,
        data,
        dataHex: hex(data, this.widths.res_data || 32),
        chanLast,
        last,
      };
      this.pushEvent(event, task);
    }

    handleDone(cycle, time) {
      // UI compatibility event: derived from the accepted final RES, never a
      // physical DONE channel. Keep it even when result details hit the cap.
      const task = this.activeTask;
      if (!task) return;
      this.trace.summary.doneEvents++;
      this.pushEvent({
        type: 'done',
        derived: true,
        source: this.trace.completionSource,
        cycle,
        time,
        taskId: task.id,
        inSeq: task.inputBeats,
        resSeq: task.resultBeats,
        status: 0,
        statusText: '成功（末结果握手推导）',
        inChannel: task.inputChannels,
        resChannel: task.resultChannels,
      }, task);
      task.statusCode = 0;
      task.statusKey = 'ok';
      task.doneStatusRaw = null;
      task.doneCycle = cycle;
      task.completionSource = this.trace.completionSource;
      this.closeTask(task, cycle, 'ok');
    }

    closeTask(task, endCycle, statusKey) {
      if (!task || task.status !== 'running') return;
      task.endCycle = Math.max(task.cmdCycle, endCycle);
      task.status = statusKey;
      if (statusKey === 'ok') task.statusText = '成功';
      else if (statusKey === 'reset') task.statusText = '复位中断';
      else if (statusKey === 'superseded') task.statusText = '被新任务覆盖';
      else if (statusKey === 'unfinished') task.statusText = '未完成';
      this.activeTask = null;
    }

    finalizeTask(task) {
      const s = this.trace.summary;
      if (task.endCycle === null) task.endCycle = s.cycleMax;
      task.duration = Math.max(1, task.endCycle - task.cmdCycle + 1);
      task.inputSpan = task.firstInputCycle === null ? 0 : task.lastInputCycle - task.firstInputCycle + 1;
      task.resultSpan = task.firstResultCycle === null ? 0 : task.lastResultCycle - task.firstResultCycle + 1;
      task.firstInputOffset = task.firstInputCycle === null ? null : task.firstInputCycle - task.cmdCycle;
      task.firstResultOffset = task.firstResultCycle === null ? null : task.firstResultCycle - task.cmdCycle;
      task.drainLag = task.lastResultCycle === null ? null : task.endCycle - task.lastResultCycle;
      task.inDensity = task.inputSpan ? task.inputBeats / task.inputSpan : 0;
      task.resDensity = task.resultSpan ? task.resultBeats / task.resultSpan : 0;
      task.stallTotal = task.stalls.in + task.stalls.res;

      if (task.status === 'ok' && task.legal) {
        if (task.expectedInput !== null && task.inputBeats !== task.expectedInput) {
          this.taskIssue(task, '成功完成但输入握手数为 ' + task.inputBeats + '，期望 ' + task.expectedInput, task.endCycle);
        }
        if (task.expectedResults !== null && task.resultBeats !== task.expectedResults) {
          this.taskIssue(task, '成功完成但结果握手数为 ' + task.resultBeats + '，期望 ' + task.expectedResults, task.endCycle);
        }
      }
      task.verified = task.status === 'ok' && task.issues.length === 0;
    }

    finish() {
      if (!this.trace) throw new Error('VCD 头部尚未完成解析');
      if (this.activeTask) {
        const task = this.activeTask;
        this.taskIssue(task, '波形结束时任务仍在运行', this.trace.summary.cycleMax);
        this.closeTask(task, this.trace.summary.cycleMax, 'unfinished');
        this.activeTask = null;
      }
      if (this.busy) {
        this.trace.busyRuns.push({ start: this.busyStart, end: this.trace.summary.cycleMax, reset: false });
        this.busy = false;
      }

      const s = this.trace.summary;
      for (const task of this.trace.tasks) this.finalizeTask(task);
      for (const task of this.trace.tasks) {
        if (task.status === 'ok') {
          s.completed++;
          if (task.verified) s.ok++;
          else s.failed++;
        }
        else if (task.status === 'reset') s.canceled++;
        else if (task.status === 'unfinished') s.unfinished++;
        else s.failed++;
      }
      s.nonResetCycles = Math.max(0, s.edges - s.resetCycles);
      s.busyRatio = s.nonResetCycles ? s.busyCycles / s.nonResetCycles : 0;
      s.inSuccess = s.tasks ? s.ok / s.tasks : 0;
      s.lastCycle = Math.max(0, s.edges - 1);
      s.clockPeriodSec = s.clockPeriodTicks !== null && this.header.timescale.seconds !== null
        ? s.clockPeriodTicks * this.header.timescale.seconds
        : null;
      s.traceSeconds = s.clockPeriodSec !== null ? s.edges * s.clockPeriodSec : null;

      if (s.droppedEvents) {
        this.addWarning('事件数量超过上限，已省略 ' + s.droppedEvents + ' 条输入/结果细节', null, 'cap');
      }
      if (!s.edges) throw new Error('波形中没有检测到 clock 上升沿');
      return this.trace;
    }
  }

  function attach(reader, options = {}) {
    const analyzer = new HandshakeAnalyzer(options);
    reader.options.onHeader = (header) => analyzer.init(header);
    reader.options.onEdge = (cycle, time, values) => analyzer.onEdge(cycle, time, values);
    return analyzer;
  }

  async function analyzeVcd(file, options = {}) {
    if (!Vcd || !Vcd.VcdReader) throw new Error('VCD 解析器未加载');
    const reader = new Vcd.VcdReader({});
    const analyzer = attach(reader, { ...options, file: { name: file.name, size: file.size } });
    await Vcd.streamFile(file, reader, options);
    return analyzer.finish();
  }

  function analyzeText(text, options = {}) {
    if (!Vcd || !Vcd.VcdReader) throw new Error('VCD 解析器未加载');
    const reader = new Vcd.VcdReader({});
    const analyzer = attach(reader, options);
    reader.feed(text);
    reader.finish();
    return analyzer.finish();
  }

  function decodeActVector(trace, index) {
    if (index === null || index === undefined || index < 0) return null;
    const bits = trace.actVectors[index];
    if (bits === undefined) return null;
    const width = trace.actWidth || bits.length;
    const lanes = Math.ceil(width / 8);
    const value = BigInt('0b' + bits);
    const out = new Array(lanes);
    for (let i = 0; i < lanes; i++) {
      let byte = Number((value >> BigInt(i * 8)) & 255n);
      if (byte >= 128) byte -= 256;
      out[i] = byte;
    }
    return out;
  }

  function actBitsLabel(trace, index) {
    if (index === null || index === undefined) return 'x';
    if (index < 0) return 'x';
    const bits = trace.actVectors[index];
    if (bits === undefined) return 'x';
    return bits;
  }

  return {
    HandshakeAnalyzer,
    attach,
    analyzeVcd,
    analyzeText,
    decodeActVector,
    actBitsLabel,
    STATUS,
    hex,
    fmtNum,
  };
});
