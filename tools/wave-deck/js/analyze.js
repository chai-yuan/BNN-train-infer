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
    1: { key: 'cfg', zh: '配置错误', short: 'CFG' },
    2: { key: 'aborted', zh: '已中止', short: 'ABRT' },
    3: { key: 'protocol', zh: '输入协议错误', short: 'PROTO' },
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
      this.prevAbort = false;
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
          unfinished: 0,
          cmdEvents: 0,
          inEvents: 0,
          actHandshakes: 0,
          weightHandshakes: 0,
          resEvents: 0,
          doneEvents: 0,
          inputBeats: 0,
          resultBeats: 0,
          busyCycles: 0,
          resetCycles: 0,
          droppedEvents: 0,
          stalls: { cmd: 0, act: 0, weight: 0, res: 0, done: 0 },
        },
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
        if (this.prevReset !== true) {
          this.marker('reset', cycle, time, '复位');
          if (this.activeTask) {
            const task = this.activeTask;
            this.taskIssue(task, '任务被复位中断，未收到完成握手', cycle);
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

      const abort = one('ctrl_abort');
      if (abort && !this.prevAbort) {
        this.marker('abort', cycle, time, '中止');
        if (this.activeTask) {
          this.activeTask.abortCycle = cycle;
          this.taskIssue(this.activeTask, '运行中收到 ctrl_abort', cycle);
        }
      }
      this.prevAbort = abort;

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
      if (one('act_valid') && !one('act_ready')) { s.stalls.act++; if (this.activeTask) this.activeTask.stalls.act++; }
      if (one('weight_valid') && !one('weight_ready')) { s.stalls.weight++; if (this.activeTask) this.activeTask.stalls.weight++; }
      if (one('res_valid') && !one('res_ready')) { s.stalls.res++; if (this.activeTask) this.activeTask.stalls.res++; }
      if (one('done_valid') && !one('done_ready')) {
        s.stalls.done++;
        if (this.activeTask) this.activeTask.doneWaitCycles++;
      }

      const cmdH = one('cmd_valid') && one('cmd_ready');
      const actH = one('act_valid') && one('act_ready');
      const wgtH = one('weight_valid') && one('weight_ready');
      const resH = one('res_valid') && one('res_ready');
      const doneH = one('done_valid') && one('done_ready');

      if (cmdH) this.handleCommand(cycle, time, raw);
      if (actH || wgtH) this.handleInput(cycle, time, raw, actH, wgtH);
      if (resH) this.handleResult(cycle, time, raw);
      if (doneH) this.handleDone(cycle, time, raw);
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
        stalls: { act: 0, weight: 0, res: 0 },
        issues: [],
        events: [],
        verified: false,
        abortCycle: null,
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

    handleInput(cycle, time, raw, actH, wgtH) {
      const s = this.trace.summary;
      s.inEvents++;
      if (actH) s.actHandshakes++;
      if (wgtH) s.weightHandshakes++;
      const task = this.activeTask;
      const actLast = raw('act_last') === '1';
      const weightLast = raw('weight_last') === '1';
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
          const expectedLast = beat === task.beatsPerChannel - 1;
          if (actLast !== weightLast) {
            this.taskIssue(task, 'act_last 与 weight_last 不一致', cycle);
          }
          if (expectedLast !== actLast || expectedLast !== weightLast) {
            this.taskIssue(task, '输入 last 与通道边界不一致（通道 ' + channel + '，拍 ' + beat + '）', cycle);
          }
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
        actH,
        wgtH,
        actLast,
        weightLast,
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

    handleDone(cycle, time, raw) {
      const s = this.trace.summary;
      s.doneEvents++;
      const code = unsigned(raw('done_status'), this.widths.done_status);
      const status = STATUS[code] || { key: 'unknown', zh: '未知状态', short: '?' };
      const task = this.activeTask;

      if (!task) {
        this.addWarning('收到无对应任务的完成握手', cycle, 'orphan-done');
      }

      if (this.canKeep('done')) {
        const event = {
          type: 'done',
          cycle,
          time,
          taskId: task ? task.id : null,
          inSeq: task ? task.inputBeats : null,
          resSeq: task ? task.resultBeats : null,
          status: code,
          statusText: status.zh,
          inChannel: task ? task.inputChannels : null,
          resChannel: task ? task.resultChannels : null,
        };
        this.pushEvent(event, task);
      }

      if (task) {
        task.statusCode = code;
        task.statusKey = status.key;
        task.statusText = status.zh;
        task.doneStatusRaw = raw('done_status');
        task.doneCycle = cycle;
        this.closeTask(task, cycle, status.key);
        this.activeTask = null;
      }
    }

    closeTask(task, endCycle, statusKey) {
      if (!task || task.status !== 'running') return;
      task.endCycle = Math.max(task.cmdCycle, endCycle);
      task.status = statusKey;
      if (statusKey === 'ok') task.statusText = '成功';
      else if (statusKey === 'cfg') task.statusText = '配置错误';
      else if (statusKey === 'aborted') task.statusText = '已中止';
      else if (statusKey === 'protocol') task.statusText = '协议错误';
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
      task.stallTotal = task.stalls.act + task.stalls.weight + task.stalls.res;

      if (task.status === 'ok' || task.status === 'aborted' || task.status === 'protocol') {
        if (task.status === 'ok' && task.legal) {
          if (task.expectedInput !== null && task.inputBeats !== task.expectedInput) {
            this.taskIssue(task, '成功完成但输入握手数为 ' + task.inputBeats + '，期望 ' + task.expectedInput, task.endCycle);
          }
          if (task.expectedResults !== null && task.resultBeats !== task.expectedResults) {
            this.taskIssue(task, '成功完成但结果握手数为 ' + task.resultBeats + '，期望 ' + task.expectedResults, task.endCycle);
          }
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
        if (task.status === 'ok' && task.verified) s.ok++;
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
