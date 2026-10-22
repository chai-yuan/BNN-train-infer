'use strict';

// Streaming VCD reader. Only direct variables of one selected scope are kept,
// so a large Verilator dump can be reduced to the top-level handshake view.
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) {
    root.WD = root.WD || {};
    Object.assign(root.WD, api);
  }
})(typeof window !== 'undefined' ? window : null, function () {
  const KNOWN_SIGNALS = [
    'clock', 'reset',
    'cmd_valid', 'cmd_ready', 'cmd_k_blocks', 'cmd_group_div', 'cmd_out_chans',
    'status_idle', 'status_busy',
    'in_valid', 'in_ready', 'act_data', 'weight_data',
    'res_valid', 'res_ready', 'res_data', 'res_last', 'res_chan_last', 'res_group_idx',
  ];
  const SCORE_SIGNALS = [
    'clock', 'reset',
    'cmd_valid', 'cmd_ready',
    'in_valid', 'in_ready',
    'res_valid', 'res_ready',
  ];
  const TIME_FACTORS = { s: 1, ms: 1e-3, us: 1e-6, ns: 1e-9, ps: 1e-12, fs: 1e-15 };

  function parseTimescale(text) {
    const m = /^\s*([0-9]+)\s*([a-z]+)\s*$/i.exec(text || '');
    if (!m) return { text: text || '', seconds: null };
    const factor = TIME_FACTORS[m[2].toLowerCase()];
    return {
      text: text.trim(),
      seconds: factor ? Number(m[1]) * factor : null,
    };
  }

  class VcdReader {
    constructor(options = {}) {
      this.options = options;
      this.phase = 'header';
      this.headerLines = [];
      this.pendingLine = '';
      this.scopes = new Map();
      this.header = null;
      this.values = [];
      this.settledValues = null;
      this.idToField = new Map();
      this.clockIndex = -1;
      this.prevClock = undefined;
      this.time = null;
      this.edgeCount = 0;
      this.firstEdgeTime = null;
      this.lastEdgeTime = null;
      this.inComment = false;
      this.scopeStack = [];
    }

    feed(text) {
      if (!text) return;
      let start = 0;
      let nl;
      while ((nl = text.indexOf('\n', start)) !== -1) {
        const line = this.pendingLine
          ? this.pendingLine + text.slice(start, nl)
          : text.slice(start, nl);
        this.pendingLine = '';
        this.processLine(line);
        start = nl + 1;
      }
      if (start < text.length) this.pendingLine += text.slice(start);
    }

    finish() {
      if (this.pendingLine) {
        this.processLine(this.pendingLine);
        this.pendingLine = '';
      }
      if (this.phase === 'header') {
        throw new Error('VCD 头部不完整：缺少 $enddefinitions');
      }
      this.finalizeTimestamp();
    }

    processLine(rawLine) {
      const line = rawLine.endsWith('\r') ? rawLine.slice(0, -1) : rawLine;
      if (this.phase === 'header') {
        if (line.trimStart().startsWith('$enddefinitions')) {
          this.headerLines.push(line);
          this.finalizeHeader();
        } else {
          this.headerLines.push(line);
        }
        return;
      }

      const s = line.trim();
      if (!s) return;

      if (this.inComment) {
        if (s.includes('$end')) this.inComment = false;
        return;
      }

      const c = s.charCodeAt(0);
      if (c === 35) { // '#'
        const tick = Number(s.slice(1));
        if (!Number.isFinite(tick)) return;
        this.finalizeTimestamp();
        this.time = tick;
        return;
      }
      if (c === 36) { // '$'
        if (s.startsWith('$comment') && !s.includes('$end')) this.inComment = true;
        return;
      }
      this.processChanges(s);
    }

    processChanges(s) {
      let p = 0;
      const n = s.length;
      while (p < n) {
        while (p < n && (s.charCodeAt(p) === 32 || s.charCodeAt(p) === 9)) p++;
        if (p >= n) break;
        const c = s.charCodeAt(p);

        if (c === 98 || c === 66) { // b/B
          const sp = this.nextSpace(s, p + 1);
          if (sp < 0) break;
          const bits = s.slice(p + 1, sp);
          p = this.skipSpaces(s, sp);
          if (p >= n) break;
          const end = this.nextSpace(s, p);
          const id = end < 0 ? s.slice(p) : s.slice(p, end);
          this.setValue(id, bits);
          p = end < 0 ? n : end;
          continue;
        }

        if (c === 114 || c === 82) { // r/R
          const sp = this.nextSpace(s, p + 1);
          if (sp < 0) break;
          const value = 'r' + s.slice(p + 1, sp);
          p = this.skipSpaces(s, sp);
          if (p >= n) break;
          const end = this.nextSpace(s, p);
          const id = end < 0 ? s.slice(p) : s.slice(p, end);
          this.setValue(id, value);
          p = end < 0 ? n : end;
          continue;
        }

        if (c === 48 || c === 49 || c === 120 || c === 88 || c === 122 || c === 90) { // 0 1 x X z Z
          const end = this.nextSpace(s, p + 1);
          const id = end < 0 ? s.slice(p + 1) : s.slice(p + 1, end);
          const scalar = c === 49 ? '1' : c === 48 ? '0' : c === 122 || c === 90 ? 'z' : 'x';
          this.setValue(id, scalar);
          p = end < 0 ? n : end;
          continue;
        }

        p = this.skipSpaces(s, p + 1);
      }
    }

    nextSpace(s, from) {
      for (let i = from; i < s.length; i++) {
        const c = s.charCodeAt(i);
        if (c === 32 || c === 9) return i;
      }
      return -1;
    }

    skipSpaces(s, from) {
      let i = from;
      while (i < s.length && (s.charCodeAt(i) === 32 || s.charCodeAt(i) === 9)) i++;
      return i;
    }

    setValue(id, value) {
      const field = this.idToField.get(id);
      if (field !== undefined) this.values[field] = value;
    }

    finalizeTimestamp() {
      const clock = this.clockIndex >= 0 ? this.values[this.clockIndex] : undefined;
      if (this.time === null) {
        this.settledValues = this.values.slice();
        if (clock !== undefined) this.prevClock = clock;
        return;
      }
      if (this.prevClock === '0' && clock === '1') {
        const cycle = this.edgeCount++;
        if (this.firstEdgeTime === null) this.firstEdgeTime = this.time;
        this.lastEdgeTime = this.time;
        // The dump at a posedge already contains post-edge flop outputs.
        // Valid/ready/payload for the transfer are the values settled before it.
        const setup = this.settledValues || this.values;
        if (this.options.onEdge) this.options.onEdge(cycle, this.time, setup);
      }
      if (clock !== undefined) this.prevClock = clock;
      this.settledValues = this.values.slice();
    }

    finalizeHeader() {
      const headerText = this.headerLines.join('\n');
      const tokens = headerText.split(/\s+/).filter(Boolean);
      const tsMatch = /\$timescale\s+([0-9]+)\s*([a-zA-Z]+)/.exec(headerText);
      const timescaleText = tsMatch ? tsMatch[1] + tsMatch[2] : '';
      let i = 0;

      while (i < tokens.length) {
        const token = tokens[i];
        if (token === '$scope') {
          const name = tokens[i + 2] || 'scope';
          this.scopeStack.push(name);
          this.ensureScope(this.scopeStack);
          i += 4;
          continue;
        }
        if (token === '$upscope') {
          this.scopeStack.pop();
          i += 2;
          continue;
        }
        if (token === '$var') {
          const width = Number(tokens[i + 2]) || 1;
          const id = tokens[i + 3];
          const name = tokens[i + 4];
          if (id && name) {
            const scope = this.ensureScope(this.scopeStack);
            scope.vars.set(name, { id, name, width });
            scope.varList.push({ id, name, width });
          }
          i += 5;
          while (i < tokens.length && tokens[i] !== '$end') i++;
          i++;
          continue;
        }
        if (token === '$timescale') {
          i += 3;
          continue;
        }
        if (token === '$enddefinitions') break;
        i++;
      }

      const candidates = [];
      for (const scope of this.scopes.values()) {
        const names = new Set(scope.varList.map((v) => v.name));
        const score = SCORE_SIGNALS.reduce((sum, name) => sum + (names.has(name) ? 1 : 0), 0);
        if (score === 0) continue;
        candidates.push({
          path: scope.path,
          module: scope.name,
          depth: scope.depth,
          score,
          missingSignals: KNOWN_SIGNALS.filter((name) => !names.has(name)),
          varCount: scope.varList.length,
          scope,
        });
      }

      const usable = candidates.filter((c) => c.missingSignals.length === 0);
      if (!usable.length) {
        const best = candidates.sort((a, b) => b.score - a.score)[0];
        const detail = best
          ? `最接近的作用域 ${best.path} 缺少信号: ${best.missingSignals.join(', ')}`
          : '没有找到包含 valid/ready 接口的作用域';
        throw new Error(`无法定位 bnn_core_top 握手接口：${detail}`);
      }

      usable.sort((a, b) => {
        const coreA = a.module === 'bnn_core_top' ? 1 : 0;
        const coreB = b.module === 'bnn_core_top' ? 1 : 0;
        if (coreA !== coreB) return coreB - coreA;
        if (a.score !== b.score) return b.score - a.score;
        return a.depth - b.depth;
      });

      const selected = usable[0];
      const scope = selected.scope;
      const fields = {};
      const missingSignals = [];

      for (const name of KNOWN_SIGNALS) {
        const variable = scope.vars.get(name);
        if (!variable) {
          missingSignals.push(name);
          continue;
        }
        const index = this.values.length;
        this.values.push(undefined);
        fields[name] = { index, width: variable.width, id: variable.id };
        if (!this.idToField.has(variable.id)) this.idToField.set(variable.id, index);
      }

      if (missingSignals.length) {
        throw new Error('选中的顶层作用域缺少必需信号: ' + missingSignals.join(', '));
      }

      this.clockIndex = fields.clock.index;
      this.header = {
        scopePath: selected.path,
        moduleName: selected.module,
        candidates: candidates.map((c) => ({
          path: c.path,
          module: c.module,
          depth: c.depth,
          score: c.score,
          varCount: c.varCount,
        })),
        fields,
        missingSignals,
        timescale: parseTimescale(timescaleText),
        varCount: scope.varList.length,
      };
      this.phase = 'body';
      if (this.options.onHeader) this.options.onHeader(this.header);
    }

    ensureScope(stack) {
      const path = stack.join('/') || 'TOP';
      let scope = this.scopes.get(path);
      if (!scope) {
        scope = {
          path,
          name: stack[stack.length - 1] || 'TOP',
          depth: Math.max(0, stack.length - 1),
          vars: new Map(),
          varList: [],
        };
        this.scopes.set(path, scope);
      }
      return scope;
    }
  }

  async function streamFile(file, reader, options = {}) {
    if (!file.stream) throw new Error('当前浏览器不支持流式读取文件');
    const streamReader = file.stream().getReader();
    const decoder = new TextDecoder('utf-8');
    let loaded = 0;
    let lastReport = 0;
    let lastYield = Date.now();
    const total = file.size || 0;

    if (options.onProgress) options.onProgress({ loaded: 0, total, phase: 'start' });

    for (;;) {
      const { done, value } = await streamReader.read();
      if (done) break;
      loaded += value.byteLength;
      reader.feed(decoder.decode(value, { stream: true }));

      const now = Date.now();
      if (loaded - lastReport >= 4 * 1024 * 1024 || now - lastYield >= 120) {
        if (options.onProgress) options.onProgress({ loaded, total, phase: 'parse' });
        lastReport = loaded;
        lastYield = now;
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }
    reader.feed(decoder.decode());
    reader.finish();
    if (options.onProgress) options.onProgress({ loaded: total, total, phase: 'done' });
  }

  return {
    VcdReader,
    streamFile,
    parseTimescale,
    KNOWN_SIGNALS,
  };
});
