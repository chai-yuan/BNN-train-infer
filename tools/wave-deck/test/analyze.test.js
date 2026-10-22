'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const { VcdReader, KNOWN_SIGNALS } = require('../js/vcd.js');
const { attach, decodeActVector, analyzeText } = require('../js/analyze.js');

function parseChunked(text, chunkSize) {
  const reader = new VcdReader({});
  const analyzer = attach(reader);
  for (let i = 0; i < text.length; i += chunkSize) reader.feed(text.slice(i, i + chunkSize));
  reader.finish();
  return analyzer.finish();
}

const WIDTHS = {
  clock: 1, reset: 1,
  cmd_valid: 1, cmd_ready: 1, cmd_k_blocks: 16, cmd_group_div: 8, cmd_out_chans: 16,
  status_idle: 1, status_busy: 1,
  in_valid: 1, in_ready: 1, act_data: 256, weight_data: 32,
  res_valid: 1, res_ready: 1, res_data: 32, res_last: 1, res_chan_last: 1, res_group_idx: 8,
};

// Each frame is the stable setup for an edge. Post-edge changes deliberately
// differ, exercising the reader's pre-edge rather than settled-edge sampling.
function syntheticVcd(frames) {
  const names = Object.keys(WIDTHS);
  const ids = Object.fromEntries(names.map((name, i) => [name, 's' + i]));
  const encode = (name, value) => {
    const width = WIDTHS[name];
    if (width === 1) return String(value) + ids[name];
    return 'b' + BigInt.asUintN(width, BigInt(value)).toString(2).padStart(width, '0') + ' ' + ids[name];
  };
  const lines = ['$timescale 1ps $end', '$scope module TOP $end'];
  for (const name of names) lines.push(`$var wire ${WIDTHS[name]} ${ids[name]} ${name} $end`);
  lines.push('$upscope $end', '$enddefinitions $end');
  const state = Object.fromEntries(names.map((name) => [name, 0]));
  for (let i = 0; i < frames.length; i++) {
    Object.assign(state, frames[i], { clock: 0 });
    lines.push('#' + i * 10000);
    for (const name of names) lines.push(encode(name, state[name]));
    lines.push('#' + (i * 10000 + 5000), encode('clock', 1));
    // Outputs change after the edge; these must not change that edge's transfer.
    lines.push(encode('cmd_ready', 0), encode('in_ready', 0), encode('res_valid', 0));
  }
  return lines.join('\n') + '\n';
}

const command = (kBlocks = 1, groupDiv = 1, outChans = 1) => ({
  cmd_valid: 1, cmd_ready: 1, cmd_k_blocks: kBlocks,
  cmd_group_div: groupDiv, cmd_out_chans: outChans, status_idle: 1, status_busy: 0,
  in_valid: 0, in_ready: 0, res_valid: 0, res_last: 0,
});
const input = (extra = {}) => ({
  cmd_valid: 0, cmd_ready: 0, status_idle: 0, status_busy: 1,
  in_valid: 1, in_ready: 1, res_valid: 0, act_data: 128, weight_data: 0xffffffff,
  ...extra,
});
const result = (extra = {}) => ({
  in_valid: 0, res_valid: 1, res_ready: 1,
  res_data: -128, res_last: 1, res_chan_last: 1, res_group_idx: 0,
  ...extra,
});

function legalFrames() {
  return [
    { reset: 1, ...command(), in_valid: 1, in_ready: 1, ...result() }, // reset masks everything
    { reset: 0, ...command(2, 1, 2) },
    input({ in_ready: 0 }),
    input(),
    input({ act_data: 129 }),
    result({ res_last: 0, res_chan_last: 0 }),
    result({ res_last: 0, res_group_idx: 1 }),
    input(),
    input({ act_data: 129 }),
    result({ res_last: 0, res_chan_last: 0 }),
    result({ res_group_idx: 1, res_ready: 0, cmd_valid: 1, cmd_ready: 0,
      cmd_k_blocks: 1, cmd_group_div: 1, cmd_out_chans: 1 }),
    {}, // hold final result and command through another blocked edge
    { res_ready: 1 }, // final RES accepted, CMD still blocked
    command(), // next edge may accept CMD
    input(),
    result(),
  ];
}

test('real new-RTL demo streams in odd chunks and has two completed tasks', () => {
  const text = fs.readFileSync(path.join(__dirname, '..', 'samples', 'demo.vcd'), 'utf8');
  const trace = parseChunked(text, 777);
  assert.equal(trace.header.scopePath, 'TOP/bnn_core_top');
  assert.equal(trace.header.timescale.text, '1ps');
  assert.deepEqual(trace.header.missingSignals, []);
  for (const removed of ['ctrl_abort', 'done_valid', 'act_valid', 'weight_valid', 'act_last', 'weight_last', 'res_error']) {
    assert.equal(trace.header.fields[removed], undefined);
    assert.ok(!text.includes(' ' + removed + ' '));
  }
  assert.equal(trace.summary.edges, 28);
  assert.equal(trace.summary.cmdEvents, 2);
  assert.equal(trace.summary.inEvents, 16);
  assert.equal(trace.summary.inHandshakes, 16);
  assert.equal(trace.summary.actHandshakes, 16);
  assert.equal(trace.summary.weightHandshakes, 16);
  assert.equal(trace.summary.resEvents, 4);
  assert.equal(trace.summary.doneEvents, 2);
  assert.equal(trace.summary.completed, 2);
  assert.equal(trace.summary.ok, 2);
  assert.equal(trace.summary.failed, 0);
  for (const task of trace.tasks) {
    assert.deepEqual(task.params, { kBlocks: 4, groupDiv: 4, outChans: 2 });
    assert.equal(task.expectedInput, 8);
    assert.equal(task.expectedResults, 2);
    assert.equal(task.inputBeats, 8);
    assert.equal(task.resultBeats, 2);
    assert.equal(task.inputChannels, 2);
    assert.equal(task.resultChannels, 2);
    assert.equal(task.status, 'ok');
    assert.equal(task.verified, true);
    assert.equal(task.doneCycle, task.lastResultCycle);
    assert.equal(task.duration, 13);
    assert.equal(task.doneStatusRaw, null);
    assert.equal(task.completionSource, 'res_valid && res_ready && res_last');
  }
  assert.deepEqual(trace.tasks.map((t) => [t.cmdCycle, t.endCycle]), [[2, 14], [15, 27]]);
  const firstAct = decodeActVector(trace, trace.events.find((ev) => ev.type === 'in').actIdx);
  assert.deepEqual(firstAct, new Array(32).fill(1));
  assert.deepEqual(trace.events.filter((ev) => ev.type === 'res').map((ev) => ev.data), [128, -128, 128, -128]);
  assert.ok(trace.markers.some((m) => m.type === 'reset'));
  assert.ok(trace.markers.some((m) => m.type === 'reset_release'));
  assert.deepEqual(trace.warnings, []);
});

test('paired input, inferred channel boundaries, backpressure and next-edge command', () => {
  const trace = parseChunked(syntheticVcd(legalFrames()), 13);
  assert.deepEqual(trace.warnings, []);
  assert.equal(trace.summary.tasks, 2);
  assert.equal(trace.summary.ok, 2);
  assert.equal(trace.summary.inHandshakes, 5);
  assert.equal(trace.summary.actHandshakes, 5);
  assert.equal(trace.summary.weightHandshakes, 5);
  assert.deepEqual(trace.summary.stalls, { cmd: 3, in: 1, act: 1, weight: 1, res: 2, done: 0 });
  assert.equal(trace.tasks[0].stallTotal, 3); // paired IN stall counts once
  assert.equal(trace.tasks[0].doneWaitCycles, 2); // final RES backpressure
  assert.deepEqual(trace.tasks.map((t) => [t.cmdCycle, t.doneCycle]), [[1, 12], [13, 15]]);
  const inputs = trace.tasks[0].events.filter((ev) => ev.type === 'in');
  assert.deepEqual(inputs.map((ev) => [ev.channel, ev.beat, ev.channelLast]),
    [[0, 0, false], [0, 1, true], [1, 0, false], [1, 1, true]]);
  for (const ev of inputs) {
    assert.equal(ev.inH, true);
    assert.equal(ev.actH, ev.inH);
    assert.equal(ev.wgtH, ev.inH);
    assert.equal(ev.actLast, ev.channelLast);
    assert.equal(ev.weightLast, ev.channelLast);
    assert.equal(ev.boundarySource, 'cmd_k_blocks + input count');
  }
  assert.equal(decodeActVector(trace, inputs[0].actIdx)[0], -128);
  const done = trace.tasks[0].events.at(-1);
  assert.equal(done.type, 'done');
  assert.equal(done.derived, true);
  assert.equal(done.source, trace.completionSource);
  assert.equal(trace.tasks[0].events.at(-2).type, 'res');
  assert.equal(trace.tasks[0].events.at(-2).cycle, done.cycle);
});

test('a blocked final result never completes a task, even at end of file', () => {
  const trace = analyzeText(syntheticVcd(legalFrames().slice(0, 12)));
  assert.equal(trace.summary.doneEvents, 0);
  assert.equal(trace.summary.completed, 0);
  assert.equal(trace.summary.unfinished, 1);
  assert.equal(trace.tasks[0].resultBeats, 3);
  assert.equal(trace.tasks[0].doneCycle, null);
  assert.equal(trace.tasks[0].completionSource, null);
});

test('reset cancels without DONE and clears input counters, config and stalled payload checks', () => {
  const trace = analyzeText(syntheticVcd([
    command(2, 1, 1), input(), input({ in_ready: 0 }),
    { reset: 1, ...command(0, 0, 0), ...result() },
    { reset: 1, act_data: 2, weight_data: 0, res_data: 7 },
    { reset: 0, cmd_valid: 0, in_valid: 0, res_valid: 0, status_busy: 0 },
    command(), input(), result(),
  ]));
  assert.equal(trace.summary.canceled, 1);
  assert.equal(trace.summary.completed, 1);
  assert.equal(trace.summary.ok, 1);
  assert.equal(trace.summary.failed, 0);
  assert.equal(trace.summary.doneEvents, 1);
  assert.equal(trace.tasks[0].status, 'reset');
  assert.equal(trace.tasks[0].doneCycle, null);
  assert.equal(trace.tasks[0].completionSource, null);
  assert.equal(trace.tasks[0].inputBeats, 1);
  assert.equal(trace.tasks[0].events.some((ev) => ev.type === 'done'), false);
  assert.deepEqual(trace.tasks[1].params, { kBlocks: 1, groupDiv: 1, outChans: 1 });
  assert.equal(trace.tasks[1].verified, true);
  assert.deepEqual(trace.tasks[1].issues, []);
  assert.equal(trace.warnings.length, 1); // only the reset cancellation
});

test('stability checks include both operands, command config, and every result sideband', () => {
  for (const [channel, fields] of Object.entries({
    cmd: ['cmd_valid', 'cmd_k_blocks', 'cmd_group_div', 'cmd_out_chans'],
    in: ['in_valid', 'act_data', 'weight_data'],
    res: ['res_valid', 'res_data', 'res_last', 'res_chan_last', 'res_group_idx'],
  })) {
    for (const field of fields) {
      let frames;
      if (channel === 'cmd') frames = [{ ...command(), cmd_ready: 0 }, {}];
      if (channel === 'in') frames = [command(), input({ in_ready: 0 }), {}];
      if (channel === 'res') frames = [command(), input(), result({ res_ready: 0 }), {}];
      frames[frames.length - 1][field] = WIDTHS[field] === 1 ? 0 : 2;
      const trace = analyzeText(syntheticVcd(frames));
      assert.ok(trace.warnings.some((w) => w.text.includes(channel + ' 反压期间')), field);
    }
  }
});

test('stalled payload must remain stable on the accepting edge too', () => {
  const trace = analyzeText(syntheticVcd([
    command(), input({ in_ready: 0 }), input({ weight_data: 7 }),
    result({ res_ready: 0 }), result({ res_data: 42 }),
  ]));
  assert.ok(trace.warnings.some((w) => w.text.includes('in 反压期间')));
  assert.ok(trace.warnings.some((w) => w.text.includes('res 反压期间')));
  assert.equal(trace.summary.completed, 1);
  assert.equal(trace.summary.ok, 0);
  assert.equal(trace.summary.failed, 1);
});

test('reset discards a stalled final result without a completion notification', () => {
  const trace = analyzeText(syntheticVcd([
    command(), input(), result({ res_ready: 0 }),
    { reset: 1, res_ready: 1 },
    { reset: 0, ...command() }, input(), result(),
  ]));
  assert.equal(trace.summary.canceled, 1);
  assert.equal(trace.summary.completed, 1);
  assert.equal(trace.tasks[0].resultBeats, 0);
  assert.equal(trace.tasks[0].doneCycle, null);
  assert.equal(trace.tasks[1].verified, true);
  assert.equal(trace.warnings.length, 1);
});

test('command on final-result edge is flagged and old result stays with old task', () => {
  const trace = analyzeText(syntheticVcd([command(), input(), { ...command(), ...result() }]));
  assert.equal(trace.tasks[0].resultBeats, 1);
  assert.equal(trace.tasks[1].resultBeats, 0);
  assert.equal(trace.tasks[0].doneCycle, 2);
  assert.ok(trace.tasks[0].issues.some((i) => i.text.includes('末结果同沿')));
});

test('completion and counts survive event capping', () => {
  const trace = analyzeText(syntheticVcd(legalFrames()), { maxEvents: 1 });
  assert.equal(trace.summary.ok, 2);
  assert.equal(trace.summary.completed, 2);
  assert.equal(trace.summary.doneEvents, 2);
  assert.equal(trace.summary.inHandshakes, 5);
  assert.equal(trace.summary.resEvents, 5);
  assert.equal(trace.summary.droppedEvents, 10);
  assert.deepEqual(trace.events.map((ev) => ev.type), ['cmd', 'done', 'cmd', 'done']);
});

test('illegal config is diagnosed without inventing a config-error completion', () => {
  const trace = analyzeText(syntheticVcd([command(1, 0, 1)]));
  assert.equal(trace.tasks[0].legal, false);
  assert.equal(trace.tasks[0].status, 'unfinished');
  assert.equal(trace.summary.doneEvents, 0);
  assert.equal(trace.tasks[0].statusCode, null);
});

test('parser requires new ports and rejects missing payload or legacy input channels', () => {
  const text = syntheticVcd([command()]);
  for (const name of KNOWN_SIGNALS) {
    const missing = text.replace(new RegExp('^\\$var[^\\n]* ' + name + ' \\$end\\n', 'm'), '');
    assert.throws(() => analyzeText(missing), /缺少|没有找到/, name);
  }
  assert.throws(() => analyzeText(text.replaceAll('in_valid', 'act_valid').replaceAll('in_ready', 'act_ready')), /in/);
});
