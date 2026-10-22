'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { VcdReader } = require('../js/vcd.js');
const { attach, decodeActVector, analyzeText } = require('../js/analyze.js');

function parseChunked(text, chunkSize) {
  const reader = new VcdReader({});
  const analyzer = attach(reader);
  for (let i = 0; i < text.length; i += chunkSize) {
    reader.feed(text.slice(i, i + chunkSize));
  }
  reader.finish();
  return analyzer.finish();
}

function testDemo() {
  const text = fs.readFileSync(path.join(__dirname, '..', 'samples', 'demo.vcd'), 'utf8');
  // Odd chunk size forces header lines and value tokens across feed() calls.
  const trace = parseChunked(text, 777);
  assert.equal(trace.header.scopePath, 'TOP/bnn_core_top');
  assert.equal(trace.header.timescale.text, '1ps');
  assert.equal(trace.summary.edges, 30);
  assert.equal(trace.summary.cmdEvents, 2);
  assert.equal(trace.summary.inEvents, 16);
  assert.equal(trace.summary.actHandshakes, 16);
  assert.equal(trace.summary.weightHandshakes, 16);
  assert.equal(trace.summary.resEvents, 4);
  assert.equal(trace.summary.doneEvents, 2);
  assert.equal(trace.summary.tasks, 2);
  assert.equal(trace.summary.ok, 2);
  assert.equal(trace.summary.failed, 0);
  assert.equal(trace.tasks.length, 2);

  for (const task of trace.tasks) {
    assert.equal(task.k, 128);
    assert.equal(task.beatsPerChannel, 4);
    assert.equal(task.groups, 1);
    assert.equal(task.n, 2);
    assert.equal(task.expectedInput, 8);
    assert.equal(task.expectedResults, 2);
    assert.equal(task.inputBeats, 8);
    assert.equal(task.resultBeats, 2);
    assert.equal(task.inputChannels, 2);
    assert.equal(task.resultChannels, 2);
    assert.equal(task.status, 'ok');
    assert.equal(task.verified, true);
    assert.equal(task.issues.length, 0);
    assert.equal(task.duration, 14);
  }
  assert.deepEqual(trace.tasks.map((t) => [t.cmdCycle, t.endCycle]), [[2, 15], [16, 29]]);

  const types = trace.events.reduce((m, ev) => (m[ev.type] = (m[ev.type] || 0) + 1, m), {});
  assert.deepEqual(types, { cmd: 2, in: 16, res: 4, done: 2 });
  assert.equal(trace.actVectors.length, 3);
  const firstAct = decodeActVector(trace, trace.events.find((ev) => ev.type === 'in').actIdx);
  assert.equal(firstAct.length, 32);
  assert.equal(firstAct[0], -128);
  assert.ok(trace.markers.some((m) => m.type === 'reset'));
  assert.ok(trace.markers.some((m) => m.type === 'reset_release'));
  assert.equal(trace.warnings.length, 0);
}

const ID = {
  clock: '!', reset: '"', abort: '#',
  cmdValid: '$', cmdReady: '%', kBlocks: '&', groupDiv: "'", outChans: '(',
  idle: ')', busy: '*', doneValid: '+', doneReady: ',', doneStatus: '-',
  actValid: '.', actReady: '/', actData: '0', actLast: '1',
  weightValid: '2', weightReady: '3', weightData: '4', weightLast: '5',
  resValid: '6', resReady: '7', resData: '8', resLast: '9', resChanLast: ':', resGroup: ';',
};

function scalar(value, id) {
  return String(value) + ID[id];
}

function vec(width, value, id) {
  return 'b' + Number(value).toString(2).padStart(width, '0') + ' ' + ID[id];
}

function syntheticVcd() {
  const header = [
    '$timescale 1ps $end',
    '$scope module TOP $end',
    `$var wire 1 ${ID.clock} clock $end`,
    `$var wire 1 ${ID.reset} reset $end`,
    `$var wire 1 ${ID.abort} ctrl_abort $end`,
    `$var wire 1 ${ID.cmdValid} cmd_valid $end`,
    `$var wire 1 ${ID.cmdReady} cmd_ready $end`,
    `$var wire 16 ${ID.kBlocks} cmd_k_blocks [15:0] $end`,
    `$var wire 8 ${ID.groupDiv} cmd_group_div [7:0] $end`,
    `$var wire 16 ${ID.outChans} cmd_out_chans [15:0] $end`,
    `$var wire 1 ${ID.idle} status_idle $end`,
    `$var wire 1 ${ID.busy} status_busy $end`,
    `$var wire 1 ${ID.doneValid} done_valid $end`,
    `$var wire 1 ${ID.doneReady} done_ready $end`,
    `$var wire 2 ${ID.doneStatus} done_status [1:0] $end`,
    `$var wire 1 ${ID.actValid} act_valid $end`,
    `$var wire 1 ${ID.actReady} act_ready $end`,
    `$var wire 256 ${ID.actData} act_data [255:0] $end`,
    `$var wire 1 ${ID.actLast} act_last $end`,
    `$var wire 1 ${ID.weightValid} weight_valid $end`,
    `$var wire 1 ${ID.weightReady} weight_ready $end`,
    `$var wire 32 ${ID.weightData} weight_data [31:0] $end`,
    `$var wire 1 ${ID.weightLast} weight_last $end`,
    `$var wire 1 ${ID.resValid} res_valid $end`,
    `$var wire 1 ${ID.resReady} res_ready $end`,
    `$var wire 32 ${ID.resData} res_data [31:0] $end`,
    `$var wire 1 ${ID.resLast} res_last $end`,
    `$var wire 1 ${ID.resChanLast} res_chan_last $end`,
    `$var wire 8 ${ID.resGroup} res_group_idx [7:0] $end`,
    '$upscope $end',
    '$enddefinitions $end',
  ];
  const zeroAct = 'b' + '0'.repeat(256) + ' ' + ID.actData;
  const body = [
    '#0',
    scalar(0, 'clock'), scalar(1, 'reset'), scalar(0, 'abort'),
    scalar(1, 'cmdValid'), scalar(1, 'cmdReady'),
    vec(16, 4, 'kBlocks'), vec(8, 0, 'groupDiv'), vec(16, 1, 'outChans'),
    scalar(1, 'idle'), scalar(0, 'busy'),
    scalar(0, 'doneValid'), scalar(1, 'doneReady'), vec(2, 0, 'doneStatus'),
    scalar(0, 'actValid'), scalar(1, 'actReady'), zeroAct, scalar(0, 'actLast'),
    scalar(0, 'weightValid'), scalar(1, 'weightReady'), vec(32, 0, 'weightData'), scalar(0, 'weightLast'),
    scalar(0, 'resValid'), scalar(1, 'resReady'), vec(32, 0, 'resData'), scalar(0, 'resLast'), scalar(0, 'resChanLast'), vec(8, 0, 'resGroup'),
    // Possege with reset high and valid/ready high: must be ignored.
    '#5000', scalar(1, 'clock'),
    // Release reset and prepare the illegal command.
    '#6000', scalar(0, 'clock'), scalar(0, 'reset'),
    '#10000', scalar(1, 'clock'), scalar(0, 'cmdValid'), scalar(0, 'cmdReady'), scalar(1, 'busy'), scalar(1, 'doneValid'), vec(2, 1, 'doneStatus'),
    '#11000', scalar(0, 'clock'), scalar(1, 'cmdReady'),
    '#15000', scalar(1, 'clock'), scalar(0, 'doneValid'),
    // Prepare a legal command.
    '#16000', scalar(0, 'clock'), vec(8, 4, 'groupDiv'), scalar(1, 'cmdValid'), scalar(1, 'cmdReady'), vec(2, 2, 'doneStatus'),
    '#20000', scalar(1, 'clock'), scalar(0, 'cmdValid'), scalar(0, 'cmdReady'),
    // Abort and complete the second task.
    '#21000', scalar(0, 'clock'), scalar(1, 'abort'), scalar(1, 'doneValid'),
    '#25000', scalar(1, 'clock'),
    '#26000', scalar(0, 'clock'),
  ];
  return header.concat(body).join('\n') + '\n';
}

function testResetAndErrors() {
  const trace = analyzeText(syntheticVcd());
  assert.equal(trace.summary.edges, 5);
  assert.equal(trace.summary.resetCycles, 1);
  assert.equal(trace.summary.cmdEvents, 2);
  assert.equal(trace.summary.doneEvents, 2);
  assert.equal(trace.tasks.length, 2);

  const first = trace.tasks[0];
  assert.equal(first.legal, false);
  assert.equal(first.status, 'cfg');
  assert.equal(first.verified, false);
  assert.ok(first.issues.length >= 1);
  assert.equal(first.inputBeats, 0);
  assert.equal(first.resultBeats, 0);

  const second = trace.tasks[1];
  assert.equal(second.legal, true);
  assert.equal(second.status, 'aborted');
  assert.equal(second.statusCode, 2);
  assert.equal(second.abortCycle, 4);
  assert.ok(trace.markers.some((m) => m.type === 'abort'));
  assert.ok(trace.warnings.length >= 2);
}

testDemo();
testResetAndErrors();
console.log('analyze.test.js: OK');
