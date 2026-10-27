'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const G = require('../js/gguf.js');

const FIXTURES = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', 'dequant.json'), 'utf8'));

const META = {
    UINT8: 0, INT8: 1, UINT16: 2, INT16: 3, UINT32: 4, INT32: 5,
    FLOAT32: 6, BOOL: 7, STRING: 8, ARRAY: 9, UINT64: 10, INT64: 11, FLOAT64: 12,
};
const TYPE = { F32: 0, F16: 1, Q1_0: 41 };

function u32(value) {
    const buffer = Buffer.alloc(4);
    buffer.writeUInt32LE(value);
    return buffer;
}

function u64(value) {
    const buffer = Buffer.alloc(8);
    buffer.writeBigUInt64LE(BigInt(value));
    return buffer;
}

function f32(value) {
    const buffer = Buffer.alloc(4);
    buffer.writeFloatLE(value);
    return buffer;
}

function ggufString(value) {
    const bytes = Buffer.from(value, 'utf8');
    return Buffer.concat([u64(bytes.length), bytes]);
}

function scalar(type, value) {
    switch (type) {
    case META.UINT8: return Buffer.from([value]);
    case META.INT8: return Buffer.from([value & 0xff]);
    case META.UINT16: {
        const buffer = Buffer.alloc(2);
        buffer.writeUInt16LE(value);
        return buffer;
    }
    case META.INT16: {
        const buffer = Buffer.alloc(2);
        buffer.writeInt16LE(value);
        return buffer;
    }
    case META.UINT32: return u32(value);
    case META.INT32: {
        const buffer = Buffer.alloc(4);
        buffer.writeInt32LE(value);
        return buffer;
    }
    case META.FLOAT32: return f32(value);
    case META.BOOL: return Buffer.from([value ? 1 : 0]);
    case META.STRING: return ggufString(value);
    case META.UINT64: return u64(value);
    case META.INT64: {
        const buffer = Buffer.alloc(8);
        buffer.writeBigInt64LE(BigInt(value));
        return buffer;
    }
    case META.FLOAT64: {
        const buffer = Buffer.alloc(8);
        buffer.writeDoubleLE(value);
        return buffer;
    }
    default: throw new Error(`unsupported scalar type ${type}`);
    }
}

function kv(key, type, value) {
    if (type === META.ARRAY) {
        const elementType = value.type;
        const payload = Buffer.concat([
            u32(elementType),
            u64(value.values.length),
            ...value.values.map(item => scalar(elementType, item)),
        ]);
        return Buffer.concat([ggufString(key), u32(type), payload]);
    }
    return Buffer.concat([ggufString(key), u32(type), scalar(type, value)]);
}

function tensorInfo(name, dimensions, typeId, offset) {
    return Buffer.concat([
        ggufString(name),
        u32(dimensions.length),
        ...dimensions.map(u64),
        u32(typeId),
        u64(offset),
    ]);
}

function buildSyntheticGguf() {
    const metadata = Buffer.concat([
        kv('general.architecture', META.STRING, 'test'),
        kv('general.alignment', META.UINT32, 32),
        kv('general.file_type', META.UINT32, 40),
        kv('test.block_count', META.UINT32, 2),
        kv('test.tags', META.ARRAY, { type: META.STRING, values: ['alpha', 'beta', '中文'] }),
        kv('test.seq', META.ARRAY, { type: META.INT32, values: [-2, 0, 9] }),
        kv('test.enabled', META.BOOL, true),
        kv('test.epsilon', META.FLOAT32, 1e-5),
        kv('test.count64', META.UINT64, 42),
        kv('tokenizer.ggml.model', META.STRING, 'gpt2'),
        kv('tokenizer.ggml.tokens', META.ARRAY, { type: META.STRING, values: ['a', 'Ġb', '中'] }),
        kv('tokenizer.ggml.token_type', META.ARRAY, { type: META.INT32, values: [1, 1, 1] }),
    ]);

    const q1Block = Buffer.from(FIXTURES.Q1_0.bytes, 'base64');
    const f16Data = Buffer.from([0x00, 0x3c, 0x00, 0xc0, 0x00, 0x38, 0x00, 0xb8]);
    const f32Data = Buffer.concat([f32(1), f32(-2), f32(0.5), f32(0.25)]);
    const tensors = Buffer.concat([
        tensorInfo('norm.weight', [4], TYPE.F32, 0),
        tensorInfo('layer.weight', [128, 2], TYPE.Q1_0, 32),
        tensorInfo('half.weight', [4], TYPE.F16, 96),
        tensorInfo('unknown.weight', [8], 77, 128),
    ]);
    const headerSize = 24;
    const infoEnd = headerSize + metadata.length + tensors.length;
    const dataOffset = Math.ceil(infoEnd / 32) * 32;
    const preData = Buffer.alloc(dataOffset - infoEnd);
    const data = Buffer.alloc(128);
    f32Data.copy(data, 0);
    q1Block.copy(data, 32);
    q1Block.copy(data, 32 + q1Block.length);
    f16Data.copy(data, 96);
    const header = Buffer.concat([
        Buffer.from('GGUF', 'ascii'),
        u32(3),
        u64(4),
        u64(12),
    ]);
    return Buffer.concat([header, metadata, tensors, preData, data]);
}

test('parses the full GGUF header, metadata and tensor index', async () => {
    const file = buildSyntheticGguf();
    const model = await G.parse(file);

    assert.equal(model.version, 3);
    assert.equal(model.header.tensorCount, 4);
    assert.equal(model.header.metadataCount, 12);
    assert.equal(model.metadata.length, 12);
    assert.equal(model.tensors.length, 4);
    assert.equal(model.alignment, 32);
    assert.equal(model.alignmentSource, 'general.alignment');
    assert.equal(model.dataOffset % 32, 0);
    assert.equal(model.metadataStart, 24);

    const arch = model.entriesByKey.get('general.architecture');
    assert.equal(arch.value, 'test');
    assert.equal(arch.typeName, 'STRING');

    const tags = model.entriesByKey.get('test.tags');
    assert.equal(tags.typeName, 'ARRAY<STRING>');
    assert.deepEqual(tags.value, ['alpha', 'beta', '中文']);
    assert.equal(tags.elementTypeId, META.STRING);
    assert.equal(tags.arrayCount, 3);
    assert.ok(tags.end > tags.valueStart);

    assert.equal(model.entriesByKey.get('test.enabled').value, true);
    assert.equal(model.entriesByKey.get('test.count64').value, 42);
    assert.equal(model.entriesByKey.get('test.epsilon').typeName, 'FLOAT32');

    const norm = model.tensorInfoById.get(0);
    const layer = model.tensorInfoById.get(1);
    const unknown = model.tensorInfoById.get(3);
    assert.equal(norm.elementCount, 4);
    assert.equal(norm.storedSize, 16);
    assert.equal(norm.absoluteStart, model.dataOffset);
    assert.equal(layer.elementCount, 256);
    assert.equal(layer.blocks, 2);
    assert.equal(layer.storedSize, 36);
    assert.equal(layer.offset, 32);
    assert.equal(layer.absoluteEnd, model.dataOffset + 68);
    assert.equal(unknown.typeInfo.known, false);
    assert.equal(unknown.storedSize, null);

    assert.equal(model.summary.architecture, 'test');
    assert.equal(model.summary.tokenizer.tokenCount, 3);
    assert.equal(model.summary.fileTypeName, 'MOSTLY_Q1_0');
    assert.equal(model.stats.parameterCount, 4 + 256 + 4 + 8);
    assert.equal(model.stats.storedBytes, 16 + 36 + 8);
    assert.equal(model.stats.knownTensorCount, 3);
    assert.equal(model.stats.unknownTensorCount, 1);
    assert.ok(model.warnings.some(item => item.title.includes('未知张量类型')));
    assert.ok(model.warnings.some(item => item.title.includes('文件末尾')));

    assert.equal(model.tensorInfoStart, model.metadataEnd);
    assert.equal(model.tensorInfoEnd + model.dataPadding, model.dataOffset);
});

test('decodePreview reads and dequantizes a request from a sliced file', async () => {
    const file = buildSyntheticGguf();
    const model = await G.parse(file);
    const layer = model.tensorInfoById.get(1);
    const preview = await G.decodePreview(model, layer, 0, 256);
    assert.equal(preview.supported, true);
    assert.equal(preview.startElement, 0);
    assert.equal(preview.count, 256);
    assert.equal(preview.blockCount, 2);
    assert.equal(preview.byteLength, 36);
    assert.deepEqual(preview.values, FIXTURES.Q1_0.expected.slice(0, 128).concat(FIXTURES.Q1_0.expected.slice(0, 128)));

    const unaligned = await G.decodePreview(model, layer, 200, 32);
    assert.equal(unaligned.startElement, 128);
    assert.equal(unaligned.aligned, false);
});

test('rejects non-GGUF, old versions and truncated files with useful errors', async () => {
    await assert.rejects(() => G.parse(Buffer.from('not a gguf file....')), /不是 GGUF/);

    const v1 = Buffer.from(buildSyntheticGguf());
    v1.writeUInt32LE(1, 4);
    await assert.rejects(() => G.parse(v1), /不支持 GGUF v1/);

    await assert.rejects(() => G.parse(Buffer.from('GGUF')), /提前结束|truncated|执行/);
});

test('all bundled quantization reference vectors decode correctly', () => {
    for (const [name, fixture] of Object.entries(FIXTURES)) {
        const typeId = Object.entries(G.TENSOR_TYPES).find(([, info]) => info.name === name)[0];
        assert.notEqual(typeId, undefined, `missing type ${name}`);
        const bytes = Buffer.from(fixture.bytes, 'base64');
        const values = G.dequantize(Number(typeId), bytes);
        assert.equal(values.length, fixture.expected.length, `${name} length`);
        for (let index = 0; index < values.length; index += 1) {
            const got = values[index];
            const expected = fixture.expected[index];
            if (Number.isNaN(expected)) {
                assert.ok(Number.isNaN(got), `${name}[${index}] NaN`);
            } else {
                const tolerance = 1e-6 * (1 + Math.abs(expected));
                assert.ok(Math.abs(got - expected) < tolerance, `${name}[${index}] expected ${expected}, got ${got}`);
            }
        }
    }
});

test('plain scalar tensor types decode with the advertised byte layout', () => {
    const f32 = Buffer.alloc(8);
    f32.writeFloatLE(1.5, 0);
    f32.writeFloatLE(-0.25, 4);
    assert.deepEqual(G.dequantize(TYPE.F32, f32), [1.5, -0.25]);

    const f16 = Buffer.from([0x00, 0x3c, 0x00, 0xc0]);
    assert.deepEqual(G.dequantize(TYPE.F16, f16), [1, -2]);

    const bf16 = Buffer.from([0x80, 0x3f, 0x00, 0xc0]);
    assert.deepEqual(G.dequantize(30, bf16), [1, -2]);

    const f64 = Buffer.alloc(16);
    f64.writeDoubleLE(3.5, 0);
    f64.writeDoubleLE(-7, 8);
    assert.deepEqual(G.dequantize(28, f64), [3.5, -7]);

    assert.deepEqual(G.dequantize(24, Buffer.from([0xff, 0x00, 0x7f, 0x80])), [-1, 0, 127, -128]);

    const i16 = Buffer.alloc(4);
    i16.writeInt16LE(-2, 0);
    i16.writeInt16LE(300, 2);
    assert.deepEqual(G.dequantize(25, i16), [-2, 300]);

    const i32 = Buffer.alloc(8);
    i32.writeInt32LE(-123456, 0);
    i32.writeInt32LE(7654321, 4);
    assert.deepEqual(G.dequantize(26, i32), [-123456, 7654321]);
});

test('Q8_1 and Q8_K preview encoding is decoded correctly', () => {
    const q8_1 = Buffer.alloc(36);
    q8_1.writeUInt16LE(0x3800, 0);
    q8_1[4] = 1;
    q8_1[5] = 0xff;
    q8_1[6] = 127;
    q8_1[7] = 128;
    assert.deepEqual(G.dequantize(9, q8_1).slice(0, 4), [0.5, -0.5, 63.5, -64]);

    const q8k = Buffer.alloc(292);
    q8k.writeFloatLE(0.25, 0);
    q8k[4] = 2;
    q8k[5] = 0xfe;
    q8k[6] = 100;
    const values = G.dequantize(15, q8k);
    assert.equal(values.length, 256);
    assert.deepEqual(values.slice(0, 3), [0.5, -0.5, 25]);
});

test('preview refuses unsupported or malformed blocks', () => {
    assert.throws(() => G.dequantize(16, Buffer.alloc(66)), /暂不支持反量化预览/);
    assert.throws(() => G.dequantize(TYPE.F32, Buffer.alloc(3)), /不是块大小/);
    assert.throws(() => G.dequantize(999, Buffer.alloc(4)), /未知张量类型/);
});

test('validates tensor type metadata and exposes block-level facts', () => {
    const q1 = G.tensorTypeInfo(41);
    assert.equal(q1.name, 'Q1_0');
    assert.equal(q1.block, 128);
    assert.equal(q1.size, 18);
    assert.equal(q1.bits, 1.125);
    assert.equal(q1.preview, true);

    const q4k = G.tensorTypeInfo(12);
    assert.equal(q4k.name, 'Q4_K');
    assert.equal(q4k.block, 256);
    assert.equal(q4k.size, 144);
    assert.equal(G.tensorTypeInfo(77).known, false);
});

test('Q1_0 stores scale first, then 128 LSB-first sign bits', () => {
    const info = G.tensorTypeInfo(41);
    assert.deepEqual(info.blockFields.map(field => field.name), ['d', 'qs']);
    assert.equal(info.blockFields[0].bytes, 2);
    assert.equal(info.blockFields[0].format, 'f16');
    assert.equal(info.blockFields[1].bytes, 16);
    assert.equal(info.blockFields[1].bits, 128);
    assert.equal(info.blockFields[1].bitOrder, 'lsb');

    const block = Buffer.alloc(18);
    block.writeUInt16LE(0x3800, 0); // FP16 +0.5
    block[2] = 0b00000101;          // element bits 0..7: 1,0,1,0,0,0,0,0
    const values = G.dequantize(41, block);
    assert.deepEqual(values.slice(0, 4), [0.5, -0.5, 0.5, -0.5]);
    assert.equal(values[7], -0.5);
    assert.equal(values[8], -0.5);
    assert.equal(values.length, 128);
});

test('every structured block-field layout covers exactly one stored block', () => {
    for (const [typeId, fields] of Object.entries(G.BLOCK_FIELDS)) {
        const info = G.tensorTypeInfo(Number(typeId));
        const bytes = fields.reduce((sum, field) => sum + field.bytes, 0);
        assert.equal(bytes, info.size, `${info.name} field bytes`);
    }
});
