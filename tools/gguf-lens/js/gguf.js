(function (root, factory) {
    const api = factory();
    if (typeof module === 'object' && module.exports) module.exports = api;
    if (typeof window !== 'undefined') window.GGUF = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
    'use strict';

    const GGUF_MAGIC = 'GGUF';
    const SUPPORTED_VERSIONS = new Set([2, 3]);
    const DEFAULT_ALIGNMENT = 32;
    const READ_CHUNK = 256 * 1024;
    const LIMITS = Object.freeze({
        metadataCount: 200000,
        tensorCount: 1000000,
        arrayElements: 30000000,
        stringBytes: 64 * 1024 * 1024,
        rank: 8,
        arrayDepth: 8,
        alignment: 1024 * 1024,
    });

    const METADATA_TYPES = {
        0: { name: 'UINT8', short: 'u8', size: 1, scalar: true },
        1: { name: 'INT8', short: 'i8', size: 1, scalar: true },
        2: { name: 'UINT16', short: 'u16', size: 2, scalar: true },
        3: { name: 'INT16', short: 'i16', size: 2, scalar: true },
        4: { name: 'UINT32', short: 'u32', size: 4, scalar: true },
        5: { name: 'INT32', short: 'i32', size: 4, scalar: true },
        6: { name: 'FLOAT32', short: 'f32', size: 4, scalar: true },
        7: { name: 'BOOL', short: 'bool', size: 1, scalar: true },
        8: { name: 'STRING', short: 'str', size: null, scalar: true },
        9: { name: 'ARRAY', short: 'arr', size: null, scalar: false },
        10: { name: 'UINT64', short: 'u64', size: 8, scalar: true },
        11: { name: 'INT64', short: 'i64', size: 8, scalar: true },
        12: { name: 'FLOAT64', short: 'f64', size: 8, scalar: true },
    };

    const TENSOR_TYPES = {
        0: { name: 'F32', block: 1, size: 4, family: 'float', layout: '4 B / 值', desc: '32 位 IEEE 754 浮点，未量化。' },
        1: { name: 'F16', block: 1, size: 2, family: 'float', layout: '2 B / 值', desc: '16 位 IEEE 754 半精度浮点。' },
        2: { name: 'Q4_0', block: 32, size: 18, family: 'q4', layout: 'd:f16 2B | qs:4bit × 32 = 16B', desc: '对称 4-bit 量化，每 32 个值共用一个 FP16 scale。' },
        3: { name: 'Q4_1', block: 32, size: 20, family: 'q4', layout: 'd:f16 2B | m:f16 2B | qs:4bit × 32 = 16B', desc: '非对称 4-bit 量化，每块含 FP16 scale 与最小值。' },
        6: { name: 'Q5_0', block: 32, size: 22, family: 'q5', layout: 'd:f16 2B | qh:高 1bit × 32 = 4B | qs:低 4bit = 16B', desc: '对称 5-bit 量化，五个 bit 拆成低 4 位与高位平面。' },
        7: { name: 'Q5_1', block: 32, size: 24, family: 'q5', layout: 'd:f16 2B | m:f16 2B | qh:4B | qs:16B', desc: '非对称 5-bit 量化，含 scale 与最小值。' },
        8: { name: 'Q8_0', block: 32, size: 34, family: 'q8', layout: 'd:f16 2B | qs:int8 × 32 = 32B', desc: '8-bit 分块量化，每 32 个值共用一个 FP16 scale。' },
        9: { name: 'Q8_1', block: 32, size: 36, family: 'q8', layout: 'd:f16 2B | s:f16 2B | qs:int8 × 32', desc: '8-bit 量化；附带的 s 字段用于激活求和的快速计算。' },
        10: { name: 'Q2_K', block: 256, size: 84, family: 'low', layout: 'scales:16B | qs:64B | d:f16 | dmin:f16', desc: 'K-quant 超级块，16 组共享 4-bit scale/min 的 2-bit 权重。' },
        11: { name: 'Q3_K', block: 256, size: 110, family: 'mid', layout: 'hmask:32B | qs:64B | scales:12B | d:f16 2B', desc: 'K-quant 超级块，2-bit 低权重加 1-bit 高掩码。' },
        12: { name: 'Q4_K', block: 256, size: 144, family: 'q4', layout: 'd:f16 | dmin:f16 | scales:12B | qs:128B', desc: 'K-quant 超级块，8 组 4-bit 量化，每组各带 scale/min。' },
        13: { name: 'Q5_K', block: 256, size: 176, family: 'q5', layout: 'd:f16 | dmin:f16 | scales:12B | qh:32B | qs:128B', desc: 'K-quant 超级块，4-bit 低权重加 1-bit 高位，8 组 scale/min。' },
        14: { name: 'Q6_K', block: 256, size: 210, family: 'q8', layout: 'ql:128B | qh:64B | scales:int8 × 16 | d:f16', desc: 'K-quant 超级块，6-bit 权重，16 个 int8 子块 scale。' },
        15: { name: 'Q8_K', block: 256, size: 292, family: 'q8', layout: 'd:f32 | qs:int8 × 256 | bsums:i16 × 16', desc: '8-bit K-quant，主要用于激活量化和 K-quant 点积。' },
        16: { name: 'IQ2_XXS', block: 256, size: 66, family: 'iq', layout: 'd:f16 | qs + scales 码本', desc: 'i-quant 2-bit 码本量化，使用 imatrix 校准。' },
        17: { name: 'IQ2_XS', block: 256, size: 74, family: 'iq', layout: 'd:f16 | qs + scales 码本', desc: 'i-quant 2-bit 码本量化，较 XXS 有更多码字。' },
        18: { name: 'IQ3_XXS', block: 256, size: 98, family: 'iq', layout: 'd:f16 | qs + scales 码本', desc: 'i-quant 3-bit 码本量化。' },
        19: { name: 'IQ1_S', block: 256, size: 50, family: 'iq', layout: 'd:f16 | qs + scales 码本', desc: 'i-quant 1-bit 码本量化，压缩率很高。' },
        20: { name: 'IQ4_NL', block: 32, size: 18, family: 'iq', layout: 'd:f16 2B | qs:4bit × 32 = 16B', desc: '4-bit 非线性码本量化，32 值共用 FP16 scale。' },
        21: { name: 'IQ3_S', block: 256, size: 110, family: 'iq', layout: 'd:f16 | qs + scales 码本', desc: 'i-quant 3-bit 码本量化。' },
        22: { name: 'IQ2_S', block: 256, size: 82, family: 'iq', layout: 'd:f16 | qs + scales 码本', desc: 'i-quant 2-bit 码本量化。' },
        23: { name: 'IQ4_XS', block: 256, size: 136, family: 'iq', layout: 'd:f16 | scales + qs 码本', desc: 'i-quant 4-bit 码本量化，超大块共享 scale。' },
        24: { name: 'I8', block: 1, size: 1, family: 'int', layout: '1 B / 值', desc: '8 位有符号整数。' },
        25: { name: 'I16', block: 1, size: 2, family: 'int', layout: '2 B / 值', desc: '16 位有符号整数。' },
        26: { name: 'I32', block: 1, size: 4, family: 'int', layout: '4 B / 值', desc: '32 位有符号整数。' },
        27: { name: 'I64', block: 1, size: 8, family: 'int', layout: '8 B / 值', desc: '64 位有符号整数。' },
        28: { name: 'F64', block: 1, size: 8, family: 'float', layout: '8 B / 值', desc: '64 位 IEEE 754 浮点。' },
        29: { name: 'IQ1_M', block: 256, size: 56, family: 'iq', layout: 'd:f16 | qs + scales 码本', desc: 'i-quant 1-bit 码本量化，带更多 scale 粒度。' },
        30: { name: 'BF16', block: 1, size: 2, family: 'float', layout: '2 B / 值', desc: 'bfloat16：8 位指数 + 7 位尾数。' },
        34: { name: 'TQ1_0', block: 256, size: 54, family: 'ternary', layout: 'qs:48B | qh:4B | d:f16 2B', desc: '三值 {-1,0,1} 量化，256 值共享一个 FP16 scale。' },
        35: { name: 'TQ2_0', block: 256, size: 66, family: 'ternary', layout: 'qs:64B | d:f16 2B', desc: '三值 {-1,0,1} 量化，每字节打包四个 2-bit 码。' },
        39: { name: 'MXFP4', block: 32, size: 17, family: 'mxfp', layout: 'e8m0:1B | qs:4bit × 32 = 16B', desc: 'OCP MX 4-bit 微缩放格式，使用 8 位 E8M0 指数 scale。' },
        40: { name: 'NVFP4', block: 64, size: 36, family: 'mxfp', layout: 'd:UE4M3 × 4 | qs:4bit × 64 = 32B', desc: 'NVIDIA FP4 微块量化：4 个 UE4M3 子块 scale + FP4 码字。' },
        41: { name: 'Q1_0', block: 128, size: 18, family: 'binary', layout: 'd:f16 2B → qs:16 B（128 × 1 bit，LSB-first）', desc: '二值量化：每个 18 B 块先放 2 B FP16 group scale，再放 16 B 符号位；bit=1 → +d，bit=0 → -d。' },
        42: { name: 'Q2_0', block: 64, size: 18, family: 'low', layout: 'd:f16 2B | qs:2bit × 64 = 16B', desc: '每 64 个值共用一个 FP16 scale，2-bit 四值码。' },
    };

    // Byte layout inside one stored block. Every field is contiguous; blocks are packed
    // back-to-back. GGML layouts are row-major with ne[0] changing fastest.
    const BLOCK_FIELDS = {
        0: [{ name: 'value', label: 'FP32 数值', bytes: 4, kind: 'value' }],
        1: [{ name: 'value', label: 'FP16 数值', bytes: 2, kind: 'value' }],
        30: [{ name: 'value', label: 'BF16 数值', bytes: 2, kind: 'value' }],
        28: [{ name: 'value', label: 'FP64 数值', bytes: 8, kind: 'value' }],
        24: [{ name: 'value', label: 'INT8 数值', bytes: 1, kind: 'value' }],
        25: [{ name: 'value', label: 'INT16 数值', bytes: 2, kind: 'value' }],
        26: [{ name: 'value', label: 'INT32 数值', bytes: 4, kind: 'value' }],
        27: [{ name: 'value', label: 'INT64 数值', bytes: 8, kind: 'value' }],
        2: [
            { name: 'd', label: 'FP16 scale', bytes: 2, kind: 'scale', format: 'f16' },
            { name: 'qs', label: '4-bit 码 × 32（低/高 nibble）', bytes: 16, kind: 'quant' },
        ],
        3: [
            { name: 'd', label: 'FP16 scale', bytes: 2, kind: 'scale', format: 'f16' },
            { name: 'm', label: 'FP16 min', bytes: 2, kind: 'min', format: 'f16' },
            { name: 'qs', label: '4-bit 码 × 32（低/高 nibble）', bytes: 16, kind: 'quant' },
        ],
        6: [
            { name: 'd', label: 'FP16 scale', bytes: 2, kind: 'scale', format: 'f16' },
            { name: 'qh', label: '高位平面 1-bit × 32', bytes: 4, kind: 'highbits' },
            { name: 'qs', label: '低 4-bit 码 × 32', bytes: 16, kind: 'quant' },
        ],
        7: [
            { name: 'd', label: 'FP16 scale', bytes: 2, kind: 'scale', format: 'f16' },
            { name: 'm', label: 'FP16 min', bytes: 2, kind: 'min', format: 'f16' },
            { name: 'qh', label: '高位平面 1-bit × 32', bytes: 4, kind: 'highbits' },
            { name: 'qs', label: '低 4-bit 码 × 32', bytes: 16, kind: 'quant' },
        ],
        8: [
            { name: 'd', label: 'FP16 scale', bytes: 2, kind: 'scale', format: 'f16' },
            { name: 'qs', label: 'INT8 量化码 × 32', bytes: 32, kind: 'quant' },
        ],
        9: [
            { name: 'd', label: 'FP16 scale', bytes: 2, kind: 'scale', format: 'f16' },
            { name: 's', label: 'FP16 sum（辅助）', bytes: 2, kind: 'aux', format: 'f16' },
            { name: 'qs', label: 'INT8 量化码 × 32', bytes: 32, kind: 'quant' },
        ],
        10: [
            { name: 'scales', label: '4-bit scale/min × 16', bytes: 16, kind: 'scales' },
            { name: 'qs', label: '2-bit 量化码 × 256', bytes: 64, kind: 'quant' },
            { name: 'd', label: 'FP16 super-scale', bytes: 2, kind: 'scale', format: 'f16' },
            { name: 'dmin', label: 'FP16 min-scale', bytes: 2, kind: 'min', format: 'f16' },
        ],
        11: [
            { name: 'hmask', label: '高 1-bit 掩码 × 256', bytes: 32, kind: 'highbits' },
            { name: 'qs', label: '2-bit 低码 × 256', bytes: 64, kind: 'quant' },
            { name: 'scales', label: '6-bit scale × 16', bytes: 12, kind: 'scales' },
            { name: 'd', label: 'FP16 super-scale', bytes: 2, kind: 'scale', format: 'f16' },
        ],
        12: [
            { name: 'd', label: 'FP16 super-scale', bytes: 2, kind: 'scale', format: 'f16' },
            { name: 'dmin', label: 'FP16 min-scale', bytes: 2, kind: 'min', format: 'f16' },
            { name: 'scales', label: '6-bit scale/min × 8', bytes: 12, kind: 'scales' },
            { name: 'qs', label: '4-bit 量化码 × 256', bytes: 128, kind: 'quant' },
        ],
        13: [
            { name: 'd', label: 'FP16 super-scale', bytes: 2, kind: 'scale', format: 'f16' },
            { name: 'dmin', label: 'FP16 min-scale', bytes: 2, kind: 'min', format: 'f16' },
            { name: 'scales', label: '6-bit scale/min × 8', bytes: 12, kind: 'scales' },
            { name: 'qh', label: '高位平面 1-bit × 256', bytes: 32, kind: 'highbits' },
            { name: 'qs', label: '4-bit 低码 × 256', bytes: 128, kind: 'quant' },
        ],
        14: [
            { name: 'ql', label: '低 4-bit 码 × 256', bytes: 128, kind: 'quant' },
            { name: 'qh', label: '高位平面 2-bit × 256', bytes: 64, kind: 'highbits' },
            { name: 'scales', label: 'INT8 子块 scale × 16', bytes: 16, kind: 'scales' },
            { name: 'd', label: 'FP16 super-scale', bytes: 2, kind: 'scale', format: 'f16' },
        ],
        15: [
            { name: 'd', label: 'FP32 super-scale', bytes: 4, kind: 'scale', format: 'f32' },
            { name: 'qs', label: 'INT8 量化码 × 256', bytes: 256, kind: 'quant' },
            { name: 'bsums', label: 'INT16 子块和 × 16', bytes: 32, kind: 'aux' },
        ],
        20: [
            { name: 'd', label: 'FP16 scale', bytes: 2, kind: 'scale', format: 'f16' },
            { name: 'qs', label: '4-bit 码本索引 × 32', bytes: 16, kind: 'quant' },
        ],
        34: [
            { name: 'qs', label: '三值码（5 字节组）', bytes: 48, kind: 'quant' },
            { name: 'qh', label: '高位三值平面', bytes: 4, kind: 'highbits' },
            { name: 'd', label: 'FP16 scale', bytes: 2, kind: 'scale', format: 'f16' },
        ],
        35: [
            { name: 'qs', label: '2-bit 三值码 × 256', bytes: 64, kind: 'quant' },
            { name: 'd', label: 'FP16 scale', bytes: 2, kind: 'scale', format: 'f16' },
        ],
        39: [
            { name: 'e', label: 'E8M0 微缩放指数', bytes: 1, kind: 'scale', format: 'e8m0' },
            { name: 'qs', label: 'FP4 码 × 32（低/高 nibble）', bytes: 16, kind: 'quant' },
        ],
        40: [
            { name: 'd', label: 'UE4M3 子块 scale × 4', bytes: 4, kind: 'scales' },
            { name: 'qs', label: 'FP4 码 × 64（低/高 nibble）', bytes: 32, kind: 'quant' },
        ],
        41: [
            { name: 'd', label: 'FP16 group scale', bytes: 2, kind: 'scale', format: 'f16' },
            { name: 'qs', label: '128 个符号位（LSB-first）', bytes: 16, kind: 'bits', bits: 128, bitOrder: 'lsb' },
        ],
        42: [
            { name: 'd', label: 'FP16 scale', bytes: 2, kind: 'scale', format: 'f16' },
            { name: 'qs', label: '2-bit 码 × 64（每字节 4 个）', bytes: 16, kind: 'quant' },
        ],
    };

    const FILE_TYPE_NAMES = {
        0: 'ALL_F32', 1: 'MOSTLY_F16', 2: 'MOSTLY_Q4_0', 3: 'MOSTLY_Q4_1',
        7: 'MOSTLY_Q8_0', 8: 'MOSTLY_Q5_0', 9: 'MOSTLY_Q5_1', 10: 'MOSTLY_Q2_K',
        11: 'MOSTLY_Q3_K_S', 12: 'MOSTLY_Q3_K_M', 13: 'MOSTLY_Q3_K_L',
        14: 'MOSTLY_Q4_K_S', 15: 'MOSTLY_Q4_K_M', 16: 'MOSTLY_Q5_K_S',
        17: 'MOSTLY_Q5_K_M', 18: 'MOSTLY_Q6_K', 19: 'MOSTLY_IQ2_XXS',
        20: 'MOSTLY_IQ2_XS', 21: 'MOSTLY_Q2_K_S', 22: 'MOSTLY_IQ3_XS',
        23: 'MOSTLY_IQ3_XXS', 24: 'MOSTLY_IQ1_S', 25: 'MOSTLY_IQ4_NL',
        26: 'MOSTLY_IQ3_S', 27: 'MOSTLY_IQ3_M', 28: 'MOSTLY_IQ2_S',
        29: 'MOSTLY_IQ2_M', 30: 'MOSTLY_IQ4_XS', 31: 'MOSTLY_IQ1_M',
        32: 'MOSTLY_BF16', 36: 'MOSTLY_TQ1_0', 37: 'MOSTLY_TQ2_0',
        38: 'MOSTLY_MXFP4_MOE', 39: 'MOSTLY_NVFP4', 40: 'MOSTLY_Q1_0',
        41: 'MOSTLY_Q2_0',
    };

    const TOKEN_TYPES = {
        1: 'NORMAL', 2: 'UNKNOWN', 3: 'CONTROL', 4: 'USER_DEFINED', 5: 'UNUSED', 6: 'BYTE',
    };

    const PREVIEW_TYPES = new Set([
        0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 24, 25, 26, 27, 28, 30,
        34, 35, 39, 40, 41, 42,
    ]);

    class GGUFError extends Error {
        constructor(message, offset) {
            super(message);
            this.name = 'GGUFError';
            this.offset = Number.isInteger(offset) ? offset : null;
        }
    }

    function fail(message, offset) {
        throw new GGUFError(message, offset);
    }

    function isSafeInteger(value) {
        return Number.isSafeInteger(value) && value >= 0;
    }

    function alignUp(value, alignment) {
        return Math.ceil(value / alignment) * alignment;
    }

    function readU16LE(bytes, offset) {
        return bytes[offset] | (bytes[offset + 1] << 8);
    }

    function readI16LE(bytes, offset) {
        const value = readU16LE(bytes, offset);
        return value & 0x8000 ? value - 0x10000 : value;
    }

    function readU32LE(bytes, offset) {
        return (bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16) | (bytes[offset + 3] << 24)) >>> 0;
    }

    function readI32LE(bytes, offset) {
        return bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16) | (bytes[offset + 3] << 24);
    }

    function readF32LE(bytes, offset) {
        const view = new DataView(bytes.buffer, bytes.byteOffset + offset, 4);
        return view.getFloat32(0, true);
    }

    function readF64LE(bytes, offset) {
        const view = new DataView(bytes.buffer, bytes.byteOffset + offset, 8);
        return view.getFloat64(0, true);
    }

    function readU64LE(bytes, offset) {
        let value = 0n;
        for (let i = 7; i >= 0; i -= 1) value = (value << 8n) | BigInt(bytes[offset + i]);
        return value;
    }

    function f16ToNumber(bits) {
        const sign = bits & 0x8000 ? -1 : 1;
        const exponent = (bits >> 10) & 0x1f;
        const fraction = bits & 0x03ff;
        if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
        if (exponent === 31) return fraction === 0 ? sign * Infinity : NaN;
        return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
    }

    function bf16ToNumber(bits) {
        const view = new DataView(new ArrayBuffer(4));
        view.setUint32(0, (bits & 0xffff) << 16, false);
        return view.getFloat32(0, false);
    }

    function e8m0ToNumber(e) {
        const view = new DataView(new ArrayBuffer(4));
        const bits = e < 2 ? (0x00200000 << e) >>> 0 : ((e - 1) << 23) >>> 0;
        view.setUint32(0, bits, false);
        return view.getFloat32(0, false);
    }

    function ue4m3ToNumber(value) {
        if (value === 0 || value === 0x7f) return 0;
        const exponent = (value >> 3) & 0x0f;
        const mantissa = value & 0x07;
        const raw = exponent === 0
            ? mantissa * Math.pow(2, -9)
            : (1 + mantissa / 8) * Math.pow(2, exponent - 7);
        return raw * 0.5;
    }

    class ByteSource {
        constructor(input) {
            this.blob = null;
            this.bytes = null;
            this.name = '';
            this.lastModified = null;
            if (typeof Blob !== 'undefined' && input instanceof Blob) {
                this.blob = input;
                this.size = input.size;
                this.name = input.name || '';
                this.lastModified = input.lastModified || null;
            } else if (input instanceof ArrayBuffer) {
                this.bytes = new Uint8Array(input);
                this.size = this.bytes.byteLength;
            } else if (ArrayBuffer.isView(input)) {
                this.bytes = new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
                this.size = this.bytes.byteLength;
            } else {
                fail('无法识别输入：请传入 File、Blob、ArrayBuffer 或 TypedArray。');
            }
        }

        async read(offset, length) {
            if (!isSafeInteger(offset) || !isSafeInteger(length) || offset + length > this.size) {
                fail(`读取越界：offset=${offset}, length=${length}, size=${this.size}`, offset);
            }
            if (this.bytes) return this.bytes.subarray(offset, offset + length);
            const buffer = await this.blob.slice(offset, offset + length).arrayBuffer();
            return new Uint8Array(buffer);
        }
    }

    class Cursor {
        constructor(source) {
            this.source = source;
            this.position = 0;
            this.cacheStart = 0;
            this.cache = null;
        }

        get size() {
            return this.source.size;
        }

        async read(length) {
            if (!isSafeInteger(length)) fail('无效的读取长度。', this.position);
            const start = this.position;
            if (start + length > this.size) {
                fail(`文件在 offset ${start} 处提前结束：需要 ${length} B，仅剩 ${this.size - start} B。`, start);
            }
            if (this.cache && start >= this.cacheStart && start + length <= this.cacheStart + this.cache.byteLength) {
                this.position += length;
                return this.cache.subarray(start - this.cacheStart, start - this.cacheStart + length);
            }
            if (length <= READ_CHUNK) {
                const chunkLength = Math.min(READ_CHUNK, this.size - start);
                this.cache = await this.source.read(start, chunkLength);
                this.cacheStart = start;
                this.position += length;
                return this.cache.subarray(0, length);
            }
            const data = await this.source.read(start, length);
            this.cache = null;
            this.position += length;
            return data;
        }

        async u8() {
            return (await this.read(1))[0];
        }

        async u16() {
            return readU16LE(await this.read(2), 0);
        }

        async u32() {
            return readU32LE(await this.read(4), 0);
        }

        async u64() {
            const raw = readU64LE(await this.read(8), 0);
            if (raw > BigInt(Number.MAX_SAFE_INTEGER)) {
                fail('文件中的 64 位计数/偏移超出 JavaScript 可安全表示的范围。', this.position - 8);
            }
            return Number(raw);
        }

        async u64Value() {
            const raw = readU64LE(await this.read(8), 0);
            return raw <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(raw) : raw;
        }

        async i8() {
            const value = await this.u8();
            return value & 0x80 ? value - 256 : value;
        }

        async i16() {
            return readI16LE(await this.read(2), 0);
        }

        async i32() {
            return readI32LE(await this.read(4), 0);
        }

        async i64() {
            const bytes = await this.read(8);
            const value = readU64LE(bytes, 0);
            const signed = value & (1n << 63n) ? value - (1n << 64n) : value;
            return signed >= BigInt(Number.MIN_SAFE_INTEGER) && signed <= BigInt(Number.MAX_SAFE_INTEGER)
                ? Number(signed)
                : signed;
        }

        async f32() {
            return readF32LE(await this.read(4), 0);
        }

        async f64() {
            return readF64LE(await this.read(8), 0);
        }

        async stringBytes() {
            const length = await this.u64();
            if (length > LIMITS.stringBytes) {
                fail(`字符串长度 ${length} B 超出查看器上限 ${LIMITS.stringBytes} B。`, this.position - 8);
            }
            return this.read(length);
        }
    }

    const textDecoder = new TextDecoder('utf-8', { fatal: false });

    function decodeString(bytes) {
        return textDecoder.decode(bytes);
    }

    function metadataTypeInfo(typeId) {
        return METADATA_TYPES[typeId] || {
            name: `UNKNOWN_${typeId}`,
            short: `?${typeId}`,
            size: null,
            scalar: false,
        };
    }

    function tensorTypeInfo(typeId) {
        const known = TENSOR_TYPES[typeId];
        if (known) {
            return {
                ...known,
                id: typeId,
                bits: known.size * 8 / known.block,
                preview: PREVIEW_TYPES.has(typeId),
                blockFields: BLOCK_FIELDS[typeId] || null,
                known: true,
            };
        }
        return {
            id: typeId,
            name: `TYPE_${typeId}`,
            block: null,
            size: null,
            bits: null,
            family: 'unknown',
            layout: '未知块结构',
            desc: '当前查看器不识别的张量类型；只能显示原始字节与文件索引。',
            preview: false,
            blockFields: null,
            known: false,
        };
    }

    async function readMetadataValue(cursor, typeId, depth, warnings, key, capture) {
        if (depth > LIMITS.arrayDepth) fail(`元数据 "${key}" 的数组嵌套过深。`, cursor.position);
        if (typeId === 8) return decodeString(await cursor.stringBytes());
        if (typeId === 9) {
            const elementTypeId = await cursor.u32();
            const count = await cursor.u64();
            const elementInfo = metadataTypeInfo(elementTypeId);
            const minimum = elementInfo.size === null
                ? (elementTypeId === 8 ? 8 : elementTypeId === 9 ? 12 : 1)
                : elementInfo.size;
            if (count > LIMITS.arrayElements) {
                fail(`数组元素数量 ${count} 超出查看器上限 ${LIMITS.arrayElements}。`, cursor.position - 8);
            }
            if (count * minimum > cursor.size - cursor.position) {
                fail(`数组 "${key}" 声明了 ${count} 个元素，但文件剩余空间不足。`, cursor.position - 8);
            }
            if (capture) {
                capture.elementTypeId = elementTypeId;
                capture.arrayCount = count;
            }
            const values = new Array(count);
            for (let i = 0; i < count; i += 1) {
                values[i] = await readMetadataValue(cursor, elementTypeId, depth + 1, warnings, key);
            }
            return values;
        }
        const info = METADATA_TYPES[typeId];
        if (!info) fail(`未知元数据类型 id=${typeId}。`, cursor.position);
        switch (typeId) {
        case 0: return cursor.u8();
        case 1: return cursor.i8();
        case 2: return cursor.u16();
        case 3: return cursor.i16();
        case 4: return cursor.u32();
        case 5: return cursor.i32();
        case 6: return cursor.f32();
        case 7: {
            const value = await cursor.u8();
            if (value > 1) {
                warnings.push({
                    severity: 'warn',
                    title: '非规范 BOOL 元数据',
                    detail: `"${key}" 使用了 ${value}；规范只允许 0 或 1，查看器按真值处理。`,
                });
            }
            return value !== 0;
        }
        case 10: return cursor.u64Value();
        case 11: return cursor.i64();
        case 12: return cursor.f64();
        default:
            fail(`未实现的元数据类型 id=${typeId}。`, cursor.position);
        }
    }

    function typeLabelForEntry(entry) {
        const info = metadataTypeInfo(entry.typeId);
        if (entry.typeId === 9) {
            return `ARRAY<${metadataTypeInfo(entry.elementTypeId).name}>`;
        }
        return info.name;
    }

    function summarizeValue(value, limit) {
        if (typeof value === 'string') {
            const compact = value.replace(/\s+/g, ' ');
            return compact.length > limit ? `${compact.slice(0, limit)}…` : compact;
        }
        if (typeof value === 'bigint') return value.toString();
        if (typeof value === 'number' || typeof value === 'boolean') return String(value);
        if (Array.isArray(value)) {
            const max = Math.min(value.length, 6);
            const items = [];
            for (let i = 0; i < max; i += 1) {
                const item = value[i];
                items.push(Array.isArray(item) ? '[…]' : summarizeValue(item, Math.max(8, Math.floor(limit / max))));
            }
            return `[${items.join(', ')}${value.length > max ? ', …' : ''}]`;
        }
        return String(value ?? '');
    }

    function parseSafeNumber(value) {
        return typeof value === 'number' && Number.isFinite(value) ? value : null;
    }

    function normalizeEntry(entry) {
        const rawSize = entry.end - entry.start;
        return {
            ...entry,
            size: rawSize,
            valueSize: entry.end - entry.valueStart,
            typeName: typeLabelForEntry(entry),
            summary: summarizeValue(entry.value, 120),
        };
    }

    function getMetaValue(entriesByKey, key) {
        const entry = entriesByKey.get(key);
        return entry ? entry.value : undefined;
    }

    function buildSummary(metadata, entriesByKey, tensors, typeStats, stats, warnings) {
        const architecture = getMetaValue(entriesByKey, 'general.architecture');
        const archPrefix = typeof architecture === 'string' && architecture ? architecture : null;
        const contextLength = archPrefix ? getMetaValue(entriesByKey, `${archPrefix}.context_length`) : undefined;
        const embeddingLength = archPrefix ? getMetaValue(entriesByKey, `${archPrefix}.embedding_length`) : undefined;
        const blockCount = archPrefix ? getMetaValue(entriesByKey, `${archPrefix}.block_count`) : undefined;
        const headCount = archPrefix ? getMetaValue(entriesByKey, `${archPrefix}.attention.head_count`) : undefined;
        const headCountKv = archPrefix ? getMetaValue(entriesByKey, `${archPrefix}.attention.head_count_kv`) : undefined;
        const tokens = getMetaValue(entriesByKey, 'tokenizer.ggml.tokens');
        const merges = getMetaValue(entriesByKey, 'tokenizer.ggml.merges');
        const tokenTypes = getMetaValue(entriesByKey, 'tokenizer.ggml.token_type');
        const fileTypeId = parseSafeNumber(getMetaValue(entriesByKey, 'general.file_type'));
        const quantVersion = parseSafeNumber(getMetaValue(entriesByKey, 'general.quantization_version'));
        const eosId = parseSafeNumber(getMetaValue(entriesByKey, 'tokenizer.ggml.eos_token_id'));
        const bosId = parseSafeNumber(getMetaValue(entriesByKey, 'tokenizer.ggml.bos_token_id'));
        const padId = parseSafeNumber(getMetaValue(entriesByKey, 'tokenizer.ggml.padding_token_id'));
        const tokenArray = Array.isArray(tokens) ? tokens : [];
        const tokenTypeCounts = {};
        if (Array.isArray(tokenTypes)) {
            for (const value of tokenTypes) {
                const id = Number(value);
                const name = TOKEN_TYPES[id] || `TYPE_${id}`;
                tokenTypeCounts[name] = (tokenTypeCounts[name] || 0) + 1;
            }
        }
        const special = [];
        const addSpecial = (label, id) => {
            if (id === null || !Number.isInteger(id)) return;
            special.push({ label, id, token: typeof tokenArray[id] === 'string' ? tokenArray[id] : null });
        };
        addSpecial('EOS', eosId);
        addSpecial('BOS', bosId);
        addSpecial('PAD', padId);
        const fileTypeName = fileTypeId === null ? null : (FILE_TYPE_NAMES[fileTypeId] || `FILE_TYPE_${fileTypeId}`);
        const dominantType = Object.values(typeStats)
            .filter(item => item.bytes > 0)
            .sort((a, b) => b.bytes - a.bytes)[0] || null;
        const tokenizer = {
            available: Array.isArray(tokens),
            model: getMetaValue(entriesByKey, 'tokenizer.ggml.model') || null,
            pre: getMetaValue(entriesByKey, 'tokenizer.ggml.pre') || null,
            tokenCount: tokenArray.length,
            mergeCount: Array.isArray(merges) ? merges.length : 0,
            tokenTypeCounts,
            special,
            hasChatTemplate: typeof getMetaValue(entriesByKey, 'tokenizer.chat_template') === 'string',
        };
        return {
            architecture,
            modelName: getMetaValue(entriesByKey, 'general.name')
                || getMetaValue(entriesByKey, 'general.basename')
                || null,
            sizeLabel: getMetaValue(entriesByKey, 'general.size_label') || null,
            description: getMetaValue(entriesByKey, 'general.description') || null,
            author: getMetaValue(entriesByKey, 'general.author') || null,
            fileTypeId,
            fileTypeName,
            quantVersion,
            dominantType,
            contextLength: parseSafeNumber(contextLength),
            embeddingLength: parseSafeNumber(embeddingLength),
            blockCount: parseSafeNumber(blockCount),
            headCount: parseSafeNumber(headCount),
            headCountKv: parseSafeNumber(headCountKv),
            tokenizer,
            metadataKeys: metadata.length,
            ...stats,
        };
    }

    async function parse(input, options = {}) {
        const started = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
        const onProgress = typeof options.onProgress === 'function' ? options.onProgress : null;
        const notify = async (message, ratio) => {
            if (onProgress) await onProgress(message, ratio);
        };
        const source = input instanceof ByteSource ? input : new ByteSource(input);
        const cursor = new Cursor(source);
        const warnings = [];

        await notify('读取 GGUF 文件头', 0.04);
        const magicBytes = await cursor.read(4);
        const magic = decodeString(magicBytes);
        if (magic !== GGUF_MAGIC) {
            if (magicBytes[0] === 0x46 && magicBytes[1] === 0x55 && magicBytes[2] === 0x47 && magicBytes[3] === 0x47) {
                fail('检测到字节序相反的 GGUF；当前查看器只支持小端（little-endian）文件。', 0);
            }
            fail(`不是 GGUF 文件：magic="${magic.replace(/[^\x20-\x7e]/g, '·')}"。`, 0);
        }
        const version = await cursor.u32();
        if (!SUPPORTED_VERSIONS.has(version)) {
            const hint = version === 1 ? 'GGUF v1 已过时且编码不同，' : '';
            fail(`${hint}不支持 GGUF v${version}；仅支持 v2 / v3。`, 4);
        }
        if (version === 2) {
            warnings.push({
                severity: 'info',
                title: 'GGUF v2 文件',
                detail: '按 v2 结构解析；v3 主要增加了大端存储支持，本项目展示为小端文件。',
            });
        }
        const tensorCount = await cursor.u64();
        const metadataCount = await cursor.u64();
        if (tensorCount > LIMITS.tensorCount) fail(`张量数量 ${tensorCount} 超出查看器上限。`, 8);
        if (metadataCount > LIMITS.metadataCount) fail(`元数据条目数量 ${metadataCount} 超出查看器上限。`, 16);
        const header = {
            start: 0,
            end: cursor.position,
            size: cursor.position,
            magic,
            version,
            tensorCount,
            metadataCount,
        };

        await notify('解析元数据键值对', 0.16);
        const metadataStart = cursor.position;
        const metadata = [];
        const entriesByKey = new Map();
        const seenKeys = new Set();
        for (let index = 0; index < metadataCount; index += 1) {
            const start = cursor.position;
            const key = decodeString(await cursor.stringBytes());
            const typeId = await cursor.u32();
            const typeInfo = metadataTypeInfo(typeId);
            if (!typeInfo || !(typeId in METADATA_TYPES)) fail(`元数据 "${key}" 使用了未知类型 id=${typeId}。`, cursor.position - 4);
            if (seenKeys.has(key)) {
                warnings.push({
                    severity: 'warn',
                    title: '重复的元数据键',
                    detail: `"${key}" 出现了不止一次；索引与详情面板使用第一次出现的值。`,
                });
            }
            seenKeys.add(key);
            const valueStart = cursor.position;
            const capture = {};
            const value = await readMetadataValue(cursor, typeId, 0, warnings, key, capture);
            const entry = {
                index,
                key,
                typeId,
                typeInfo,
                value,
                elementTypeId: typeId === 9 ? capture.elementTypeId : null,
                arrayCount: typeId === 9 ? capture.arrayCount : null,
                start,
                end: cursor.position,
                valueStart,
            };
            const normalized = normalizeEntry(entry);
            metadata.push(normalized);
            if (!entriesByKey.has(key)) entriesByKey.set(key, normalized);
            if ((index & 63) === 0) await notify(`解析元数据 ${index}/${metadataCount}`, 0.16 + 0.22 * (index / Math.max(1, metadataCount)));
        }
        const metadataEnd = cursor.position;

        const alignmentEntry = entriesByKey.get('general.alignment');
        let alignment = DEFAULT_ALIGNMENT;
        let alignmentSource = '默认值';
        if (alignmentEntry) {
            const rawAlignment = alignmentEntry.value;
            const numeric = typeof rawAlignment === 'bigint' ? Number(rawAlignment) : Number(rawAlignment);
            if (!Number.isInteger(numeric) || numeric <= 0) {
                warnings.push({
                    severity: 'warn',
                    title: '无效的对齐值',
                    detail: '"general.alignment" 不是正整数，已回退到 32 B。',
                });
            } else if (numeric > LIMITS.alignment) {
                warnings.push({
                    severity: 'warn',
                    title: '异常的对齐值',
                    detail: `对齐值 ${numeric} B 超出常规范围，已回退到 32 B 以避免错误索引。`,
                });
            } else {
                alignment = numeric;
                alignmentSource = 'general.alignment';
                if (numeric < 8 || (numeric & (numeric - 1)) !== 0) {
                    warnings.push({
                        severity: 'warn',
                        title: '非规范对齐值',
                        detail: `GGUF 规范要求 8 的倍数（通常为 2 的幂）；文件中写的是 ${numeric} B，仍按此值计算。`,
                    });
                }
            }
        }

        await notify('读取张量索引', 0.45);
        const tensorInfoStart = cursor.position;
        const rawTensors = [];
        const tensorNames = new Set();
        for (let index = 0; index < tensorCount; index += 1) {
            const infoStart = cursor.position;
            const name = decodeString(await cursor.stringBytes());
            const rank = await cursor.u32();
            if (rank > LIMITS.rank) fail(`张量 "${name}" 的维数 ${rank} 超出查看器上限 ${LIMITS.rank}。`, cursor.position - 4);
            const dimensions = [];
            for (let dim = 0; dim < rank; dim += 1) dimensions.push(await cursor.u64());
            const typeId = await cursor.u32();
            const offset = await cursor.u64();
            const infoEnd = cursor.position;
            if (tensorNames.has(name)) {
                warnings.push({
                    severity: 'warn',
                    title: '重复的张量名',
                    detail: `"${name}" 在索引中出现多次；按位置区分，按名称查找会存在歧义。`,
                });
            }
            tensorNames.add(name);
            if (rank > 4) {
                warnings.push({
                    severity: 'info',
                    title: '高维张量',
                    detail: `"${name}" 有 ${rank} 维；GGML 传统上限制为 4 维，查看器仍按索引展示。`,
                });
            }
            rawTensors.push({
                index,
                name,
                rank,
                dimensions,
                typeId,
                offset,
                infoStart,
                infoEnd,
            });
            if ((index & 127) === 0) await notify(`读取张量索引 ${index}/${tensorCount}`, 0.45 + 0.2 * (index / Math.max(1, tensorCount)));
        }
        const tensorInfoEnd = cursor.position;
        const dataOffset = alignUp(tensorInfoEnd, alignment);
        if (dataOffset > source.size) {
            fail(`张量数据区起点 ${dataOffset} 超出文件大小 ${source.size}。`, tensorInfoEnd);
        }
        const dataPadding = dataOffset - tensorInfoEnd;

        const tensorInfoById = new Map();
        const typeStats = {};
        let parameterCount = 0;
        let storedBytes = 0;
        let knownTensorCount = 0;
        let unknownTensorCount = 0;
        let maxDataEnd = 0;
        let tensorCountWithElements = 0;

        for (const raw of rawTensors) {
            const typeInfo = tensorTypeInfo(raw.typeId);
            const preview = typeInfo;
            let elementCount = 1;
            let elementOverflow = false;
            for (const dim of raw.dimensions) {
                if (dim === 0) {
                    warnings.push({
                        severity: 'warn',
                        title: '零长度维度',
                        detail: `张量 "${raw.name}" 的维度包含 0。`,
                    });
                }
                if (!Number.isSafeInteger(elementCount * dim)) {
                    elementOverflow = true;
                    break;
                }
                elementCount *= dim;
            }
            if (!raw.dimensions.length) elementCount = 0;
            if (elementOverflow) {
                warnings.push({
                    severity: 'warn',
                    title: '张量元素数溢出',
                    detail: `张量 "${raw.name}" 的元素数量超出 JavaScript 安全整数范围。`,
                });
                elementCount = null;
            }
            let storedSize = null;
            let blocks = null;
            if (typeInfo.known && elementCount !== null) {
                if (raw.rank > 0 && raw.dimensions[0] % typeInfo.block !== 0) {
                    warnings.push({
                        severity: 'warn',
                        title: '行宽与量化块不匹配',
                        detail: `"${raw.name}" 的 ne[0]=${raw.dimensions[0]} 不是 ${typeInfo.name} 块大小 ${typeInfo.block} 的整数倍。`,
                    });
                }
                if (elementCount % typeInfo.block === 0) {
                    blocks = elementCount / typeInfo.block;
                    storedSize = blocks * typeInfo.size;
                } else {
                    blocks = Math.ceil(elementCount / typeInfo.block);
                    storedSize = blocks * typeInfo.size;
                    warnings.push({
                        severity: 'warn',
                        title: '不完整的量化块',
                        detail: `"${raw.name}" 的元素总数未对齐到 ${typeInfo.block}，尺寸按向上取整估算。`,
                    });
                }
            }
            const absoluteStart = dataOffset + raw.offset;
            const absoluteEnd = storedSize === null ? null : absoluteStart + storedSize;
            const outOfBounds = absoluteEnd !== null && absoluteEnd > source.size;
            if (outOfBounds) {
                warnings.push({
                    severity: 'warn',
                    title: '张量数据越界',
                    detail: `"${raw.name}" 的数据区间 ${absoluteStart}..${absoluteEnd} 超出文件末尾 ${source.size}。`,
                });
            }
            if (raw.offset % alignment !== 0) {
                warnings.push({
                    severity: 'warn',
                    title: '张量偏移未对齐',
                    detail: `"${raw.name}" 的 offset=${raw.offset} 不是对齐值 ${alignment} 的整数倍。`,
                });
            }
            const tensor = {
                ...raw,
                typeInfo,
                preview,
                elementCount,
                blocks,
                storedSize,
                bitsPerWeight: typeInfo.bits,
                absoluteStart,
                absoluteEnd,
                outOfBounds,
            };
            tensorInfoById.set(tensor.index, tensor);

            if (tensor.elementCount !== null) {
                parameterCount += tensor.elementCount;
                tensorCountWithElements += 1;
            }
            if (typeInfo.known) {
                knownTensorCount += 1;
                if (storedSize !== null) {
                    storedBytes += storedSize;
                    maxDataEnd = Math.max(maxDataEnd, raw.offset + storedSize);
                }
                if (!typeStats[raw.typeId]) {
                    typeStats[raw.typeId] = {
                        id: raw.typeId,
                        name: typeInfo.name,
                        family: typeInfo.family,
                        bits: typeInfo.bits,
                        tensorCount: 0,
                        parameters: 0,
                        bytes: 0,
                    };
                }
                typeStats[raw.typeId].tensorCount += 1;
                if (tensor.elementCount !== null) typeStats[raw.typeId].parameters += tensor.elementCount;
                if (storedSize !== null) typeStats[raw.typeId].bytes += storedSize;
            } else {
                unknownTensorCount += 1;
            }
        }

        await notify('校验数据区与对齐', 0.72);
        const sortedTensors = [...rawTensors]
            .filter(tensor => tensorInfoById.get(tensor.index).storedSize !== null)
            .sort((a, b) => a.offset - b.offset);
        let overlapBytes = 0;
        let dataGaps = 0;
        let previousEnd = 0;
        let previousName = null;
        for (const raw of sortedTensors) {
            const tensor = tensorInfoById.get(raw.index);
            if (raw.offset < previousEnd) {
                const overlap = previousEnd - raw.offset;
                overlapBytes += overlap;
                warnings.push({
                    severity: 'warn',
                    title: '张量数据区间重叠',
                    detail: `"${raw.name}" 与 "${previousName}" 的数据区间重叠 ${overlap} B。`,
                });
            } else if (raw.offset > previousEnd) {
                dataGaps += raw.offset - previousEnd;
            }
            previousEnd = Math.max(previousEnd, raw.offset + tensor.storedSize);
            previousName = raw.name;
        }
        const dataEnd = dataOffset + maxDataEnd;
        const trailingBytes = Math.max(0, source.size - dataEnd);
        if (dataEnd > source.size) {
            warnings.push({
                severity: 'warn',
                title: '数据区越过文件末尾',
                detail: `计算得到的数据末尾为 ${dataEnd} B，但文件只有 ${source.size} B。`,
            });
        }
        if (trailingBytes > 0) {
            warnings.push({
                severity: 'info',
                title: '文件末尾存在多余字节',
                detail: `最后一个张量之后还有 ${trailingBytes} B；可能是附带的元数据、分片信息或填充，查看器未将其归入任何张量。`,
            });
        }
        if (overlapBytes > 0) {
            warnings.push({
                severity: 'warn',
                title: '张量区间共享或重叠',
                detail: `检测到 ${overlapBytes} B 的重叠数据；某些文件会复用权重区间，但通常意味着 offset 索引异常。`,
            });
        }
        if (unknownTensorCount > 0) {
            warnings.push({
                severity: 'info',
                title: '存在未知张量类型',
                detail: `${unknownTensorCount} 个张量的类型不在内建类型表中；总字节数只是已知类型的合计。`,
            });
        }
        const alignmentPadding = dataGaps + dataPadding;
        const usedRatio = source.size > 0 ? (storedBytes / source.size) * 100 : 0;
        const bitsPerWeight = parameterCount > 0 ? (storedBytes * 8 / parameterCount) : null;
        const stats = {
            parameterCount,
            storedBytes,
            dataBytes: Math.max(0, dataEnd - dataOffset),
            knownTensorCount,
            unknownTensorCount,
            tensorCountWithElements,
            dataOffset,
            dataEnd,
            dataGaps,
            dataPadding,
            alignmentPadding,
            overlapBytes,
            trailingBytes,
            usedRatio,
            bitsPerWeight,
            metadataBytes: metadataEnd - metadataStart,
            tensorInfoBytes: tensorInfoEnd - tensorInfoStart,
            headerBytes: header.size,
        };

        const model = {
            version,
            sourceName: source.name || options.sourceName || '',
            fileSize: source.size,
            header,
            metadata,
            metadataStart,
            metadataEnd,
            tensorInfoStart,
            tensorInfoEnd,
            alignment,
            alignmentSource,
            dataPadding,
            dataOffset,
            tensors: rawTensors.map(raw => tensorInfoById.get(raw.index)),
            tensorInfoById: tensorInfoById,
            typeStats,
            warnings,
            stats,
            readAt: (offset, length) => source.read(offset, length),
            parseMs: ((typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now()) - started,
        };
        model.entriesByKey = entriesByKey;
        model.summary = buildSummary(metadata, entriesByKey, model.tensors, typeStats, stats, warnings);

        await notify('完成', 1);
        return model;
    }

    async function decodePreview(model, tensor, startElement = 0, maxValues = 256) {
        const typeInfo = tensor.typeInfo || tensorTypeInfo(tensor.typeId);
        if (!typeInfo.known || !typeInfo.preview) {
            return {
                supported: false,
                reason: typeInfo.known ? '该量化类型暂不支持数值预览。' : '未知类型无法反量化。',
            };
        }
        if (tensor.elementCount === null || tensor.storedSize === null) {
            return { supported: false, reason: '张量尺寸无法确定。' };
        }
        const requested = Math.max(0, Math.floor(Number(startElement) || 0));
        const block = typeInfo.block;
        const firstBlock = Math.floor(requested / block);
        const alignedStart = firstBlock * block;
        const availableElements = tensor.elementCount - alignedStart;
        if (availableElements <= 0) {
            return { supported: false, reason: '起始元素超出张量范围。' };
        }
        const count = Math.min(Math.max(1, Math.floor(maxValues)), availableElements);
        const blockCount = Math.ceil(count / block);
        const byteStart = tensor.absoluteStart + firstBlock * typeInfo.size;
        const byteLength = blockCount * typeInfo.size;
        if (tensor.absoluteEnd !== null && byteStart + byteLength > tensor.absoluteEnd) {
            return { supported: false, reason: '读取区间超出张量数据。' };
        }
        const bytes = await model.readAt(byteStart, byteLength);
        const decoded = dequantize(typeInfo.id, bytes);
        return {
            supported: true,
            values: decoded.slice(0, count),
            startElement: alignedStart,
            count,
            byteStart,
            byteLength,
            blockCount,
            typeInfo,
            aligned: alignedStart === requested,
        };
    }

    function dequantize(typeId, bytes) {
        const info = tensorTypeInfo(typeId);
        if (!info.known) throw new GGUFError(`未知张量类型 id=${typeId}。`);
        if (!info.preview) throw new GGUFError(`${info.name} 暂不支持反量化预览。`);
        if (bytes.byteLength === 0) return [];
        if (bytes.byteLength % info.size !== 0) {
            throw new GGUFError(`${info.name} 字节长度 ${bytes.byteLength} 不是块大小 ${info.size} 的整数倍。`);
        }
        const blockCount = bytes.byteLength / info.size;
        const output = new Float64Array(blockCount * info.block);
        for (let blockIndex = 0; blockIndex < blockCount; blockIndex += 1) {
            const base = blockIndex * info.size;
            const outBase = blockIndex * info.block;
            dequantizeBlock(typeId, bytes, base, output, outBase);
        }
        return Array.from(output);
    }

    function dequantizeBlock(typeId, bytes, base, output, out) {
        switch (typeId) {
        case 0:
            for (let i = 0; i < 1; i += 1) output[out + i] = readF32LE(bytes, base + i * 4);
            return;
        case 1:
            for (let i = 0; i < 1; i += 1) output[out + i] = f16ToNumber(readU16LE(bytes, base + i * 2));
            return;
        case 30:
            for (let i = 0; i < 1; i += 1) output[out + i] = bf16ToNumber(readU16LE(bytes, base + i * 2));
            return;
        case 28:
            for (let i = 0; i < 1; i += 1) output[out + i] = readF64LE(bytes, base + i * 8);
            return;
        case 24:
            for (let i = 0; i < 1; i += 1) {
                const value = bytes[base + i];
                output[out + i] = value & 0x80 ? value - 256 : value;
            }
            return;
        case 25:
            for (let i = 0; i < 1; i += 1) output[out + i] = readI16LE(bytes, base + i * 2);
            return;
        case 26:
            for (let i = 0; i < 1; i += 1) output[out + i] = readI32LE(bytes, base + i * 4);
            return;
        case 27:
            for (let i = 0; i < 1; i += 1) {
                let value = 0n;
                for (let k = 7; k >= 0; k -= 1) value = (value << 8n) | BigInt(bytes[base + i * 8 + k]);
                const signed = value & (1n << 63n) ? value - (1n << 64n) : value;
                output[out + i] = Number(signed);
            }
            return;
        case 2: {
            const d = f16ToNumber(readU16LE(bytes, base));
            for (let i = 0; i < 16; i += 1) {
                const q = bytes[base + 2 + i];
                output[out + i] = ((q & 0x0f) - 8) * d;
                output[out + 16 + i] = ((q >> 4) - 8) * d;
            }
            return;
        }
        case 3: {
            const d = f16ToNumber(readU16LE(bytes, base));
            const m = f16ToNumber(readU16LE(bytes, base + 2));
            for (let i = 0; i < 16; i += 1) {
                const q = bytes[base + 4 + i];
                output[out + i] = (q & 0x0f) * d + m;
                output[out + 16 + i] = (q >> 4) * d + m;
            }
            return;
        }
        case 6: {
            const d = f16ToNumber(readU16LE(bytes, base));
            const high = readU32LE(bytes, base + 2);
            for (let i = 0; i < 32; i += 1) {
                const low = (bytes[base + 6 + (i % 16)] >> (i < 16 ? 0 : 4)) & 0x0f;
                const highBit = (high >>> i) & 1;
                output[out + i] = ((low | (highBit << 4)) - 16) * d;
            }
            return;
        }
        case 7: {
            const d = f16ToNumber(readU16LE(bytes, base));
            const m = f16ToNumber(readU16LE(bytes, base + 2));
            const high = readU32LE(bytes, base + 4);
            for (let i = 0; i < 32; i += 1) {
                const low = (bytes[base + 8 + (i % 16)] >> (i < 16 ? 0 : 4)) & 0x0f;
                const highBit = (high >>> i) & 1;
                output[out + i] = (low | (highBit << 4)) * d + m;
            }
            return;
        }
        case 8: {
            const d = f16ToNumber(readU16LE(bytes, base));
            for (let i = 0; i < 32; i += 1) {
                const q = bytes[base + 2 + i];
                output[out + i] = (q & 0x80 ? q - 256 : q) * d;
            }
            return;
        }
        case 9: {
            const d = f16ToNumber(readU16LE(bytes, base));
            for (let i = 0; i < 32; i += 1) {
                const q = bytes[base + 4 + i];
                output[out + i] = (q & 0x80 ? q - 256 : q) * d;
            }
            return;
        }
        case 20: {
            const d = f16ToNumber(readU16LE(bytes, base));
            const kvalues = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
            for (let i = 0; i < 16; i += 1) {
                const q = bytes[base + 2 + i];
                output[out + i] = d * kvalues[q & 0x0f];
                output[out + 16 + i] = d * kvalues[q >> 4];
            }
            return;
        }
        case 39: {
            const d = e8m0ToNumber(bytes[base]);
            const kvalues = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
            for (let i = 0; i < 16; i += 1) {
                const q = bytes[base + 1 + i];
                output[out + i] = d * kvalues[q & 0x0f];
                output[out + 16 + i] = d * kvalues[q >> 4];
            }
            return;
        }
        case 40: {
            const kvalues = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
            for (let s = 0; s < 4; s += 1) {
                const d = ue4m3ToNumber(bytes[base + s]);
                const subBase = base + 4 + s * 8;
                const subOut = out + s * 16;
                for (let i = 0; i < 8; i += 1) {
                    const q = bytes[subBase + i];
                    output[subOut + i] = d * kvalues[q & 0x0f];
                    output[subOut + 8 + i] = d * kvalues[q >> 4];
                }
            }
            return;
        }
        case 41: {
            const d = f16ToNumber(readU16LE(bytes, base));
            for (let i = 0; i < 128; i += 1) {
                const bit = (bytes[base + 2 + (i >> 3)] >>> (i & 7)) & 1;
                output[out + i] = bit ? d : -d;
            }
            return;
        }
        case 42: {
            const d = f16ToNumber(readU16LE(bytes, base));
            for (let i = 0; i < 16; i += 1) {
                const q = bytes[base + 2 + i];
                output[out + i * 4 + 0] = ((q >> 0) & 0x03) * d - d;
                output[out + i * 4 + 1] = ((q >> 2) & 0x03) * d - d;
                output[out + i * 4 + 2] = ((q >> 4) & 0x03) * d - d;
                output[out + i * 4 + 3] = ((q >> 6) & 0x03) * d - d;
            }
            return;
        }
        case 34: {
            const d = f16ToNumber(readU16LE(bytes, base + 52));
            const pow3 = [1, 3, 9, 27, 81, 243];
            let p = 0;
            for (let j = 0; j < 32; j += 32) {
                for (let n = 0; n < 5; n += 1) {
                    for (let m = 0; m < 32; m += 1) {
                        const q = (bytes[base + j + m] * pow3[n]) & 0xff;
                        const xi = ((q * 3) >>> 8);
                        output[out + p] = (xi - 1) * d;
                        p += 1;
                    }
                }
            }
            for (let j = 32; j < 48; j += 16) {
                for (let n = 0; n < 5; n += 1) {
                    for (let m = 0; m < 16; m += 1) {
                        const q = (bytes[base + j + m] * pow3[n]) & 0xff;
                        const xi = ((q * 3) >>> 8);
                        output[out + p] = (xi - 1) * d;
                        p += 1;
                    }
                }
            }
            for (let n = 0; n < 4; n += 1) {
                for (let j = 0; j < 4; j += 1) {
                    const q = (bytes[base + 48 + j] * pow3[n]) & 0xff;
                    const xi = ((q * 3) >>> 8);
                    output[out + p] = (xi - 1) * d;
                    p += 1;
                }
            }
            return;
        }
        case 35: {
            const d = f16ToNumber(readU16LE(bytes, base + 64));
            for (let chunk = 0; chunk < 2; chunk += 1) {
                for (let plane = 0; plane < 4; plane += 1) {
                    for (let i = 0; i < 32; i += 1) {
                        const q = (bytes[base + chunk * 32 + i] >> (plane * 2)) & 0x03;
                        output[out + chunk * 128 + plane * 32 + i] = (q - 1) * d;
                    }
                }
            }
            return;
        }
        case 10: {
            const scales = bytes.subarray(base, base + 16);
            const qsBase = base + 16;
            const d = f16ToNumber(readU16LE(bytes, base + 80));
            const dmin = f16ToNumber(readU16LE(bytes, base + 82));
            let scaleIndex = 0;
            let p = 0;
            for (let group = 0; group < 2; group += 1) {
                let shift = 0;
                const qBase = qsBase + group * 32;
                for (let j = 0; j < 4; j += 1) {
                    let sc = scales[scaleIndex];
                    scaleIndex += 1;
                    let dl = d * (sc & 0x0f);
                    let ml = dmin * (sc >> 4);
                    for (let l = 0; l < 16; l += 1) {
                        output[out + p] = dl * ((bytes[qBase + l] >> shift) & 0x03) - ml;
                        p += 1;
                    }
                    sc = scales[scaleIndex];
                    scaleIndex += 1;
                    dl = d * (sc & 0x0f);
                    ml = dmin * (sc >> 4);
                    for (let l = 0; l < 16; l += 1) {
                        output[out + p] = dl * ((bytes[qBase + 16 + l] >> shift) & 0x03) - ml;
                        p += 1;
                    }
                    shift += 2;
                }
            }
            return;
        }
        case 11: {
            const hmask = bytes.subarray(base, base + 32);
            const qsBase = base + 32;
            const scalesBytes = bytes.subarray(base + 96, base + 108);
            const d = f16ToNumber(readU16LE(bytes, base + 108));
            const low = new Array(16);
            for (let i = 0; i < 8; i += 1) {
                low[i] = scalesBytes[i] & 0x0f;
                low[i + 8] = scalesBytes[i] >> 4;
            }
            const scales = new Array(16);
            for (let k = 0; k < 4; k += 1) {
                for (let i = 0; i < 4; i += 1) {
                    const high = (scalesBytes[8 + i] >> (k * 2)) & 0x03;
                    scales[k * 4 + i] = (low[k * 4 + i] | (high << 4)) - 32;
                }
            }
            let scaleIndex = 0;
            let p = 0;
            let m = 1;
            for (let group = 0; group < 2; group += 1) {
                const qBase = qsBase + group * 32;
                let shift = 0;
                for (let j = 0; j < 4; j += 1) {
                    let dl = d * scales[scaleIndex];
                    scaleIndex += 1;
                    for (let l = 0; l < 16; l += 1) {
                        const q = (bytes[qBase + l] >> shift) & 0x03;
                        const sub = hmask[l] & m ? 0 : 4;
                        output[out + p] = dl * (q - sub);
                        p += 1;
                    }
                    dl = d * scales[scaleIndex];
                    scaleIndex += 1;
                    for (let l = 0; l < 16; l += 1) {
                        const q = (bytes[qBase + 16 + l] >> shift) & 0x03;
                        const sub = hmask[16 + l] & m ? 0 : 4;
                        output[out + p] = dl * (q - sub);
                        p += 1;
                    }
                    shift += 2;
                    m <<= 1;
                }
            }
            return;
        }
        case 12:
        case 13: {
            const d = f16ToNumber(readU16LE(bytes, base));
            const dmin = f16ToNumber(readU16LE(bytes, base + 2));
            const scalesRaw = bytes.subarray(base + 4, base + 16);
            const qlBase = base + 16 + (typeId === 13 ? 32 : 0);
            const qhBase = base + 16;
            const sc = new Array(8);
            const mn = new Array(8);
            for (let j = 0; j < 4; j += 1) {
                sc[j] = scalesRaw[j] & 0x3f;
                mn[j] = scalesRaw[4 + j] & 0x3f;
                sc[j + 4] = (scalesRaw[8 + j] & 0x0f) | ((scalesRaw[j] >> 2) & 0x30);
                mn[j + 4] = (scalesRaw[8 + j] >> 4) | ((scalesRaw[4 + j] >> 2) & 0x30);
            }
            let p = 0;
            for (let group = 0; group < 4; group += 1) {
                const d1 = d * sc[group * 2];
                const m1 = dmin * mn[group * 2];
                const d2 = d * sc[group * 2 + 1];
                const m2 = dmin * mn[group * 2 + 1];
                if (typeId === 12) {
                    for (let l = 0; l < 32; l += 1) {
                        output[out + p] = d1 * (bytes[qlBase + group * 32 + l] & 0x0f) - m1;
                        p += 1;
                    }
                    for (let l = 0; l < 32; l += 1) {
                        output[out + p] = d2 * (bytes[qlBase + group * 32 + l] >> 4) - m2;
                        p += 1;
                    }
                } else {
                    const bit1 = 1 << (group * 2);
                    const bit2 = 2 << (group * 2);
                    for (let l = 0; l < 32; l += 1) {
                        const high = (bytes[qhBase + l] & bit1) ? 16 : 0;
                        output[out + p] = d1 * ((bytes[qlBase + group * 32 + l] & 0x0f) + high) - m1;
                        p += 1;
                    }
                    for (let l = 0; l < 32; l += 1) {
                        const high = (bytes[qhBase + l] & bit2) ? 16 : 0;
                        output[out + p] = d2 * ((bytes[qlBase + group * 32 + l] >> 4) + high) - m2;
                        p += 1;
                    }
                }
            }
            return;
        }
        case 14: {
            const d = f16ToNumber(readU16LE(bytes, base + 208));
            for (let group = 0; group < 2; group += 1) {
                const qlBase = base + group * 64;
                const qhBase = base + 128 + group * 32;
                const scaleBase = base + 192 + group * 8;
                const outBase = out + group * 128;
                for (let l = 0; l < 32; l += 1) {
                    const is = l >> 4;
                    const q1 = ((bytes[qlBase + l] & 0x0f) | (((bytes[qhBase + l] >> 0) & 0x03) << 4)) - 32;
                    const q2 = ((bytes[qlBase + l + 32] & 0x0f) | (((bytes[qhBase + l] >> 2) & 0x03) << 4)) - 32;
                    const q3 = ((bytes[qlBase + l] >> 4) | (((bytes[qhBase + l] >> 4) & 0x03) << 4)) - 32;
                    const q4 = ((bytes[qlBase + l + 32] >> 4) | (((bytes[qhBase + l] >> 6) & 0x03) << 4)) - 32;
                    const sc = (offset) => (bytes[scaleBase + offset] & 0x80
                        ? bytes[scaleBase + offset] - 256
                        : bytes[scaleBase + offset]);
                    output[outBase + l] = d * sc(is) * q1;
                    output[outBase + 32 + l] = d * sc(is + 2) * q2;
                    output[outBase + 64 + l] = d * sc(is + 4) * q3;
                    output[outBase + 96 + l] = d * sc(is + 6) * q4;
                }
            }
            return;
        }
        case 15: {
            const d = readF32LE(bytes, base);
            for (let i = 0; i < 256; i += 1) {
                const q = bytes[base + 4 + i];
                output[out + i] = (q & 0x80 ? q - 256 : q) * d;
            }
            return;
        }
        default:
            throw new GGUFError(`反量化尚未实现：${tensorTypeInfo(typeId).name}。`);
        }
    }

    function formatBytes(value, digits = 1) {
        if (value === null || value === undefined || !Number.isFinite(Number(value))) return '--';
        const number = Number(value);
        const sign = number < 0 ? '-' : '';
        const absolute = Math.abs(number);
        if (absolute < 1024) return `${sign}${Math.round(absolute)} B`;
        const units = ['KiB', 'MiB', 'GiB', 'TiB', 'PiB'];
        let scaled = absolute / 1024;
        let unitIndex = 0;
        while (scaled >= 1024 && unitIndex < units.length - 1) {
            scaled /= 1024;
            unitIndex += 1;
        }
        return `${sign}${scaled.toFixed(scaled >= 100 ? 0 : digits)} ${units[unitIndex]}`;
    }

    function formatNumber(value, digits = 0) {
        if (value === null || value === undefined) return '--';
        if (typeof value === 'bigint') return value.toLocaleString('en-US');
        const number = Number(value);
        if (!Number.isFinite(number)) return String(value);
        return number.toLocaleString('en-US', {
            minimumFractionDigits: digits,
            maximumFractionDigits: digits,
        });
    }

    function formatCompact(value) {
        if (value === null || value === undefined) return '--';
        const number = Number(value);
        if (!Number.isFinite(number)) return String(value);
        const abs = Math.abs(number);
        if (abs >= 1e12) return `${(number / 1e12).toFixed(abs >= 1e13 ? 0 : 2)}T`;
        if (abs >= 1e9) return `${(number / 1e9).toFixed(abs >= 1e10 ? 0 : 2)}B`;
        if (abs >= 1e6) return `${(number / 1e6).toFixed(abs >= 1e7 ? 0 : 2)}M`;
        if (abs >= 1e3) return `${(number / 1e3).toFixed(abs >= 1e4 ? 0 : 2)}K`;
        return formatNumber(number);
    }

    function hex(value, width = 0) {
        if (typeof value === 'bigint') return `0x${value.toString(16).padStart(width, '0')}`;
        return `0x${Math.trunc(Number(value)).toString(16).padStart(width, '0')}`;
    }

    return {
        GGUFError,
        GGUF_MAGIC,
        DEFAULT_ALIGNMENT,
        METADATA_TYPES,
        TENSOR_TYPES,
        BLOCK_FIELDS,
        FILE_TYPE_NAMES,
        TOKEN_TYPES,
        LIMITS,
        parse,
        decodePreview,
        dequantize,
        metadataTypeInfo,
        tensorTypeInfo,
        typeLabelForEntry,
        summarizeValue,
        f16ToNumber,
        e8m0ToNumber,
        ue4m3ToNumber,
        formatBytes,
        formatNumber,
        formatCompact,
        hex,
    };
});
