(function () {
    'use strict';

    const G = window.GGUF;
    const $ = (selector, root = document) => root.querySelector(selector);
    const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

    const TYPE_COLORS = {
        F32: '#4c7184',
        F16: '#698b9d',
        BF16: '#5e8c8e',
        F64: '#3e6074',
        Q1_0: '#a9523d',
        Q2_0: '#b96b45',
        Q2_K: '#a7653d',
        Q3_K: '#a88637',
        Q4_0: '#9a772f',
        Q4_1: '#aa8231',
        Q4_K: '#84752f',
        Q5_0: '#647d34',
        Q5_1: '#718b3c',
        Q5_K: '#56864b',
        Q6_K: '#2f8257',
        Q8_0: '#257247',
        Q8_1: '#2b6a55',
        Q8_K: '#1f6a5c',
        TQ1_0: '#7c688b',
        TQ2_0: '#8c6d95',
        MXFP4: '#7a5d79',
        NVFP4: '#8d5f7e',
    };

    const FAMILY_COLORS = {
        float: '#54798a',
        int: '#68796a',
        low: '#b06a49',
        mid: '#9b8434',
        q4: '#9a772f',
        q5: '#6f843a',
        q8: '#2f8257',
        iq: '#2f6b60',
        ternary: '#7c688b',
        binary: '#a9523d',
        mxfp: '#7a5d79',
        unknown: '#829487',
    };

    const TOKEN_TYPE_COLORS = {
        NORMAL: '#5f9773',
        UNKNOWN: '#a2b0a4',
        CONTROL: '#b3543f',
        USER_DEFINED: '#755e84',
        UNUSED: '#c3b57f',
        BYTE: '#4c7184',
    };

    const GPT2_BYTE_TO_CHAR = (() => {
        const direct = [];
        for (let code = 33; code <= 126; code += 1) direct.push(code);
        for (let code = 161; code <= 172; code += 1) direct.push(code);
        for (let code = 174; code <= 255; code += 1) direct.push(code);
        const mapped = new Set(direct);
        const bytes = direct.slice();
        let extra = 0;
        for (let code = 0; code < 256; code += 1) {
            if (!mapped.has(code)) {
                bytes.push(code);
                direct.push(256 + extra);
                extra += 1;
            }
        }
        return new Map(bytes.map((byte, index) => [String.fromCodePoint(direct[index]), byte]));
    })();
    const GPT2_DECODE_CACHE = new Map();
    const GPT2_UTF8_DECODER = new TextDecoder('utf-8', { fatal: true });

    const METADATA_TYPE_NOTES = {
        0: '无符号 8 位整数',
        1: '有符号 8 位整数',
        2: '无符号 16 位整数',
        3: '有符号 16 位整数',
        4: '无符号 32 位整数',
        5: '有符号 32 位整数',
        6: 'IEEE 754 单精度浮点',
        7: '1 字节布尔，仅 0 / 1',
        8: '8 字节长度 + UTF-8 字节（无 NUL）',
        9: 'u32 元素类型 + u64 元素个数 + 连续元素',
        10: '无符号 64 位整数',
        11: '有符号 64 位整数',
        12: 'IEEE 754 双精度浮点',
    };

    const state = {
        model: null,
        loadSeq: 0,
        activeView: 'overview',
        lastFile: null,
        lastError: '',
        meta: {
            query: '',
            filter: 'all',
            selectedIndex: null,
            arrayQuery: '',
            arrayPage: 0,
        },
        tensor: {
            query: '',
            typeFilter: 'all',
            sortKey: null,
            sortDir: 1,
            selectedIndex: null,
            limit: 800,
            previewTab: 'values',
            previewStart: 0,
            previewByte: 0,
            blockStart: 0,
            previewSeq: 0,
        },
        hex: {
            offset: 0,
            cols: 16,
            seq: 0,
        },
    };

    const el = {};
    let toastTimer = null;
    let tensorSearchTimer = null;
    let metaSearchTimer = null;
    let lastMetaSelection = null;

    const ESCAPE_MAP = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };

    function esc(value) {
        return String(value ?? '').replace(/[&<>"']/g, character => ESCAPE_MAP[character]);
    }

    function formatBytes(value, digits = 1) {
        return G.formatBytes(value, digits);
    }

    function formatNumber(value, digits = 0) {
        return G.formatNumber(value, digits);
    }

    function compact(value) {
        return G.formatCompact(value);
    }

    function hexOffset(value, width = 8) {
        return G.hex(value, width);
    }

    function formatShape(dimensions) {
        if (!dimensions || !dimensions.length) return 'scalar';
        return dimensions.map(value => formatNumber(value)).join(' × ');
    }

    function formatFloat(value, precision = 6) {
        if (typeof value === 'bigint') return value.toString();
        if (value === null || value === undefined) return '--';
        const number = Number(value);
        if (!Number.isFinite(number)) return String(number);
        if (Object.is(number, -0)) return '-0';
        if (number === 0) return '0';
        const absolute = Math.abs(number);
        if (absolute < 1e-4 || absolute >= 1e7) {
            return number.toExponential(Math.max(1, precision - 2))
                .replace(/e([+-])0+(\d+)/, 'e$1$2')
                .replace(/(\.\d*?)0+e/, '$1e')
                .replace(/\.e/, 'e');
        }
        return String(Number(number.toPrecision(precision)));
    }

    function cleanFileType(name) {
        if (!name) return 'UNSPECIFIED';
        return String(name).replace(/^MOSTLY_/, '');
    }

    function truncate(value, limit = 120) {
        const text = String(value ?? '');
        return text.length > limit ? `${text.slice(0, limit)}…` : text;
    }

    function metaTypeClass(entry) {
        if (entry.typeId === 8) return 'meta-string';
        if (entry.typeId === 9) return 'meta-array';
        if (entry.typeId === 7) return 'meta-bool';
        return 'meta-num';
    }

    function tensorTypeClass(typeInfo) {
        return `t-family-${typeInfo.family || 'unknown'}`;
    }

    function tensorBadge(typeInfo) {
        return `<span class="type-badge ${tensorTypeClass(typeInfo)}">${esc(typeInfo.name)}</span>`;
    }

    function typeColor(typeInfo) {
        return TYPE_COLORS[typeInfo.name] || FAMILY_COLORS[typeInfo.family] || FAMILY_COLORS.unknown;
    }

    function supportsBlockView(typeInfo) {
        if (!typeInfo || !typeInfo.known || typeInfo.block <= 1 || !Array.isArray(typeInfo.blockFields)) return false;
        return typeInfo.blockFields.reduce((sum, field) => sum + field.bytes, 0) === typeInfo.size;
    }

    function defaultPreviewTab(tensor) {
        if (!tensor) return 'raw';
        if (supportsBlockView(tensor.typeInfo)) return 'blocks';
        return tensor.typeInfo.preview ? 'values' : 'raw';
    }

    function fieldAtByte(typeInfo, offsetInBlock) {
        if (!typeInfo || !typeInfo.blockFields) return null;
        let start = 0;
        for (const field of typeInfo.blockFields) {
            if (offsetInBlock >= start && offsetInBlock < start + field.bytes) {
                return { field, start, indexInField: offsetInBlock - start };
            }
            start += field.bytes;
        }
        return null;
    }

    function fieldKindClass(kind) {
        return `field-kind-${kind || 'quant'}`;
    }

    function readU16LE(bytes, offset) {
        return (bytes[offset] | (bytes[offset + 1] << 8)) >>> 0;
    }

    function decodeFieldScalar(field, bytes, offset) {
        try {
            if (field.format === 'f16') return G.f16ToNumber(readU16LE(bytes, offset));
            if (field.format === 'f32') {
                const view = new DataView(bytes.buffer, bytes.byteOffset + offset, Math.min(4, bytes.byteLength - offset));
                return view.getFloat32(0, true);
            }
            if (field.format === 'e8m0') return G.e8m0ToNumber(bytes[offset]);
            if (field.format === 'ue4m3') return G.ue4m3ToNumber(bytes[offset]);
        } catch (_error) {
            return null;
        }
        return null;
    }

    function fieldValueText(field, bytes, start) {
        const value = decodeFieldScalar(field, bytes, start);
        return value === null ? '' : formatFloat(value, 8);
    }

    function blockLocation(tensor, blockIndex) {
        const type = tensor.typeInfo;
        const dims = tensor.dimensions;
        const result = {
            blockIndex,
            blockCount: tensor.blocks,
            elementStart: blockIndex * type.block,
            elementEnd: Math.min(tensor.elementCount, (blockIndex + 1) * type.block) - 1,
            group: null,
            groupsPerRow: null,
            flatRow: null,
            columnStart: null,
            columnEnd: null,
            coords: null,
        };
        if (dims.length > 0 && dims[0] % type.block === 0) {
            const groupsPerRow = dims[0] / type.block;
            result.groupsPerRow = groupsPerRow;
            result.group = blockIndex % groupsPerRow;
            result.flatRow = Math.floor(blockIndex / groupsPerRow);
            result.columnStart = result.group * type.block;
            result.columnEnd = result.columnStart + type.block - 1;
            if (dims.length > 1) {
                const coords = new Array(dims.length).fill(0);
                let remainder = result.flatRow;
                for (let i = 1; i < dims.length; i += 1) {
                    coords[i] = remainder % dims[i];
                    remainder = Math.floor(remainder / dims[i]);
                }
                result.coords = coords;
            }
        }
        return result;
    }

    function blockLocationText(tensor, location) {
        if (location.groupsPerRow === null) {
            return `元素 ${formatNumber(location.elementStart)}–${formatNumber(location.elementEnd)}`;
        }
        const coords = location.coords
            ? location.coords.slice(1).map((value, index) => `D${index + 1}=${formatNumber(value)}`).join(', ')
            : '';
        return `row ${formatNumber(location.flatRow)}${coords ? ` (${coords})` : ''} · group ${formatNumber(location.group)} / ${formatNumber(location.groupsPerRow)} · 列 ${formatNumber(location.columnStart)}–${formatNumber(location.columnEnd)}`;
    }

    function tensorByteAnnotation(tensor, absoluteOffset) {
        if (!tensor || absoluteOffset < tensor.absoluteStart || absoluteOffset >= tensor.absoluteStart + (tensor.storedSize || 0)) return null;
        const type = tensor.typeInfo;
        if (!type.blockFields) return null;
        const relative = absoluteOffset - tensor.absoluteStart;
        const blockIndex = Math.floor(relative / type.size);
        const offsetInBlock = relative % type.size;
        const located = fieldAtByte(type, offsetInBlock);
        if (!located) return null;
        let detail = '';
        if (type.name === 'Q1_0' && located.field.name === 'qs') {
            const firstBit = located.indexInField * 8;
            detail = ` · bits ${formatNumber(firstBit)}–${formatNumber(firstBit + 7)}`;
        }
        return {
            kind: located.field.kind,
            field: located.field,
            blockIndex,
            offsetInBlock,
            blockStart: offsetInBlock === 0,
            title: `${type.name} · block #${formatNumber(blockIndex)} · block byte +${formatNumber(offsetInBlock)} · ${located.field.name} · ${located.field.label}${detail}`,
        };
    }

    function renderFieldLegend(typeInfo, options = {}) {
        if (!typeInfo || !typeInfo.blockFields) return '';
        const note = options.note === false ? '' : `<span class="field-legend-note">每 ${formatNumber(typeInfo.size)} B 一块 · 块内左→右</span>`;
        return `<div class="field-legend">${typeInfo.blockFields.map(field => `
            <span class="field-legend-item ${fieldKindClass(field.kind)}">
                <i class="field-swatch"></i>
                <b>${esc(field.name)}</b>
                <span>${esc(field.label)}</span>
                <em>${formatNumber(field.bytes)} B</em>
            </span>`).join('')}${note}</div>`;
    }

    function getMetaEntry(key) {
        return state.model ? state.model.entriesByKey.get(key) : null;
    }

    function getMetaValue(key) {
        const entry = getMetaEntry(key);
        return entry ? entry.value : undefined;
    }

    function activeTensor() {
        if (!state.model || state.tensor.selectedIndex === null) return null;
        return state.model.tensorInfoById.get(state.tensor.selectedIndex) || null;
    }

    function activeMeta() {
        if (!state.model || state.meta.selectedIndex === null) return null;
        return state.model.metadata.find(entry => entry.index === state.meta.selectedIndex) || null;
    }

    function setText(node, value) {
        if (node) node.textContent = value;
    }

    function showToast(message, variant = '') {
        clearTimeout(toastTimer);
        el.toast.textContent = message;
        el.toast.className = `toast show ${variant}`.trim();
        toastTimer = setTimeout(() => {
            el.toast.className = 'toast';
        }, 2800);
    }

    function delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    function nextFrame() {
        return new Promise(resolve => requestAnimationFrame(() => resolve()));
    }

    function copyFallback(text) {
        const area = document.createElement('textarea');
        area.value = text;
        area.setAttribute('readonly', '');
        area.style.position = 'fixed';
        area.style.top = '-1000px';
        area.style.opacity = '0';
        document.body.appendChild(area);
        area.select();
        let copied = false;
        try {
            copied = document.execCommand('copy');
        } catch (_error) {
            copied = false;
        }
        area.remove();
        return copied;
    }

    async function copyText(text, label) {
        if (!text && text !== '') return;
        try {
            if (navigator.clipboard && navigator.clipboard.writeText) {
                await navigator.clipboard.writeText(text);
                showToast(`${label || '内容'}已复制`);
                return;
            }
        } catch (_error) {
            // Fall back for file:// pages where the async clipboard API is unavailable.
        }
        if (copyFallback(text)) {
            showToast(`${label || '内容'}已复制`);
        } else {
            showToast('浏览器拒绝了剪贴板访问', 'error');
        }
    }

    function setLed(mode) {
        el.fileLed.classList.remove('on', 'err');
        if (mode) el.fileLed.classList.add(mode);
    }

    function setProgress(ratio, step) {
        const safeRatio = Math.max(0, Math.min(1, Number(ratio) || 0));
        const percentage = Math.round(safeRatio * 100);
        el.loadBar.style.width = `${percentage}%`;
        el.loadPct.textContent = `${percentage}%`;
        if (step) el.loadStep.textContent = step;
    }

    function openFilePicker() {
        el.fileInput.click();
    }

    function showLoading(file) {
        el.loadOverlay.hidden = false;
        el.loadFile.textContent = `${file.name} · ${formatBytes(file.size)}`;
        el.loadStep.textContent = '读取文件头';
        el.loadBar.style.width = '2%';
        el.loadPct.textContent = '2%';
        setText(el.statusText, `正在解析 ${file.name}`);
    }

    function hideLoading() {
        el.loadOverlay.hidden = true;
    }

    function errorMessage(error) {
        let message = error && error.message ? error.message : String(error);
        if (error && Number.isInteger(error.offset)) {
            message += `\n位置：byte ${formatNumber(error.offset)} (${hexOffset(error.offset)})`;
        }
        return message;
    }

    async function loadFile(file) {
        if (!file) return;
        const sequence = ++state.loadSeq;
        state.lastError = '';
        el.welcomeError.hidden = true;
        showLoading(file);
        await nextFrame();
        const started = performance.now();
        try {
            const model = await G.parse(file, {
                onProgress: (message, ratio) => {
                    if (sequence !== state.loadSeq) return;
                    const current = parseInt(el.loadPct.textContent, 10) || 0;
                    setProgress(Math.max(current / 100, ratio), message);
                },
            });
            if (sequence !== state.loadSeq) return;
            const elapsed = performance.now() - started;
            if (elapsed < 220) await delay(220 - elapsed);
            setProgress(1, '渲染视图');
            await nextFrame();
            applyModel(model, file);
            hideLoading();
        } catch (error) {
            if (sequence !== state.loadSeq) return;
            hideLoading();
            state.lastError = `${file.name}\n${errorMessage(error)}`;
            if (!state.model) {
                setLed('err');
                setText(el.fileName, '未载入文件');
                setText(el.fileSize, '');
                setText(el.fileSub, '文件读取失败 · 检查上方提示');
                setText(el.statusText, `解析失败 · ${error && error.message ? error.message : '未知错误'}`);
                showWelcomeError();
                activateView('overview');
            } else {
                showToast(`解析失败：${error && error.message ? error.message : '未知错误'}`, 'error');
                setText(el.statusText, `解析失败 · 已保留当前文件`);
            }
            console.error(error);
        } finally {
            if (sequence === state.loadSeq) {
                el.fileInput.value = '';
            }
        }
    }

    function showWelcomeError() {
        if (!state.lastError) {
            el.welcomeError.hidden = true;
            return;
        }
        el.welcomeError.textContent = `无法解析文件\n${state.lastError}`;
        el.welcomeError.hidden = false;
    }

    function applyModel(model, file) {
        state.model = model;
        state.lastFile = file;
        state.lastError = '';
        el.welcomeError.hidden = true;
        state.meta.query = '';
        state.meta.filter = 'all';
        state.meta.arrayQuery = '';
        state.meta.arrayPage = 0;
        const preferred = model.metadata.find(entry => entry.key === 'general.architecture')
            || model.metadata[0]
            || null;
        state.meta.selectedIndex = preferred ? preferred.index : null;
        lastMetaSelection = state.meta.selectedIndex;

        state.tensor.query = '';
        state.tensor.typeFilter = 'all';
        state.tensor.sortKey = null;
        state.tensor.sortDir = 1;
        state.tensor.limit = 800;
        state.tensor.selectedIndex = model.tensors.length ? model.tensors[0].index : null;
        const firstType = state.tensor.selectedIndex !== null
            ? model.tensorInfoById.get(state.tensor.selectedIndex)
            : null;
        state.tensor.previewTab = defaultPreviewTab(firstType);
        state.tensor.previewStart = 0;
        state.tensor.previewByte = 0;
        state.tensor.blockStart = 0;
        state.hex.offset = 0;
        state.hex.cols = 16;

        el.metaSearch.value = '';
        el.tensorSearch.value = '';
        el.tensorFilterReset.disabled = false;
        setDataTabsEnabled(true);
        renderHeader(model);
        renderOverview(model);
        renderMetadata();
        populateTensorTypeFilter();
        renderTensorTable();
        renderTensorInspector(true);
        renderStructure();
        renderFormatCurrent();
        renderQ1Anatomy();
        activateView('overview', { keepScroll: true });
        setText(el.statusText, `已载入 ${file.name} · ${model.tensors.length} 张量 · ${model.metadata.length} 元数据项`);
    }

    function setDataTabsEnabled(enabled) {
        $$('.view-tab').forEach(tab => {
            if (tab.dataset.view === 'format' || tab.dataset.view === 'overview') return;
            if (enabled) tab.removeAttribute('aria-disabled');
            else tab.setAttribute('aria-disabled', 'true');
        });
    }

    function activateView(view, options = {}) {
        if (!state.model && view !== 'overview' && view !== 'format') {
            showToast('请先载入一个 GGUF 文件', 'error');
            return;
        }
        if (state.activeView === view && !options.force) {
            if (options.scrollHex) scrollToHex();
            return;
        }
        state.activeView = view;
        $$('.view-tab').forEach(tab => {
            const active = tab.dataset.view === view;
            tab.classList.toggle('active', active);
            tab.setAttribute('aria-selected', active ? 'true' : 'false');
        });
        $$('.view').forEach(panel => {
            const active = panel.id === `view${view.charAt(0).toUpperCase()}${view.slice(1)}`;
            panel.classList.toggle('active', active);
            panel.hidden = !active;
        });
        if (!options.keepScroll) el.main.scrollTop = 0;
        if (view === 'structure') {
            loadStructureHex();
        }
        if (view === 'tensors' && state.model && activeTensor()) {
            loadTensorPreview();
        }
        if (view === 'overview') {
            renderOverviewEmptyState();
        }
        if (options.scrollHex) {
            requestAnimationFrame(scrollToHex);
        }
    }

    function scrollToHex() {
        const panel = $('.hex-panel');
        if (panel) panel.scrollIntoView({ block: 'start', behavior: 'smooth' });
    }

    function renderOverviewEmptyState() {
        if (state.model) {
            el.overviewEmpty.hidden = true;
            el.overviewContent.hidden = false;
        } else {
            el.overviewEmpty.hidden = false;
            el.overviewContent.hidden = true;
        }
        showWelcomeError();
    }

    function renderHeader(model) {
        const name = model.sourceName || state.lastFile && state.lastFile.name || '未命名.gguf';
        setText(el.fileName, name);
        setText(el.fileSize, formatBytes(model.fileSize));
        setText(el.fileSub, `GGUF v${model.version} · ${formatNumber(model.tensors.length)} 张量 · ${formatNumber(model.metadata.length)} 元数据项 · 小端`);
        setLed('on');
        el.navStats.innerHTML = [
            `<span><b>v${model.version}</b> · LITTLE ENDIAN</span>`,
            '<span class="nav-divider"></span>',
            `<span><b>${formatNumber(model.tensors.length)}</b> TENSORS</span>`,
            '<span class="nav-divider"></span>',
            `<span><b>${formatNumber(model.metadata.length)}</b> KV</span>`,
        ].join('');
        setText(el.parseInfo, `PARSE ${formatNumber(model.parseMs, 0)} ms · ${formatBytes(model.fileSize)} · LOCAL`);
        document.title = `${name} · GGUF/LENS`;
    }

    function renderOverview(model) {
        el.overviewEmpty.hidden = true;
        el.overviewContent.hidden = false;
        const summary = model.summary;
        const stats = model.stats;
        const types = Object.values(model.typeStats).sort((a, b) => b.bytes - a.bytes);
        const typeNames = types.map(type => `${type.name} ${formatNumber(type.tensorCount)}`);
        const quantLabel = cleanFileType(summary.fileTypeName);
        const modelName = summary.modelName || model.sourceName || '未命名模型';

        setText(el.profileKicker, `MODEL PROFILE · GGUF v${model.version} · ${quantLabel}`);
        setText(el.profileName, modelName);
        setText(el.profileDesc, summary.description || [
            summary.architecture ? `${summary.architecture} 架构` : '未知架构',
            summary.sizeLabel ? `${summary.sizeLabel} 规模` : '',
            `${formatNumber(model.tensors.length)} 个张量保存在 ${formatBytes(model.fileSize)} 文件中。`,
        ].filter(Boolean).join(' · '));

        const tags = [];
        if (summary.architecture) tags.push(tag('ARCH', summary.architecture, 'accent'));
        tags.push(tag('GGUF', `v${model.version} · LE`, ''));
        if (quantLabel && quantLabel !== 'UNSPECIFIED') tags.push(tag('QUANT', quantLabel, 'warm'));
        if (summary.sizeLabel) tags.push(tag('SIZE', summary.sizeLabel, ''));
        if (summary.tokenizer.available) tags.push(tag('VOCAB', formatNumber(summary.tokenizer.tokenCount), ''));
        el.profileTags.innerHTML = tags.join('');

        const context = summary.contextLength ? `${formatNumber(summary.contextLength)} tokens` : '--';
        const layer = summary.blockCount ? `${formatNumber(summary.blockCount)} 层` : '--';
        const heads = summary.headCount
            ? `${formatNumber(summary.headCount)}${summary.headCountKv ? ` / ${formatNumber(summary.headCountKv)} KV` : ''}`
            : '--';
        el.profileFacts.innerHTML = [
            profileFact('CONTEXT', context),
            profileFact('LAYERS', layer),
            profileFact('ATTENTION', heads),
            profileFact('ALIGN', `${formatNumber(model.alignment)} B`),
        ].join('');

        setText(el.kpiParams, compact(stats.parameterCount));
        setText(el.kpiParamsExact, `${formatNumber(stats.parameterCount)} 个元素`);
        setText(el.kpiBpw, stats.bitsPerWeight === null ? '--' : formatNumber(stats.bitsPerWeight, 3));
        setText(el.kpiTensors, formatNumber(model.tensors.length));
        setText(el.kpiTensorTypes, typeNames.slice(0, 3).join(' · ') + (typeNames.length > 3 ? ' …' : ''));
        setText(el.kpiMeta, formatNumber(model.metadata.length));
        setText(el.kpiMetaSize, `${formatBytes(model.metadataEnd - model.metadataStart)} · ${formatNumber(model.metadata.length)} KEYS`);

        renderFileMap(el.overviewMap, el.overviewMapLegend, model);
        renderMix(model);
        renderArchitecture(model);
        renderTokenizer(model);
        renderLargest(model);
        renderDiagnostics(model);
    }

    function tag(label, value, variant) {
        return `<span class="tag ${variant}"><span class="tag-label">${esc(label)}</span>${esc(value)}</span>`;
    }

    function profileFact(label, value) {
        return `<span class="profile-fact"><span class="label">${esc(label)}</span><b>${esc(value)}</b></span>`;
    }

    function buildSegments(model) {
        const segments = [];
        const dataStart = model.dataOffset;
        const dataEnd = Math.min(Math.max(model.stats.dataEnd, dataStart), model.fileSize);
        const push = (name, cls, start, end, note) => {
            const safeStart = Math.max(0, Math.min(start, model.fileSize));
            const safeEnd = Math.max(safeStart, Math.min(end, model.fileSize));
            if (safeEnd <= safeStart) return;
            segments.push({ name, cls, start: safeStart, end: safeEnd, size: safeEnd - safeStart, note: note || '' });
        };
        push('文件头', 'seg-header', 0, 24, 'magic + version + counts');
        push('元数据', 'seg-metadata', model.metadataStart, model.metadataEnd, `${formatNumber(model.metadata.length)} key-value records`);
        push('张量索引', 'seg-index', model.tensorInfoStart, model.tensorInfoEnd, `${formatNumber(model.tensors.length)} tensor info records`);
        push('对齐填充', 'seg-padding', model.tensorInfoEnd, dataStart, `padding ${formatNumber(model.dataPadding)} B`);
        push('张量数据', 'seg-data', dataStart, dataEnd, `${formatBytes(model.stats.storedBytes)} accounted`);
        push('尾部', 'seg-tail', dataEnd, model.fileSize, 'unreferenced / trailing bytes');
        return segments;
    }

    function renderFileMap(container, legend, model, tall) {
        if (!container) return;
        const segments = buildSegments(model);
        const total = model.fileSize || 1;
        container.innerHTML = segments.map(segment => {
            const title = `${segment.name} · ${formatNumber(segment.start)}..${formatNumber(segment.end)} · ${formatBytes(segment.size)} ${segment.note ? `· ${segment.note}` : ''}`;
            const share = segment.size / total;
            return `<div class="map-seg ${segment.cls}" style="flex-grow:${Math.max(segment.size, 1)}" title="${esc(title)}" aria-label="${esc(title)}"><span>${share > 0.035 ? esc(segment.name) : ''}</span></div>`;
        }).join('');
        if (tall) container.classList.add('tall');
        if (legend) {
            legend.innerHTML = segments.map(segment => `
                <div class="map-legend-item" title="${esc(segment.note)}">
                    <i class="map-swatch" style="background:${segmentColor(segment.cls)}"></i>
                    <span class="map-name">${esc(segment.name)}</span>
                    <span class="map-value"><b>${formatBytes(segment.size)}</b> · ${hexOffset(segment.start, 8)}</span>
                </div>`).join('');
        }
    }

    function segmentColor(cls) {
        return {
            'seg-header': '#587866',
            'seg-metadata': '#a8c4af',
            'seg-index': '#d4c997',
            'seg-padding': '#e7eee3',
            'seg-data': '#a9cbb4',
            'seg-tail': '#d4ddd2',
        }[cls] || '#a9baac';
    }

    function groupTypesForMix(model) {
        const types = Object.values(model.typeStats)
            .filter(type => type.bytes > 0)
            .sort((a, b) => b.bytes - a.bytes);
        if (types.length <= 6) return { groups: types, hidden: 0 };
        const groups = types.slice(0, 5);
        const rest = types.slice(5);
        groups.push({
            id: -1,
            name: 'OTHER',
            family: 'unknown',
            tensorCount: rest.reduce((sum, type) => sum + type.tensorCount, 0),
            parameters: rest.reduce((sum, type) => sum + type.parameters, 0),
            bytes: rest.reduce((sum, type) => sum + type.bytes, 0),
            hidden: rest.length,
        });
        return { groups, hidden: rest.length };
    }

    function renderMix(model) {
        const { groups, hidden } = groupTypesForMix(model);
        const total = groups.reduce((sum, type) => sum + type.bytes, 0);
        let cursor = 0;
        const stops = [];
        for (const group of groups) {
            const start = cursor;
            cursor += total > 0 ? (group.bytes / total) * 100 : 0;
            const color = group.id === -1 ? '#a9b6aa' : typeColor({ name: group.name, family: group.family });
            stops.push(`${color} ${start.toFixed(3)}% ${cursor.toFixed(3)}%`);
        }
        if (!stops.length) stops.push('#d9e4d8 0% 100%');
        el.mixChart.style.background = `conic-gradient(${stops.join(', ')})`;
        setText(el.mixCenterBpw, model.stats.bitsPerWeight === null ? '--' : formatNumber(model.stats.bitsPerWeight, 3));
        el.mixNote.textContent = model.stats.unknownTensorCount ? 'KNOWN BYTES ONLY' : 'BY BYTES';
        if (!groups.length) {
            el.mixLegend.innerHTML = '<div class="fact-row"><span>没有可统计的已知张量类型</span><b>--</b></div>';
            return;
        }
        el.mixLegend.innerHTML = groups.map(group => {
            const share = total ? (group.bytes / total) * 100 : 0;
            const label = group.id === -1 ? `其他 ${group.hidden} 种` : group.name;
            const color = group.id === -1 ? '#a9b6aa' : typeColor({ name: group.name, family: group.family });
            const sub = group.id === -1
                ? `${formatNumber(group.tensorCount)} 个张量`
                : `${formatNumber(group.tensorCount)} 个张量 · ${formatNumber(group.parameters)} 参数`;
            return `<div class="mix-row">
                <i class="mix-color" style="background:${color}"></i>
                <span class="mix-type"><b>${esc(label)}</b><small>${esc(sub)}</small></span>
                <span class="mix-share"><b>${share.toFixed(1)}%</b>${formatBytes(group.bytes)}</span>
            </div>`;
        }).join('') + (hidden ? '' : '');
    }

    function renderArchitecture(model) {
        const summary = model.summary;
        const arch = summary.architecture;
        const get = suffix => {
            if (!arch) return undefined;
            return getMetaValue(`${arch}.${suffix}`);
        };
        const rows = [
            ['架构 architecture', summary.architecture || '--'],
            ['上下文 context', summary.contextLength ? `${formatNumber(summary.contextLength)} tokens` : '--'],
            ['隐藏维度 embedding', summary.embeddingLength ? formatNumber(summary.embeddingLength) : '--'],
            ['FFN 维度', Number.isFinite(Number(get('feed_forward_length'))) ? formatNumber(get('feed_forward_length')) : '--'],
            ['注意力头', summary.headCount ? `${formatNumber(summary.headCount)}${summary.headCountKv ? ` / ${formatNumber(summary.headCountKv)} KV` : ''}` : '--'],
            ['层数 blocks', summary.blockCount ? formatNumber(summary.blockCount) : '--'],
            ['量化 profile', cleanFileType(summary.fileTypeName) + (summary.quantVersion !== null ? ` · QV${formatNumber(summary.quantVersion)}` : '')],
            ['数据对齐', `${formatNumber(model.alignment)} B${model.alignmentSource === '默认值' ? '（默认）' : ''} · 小端`],
        ];
        el.archFacts.innerHTML = rows.map(([label, value]) => `
            <div class="fact-row"><span>${esc(label)}</span><b title="${esc(value)}">${esc(value)}</b></div>
        `).join('');
    }

    function renderTokenizer(model) {
        const tokenizer = model.summary.tokenizer;
        const flavor = tokenizer.model
            ? `${String(tokenizer.model).toUpperCase()}${tokenizer.pre ? ` / ${tokenizer.pre}` : ''}`
            : 'NO TOKENIZER';
        setText(el.tokenizerFlavor, flavor);
        if (!tokenizer.available) {
            el.tokenizerOverview.innerHTML = '<div class="token-empty">此文件没有 <code>tokenizer.ggml.tokens</code> 元数据，可能是纯权重、LoRA 或自定义结构文件。</div>';
            return;
        }
        const typeEntries = Object.entries(tokenizer.tokenTypeCounts)
            .filter(([, count]) => count > 0)
            .sort((a, b) => b[1] - a[1]);
        const typeTotal = typeEntries.reduce((sum, [, count]) => sum + count, 0);
        const bar = typeEntries.length && typeTotal ? `
            <div class="token-bar">${typeEntries.map(([name, count]) => `<i style="width:${(count / typeTotal) * 100}%;background:${TOKEN_TYPE_COLORS[name] || '#a9b6aa'}" title="${esc(name)} · ${formatNumber(count)}"></i>`).join('')}</div>
            <div class="token-bar-legend">${typeEntries.map(([name, count]) => `<span><i style="background:${TOKEN_TYPE_COLORS[name] || '#a9b6aa'}"></i>${esc(name)} ${formatNumber(count)}</span>`).join('')}</div>
        ` : '';
        const specials = tokenizer.special.length ? `
            <div class="token-special">${tokenizer.special.map(item => `<span class="special"><span>${esc(item.label)}</span><b title="${esc(item.token || '')}">${esc(item.token ?? '--')}</b><span>#${formatNumber(item.id)}</span></span>`).join('')}</div>
        ` : '';
        el.tokenizerOverview.innerHTML = `
            <div class="token-stat-grid">
                <div class="token-stat"><span>TOKENS</span><b>${formatNumber(tokenizer.tokenCount)}</b></div>
                <div class="token-stat"><span>MERGES</span><b>${formatNumber(tokenizer.mergeCount)}</b></div>
                <div class="token-stat"><span>CHAT TEMPLATE</span><b>${tokenizer.hasChatTemplate ? 'YES' : 'NO'}</b></div>
            </div>
            ${bar}
            ${specials}
        `;
    }

    function renderLargest(model) {
        const items = model.tensors
            .filter(tensor => tensor.storedSize !== null)
            .slice()
            .sort((a, b) => b.storedSize - a.storedSize)
            .slice(0, 6);
        if (!items.length) {
            el.largestList.innerHTML = '<div class="token-empty">没有可显示的张量。</div>';
            return;
        }
        const max = items[0].storedSize || 1;
        el.largestList.innerHTML = items.map(tensor => `
            <button class="largest-row" type="button" data-tensor-goto="${tensor.index}">
                <span class="largest-line">
                    <span class="largest-name" title="${esc(tensor.name)}">${esc(tensor.name)}</span>
                    <span class="largest-size">${formatBytes(tensor.storedSize)}</span>
                </span>
                <span class="largest-sub">
                    ${tensorBadge(tensor.typeInfo)}
                    <span class="largest-shape">[${esc(formatShape(tensor.dimensions))}]</span>
                    <span class="largest-bar"><i style="width:${Math.max(2, (tensor.storedSize / max) * 100)}%"></i></span>
                </span>
            </button>
        `).join('');
    }

    function renderDiagnostics(model) {
        const warnings = model.warnings || [];
        const warnCount = warnings.filter(item => item.severity === 'warn').length;
        const infoCount = warnings.filter(item => item.severity === 'info').length;
        el.diagnosticSummary.classList.remove('warn', 'info');
        if (warnCount) {
            el.diagnosticSummary.textContent = `${warnCount} WARN`;
            el.diagnosticSummary.classList.add('warn');
        } else if (infoCount) {
            el.diagnosticSummary.textContent = `${infoCount} NOTE`;
            el.diagnosticSummary.classList.add('info');
        } else {
            el.diagnosticSummary.textContent = 'PASS';
        }
        if (!warnings.length) {
            el.diagnosticsList.innerHTML = `
                <div class="diag-ok">
                    <span class="diag-icon">OK</span>
                    <span>头部、元数据、张量索引、偏移对齐与数据边界检查通过。</span>
                </div>
                <div class="diag-item">
                    <span class="diag-icon">i</span>
                    <div><b>文件重心：张量数据</b><p>${formatBytes(model.stats.storedBytes)} 已知权重占文件的 ${model.stats.usedRatio.toFixed(1)}%，平均 ${formatNumber(model.stats.bitsPerWeight, 3)} bit/权重。</p></div>
                </div>
            `;
            return;
        }
        const shown = warnings.slice(0, 12);
        el.diagnosticsList.innerHTML = shown.map(item => `
            <div class="diag-item ${item.severity}">
                <span class="diag-icon">${item.severity === 'warn' ? '!' : 'i'}</span>
                <div><b>${esc(item.title)}</b><p>${esc(item.detail)}</p></div>
            </div>
        `).join('') + (warnings.length > shown.length
            ? `<div class="diag-ok" style="grid-column:1/-1;color:var(--muted)"><span class="diag-icon">+</span><span>另有 ${warnings.length - shown.length} 条信息未展开。</span></div>`
            : '');
    }

    function metaFilteredEntries() {
        if (!state.model) return [];
        const query = state.meta.query.trim().toLowerCase();
        return state.model.metadata.filter(entry => {
            if (state.meta.filter === 'string' && entry.typeId !== 8) return false;
            if (state.meta.filter === 'array' && entry.typeId !== 9) return false;
            if (state.meta.filter === 'scalar' && (entry.typeId === 8 || entry.typeId === 9)) return false;
            if (!query) return true;
            const haystack = `${entry.key}\n${entry.summary}`.toLowerCase();
            return haystack.includes(query);
        });
    }

    function renderMetadata() {
        if (!state.model) return;
        const counts = {
            all: state.model.metadata.length,
            string: state.model.metadata.filter(entry => entry.typeId === 8).length,
            array: state.model.metadata.filter(entry => entry.typeId === 9).length,
            scalar: state.model.metadata.filter(entry => entry.typeId !== 8 && entry.typeId !== 9).length,
        };
        $$('#metaFilters button').forEach(button => {
            const filter = button.dataset.metaFilter;
            button.classList.toggle('active', filter === state.meta.filter);
            const counter = $('i', button);
            if (counter) counter.textContent = formatNumber(counts[filter] || 0);
        });
        const filtered = metaFilteredEntries();
        if (!filtered.some(entry => entry.index === state.meta.selectedIndex)) {
            state.meta.selectedIndex = filtered.length ? filtered[0].index : null;
            state.meta.arrayQuery = '';
            state.meta.arrayPage = 0;
        }
        el.metaList.innerHTML = filtered.length ? filtered.map(entry => {
            const selected = entry.index === state.meta.selectedIndex;
            return `<button class="metadata-row${selected ? ' selected' : ''}" type="button" data-meta-index="${entry.index}">
                <span class="metadata-index">${String(entry.index + 1).padStart(2, '0')}</span>
                <span class="metadata-key">${esc(entry.key)}</span>
                <span><span class="type-badge ${metaTypeClass(entry)}">${esc(entry.typeName)}</span></span>
                <span class="metadata-preview" title="${esc(entry.summary)}">${esc(truncate(entry.summary, 100))}</span>
                <span class="metadata-size">${formatBytes(entry.size)}</span>
            </button>`;
        }).join('') : '<div class="empty-result">没有匹配的元数据字段</div>';
        const rawBytes = state.model.metadataEnd - state.model.metadataStart;
        setText(el.metaCount, `显示 ${formatNumber(filtered.length)} / ${formatNumber(state.model.metadata.length)} 项 · 元数据区 ${formatBytes(rawBytes)}`);
        setText(el.metaStat, `GGUF v${state.model.version} · ${formatNumber(state.model.metadata.length)} FIELDS · ${formatBytes(rawBytes)}`);
        el.metaList.scrollTop = 0;
        renderMetaInspector();
    }

    function renderMetaInspector() {
        const entry = activeMeta();
        if (!entry) {
            const changed = lastMetaSelection !== null;
            lastMetaSelection = null;
            el.metaInspector.innerHTML = `<div class="inspector-empty">
                <span class="empty-mark">KV</span>
                <b>选择一个元数据字段</b>
                <p>左侧列表展示类型与摘要，点击字段可查看完整值、数组页和原始偏移。</p>
            </div>`;
            if (changed) el.metaInspector.scrollTop = 0;
            return;
        }
        if (lastMetaSelection !== entry.index) {
            state.meta.arrayQuery = '';
            state.meta.arrayPage = 0;
            lastMetaSelection = entry.index;
        }
        const isArray = entry.typeId === 9;
        const actions = [];
        actions.push('<button class="mini-btn" type="button" data-meta-action="copy-key">复制键</button>');
        if (!isArray) actions.push('<button class="mini-btn" type="button" data-meta-action="copy-value">复制值</button>');
        actions.push('<button class="mini-btn" type="button" data-meta-action="locate">结构中查看</button>');
        let valueHtml = '';
        if (isArray) {
            valueHtml = renderMetaArray(entry);
        } else {
            valueHtml = renderMetaScalar(entry);
        }
        el.metaInspector.innerHTML = `
            <div class="inspector-head">
                <div>
                    <span class="panel-kicker">FIELD INSPECTION · OFFSET ${hexOffset(entry.start)}</span>
                    <h2 class="inspector-title">${esc(entry.key)}</h2>
                    <div class="inspector-sub">
                        <span class="type-badge ${metaTypeClass(entry)}">${esc(entry.typeName)}</span>
                        <span>#${formatNumber(entry.index)}</span>
                        <span>${formatBytes(entry.size)}</span>
                    </div>
                    <div class="inspector-actions">${actions.join('')}</div>
                </div>
            </div>
            <div class="inspector-section">
                <div class="inspector-section-head"><h3>VALUE</h3><span>${isArray ? 'ARRAY' : 'SCALAR'}</span></div>
                ${valueHtml}
            </div>
            <div class="inspector-section">
                <div class="inspector-section-head"><h3>BYTE SPAN</h3><span>${formatBytes(entry.valueSize)} VALUE</span></div>
                <div class="byte-range">字段 <code>${hexOffset(entry.start)}</code> → <code>${hexOffset(entry.end)}</code> · 值从 <code>${hexOffset(entry.valueStart)}</code> 开始</div>
                <div class="value-meta">
                    <span>类型 ID <b>${entry.typeId}</b></span>
                    ${isArray ? `<span>元素类型 <b>${esc(G.metadataTypeInfo(entry.elementTypeId).name)}</b></span><span>元素数 <b>${formatNumber(entry.arrayCount)}</b></span>` : ''}
                    <span>字段字节 <b>${formatNumber(entry.size)}</b></span>
                </div>
            </div>
        `;
    }

    function renderMetaScalar(entry) {
        const value = entry.value;
        let display = '';
        let extra = '';
        if (typeof value === 'string') {
            const longString = value.includes('\n') || value.length > 180;
            display = `<div class="value-block${longString ? ' long-string' : ''}">${esc(value || '(empty string)')}</div>`;
            extra = `<div class="value-meta"><span>字符数 <b>${formatNumber(value.length)}</b></span><span>UTF-8 字节 <b>${formatNumber(entry.valueSize - 8)}</b></span></div>`;
        } else if (typeof value === 'boolean') {
            display = `<div class="value-block"><span class="big-bool">${value ? 'TRUE' : 'FALSE'}</span> <span style="color:var(--faint)">= ${value ? 1 : 0}</span></div>`;
        } else if (typeof value === 'bigint') {
            display = `<div class="value-block">${esc(value.toString())}</div>`;
        } else if (typeof value === 'number') {
            display = `<div class="value-block">${esc(formatFloat(value, 9))}</div>`;
            extra = `<div class="value-meta"><span>原始类型 <b>${esc(entry.typeName)}</b></span><span>${Number.isInteger(value) ? '整数' : '浮点'}</span></div>`;
        } else {
            display = `<div class="value-block">${esc(String(value))}</div>`;
        }
        return display + extra;
    }

    function renderMetaArray(entry) {
        const values = Array.isArray(entry.value) ? entry.value : [];
        const query = state.meta.arrayQuery.trim().toLowerCase();
        const isTokenType = entry.key === 'tokenizer.ggml.token_type';
        const isTokenArray = entry.key === 'tokenizer.ggml.tokens';
        const isMergeArray = entry.key === 'tokenizer.ggml.merges';
        const matches = [];
        for (let index = 0; index < values.length; index += 1) {
            const value = values[index];
            if (query && !arrayValueSearchText(value, isTokenArray).includes(query)) continue;
            matches.push({ index, value });
        }
        const pageSize = 40;
        const pageCount = Math.max(1, Math.ceil(matches.length / pageSize));
        state.meta.arrayPage = Math.max(0, Math.min(state.meta.arrayPage, pageCount - 1));
        const start = state.meta.arrayPage * pageSize;
        const pageItems = matches.slice(start, start + pageSize);
        const rows = pageItems.length ? pageItems.map(item => {
            let suffix = '';
            if (isTokenType) {
                const typeName = G.TOKEN_TYPES[item.value] || `TYPE_${item.value}`;
                suffix = `<span class="token-kind">${esc(typeName)}</span>`;
            }
            return `<div class="array-row">
                <span class="array-index">#${formatNumber(item.index)}</span>
                <span class="array-value${typeof item.value === 'number' ? ' numeric' : ''}">${arrayElementHtml(item.value, isTokenArray)}${suffix}</span>
            </div>`;
        }).join('') : '<div class="array-empty">没有匹配的数组元素</div>';
        const range = matches.length ? `${formatNumber(start + 1)}–${formatNumber(Math.min(start + pageSize, matches.length))} / ${formatNumber(matches.length)}` : '0 / 0';
        return `
            <div class="array-summary"><span><b>${formatNumber(values.length)}</b> 个元素</span><span>${esc(G.metadataTypeInfo(entry.elementTypeId).name)}</span></div>
            <div class="array-toolbar">
                <input class="array-search" id="arraySearch" type="search" placeholder="在数组值中搜索" value="${esc(state.meta.arrayQuery)}" autocomplete="off" spellcheck="false">
                <div class="array-pager">
                    <button class="mini-btn" type="button" data-meta-action="array-prev" ${state.meta.arrayPage <= 0 ? 'disabled' : ''}>← 上页</button>
                    <span>${range}${query ? ' · 过滤中' : ''}</span>
                    <button class="mini-btn" type="button" data-meta-action="array-next" ${state.meta.arrayPage >= pageCount - 1 ? 'disabled' : ''}>下页 →</button>
                </div>
            </div>
            <div class="array-list">${rows}</div>
            ${isTokenArray ? '<div class="value-meta"><span>GPT-2 byte-level 编码 · 箭头后为解码文本 · 索引即 token id</span></div>' : (isMergeArray || isTokenType) ? '<div class="value-meta"><span>显示索引即原始数组下标</span></div>' : ''}
        `;
    }

    function decodeGPT2Token(value) {
        if (typeof value !== 'string') return null;
        if (GPT2_DECODE_CACHE.has(value)) return GPT2_DECODE_CACHE.get(value);
        const bytes = [];
        for (const character of value) {
            const byte = GPT2_BYTE_TO_CHAR.get(character);
            if (byte === undefined) {
                GPT2_DECODE_CACHE.set(value, null);
                return null;
            }
            bytes.push(byte);
        }
        let decoded = null;
        try {
            decoded = GPT2_UTF8_DECODER.decode(Uint8Array.from(bytes));
        } catch (_error) {
            decoded = null;
        }
        GPT2_DECODE_CACHE.set(value, decoded);
        return decoded;
    }

    function arrayValueSearchText(value, isTokenArray = false) {
        if (typeof value === 'string') {
            const raw = value.toLowerCase();
            if (!isTokenArray) return raw;
            const decoded = decodeGPT2Token(value);
            return decoded === null ? raw : `${raw}\u0000${decoded.toLowerCase()}`;
        }
        if (Array.isArray(value)) return G.summarizeValue(value, 500).toLowerCase();
        return String(value).toLowerCase();
    }

    function arrayElementHtml(value, isTokenArray = false) {
        if (typeof value === 'string') {
            const text = value.length > 320 ? `${value.slice(0, 320)}…` : value;
            const raw = esc(JSON.stringify(text));
            if (isTokenArray) {
                const decoded = decodeGPT2Token(value);
                if (decoded !== null && decoded !== value) {
                    return `${raw}<span class="token-decoded">→ ${esc(JSON.stringify(decoded))}</span>`;
                }
            }
            return raw;
        }
        if (typeof value === 'number' || typeof value === 'boolean' || typeof value === 'bigint') {
            return esc(String(value));
        }
        if (Array.isArray(value)) return esc(G.summarizeValue(value, 320));
        return esc(String(value));
    }

    function selectMetadata(index) {
        if (!state.model) return;
        clearTimeout(metaSearchTimer);
        state.meta.selectedIndex = Number(index);
        state.meta.arrayQuery = '';
        state.meta.arrayPage = 0;
        renderMetadata();
    }

    function populateTensorTypeFilter() {
        if (!state.model) return;
        const counts = new Map();
        for (const tensor of state.model.tensors) {
            counts.set(tensor.typeId, (counts.get(tensor.typeId) || 0) + 1);
        }
        const options = [...counts.keys()].sort((a, b) => a - b).map(typeId => {
            const typeInfo = G.tensorTypeInfo(typeId);
            return `<option value="${typeId}">${esc(typeInfo.name)} · ${formatNumber(counts.get(typeId))}</option>`;
        });
        el.tensorTypeFilter.innerHTML = `<option value="all">全部类型</option>${options.join('')}`;
        if (state.tensor.typeFilter !== 'all' && !counts.has(Number(state.tensor.typeFilter))) {
            state.tensor.typeFilter = 'all';
        }
        el.tensorTypeFilter.value = state.tensor.typeFilter;
    }

    function filteredTensors() {
        if (!state.model) return [];
        const query = state.tensor.query.trim().toLowerCase();
        let tensors = state.model.tensors.filter(tensor => {
            if (state.tensor.typeFilter !== 'all' && tensor.typeId !== Number(state.tensor.typeFilter)) return false;
            if (query && !tensor.name.toLowerCase().includes(query)) return false;
            return true;
        });
        if (state.tensor.sortKey) {
            const key = state.tensor.sortKey;
            const direction = state.tensor.sortDir;
            tensors = tensors.slice().sort((a, b) => {
                let valueA;
                let valueB;
                if (key === 'name') {
                    valueA = a.name;
                    valueB = b.name;
                    return valueA.localeCompare(valueB) * direction;
                }
                if (key === 'shape') {
                    valueA = a.elementCount ?? -1;
                    valueB = b.elementCount ?? -1;
                } else if (key === 'type') {
                    valueA = a.typeInfo.name;
                    valueB = b.typeInfo.name;
                    return valueA.localeCompare(valueB) * direction;
                } else if (key === 'params') {
                    valueA = a.elementCount ?? -1;
                    valueB = b.elementCount ?? -1;
                } else if (key === 'bytes') {
                    valueA = a.storedSize ?? -1;
                    valueB = b.storedSize ?? -1;
                } else if (key === 'offset') {
                    valueA = a.offset;
                    valueB = b.offset;
                } else {
                    valueA = a.index;
                    valueB = b.index;
                }
                return ((valueA - valueB) || (a.index - b.index)) * direction;
            });
        }
        return tensors;
    }

    function updateSortHeaders() {
        $$('.sort-btn').forEach(button => {
            const active = button.dataset.sort === state.tensor.sortKey;
            button.classList.toggle('active', active);
            const icon = $('i', button);
            if (icon) icon.textContent = active ? (state.tensor.sortDir > 0 ? '↑' : '↓') : '↕';
        });
    }

    function renderTensorTable(preserveScroll = false) {
        if (!state.model) return;
        const previousScroll = el.tensorTableWrap.scrollTop;
        const filtered = filteredTensors();
        const totalFilteredBytes = filtered.reduce((sum, tensor) => sum + (tensor.storedSize || 0), 0);
        const shown = filtered.slice(0, state.tensor.limit);
        const selected = state.tensor.selectedIndex;
        updateSortHeaders();
        el.tensorRows.innerHTML = shown.length ? shown.map(tensor => {
            const shape = formatShape(tensor.dimensions);
            return `<tr data-tensor="${tensor.index}" class="${tensor.index === selected ? 'selected' : ''}" title="${esc(tensor.name)}">
                <td class="name-cell"><span class="tensor-name">${esc(tensor.name)}</span></td>
                <td><span class="tensor-shape">[${esc(shape)}]</span></td>
                <td class="tensor-type-cell">${tensorBadge(tensor.typeInfo)}</td>
                <td class="num" title="${formatNumber(tensor.elementCount)}">${tensor.elementCount === null ? '--' : compact(tensor.elementCount)}</td>
                <td class="num tensor-bytes">${tensor.storedSize === null ? '--' : formatBytes(tensor.storedSize)}</td>
                <td class="num tensor-offset">${hexOffset(tensor.offset)}</td>
            </tr>`;
        }).join('') : `<tr><td colspan="6"><div class="empty-result">没有匹配的张量</div></td></tr>`;
        const more = filtered.length > shown.length;
        el.tensorCount.innerHTML = `显示 <b>${formatNumber(shown.length)}</b> / ${formatNumber(filtered.length)} 个张量 · 合计 ${formatBytes(totalFilteredBytes)}${more ? ' <button class="mini-btn" type="button" data-tensor-more>显示更多</button>' : ''}`;
        setText(el.tensorStat, `${formatNumber(state.model.tensors.length)} TENSORS · ${compact(state.model.stats.parameterCount)} PARAMS · ${formatBytes(state.model.stats.storedBytes)}`);
        el.tensorTableWrap.scrollTop = preserveScroll ? previousScroll : 0;
    }

    function selectTensor(index, options = {}) {
        if (!state.model) return;
        const tensor = state.model.tensorInfoById.get(Number(index));
        if (!tensor) return;
        if (options.fromOverview) {
            state.tensor.query = '';
            state.tensor.typeFilter = 'all';
            state.tensor.sortKey = null;
            state.tensor.sortDir = 1;
            state.tensor.limit = 800;
            el.tensorSearch.value = '';
            el.tensorTypeFilter.value = 'all';
        }
        state.tensor.selectedIndex = tensor.index;
        state.tensor.previewStart = 0;
        state.tensor.previewByte = 0;
        state.tensor.blockStart = 0;
        state.tensor.previewTab = defaultPreviewTab(tensor);
        if (options.fromOverview) renderTensorTable();
        $$('#tensorRows tr[data-tensor]').forEach(row => {
            row.classList.toggle('selected', Number(row.dataset.tensor) === tensor.index);
        });
        renderTensorInspector(true);
        if (options.fromOverview) {
            activateView('tensors');
            const row = $(`#tensorRows tr[data-tensor="${tensor.index}"]`);
            if (row) row.scrollIntoView({ block: 'center' });
        }
    }

    function axisInfo(tensor) {
        const dims = tensor.dimensions;
        const type = tensor.typeInfo;
        if (!type.known || !type.block || !type.size) {
            return dims.map((dim, index) => ({ dim, index, strideElements: index === 0 ? 1 : null, strideBytes: null }));
        }
        return dims.map((dim, index) => {
            if (index === 0) return { dim, index, strideElements: 1, strideBytes: null };
            let inner = 1;
            for (let i = 0; i < index; i += 1) inner *= dims[i] || 1;
            const blocks = Math.ceil(inner / type.block);
            return { dim, index, strideElements: inner, strideBytes: blocks * type.size };
        });
    }

    function renderTensorInspector(scrollTop = false) {
        const tensor = activeTensor();
        if (!tensor) {
            el.tensorInspector.innerHTML = `<div class="inspector-empty">
                <span class="empty-mark">B×N</span>
                <b>选择一个张量</b>
                <p>这里会显示维度含义、量化块结构、反量化预览和原始字节。</p>
            </div>`;
            return;
        }
        const type = tensor.typeInfo;
        const canBlocks = supportsBlockView(type);
        let previewTab = state.tensor.previewTab;
        if (previewTab === 'blocks' && !canBlocks) previewTab = type.preview ? 'values' : 'raw';
        if (previewTab === 'values' && !type.preview) previewTab = canBlocks ? 'blocks' : 'raw';
        if (!['values', 'blocks', 'raw'].includes(previewTab)) previewTab = 'raw';
        state.tensor.previewTab = previewTab;
        const factRows = [
            ['SHAPE', `[${formatShape(tensor.dimensions)}]`],
            ['PARAMETERS', tensor.elementCount === null ? '--' : formatNumber(tensor.elementCount)],
            ['STORED BYTES', tensor.storedSize === null ? '未知' : `${formatNumber(tensor.storedSize)} B`],
            ['DATA RANGE', `${hexOffset(tensor.absoluteStart)} → ${tensor.absoluteEnd === null ? '?' : hexOffset(tensor.absoluteEnd)}`],
            ['REL. OFFSET', hexOffset(tensor.offset)],
            ['BLOCKS', tensor.blocks === null ? '--' : formatNumber(tensor.blocks)],
            ['BITS / WEIGHT', tensor.bitsPerWeight === null ? '--' : formatNumber(tensor.bitsPerWeight, 3)],
            ['TYPE ID', String(tensor.typeId)],
        ];
        const axes = axisInfo(tensor).map(axis => {
            let note = '';
            if (axis.index === 0) {
                note = axis.strideBytes === null && type.known
                    ? `最内层连续轴 · 一整行 ${formatNumber(Math.ceil(axis.dim / type.block) * type.size)} B`
                    : '最内层连续轴';
            } else if (axis.strideBytes !== null) {
                note = `跨过 ${formatNumber(axis.strideElements)} 个值 · 步长 ${formatNumber(axis.strideBytes)} B`;
            }
            return `<div class="axis-row">
                <span class="axis-label">D${axis.index}</span>
                <span class="axis-name">ne[${axis.index}] ${axis.index === tensor.dimensions.length - 1 ? '· outermost' : ''}</span>
                <span class="axis-value">${formatNumber(axis.dim)}</span>
                ${note ? `<span class="axis-note">${esc(note)}</span>` : ''}
            </div>`;
        }).join('');
        const layoutChips = type.blockFields
            ? type.blockFields.map((field, index) => {
                const chip = `<span class="quant-field ${fieldKindClass(field.kind)}" title="${esc(field.label)} · ${formatNumber(field.bytes)} B"><b>${esc(field.name)}</b><i>${formatNumber(field.bytes)} B</i></span>`;
                return `${chip}${index < type.blockFields.length - 1 ? '<span class="quant-arrow">→</span>' : ''}`;
            }).join('')
            : type.layout.split('|').map(part => `<span>${esc(part.trim())}</span>`).join('');
        const orderNote = type.blockFields
            ? `<div class="quant-order-note">块内按上面的物理顺序排列，每 ${formatNumber(type.size)} B 为一块，块与块首尾相接；不会把所有 scale 汇总到独立的文件区。${type.name === 'Q1_0' ? ' 本类型每字节按 LSB-first 解码 8 个元素，bit=1 → +d，bit=0 → −d。' : ''}</div>`
            : '';
        const specs = [
            ['ID', String(type.id)],
            ['BLOCK', type.block === null ? '?' : `${formatNumber(type.block)} 值`],
            ['BLOCK SIZE', type.size === null ? '?' : `${formatNumber(type.size)} B`],
            ['BPW', type.bits === null ? '?' : formatNumber(type.bits, 3)],
        ].map(([label, value]) => `<span class="quant-spec">${esc(label)} <b>${esc(value)}</b></span>`).join('');
        const actions = [
            '<button class="mini-btn" type="button" data-tensor-action="copy-name">复制名称</button>',
            canBlocks ? '<button class="mini-btn" type="button" data-tensor-action="view-blocks">按块查看</button>' : '',
            '<button class="mini-btn" type="button" data-tensor-action="view-bytes">结构中查看</button>',
        ].filter(Boolean);
        el.tensorInspector.innerHTML = `
            <div class="inspector-head">
                <div>
                    <span class="panel-kicker">TENSOR INSPECTION · #${formatNumber(tensor.index)}</span>
                    <h2 class="tensor-name-big">${esc(tensor.name)}</h2>
                    <div class="tensor-head-meta">
                        ${tensorBadge(type)}
                        <span>${formatNumber(tensor.rank)} 维</span>
                        <span>相对偏移 ${hexOffset(tensor.offset)}</span>
                    </div>
                    <div class="inspector-actions">${actions.join('')}</div>
                </div>
            </div>
            <div class="inspector-section">
                <div class="inspector-section-head"><h3>索引信息</h3><span>${formatBytes(tensor.storedSize)}</span></div>
                <div class="tensor-fact-grid">${factRows.map(([label, value]) => `<div class="tensor-fact"><span>${esc(label)}</span><b title="${esc(value)}">${esc(value)}</b></div>`).join('')}</div>
            </div>
            <div class="inspector-section">
                <div class="inspector-section-head"><h3>存储格式</h3><span>${esc(type.name)}</span></div>
                <div class="quant-card">
                    <div class="quant-card-top">
                        <div>
                            <div class="quant-type">${esc(type.name)}</div>
                            <p class="quant-desc">${esc(type.desc)}</p>
                            <div class="quant-specs">${specs}</div>
                        </div>
                    </div>
                    <div class="quant-layout">${layoutChips}</div>
                    ${orderNote}
                </div>
            </div>
            <div class="inspector-section">
                <div class="preview-toolbar">
                    <div class="preview-tabs">
                        <button class="preview-tab ${previewTab === 'values' ? 'active' : ''}" type="button" data-preview-tab="values" ${type.preview ? '' : 'disabled'}>数值预览</button>
                        <button class="preview-tab ${previewTab === 'blocks' ? 'active' : ''}" type="button" data-preview-tab="blocks" ${canBlocks ? '' : 'disabled'}>分块字节</button>
                        <button class="preview-tab ${previewTab === 'raw' ? 'active' : ''}" type="button" data-preview-tab="raw">原始字节</button>
                    </div>
                    <span class="micro-note">ON-DEMAND READ</span>
                </div>
                ${renderPreviewControls(previewTab, tensor)}
                <div class="preview-status" id="tensorPreviewStatus">正在读取张量字节…</div>
                <div id="tensorPreviewBody"></div>
            </div>
            <div class="inspector-section">
                <div class="inspector-section-head"><h3>维度 / GGML 轴</h3><span>ne[0] FASTEST</span></div>
                <div class="axis-list">${axes || '<div class="token-empty">没有维度信息。</div>'}</div>
            </div>
        `;
        if (scrollTop) el.tensorInspector.scrollTop = 0;
        loadTensorPreview();
    }

    function renderPreviewControls(tab, tensor) {
        if (tab === 'values') {
            if (!tensor.typeInfo.preview) {
                return '<div class="preview-note">该类型暂不支持数值预览；可以查看右侧相邻的原始字节，它仍会按真实量化块存储。</div>';
            }
            return `<div class="preview-controls">
                <label><span>ELEMENT</span><input id="previewStartInput" type="text" inputmode="numeric" value="${formatNumber(state.tensor.previewStart)}" autocomplete="off"></label>
                <button class="mini-btn" type="button" data-preview-action="prev-values" title="向前 256 个元素">−256</button>
                <button class="mini-btn" type="button" data-preview-action="apply-values">读取</button>
                <button class="mini-btn" type="button" data-preview-action="next-values" title="向后 256 个元素">+256</button>
            </div>`;
        }
        if (tab === 'blocks') {
            const canBlocks = supportsBlockView(tensor.typeInfo);
            const block = tensor.blocks === null ? 0 : Math.max(0, Math.min(state.tensor.blockStart, tensor.blocks - 1));
            return `<div class="preview-controls">
                <label><span>BLOCK</span><input id="previewBlockInput" type="text" inputmode="numeric" value="${formatNumber(block)}" autocomplete="off"></label>
                <button class="mini-btn" type="button" data-preview-action="prev-block" ${!canBlocks || block <= 0 ? 'disabled' : ''}>−1</button>
                <button class="mini-btn" type="button" data-preview-action="apply-block" ${canBlocks ? '' : 'disabled'}>读取</button>
                <button class="mini-btn" type="button" data-preview-action="next-block" ${!canBlocks || tensor.blocks === null || block >= tensor.blocks - 1 ? 'disabled' : ''}>+1</button>
            </div>`;
        }
        return `<div class="preview-controls">
            <label><span>BYTE</span><input id="previewByteInput" type="text" inputmode="numeric" value="${formatNumber(state.tensor.previewByte)}" autocomplete="off"></label>
            <button class="mini-btn" type="button" data-preview-action="prev-bytes">−512</button>
            <button class="mini-btn" type="button" data-preview-action="apply-bytes">读取</button>
            <button class="mini-btn" type="button" data-preview-action="next-bytes">+512</button>
        </div>`;
    }

    async function loadTensorPreview() {
        const tensor = activeTensor();
        if (!tensor) return;
        const body = $('#tensorPreviewBody');
        const status = $('#tensorPreviewStatus');
        if (!body) return;
        let tab = state.tensor.previewTab;
        if (tab === 'blocks' && !supportsBlockView(tensor.typeInfo)) tab = tensor.typeInfo.preview ? 'values' : 'raw';
        if (tab === 'values' && !tensor.typeInfo.preview) tab = supportsBlockView(tensor.typeInfo) ? 'blocks' : 'raw';
        if (!['values', 'blocks', 'raw'].includes(tab)) tab = 'raw';
        state.tensor.previewTab = tab;
        const sequence = ++state.tensor.previewSeq;
        body.innerHTML = '<div class="preview-status">读取中…</div>';
        if (status) {
            status.className = 'preview-status';
            status.textContent = tab === 'values'
                ? '从文件切片读取量化块…'
                : tab === 'blocks' ? '读取单个量化块…' : '从文件切片读取原始字节…';
        }
        try {
            if (tab === 'blocks') {
                const type = tensor.typeInfo;
                const totalBlocks = tensor.blocks === null ? 0 : tensor.blocks;
                const blockIndex = Math.max(0, Math.min(Number(state.tensor.blockStart) || 0, Math.max(0, totalBlocks - 1)));
                const absolute = tensor.absoluteStart + blockIndex * type.size;
                const bytes = await state.model.readAt(absolute, type.size);
                if (sequence !== state.tensor.previewSeq || activeTensor() !== tensor || state.tensor.previewTab !== 'blocks') return;
                state.tensor.blockStart = blockIndex;
                const input = $('#previewBlockInput');
                if (input) input.value = formatNumber(blockIndex);
                const location = blockLocation(tensor, blockIndex);
                if (status) {
                    status.className = 'preview-status';
                    status.textContent = `块 #${formatNumber(blockIndex)} / ${formatNumber(totalBlocks)} · 张量内 +${formatNumber(blockIndex * type.size)} B · ${blockLocationText(tensor, location)}`;
                }
                body.innerHTML = renderBlockPreview(tensor, blockIndex, bytes, absolute);
                return;
            }
            if (tab === 'values') {
                const result = await G.decodePreview(state.model, tensor, state.tensor.previewStart, 256);
                if (sequence !== state.tensor.previewSeq || activeTensor() !== tensor || state.tensor.previewTab !== 'values') return;
                if (!result.supported) {
                    if (status) {
                        status.className = 'preview-status warn';
                        status.textContent = result.reason || '无法预览';
                    }
                    body.innerHTML = '<div class="preview-note">切换到“原始字节”可以看到它在文件中的真实编码。</div>';
                    return;
                }
                state.tensor.previewStart = result.startElement;
                updatePreviewInputs('element');
                renderValuePreview(result);
            } else {
                const startInTensor = Math.max(0, Math.min(Number(state.tensor.previewByte) || 0, tensor.storedSize === null ? Math.max(0, state.model.fileSize - tensor.absoluteStart - 1) : Math.max(0, tensor.storedSize - 1)));
                const remaining = tensor.storedSize === null
                    ? state.model.fileSize - tensor.absoluteStart - startInTensor
                    : tensor.storedSize - startInTensor;
                const length = Math.max(1, Math.min(512, remaining));
                const absolute = tensor.absoluteStart + startInTensor;
                const bytes = await state.model.readAt(absolute, length);
                if (sequence !== state.tensor.previewSeq || activeTensor() !== tensor || state.tensor.previewTab !== 'raw') return;
                state.tensor.previewByte = startInTensor;
                if (status) {
                    status.className = 'preview-status';
                    status.textContent = `绝对偏移 ${hexOffset(absolute)} · 读取 ${formatNumber(bytes.byteLength)} B`;
                }
                const band = tensor.typeInfo.blockFields
                    ? `<div class="hex-field-band inline">${renderHexFieldBandContent(tensor, absolute)}</div>`
                    : '';
                body.innerHTML = `${band}<div class="hex-mini">${renderHexRows(bytes, absolute, 16, value => tensorByteAnnotation(tensor, value))}</div>
                    <div class="hex-mini-foot"><span>张量内偏移 ${formatNumber(startInTensor)} B</span><span>${formatBytes(tensor.storedSize)}</span></div>`;
            }
        } catch (error) {
            if (sequence !== state.tensor.previewSeq) return;
            if (status) {
                status.className = 'preview-status warn';
                status.textContent = error.message || '读取失败';
            }
            body.innerHTML = `<div class="preview-note">${esc(error.message || '读取失败')}</div>`;
        }
    }

    function renderBlockPreview(tensor, blockIndex, bytes, absoluteStart) {
        const type = tensor.typeInfo;
        const location = blockLocation(tensor, blockIndex);
        const fields = type.blockFields || [];
        const orderChips = fields.map(field => `<span class="quant-field ${fieldKindClass(field.kind)}"><b>${esc(field.name)}</b><i>${formatNumber(field.bytes)} B</i></span>`).join('<span class="quant-arrow">→</span>');
        let cursor = 0;
        const fieldRows = fields.map(field => {
            const fieldStart = cursor;
            const fieldBytes = bytes.subarray(fieldStart, fieldStart + field.bytes);
            const decoded = fieldValueText(field, bytes, fieldStart);
            const byteCells = Array.from(fieldBytes).map((byte, index) => {
                let title = `block +${fieldStart + index} · ${field.name} · 字节 ${index + 1}/${field.bytes}`;
                if (type.name === 'Q1_0' && field.name === 'qs') {
                    const firstBit = index * 8;
                    title = `qs[${index}] = 0x${byte.toString(16).padStart(2, '0')} · 元素 ${formatNumber(location.elementStart + firstBit)}–${formatNumber(location.elementStart + firstBit + 7)} · LSB-first`;
                }
                return `<i class="block-byte" title="${esc(title)}">${byte.toString(16).padStart(2, '0')}</i>`;
            }).join('');
            cursor += field.bytes;
            return `<div class="block-field-row ${fieldKindClass(field.kind)}">
                <div class="block-field-info">
                    <div class="block-field-name"><b>${esc(field.name)}</b><span>${esc(field.label)}</span></div>
                    <div class="block-field-offset">block +${formatNumber(fieldStart)}…+${formatNumber(fieldStart + field.bytes - 1)} · ${formatNumber(field.bytes)} B${field.bits ? ` · ${formatNumber(field.bits)} bit` : ''}</div>
                    ${decoded ? `<div class="block-field-decoded">${esc(field.name)} = ${esc(decoded)}</div>` : ''}
                </div>
                <div class="block-field-bytes">${byteCells}</div>
            </div>`;
        }).join('');
        const q1Map = type.name === 'Q1_0' ? renderQ1SignMap(tensor, bytes, location) : '';
        return `<div class="block-inspect">
            <div class="block-summary">
                <div><span>BLOCK</span><b>#${formatNumber(blockIndex)} <small>/ ${formatNumber(tensor.blocks)}</small></b></div>
                <div><span>BLOCK BYTES</span><b>${formatNumber(type.size)} B</b></div>
                <div><span>ELEMENTS</span><b>${formatNumber(location.elementStart)}–${formatNumber(location.elementEnd)}</b></div>
                <div><span>ABS OFFSET</span><b>${hexOffset(absoluteStart)}</b></div>
            </div>
            <div class="block-location">${esc(blockLocationText(tensor, location))}</div>
            <div class="block-physical">
                <span class="block-physical-label">块内物理顺序</span>
                ${orderChips}
                <span class="block-repeat">每 ${formatNumber(type.size)} B 重复</span>
            </div>
            <div class="block-field-list">${fieldRows}</div>
            ${q1Map}
            <div class="hex-mini-foot"><span>块内字节 +0…+${formatNumber(type.size - 1)}</span><span>下一块从 +${formatNumber(type.size)} 开始</span></div>
        </div>`;
    }

    function renderQ1SignMap(tensor, bytes, location) {
        const d = G.f16ToNumber(readU16LE(bytes, 0));
        const signBytes = bytes.subarray(2, 18);
        const rows = [];
        for (let byteIndex = 0; byteIndex < signBytes.length; byteIndex += 1) {
            const byte = signBytes[byteIndex];
            const cells = [];
            for (let bit = 0; bit < 8; bit += 1) {
                const set = ((byte >> bit) & 1) === 1;
                const element = location.elementStart + byteIndex * 8 + bit;
                const value = set ? d : -d;
                cells.push(`<i class="bit-cell ${set ? 'plus' : 'minus'}" title="bit ${formatNumber(byteIndex * 8 + bit)} = ${set ? 1 : 0} · 元素 #${formatNumber(element)} → ${esc(formatFloat(value))}">${set ? '+' : '−'}</i>`);
            }
            rows.push(`<div class="bit-row">
                <span class="bit-byte-label"><b>qs[${byteIndex}]</b><em>#${formatNumber(location.elementStart + byteIndex * 8)}–${formatNumber(location.elementStart + byteIndex * 8 + 7)}</em><small>0x${byte.toString(16).padStart(2, '0')}</small></span>
                <div class="bit-cells">${cells.join('')}</div>
            </div>`);
        }
        const firstValues = [];
        for (let bit = 0; bit < 16; bit += 1) {
            firstValues.push(((signBytes[bit >> 3] >> (bit & 7)) & 1) ? d : -d);
        }
        return `<div class="sign-map">
            <div class="sign-map-head">
                <div><b>128 个符号位 → 128 个元素</b><span>qs[0] 到 qs[15]，每字节 bit0 先出现（LSB-first）</span></div>
                <span class="sign-rule">1 → +d&nbsp;&nbsp;0 → −d</span>
            </div>
            <div class="bit-rows">${rows.join('')}</div>
            <div class="decoded-sample"><span>前 16 个解码值</span><code>${esc(firstValues.map(value => formatFloat(value)).join('  '))}</code></div>
        </div>`;
    }

    function renderHexFieldBandContent(tensor, absoluteStart) {
        const type = tensor.typeInfo;
        if (!type.blockFields) return '';
        const field = tensorByteAnnotation(tensor, absoluteStart);
        const relative = absoluteStart - tensor.absoluteStart;
        const blockIndex = Math.floor(relative / type.size);
        const offsetInBlock = relative % type.size;
        const current = field ? `${esc(field.field.name)} · block +${formatNumber(offsetInBlock)}` : `block +${formatNumber(offsetInBlock)}`;
        return `<div class="hex-field-band-head">
                <div><b>${esc(tensor.name)}</b><span>${esc(type.name)} · ${formatNumber(type.size)} B / block · 当前 ${current}</span></div>
                <span class="micro-note">按块字段着色</span>
            </div>
            ${renderFieldLegend(type)}
            <div class="field-band-help">十六进制中的颜色对应上面的字段；悬停任意字节可查看块号、字段和块内偏移。${type.name === 'Q1_0' ? ' Q1_0 每块 = [d:f16 2 B] → [qs:16 B 符号位]；位序为 LSB-first。' : ''}</div>`;
    }

    function renderValuePreview(result) {
        const body = $('#tensorPreviewBody');
        const status = $('#tensorPreviewStatus');
        if (!body) return;
        const values = result.values || [];
        let min = Infinity;
        let max = -Infinity;
        let maxAbs = 0;
        let sum = 0;
        let finiteCount = 0;
        let nanCount = 0;
        let zeroCount = 0;
        for (const raw of values) {
            const value = Number(raw);
            if (!Number.isFinite(value)) {
                nanCount += 1;
                continue;
            }
            finiteCount += 1;
            sum += value;
            if (value < min) min = value;
            if (value > max) max = value;
            const absolute = Math.abs(value);
            if (absolute > maxAbs) maxAbs = absolute;
            if (value === 0) zeroCount += 1;
        }
        const mean = finiteCount ? sum / finiteCount : null;
        const cells = values.map((raw, index) => {
            const value = Number(raw);
            const elementIndex = result.startElement + index;
            let cls = 'value-cell';
            let opacity = 1;
            if (!Number.isFinite(value)) {
                cls += ' nan';
            } else if (value === 0) {
                cls += ' zero';
            } else if (value > 0) {
                cls += ' pos';
                opacity = 0.24 + 0.76 * Math.min(1, maxAbs ? value / maxAbs : 0);
            } else {
                cls += ' neg';
                opacity = 0.24 + 0.76 * Math.min(1, maxAbs ? Math.abs(value) / maxAbs : 0);
            }
            return `<i class="${cls}" style="opacity:${opacity.toFixed(3)}" title="#${formatNumber(elementIndex)} · ${esc(formatFloat(value))}"></i>`;
        }).join('');
        const displayCount = Math.min(64, values.length);
        const entries = values.slice(0, displayCount).map((value, index) => `
            <div class="value-entry"><span class="idx">#${formatNumber(result.startElement + index)}</span><span class="val">${esc(formatFloat(value))}</span></div>
        `).join('');
        if (status) {
            status.className = 'preview-status';
            status.textContent = `${result.aligned ? '' : `起始位置已对齐到块边界 #${formatNumber(result.startElement)} · `}块内 ${formatNumber(result.blockCount)} 个 · ${formatNumber(result.byteLength)} B`;
        }
        body.innerHTML = `
            <div class="value-strip" aria-label="数值热力图">${cells}</div>
            <div class="value-stats">
                <span>MIN <b>${min === Infinity ? '--' : formatFloat(min)}</b></span>
                <span>MAX <b>${max === -Infinity ? '--' : formatFloat(max)}</b></span>
                <span>MEAN <b>${mean === null ? '--' : formatFloat(mean)}</b></span>
                <span>ABS MAX <b>${formatFloat(maxAbs)}</b></span>
                <span>ZERO <b>${formatNumber(zeroCount)}</b></span>
                ${nanCount ? `<span>NON-FINITE <b>${formatNumber(nanCount)}</b></span>` : ''}
            </div>
            <div class="value-grid">${entries}</div>
            <div class="value-meta"><span>显示前 ${formatNumber(displayCount)} / ${formatNumber(values.length)} 个采样值</span><span>总元素 ${formatNumber(activeTensor() ? activeTensor().elementCount : 0)}</span><span>起始 #${formatNumber(result.startElement)}</span></div>
        `;
    }

    function parseNumberInput(text, maximum) {
        const cleaned = String(text || '').trim().replace(/[,_\s]/g, '');
        if (!cleaned) return null;
        let value;
        if (/^0x[0-9a-f]+$/i.test(cleaned)) value = Number.parseInt(cleaned.slice(2), 16);
        else if (/^\d+$/.test(cleaned)) value = Number(cleaned);
        else return null;
        if (!Number.isSafeInteger(value) || value < 0) return null;
        if (maximum !== undefined && value > maximum) return null;
        return value;
    }

    function renderStructure() {
        if (!state.model) return;
        const model = state.model;
        const stats = model.stats;
        const segments = buildSegments(model);
        const summary = `GGUF v${model.version} · ${formatBytes(model.fileSize)} · ${formatNumber(model.alignment)} B ALIGN`;
        setText(el.structureStat, summary);
        setText(el.structureRulerNote, `${formatNumber(model.fileSize)} BYTES · LITTLE ENDIAN`);
        renderFileMap(el.structureRuler, null, model, true);
        const knownShare = model.fileSize ? (model.stats.storedBytes / model.fileSize) * 100 : 0;
        el.structureFacts.innerHTML = `
            <div class="structure-fact"><span>ALIGNMENT</span><b>${formatNumber(model.alignment)} B · ${model.alignmentSource === '默认值' ? '默认' : '元数据'}</b></div>
            <div class="structure-fact"><span>数据前对齐填充</span><b>${formatNumber(model.stats.dataPadding)} B</b></div>
            <div class="structure-fact"><span>数据区间内部空隙</span><b>${formatNumber(model.stats.dataGaps)} B</b></div>
            <div class="structure-fact"><span>尾部未引用字节</span><b>${formatNumber(model.stats.trailingBytes)} B</b></div>
            <div class="structure-fact"><span>已知权重字节</span><b>${formatNumber(model.stats.storedBytes)} B</b></div>
            <div class="structure-fact"><span>已知权重 / 文件</span><b>${knownShare.toFixed(2)}%</b></div>
            <div class="structure-fact"><span>张量索引项</span><b>${formatNumber(model.tensors.length)}</b></div>
            <div class="structure-fact"><span>已知 / 未知类型</span><b>${formatNumber(model.stats.knownTensorCount)} / ${formatNumber(model.stats.unknownTensorCount)}</b></div>
        `;
        el.headerFields.innerHTML = `
            <thead><tr><th>OFFSET</th><th>SIZE</th><th>FIELD</th><th>VALUE</th></tr></thead>
            <tbody>
                <tr><td>${hexOffset(0)}</td><td>4 B</td><td><code>magic</code></td><td><span class="value-strong">GGUF</span> <span style="color:var(--faint)">47 47 55 46</span></td></tr>
                <tr><td>${hexOffset(4)}</td><td>4 B</td><td><code>version</code></td><td>u32 = ${formatNumber(model.version)}</td></tr>
                <tr><td>${hexOffset(8)}</td><td>8 B</td><td><code>tensor_count</code></td><td>u64 = ${formatNumber(model.tensors.length)}</td></tr>
                <tr><td>${hexOffset(16)}</td><td>8 B</td><td><code>metadata_kv_count</code></td><td>u64 = ${formatNumber(model.metadata.length)}</td></tr>
                <tr><td>${hexOffset(model.metadataStart)}</td><td>${formatBytes(model.metadataEnd - model.metadataStart)}</td><td><code>metadata_kv[]</code></td><td>${formatNumber(model.metadata.length)} 个带类型键值</td></tr>
                <tr><td>${hexOffset(model.tensorInfoStart)}</td><td>${formatBytes(model.tensorInfoEnd - model.tensorInfoStart)}</td><td><code>tensor_info[]</code></td><td>${formatNumber(model.tensors.length)} 个索引项</td></tr>
                <tr><td>${hexOffset(model.tensorInfoEnd)}</td><td>${formatBytes(model.dataPadding)}</td><td><code>alignment padding</code></td><td>补齐至 ${formatNumber(model.alignment)} B</td></tr>
                <tr><td>${hexOffset(model.dataOffset)}</td><td>${formatBytes(stats.dataBytes)}</td><td><code>tensor_data</code></td><td>offset 基准 · 总已知权重 ${formatBytes(stats.storedBytes)}</td></tr>
            </tbody>
        `;
        el.layoutFields.innerHTML = `
            <thead><tr><th>SECTION</th><th>START</th><th>END</th><th>SIZE</th><th>SHARE</th></tr></thead>
            <tbody>${segments.map(segment => `<tr>
                <td>${esc(segment.name)}</td>
                <td>${hexOffset(segment.start)}</td>
                <td>${hexOffset(segment.end)}</td>
                <td>${formatBytes(segment.size)}</td>
                <td>${((segment.size / (model.fileSize || 1)) * 100).toFixed(2)}%</td>
            </tr>`).join('')}</tbody>
        `;
        updateHexInput();
        updateHexRegionButtons();
    }

    function tensorAtOffset(offset) {
        if (!state.model) return null;
        return state.model.tensors.find(tensor => tensor.absoluteEnd !== null && offset >= tensor.absoluteStart && offset < tensor.absoluteEnd) || null;
    }

    function regionForOffset(offset) {
        const model = state.model;
        if (!model) return null;
        if (offset < model.metadataStart) return { key: 'header', label: '文件头' };
        if (offset < model.metadataEnd) return { key: 'metadata', label: '元数据区' };
        if (offset < model.tensorInfoEnd) return { key: 'index', label: '张量索引区' };
        if (offset < model.dataOffset) return { key: null, label: '对齐填充' };
        if (offset < model.stats.dataEnd) return { key: 'data', label: '张量数据区' };
        return { key: 'tail', label: '文件尾部' };
    }

    function updateHexInput() {
        if (!state.model) return;
        el.hexInput.value = hexOffset(state.hex.offset, 8);
    }

    function updateHexRegionButtons() {
        const region = regionForOffset(state.hex.offset);
        $$('#hexRegions button').forEach(button => {
            button.classList.toggle('active', button.dataset.hexRegion === (region && region.key));
        });
    }

    async function loadStructureHex() {
        if (!state.model) return;
        const sequence = ++state.hex.seq;
        const model = state.model;
        const cols = state.hex.cols;
        let offset = Math.max(0, Math.min(state.hex.offset || 0, Math.max(0, model.fileSize - 1)));
        state.hex.offset = offset;
        updateHexInput();
        updateHexRegionButtons();
        const length = Math.min(512, model.fileSize - offset);
        const pageEnd = offset + length;
        const region = regionForOffset(offset);
        const tensor = tensorAtOffset(offset);
        const pageTensors = model.tensors.filter(candidate => (
            candidate.absoluteEnd !== null
            && candidate.absoluteStart < pageEnd
            && candidate.absoluteEnd > offset
            && candidate.typeInfo.blockFields
        ));
        const primary = tensor && tensor.typeInfo.blockFields ? tensor : pageTensors[0] || null;
        if (primary) {
            const bandOffset = Math.max(offset, primary.absoluteStart);
            el.hexFieldBand.hidden = false;
            el.hexFieldBand.innerHTML = renderHexFieldBandContent(primary, bandOffset);
        } else {
            el.hexFieldBand.hidden = true;
            el.hexFieldBand.innerHTML = '';
        }
        el.hexContext.innerHTML = [
            `<b style="color:var(--ink-soft);font-weight:400">${esc(region ? region.label : '文件字节')}</b>`,
            `<span> · ${hexOffset(offset)} / ${hexOffset(model.fileSize)}</span>`,
            tensor ? `<span> · ${esc(tensor.name)} + ${formatNumber(offset - tensor.absoluteStart)} B</span>` : '',
        ].join('');
        el.hexView.innerHTML = '<div class="hex-empty">DECODING…</div>';
        const annotate = pageTensors.length
            ? absolute => {
                for (const candidate of pageTensors) {
                    if (absolute < candidate.absoluteStart || absolute >= candidate.absoluteEnd) continue;
                    const annotation = tensorByteAnnotation(candidate, absolute);
                    if (annotation) return annotation;
                }
                return null;
            }
            : null;
        try {
            const bytes = await model.readAt(offset, length);
            if (sequence !== state.hex.seq) return;
            el.hexView.innerHTML = renderHexRows(bytes, offset, cols, annotate);
            setText(el.hexRange, `${hexOffset(offset)} → ${hexOffset(offset + length)} · ${formatNumber(length)} B / ${formatNumber(model.fileSize)} B`);
        } catch (error) {
            if (sequence !== state.hex.seq) return;
            el.hexView.innerHTML = `<div class="hex-empty">${esc(error.message || '读取失败')}</div>`;
        }
    }

    function renderHexRows(bytes, baseOffset, columns, annotate) {
        const rows = [];
        for (let index = 0; index < bytes.length; index += columns) {
            const line = bytes.subarray(index, Math.min(index + columns, bytes.length));
            const address = hexOffset(baseOffset + index, 8);
            const byteHtml = Array.from(line).map((byte, byteIndex) => {
                const annotation = typeof annotate === 'function' ? annotate(baseOffset + index + byteIndex) : null;
                const classes = ['hex-byte'];
                if (byte === 0) classes.push('zero');
                let title = '';
                if (annotation) {
                    classes.push(fieldKindClass(annotation.kind));
                    if (annotation.blockStart) classes.push('block-start');
                    title = annotation.title;
                }
                return `<span class="${classes.join(' ')}"${title ? ` title="${esc(title)}"` : ''}>${byte.toString(16).padStart(2, '0')}</span>`;
            }).join('');
            const ascii = Array.from(line).map(byte => byte >= 32 && byte <= 126 ? esc(String.fromCharCode(byte)) : '.').join('');
            rows.push(`<div class="hex-row"><span class="hex-address-cell">${address}</span><span class="hex-bytes-cell">${byteHtml}</span><span class="hex-ascii-cell">${ascii}</span></div>`);
        }
        return rows.join('');
    }

    function gotoHex(offset) {
        if (!state.model) return;
        const clamped = Math.max(0, Math.min(offset, Math.max(0, state.model.fileSize - 1)));
        state.hex.offset = clamped;
        updateHexInput();
        loadStructureHex();
    }

    function renderFormatStatic() {
        el.formatHeaderTable.innerHTML = `
            <thead><tr><th>OFFSET</th><th>SIZE</th><th>FIELD</th><th>ENCODING</th></tr></thead>
            <tbody>
                <tr><td>0x00</td><td>4 B</td><td><code>magic</code></td><td>ASCII “GGUF”</td></tr>
                <tr><td>0x04</td><td>4 B</td><td><code>version</code></td><td>u32，当前为 2 / 3</td></tr>
                <tr><td>0x08</td><td>8 B</td><td><code>tensor_count</code></td><td>u64</td></tr>
                <tr><td>0x10</td><td>8 B</td><td><code>metadata_kv_count</code></td><td>u64</td></tr>
                <tr><td>0x18</td><td>可变</td><td><code>metadata_kv[]</code></td><td>字符串 key + u32 类型 + value</td></tr>
                <tr><td>—</td><td>可变</td><td><code>tensor_info[]</code></td><td>名称 / 维数 / 维度 / 类型 / 相对偏移</td></tr>
                <tr><td>—</td><td>0…align-1</td><td><code>padding</code></td><td>0x00 补齐</td></tr>
                <tr><td>—</td><td>可变</td><td><code>tensor_data</code></td><td>量化块，按索引 offset 定位</td></tr>
            </tbody>
        `;
        el.formatMetadataTypes.innerHTML = `
            <thead><tr><th>ID</th><th>TYPE</th><th>WIRE SIZE</th><th>说明</th></tr></thead>
            <tbody>${Object.entries(G.METADATA_TYPES).map(([id, info]) => `<tr>
                <td>${id}</td><td><code>${esc(info.name)}</code></td><td>${info.size === null ? '可变' : `${info.size} B`}</td><td>${esc(METADATA_TYPE_NOTES[id] || '')}</td>
            </tr>`).join('')}</tbody>
        `;
        renderFormatTensorTypes();
    }

    function renderFormatTensorTypes() {
        const model = state.model;
        const used = model ? model.typeStats : {};
        const count = Object.keys(G.TENSOR_TYPES).length;
        setText(el.formatTypeNote, model ? `CURRENT FILE · ${Object.keys(used).length} / ${count} KNOWN TYPES` : `${count} KNOWN TYPES · STATIC`);
        el.formatTensorTypes.innerHTML = `
            <thead><tr><th>TYPE</th><th>ID</th><th>BLOCK</th><th>BYTES</th><th>BPW</th><th>块结构 / 说明</th></tr></thead>
            <tbody>${Object.keys(G.TENSOR_TYPES).map(Number).sort((a, b) => a - b).map(typeId => {
                const info = G.tensorTypeInfo(typeId);
                const active = used[typeId];
                return `<tr class="${active ? 'type-active' : ''}">
                    <td><span class="type-badge ${tensorTypeClass(info)}">${esc(info.name)}</span>${active ? '<span class="type-active-mark">USED</span>' : ''}</td>
                    <td>${info.id}</td>
                    <td>${formatNumber(info.block)}</td>
                    <td>${formatNumber(info.size)} B</td>
                    <td>${formatNumber(info.bits, 3)}</td>
                    <td><code>${esc(info.layout)}</code><div class="type-layout">${esc(info.desc)}</div></td>
                </tr>`;
            }).join('')}</tbody>
        `;
    }

    async function renderQ1Anatomy() {
        if (!el.q1AnatomyBody) return;
        const model = state.model;
        const q1Tensor = model ? model.tensors.find(tensor => tensor.typeId === 41 && tensor.storedSize !== null) : null;
        let sample = null;
        let sampleError = '';
        if (q1Tensor) {
            try {
                const bytes = await model.readAt(q1Tensor.absoluteStart, Math.min(18, q1Tensor.storedSize));
                if (state.model !== model) return;
                if (bytes.byteLength >= 18) {
                    const d = G.f16ToNumber(readU16LE(bytes, 0));
                    const signByte = bytes[2];
                    const signs = [];
                    for (let bit = 0; bit < 8; bit += 1) signs.push(((signByte >> bit) & 1) === 1 ? d : -d);
                    sample = { tensor: q1Tensor, bytes, d, signs };
                }
            } catch (error) {
                if (state.model !== model) return;
                sampleError = error.message || '读取示例块失败';
            }
        }
        const fieldScale = 'q1-block-segment scale';
        const fieldSigns = 'q1-block-segment signs';
        const actual = sample ? `
            <div class="q1-sample-head">
                <div>
                    <span class="panel-kicker">LIVE BYTES · BLOCK #0</span>
                    <b>${esc(sample.tensor.name)}</b>
                    <span class="q1-sample-meta">[${esc(formatShape(sample.tensor.dimensions))}] · 每行 ${formatNumber(Math.floor(sample.tensor.dimensions[0] / 128))} 个块</span>
                </div>
                <button class="mini-btn" type="button" data-tensor-goto="${sample.tensor.index}">在张量检视中打开</button>
            </div>
            <div class="q1-byte-strip">
                <div class="${fieldScale}">
                    <span>+0 … +1</span>
                    <b>d · FP16 scale</b>
                    <div class="q1-byte-cells">${Array.from(sample.bytes.subarray(0, 2)).map(byte => `<i>${byte.toString(16).padStart(2, '0')}</i>`).join('')}</div>
                    <em>d = ${esc(formatFloat(sample.d, 8))}</em>
                </div>
                <div class="${fieldSigns}">
                    <span>+2 … +17</span>
                    <b>qs · 128 个符号位</b>
                    <div class="q1-byte-cells sign-bytes">${Array.from(sample.bytes.subarray(2, 18)).map((byte, index) => `<i title="qs[${index}] · elements ${index * 8}–${index * 8 + 7}">${byte.toString(16).padStart(2, '0')}</i>`).join('')}</div>
                    <em>qs[0] 的 8 个 bit 先对应元素 0–7</em>
                </div>
            </div>
            <div class="q1-bit-example">
                <div class="q1-bit-example-head"><b>qs[0] = 0x${sample.bytes[2].toString(16).padStart(2, '0')}</b><span>按 bit0 → bit7 展开</span></div>
                <div class="q1-bit-example-cells">${sample.signs.map((value, bit) => `<i class="${value >= 0 ? 'plus' : 'minus'}" title="bit ${bit} · 元素 ${bit} → ${esc(formatFloat(value))}">${(((sample.bytes[2] >> bit) & 1) === 1) ? '1' : '0'}<small>${value >= 0 ? '+' : '−'}d</small></i>`).join('')}</div>
                <div class="q1-bit-example-foot">bit=1 → +d = ${esc(formatFloat(sample.d, 8))}；bit=0 → −d = ${esc(formatFloat(-sample.d, 8))}</div>
            </div>
        ` : `
            <div class="q1-sample-head">
                <div>
                    <span class="panel-kicker">LIVE BYTES · BLOCK #0</span>
                    <b>示例占位</b>
                </div>
            </div>
            <div class="q1-byte-strip">
                <div class="${fieldScale}">
                    <span>+0 … +1</span><b>d · FP16 scale</b>
                    <div class="q1-byte-cells"><i>??</i><i>??</i></div>
                    <em>载入包含 Q1_0 的文件后读取实际字节</em>
                </div>
                <div class="${fieldSigns}">
                    <span>+2 … +17</span><b>qs · 128 个符号位</b>
                    <div class="q1-byte-cells sign-bytes">${Array.from({ length: 16 }, () => '<i>??</i>').join('')}</div>
                    <em>每字节 8 个 bit，LSB-first</em>
                </div>
            </div>
            ${sampleError ? `<div class="q1-sample-error">${esc(sampleError)}</div>` : ''}
        `;
        el.q1AnatomyBody.innerHTML = `
            <div class="q1-anatomy-grid">
                <div class="q1-copy">
                    <p class="q1-lead">Q1_0 的数据不是“所有 scale 一段、所有 bit 一段”，而是每 <b>128 个元素</b>切一个 <b>18 B 的块</b>，块内固定按 <b>scale → 符号位</b>排列。</p>
                    <div class="q1-order-strip">
                        <div class="q1-order-block d"><b>d</b><span>FP16 scale</span><em>2 B</em><small>+0, +1</small></div>
                        <div class="q1-order-arrow">→</div>
                        <div class="q1-order-block qs"><b>qs</b><span>16 × 8 = 128 bits</span><em>16 B</em><small>+2 … +17</small></div>
                        <div class="q1-order-repeat">重复 × N<br><small>下一块从 +18 开始</small></div>
                    </div>
                    <ul class="q1-facts">
                        <li><b>块内顺序</b><span>先是 little-endian FP16 scale <code>d</code>，再接 16 B 符号位。每个块都重复这两个字段，所以 scale 与符号位是交错存放的。</span></li>
                        <li><b>位顺序</b><span>每字节 bit0 先对应较小元素下标。例如 <code>qs[0]</code> 的 bit0…bit7 → 元素 0…7；<code>qs[1]</code> bit0 → 元素 8。</span></li>
                        <li><b>数值规则</b><span>bit=1 → <code>+d</code>，bit=0 → <code>−d</code>。每 128 个元素共用一个 scale，不是每字节一个。</span></li>
                        <li><b>张量中的顺序</b><span>GGML 的 <code>ne[0]</code> 连续变化。若权重形状是 <code>[in, out]</code>，每行的 in 维按 <code>in / 128</code> 个块连续排，然后才进入下一行。</span></li>
                    </ul>
                </div>
                <div class="q1-sample">${actual}</div>
            </div>
        `;
    }

    function renderFormatCurrent() {
        if (!state.model) {
            setText(el.formatCurrent, 'STATIC REFERENCE');
            renderFormatTensorTypes();
            return;
        }
        const model = state.model;
        el.formatCurrent.textContent = `THIS FILE · v${model.version} · ${formatNumber(model.tensors.length)} TENSORS · ${formatNumber(Object.keys(model.typeStats).length)} USED TYPES`;
        renderFormatTensorTypes();
    }

    function bindEvents() {
        $('#openBtn').addEventListener('click', openFilePicker);
        $('#welcomeOpenBtn').addEventListener('click', openFilePicker);
        $('#welcomeDropBtn').addEventListener('click', event => {
            event.stopPropagation();
            openFilePicker();
        });
        el.welcomeDrop.addEventListener('click', event => {
            if (event.target.closest('button')) return;
            openFilePicker();
        });
        $('#welcomeFormatBtn').addEventListener('click', () => activateView('format'));
        el.fileInput.addEventListener('change', () => {
            const file = el.fileInput.files && el.fileInput.files[0];
            if (file) loadFile(file);
        });

        let dragDepth = 0;
        window.addEventListener('dragenter', event => {
            if (!event.dataTransfer || !Array.from(event.dataTransfer.types || []).includes('Files')) return;
            event.preventDefault();
            dragDepth += 1;
            el.dropOverlay.hidden = false;
        });
        window.addEventListener('dragover', event => {
            if (!event.dataTransfer || !Array.from(event.dataTransfer.types || []).includes('Files')) return;
            event.preventDefault();
            event.dataTransfer.dropEffect = 'copy';
        });
        window.addEventListener('dragleave', event => {
            if (!event.dataTransfer) return;
            dragDepth = Math.max(0, dragDepth - 1);
            if (dragDepth === 0) el.dropOverlay.hidden = true;
        });
        window.addEventListener('drop', event => {
            event.preventDefault();
            dragDepth = 0;
            el.dropOverlay.hidden = true;
            const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
            if (file) loadFile(file);
        });
        window.addEventListener('keydown', event => {
            if (event.key === 'Escape' && !el.dropOverlay.hidden) {
                dragDepth = 0;
                el.dropOverlay.hidden = true;
            }
        });

        el.viewTabs.addEventListener('click', event => {
            const tab = event.target.closest('.view-tab');
            if (!tab) return;
            if (tab.getAttribute('aria-disabled') === 'true') return;
            activateView(tab.dataset.view);
        });
        el.viewTabs.addEventListener('keydown', event => {
            if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
            const tabs = $$('.view-tab:not([aria-disabled="true"])', el.viewTabs);
            const current = document.activeElement;
            const index = tabs.indexOf(current);
            if (index === -1) return;
            event.preventDefault();
            const next = tabs[(index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length];
            next.focus();
            activateView(next.dataset.view);
        });

        document.addEventListener('click', event => {
            const goto = event.target.closest('[data-goto]');
            if (goto) {
                activateView(goto.dataset.goto);
                return;
            }
            const tensorGoto = event.target.closest('[data-tensor-goto]');
            if (tensorGoto) {
                selectTensor(Number(tensorGoto.dataset.tensorGoto), { fromOverview: true });
            }
        });

        el.metaSearch.addEventListener('input', () => {
            state.meta.query = el.metaSearch.value;
            renderMetadata();
        });
        $('#metaFilters').addEventListener('click', event => {
            const button = event.target.closest('button[data-meta-filter]');
            if (!button) return;
            clearTimeout(metaSearchTimer);
            state.meta.filter = button.dataset.metaFilter;
            renderMetadata();
        });
        el.metaList.addEventListener('click', event => {
            const row = event.target.closest('.metadata-row');
            if (row) selectMetadata(row.dataset.metaIndex);
        });
        el.metaInspector.addEventListener('click', async event => {
            const button = event.target.closest('[data-meta-action]');
            if (!button) return;
            const entry = activeMeta();
            if (!entry) return;
            const action = button.dataset.metaAction;
            if (action === 'copy-key') {
                await copyText(entry.key, '键名');
            } else if (action === 'copy-value') {
                await copyText(metaValueAsText(entry.value), '字段值');
            } else if (action === 'locate') {
                state.hex.offset = entry.start;
                activateView('structure', { scrollHex: true, force: true });
            } else if (action === 'array-prev') {
                state.meta.arrayPage = Math.max(0, state.meta.arrayPage - 1);
                renderMetaInspector();
            } else if (action === 'array-next') {
                state.meta.arrayPage += 1;
                renderMetaInspector();
            }
        });
        el.metaInspector.addEventListener('input', event => {
            if (!event.target.matches('#arraySearch')) return;
            const value = event.target.value;
            clearTimeout(metaSearchTimer);
            metaSearchTimer = setTimeout(() => {
                state.meta.arrayQuery = value;
                state.meta.arrayPage = 0;
                renderMetaInspector();
                const search = $('#arraySearch');
                if (search) {
                    search.focus();
                    search.setSelectionRange(search.value.length, search.value.length);
                }
            }, 110);
        });

        el.tensorSearch.addEventListener('input', () => {
            clearTimeout(tensorSearchTimer);
            tensorSearchTimer = setTimeout(() => {
                state.tensor.query = el.tensorSearch.value;
                state.tensor.limit = 800;
                renderTensorTable();
            }, 110);
        });
        el.tensorTypeFilter.addEventListener('change', () => {
            state.tensor.typeFilter = el.tensorTypeFilter.value;
            state.tensor.limit = 800;
            renderTensorTable();
        });
        el.tensorFilterReset.addEventListener('click', () => {
            clearTimeout(tensorSearchTimer);
            state.tensor.query = '';
            state.tensor.typeFilter = 'all';
            state.tensor.sortKey = null;
            state.tensor.sortDir = 1;
            state.tensor.limit = 800;
            el.tensorSearch.value = '';
            el.tensorTypeFilter.value = 'all';
            renderTensorTable();
        });
        $$('.sort-btn').forEach(button => {
            button.addEventListener('click', () => {
                const key = button.dataset.sort;
                if (state.tensor.sortKey === key) {
                    state.tensor.sortDir *= -1;
                } else {
                    state.tensor.sortKey = key;
                    state.tensor.sortDir = ['params', 'bytes', 'offset'].includes(key) ? -1 : 1;
                }
                state.tensor.limit = 800;
                renderTensorTable();
            });
        });
        el.tensorRows.addEventListener('click', event => {
            const row = event.target.closest('tr[data-tensor]');
            if (row) selectTensor(Number(row.dataset.tensor));
        });
        el.tensorCount.addEventListener('click', event => {
            if (!event.target.closest('[data-tensor-more]')) return;
            state.tensor.limit += 800;
            renderTensorTable(true);
        });
        el.tensorInspector.addEventListener('click', async event => {
            const tensor = activeTensor();
            if (!tensor) return;
            const tab = event.target.closest('[data-preview-tab]');
            if (tab) {
                if (tab.disabled) return;
                state.tensor.previewTab = tab.dataset.previewTab;
                renderTensorInspector();
                return;
            }
            const actionButton = event.target.closest('[data-tensor-action]');
            if (actionButton) {
                const action = actionButton.dataset.tensorAction;
                if (action === 'copy-name') await copyText(tensor.name, '张量名');
                if (action === 'view-blocks') {
                    state.tensor.previewTab = 'blocks';
                    renderTensorInspector();
                }
                if (action === 'view-bytes') {
                    state.hex.offset = tensor.absoluteStart;
                    activateView('structure', { scrollHex: true, force: true });
                }
                return;
            }
            const previewButton = event.target.closest('[data-preview-action]');
            if (!previewButton) return;
            const action = previewButton.dataset.previewAction;
            if (action === 'prev-values') {
                state.tensor.previewStart = Math.max(0, state.tensor.previewStart - 256);
                updatePreviewInputs('element');
                loadTensorPreview();
            } else if (action === 'next-values') {
                state.tensor.previewStart = Math.min(Math.max(0, tensor.elementCount - 1), state.tensor.previewStart + 256);
                updatePreviewInputs('element');
                loadTensorPreview();
            } else if (action === 'apply-values') {
                const parsed = parseNumberInput($('#previewStartInput') && $('#previewStartInput').value, Math.max(0, tensor.elementCount - 1));
                if (parsed === null) {
                    showToast('请输入有效的元素下标', 'error');
                    return;
                }
                state.tensor.previewStart = parsed;
                loadTensorPreview();
            } else if (action === 'prev-block') {
                state.tensor.blockStart = Math.max(0, state.tensor.blockStart - 1);
                updatePreviewInputs('block');
                loadTensorPreview();
            } else if (action === 'next-block') {
                state.tensor.blockStart = Math.min(Math.max(0, (tensor.blocks || 1) - 1), state.tensor.blockStart + 1);
                updatePreviewInputs('block');
                loadTensorPreview();
            } else if (action === 'apply-block') {
                const parsed = parseNumberInput($('#previewBlockInput') && $('#previewBlockInput').value, Math.max(0, (tensor.blocks || 1) - 1));
                if (parsed === null) {
                    showToast('请输入有效的块下标', 'error');
                    return;
                }
                state.tensor.blockStart = parsed;
                loadTensorPreview();
            } else if (action === 'prev-bytes') {
                state.tensor.previewByte = Math.max(0, state.tensor.previewByte - 512);
                updatePreviewInputs('byte');
                loadTensorPreview();
            } else if (action === 'next-bytes') {
                const maxByte = tensor.storedSize === null ? state.model.fileSize - tensor.absoluteStart : tensor.storedSize;
                state.tensor.previewByte = Math.max(0, Math.min(Math.max(0, maxByte - 1), state.tensor.previewByte + 512));
                updatePreviewInputs('byte');
                loadTensorPreview();
            } else if (action === 'apply-bytes') {
                const maxByte = tensor.storedSize === null ? state.model.fileSize - tensor.absoluteStart : tensor.storedSize;
                const parsed = parseNumberInput($('#previewByteInput') && $('#previewByteInput').value, Math.max(0, maxByte));
                if (parsed === null) {
                    showToast('请输入有效的字节偏移', 'error');
                    return;
                }
                state.tensor.previewByte = parsed;
                loadTensorPreview();
            }
        });
        el.tensorInspector.addEventListener('keydown', event => {
            if (event.key !== 'Enter') return;
            if (event.target.matches('#previewStartInput')) {
                event.preventDefault();
                const button = $('[data-preview-action="apply-values"]');
                if (button) button.click();
            } else if (event.target.matches('#previewByteInput')) {
                event.preventDefault();
                const button = $('[data-preview-action="apply-bytes"]');
                if (button) button.click();
            } else if (event.target.matches('#previewBlockInput')) {
                event.preventDefault();
                const button = $('[data-preview-action="apply-block"]');
                if (button) button.click();
            }
        });

        el.hexRegions.addEventListener('click', event => {
            const button = event.target.closest('button[data-hex-region]');
            if (!button || !state.model) return;
            const region = button.dataset.hexRegion;
            if (region === 'header') gotoHex(0);
            else if (region === 'metadata') gotoHex(state.model.metadataStart);
            else if (region === 'index') gotoHex(state.model.tensorInfoStart);
            else if (region === 'data') gotoHex(state.model.tensors.length ? state.model.tensors[0].absoluteStart : state.model.dataOffset);
            else if (region === 'tail') gotoHex(Math.max(0, state.model.fileSize - 512));
        });
        $('#hexGoto').addEventListener('click', () => {
            if (!state.model) return;
            const parsed = parseNumberInput(el.hexInput.value, Math.max(0, state.model.fileSize - 1));
            if (parsed === null) {
                showToast('请输入文件范围内的十进制或 0x 十六进制偏移', 'error');
                return;
            }
            gotoHex(parsed);
        });
        el.hexInput.addEventListener('keydown', event => {
            if (event.key === 'Enter') {
                event.preventDefault();
                $('#hexGoto').click();
            }
        });
        $('#hexPrev').addEventListener('click', () => gotoHex(Math.max(0, state.hex.offset - 512)));
        $('#hexNext').addEventListener('click', () => gotoHex(Math.min(Math.max(0, state.model.fileSize - 1), state.hex.offset + 512)));
        el.hexCols.addEventListener('change', () => {
            state.hex.cols = Number(el.hexCols.value);
            loadStructureHex();
        });
    }

    function updatePreviewInputs(kind) {
        if (kind === 'element') {
            const input = $('#previewStartInput');
            if (input) input.value = formatNumber(state.tensor.previewStart);
        } else if (kind === 'block') {
            const input = $('#previewBlockInput');
            if (input) input.value = formatNumber(state.tensor.blockStart);
        } else {
            const input = $('#previewByteInput');
            if (input) input.value = formatNumber(state.tensor.previewByte);
        }
    }

    function metaValueAsText(value) {
        if (typeof value === 'string') return value;
        if (typeof value === 'bigint') return value.toString();
        if (typeof value === 'boolean' || typeof value === 'number') return String(value);
        try {
            return JSON.stringify(value, (_key, item) => typeof item === 'bigint' ? item.toString() : item, 2);
        } catch (_error) {
            return String(value);
        }
    }

    function cacheElements() {
        el.fileLed = $('#fileLed');
        el.fileName = $('#fileName');
        el.fileSize = $('#fileSize');
        el.fileSub = $('#fileSub');
        el.fileInput = $('#fileInput');
        el.navStats = $('#navStats');
        el.viewTabs = $('#viewTabs');
        el.main = $('#main');
        el.statusText = $('#statusText');
        el.parseInfo = $('#parseInfo');
        el.overviewEmpty = $('#overviewEmpty');
        el.overviewContent = $('#overviewContent');
        el.welcomeDrop = $('#welcomeDrop');
        el.welcomeError = $('#welcomeError');
        el.profileKicker = $('#profileKicker');
        el.profileName = $('#profileName');
        el.profileDesc = $('#profileDesc');
        el.profileTags = $('#profileTags');
        el.profileFacts = $('#profileFacts');
        el.kpiParams = $('#kpiParams');
        el.kpiParamsExact = $('#kpiParamsExact');
        el.kpiBpw = $('#kpiBpw');
        el.kpiTensors = $('#kpiTensors');
        el.kpiTensorTypes = $('#kpiTensorTypes');
        el.kpiMeta = $('#kpiMeta');
        el.kpiMetaSize = $('#kpiMetaSize');
        el.overviewMap = $('#overviewMap');
        el.overviewMapLegend = $('#overviewMapLegend');
        el.mixChart = $('#mixChart');
        el.mixCenterBpw = $('#mixCenterBpw');
        el.mixLegend = $('#mixLegend');
        el.mixNote = $('#mixNote');
        el.archFacts = $('#archFacts');
        el.tokenizerFlavor = $('#tokenizerFlavor');
        el.tokenizerOverview = $('#tokenizerOverview');
        el.largestList = $('#largestList');
        el.diagnosticSummary = $('#diagnosticSummary');
        el.diagnosticsList = $('#diagnosticsList');
        el.metaStat = $('#metaStat');
        el.metaSearch = $('#metaSearch');
        el.metaList = $('#metaList');
        el.metaCount = $('#metaCount');
        el.metaInspector = $('#metaInspector');
        el.tensorStat = $('#tensorStat');
        el.tensorSearch = $('#tensorSearch');
        el.tensorTypeFilter = $('#tensorTypeFilter');
        el.tensorFilterReset = $('#tensorFilterReset');
        el.tensorTableWrap = $('#tensorTableWrap');
        el.tensorCount = $('#tensorCount');
        el.tensorRows = $('#tensorRows');
        el.tensorInspector = $('#tensorInspector');
        el.structureStat = $('#structureStat');
        el.structureRuler = $('#structureRuler');
        el.structureRulerNote = $('#structureRulerNote');
        el.structureFacts = $('#structureFacts');
        el.headerFields = $('#headerFields');
        el.layoutFields = $('#layoutFields');
        el.hexRegions = $('#hexRegions');
        el.hexInput = $('#hexInput');
        el.hexCols = $('#hexCols');
        el.hexContext = $('#hexContext');
        el.hexView = $('#hexView');
        el.hexRange = $('#hexRange');
        el.hexFieldBand = $('#hexFieldBand');
        el.formatCurrent = $('#formatCurrent');
        el.formatTypeNote = $('#formatTypeNote');
        el.formatHeaderTable = $('#formatHeaderTable');
        el.formatMetadataTypes = $('#formatMetadataTypes');
        el.formatTensorTypes = $('#formatTensorTypes');
        el.q1AnatomyBody = $('#q1AnatomyBody');
        el.dropOverlay = $('#dropOverlay');
        el.loadOverlay = $('#loadOverlay');
        el.loadFile = $('#loadFile');
        el.loadStep = $('#loadStep');
        el.loadPct = $('#loadPct');
        el.loadBar = $('#loadBar');
        el.toast = $('#toast');
    }

    function init() {
        if (!G) {
            document.body.innerHTML = '<pre style="padding:20px">GGUF parser failed to load.</pre>';
            return;
        }
        cacheElements();
        renderFormatStatic();
        renderQ1Anatomy();
        bindEvents();
        setDataTabsEnabled(false);
        setLed('');
        setText(el.statusText, '就绪 · 等待载入文件');
        setText(el.parseInfo, '--');
        renderOverviewEmptyState();
        activateView('overview', { keepScroll: true });
    }

    init();
})();
