/* bnn-MNIST-CNN.c — W1A8 MNIST CNN 纯 C 推理器（BNN2 格式：conv + pool + fc）
 * 编译: gcc -O2 -std=c99 -Wall -Wextra -o bnn-MNIST-CNN \
 *           bnn-MNIST-CNN.c bnn-accelerator/soft-bnn-accelerator.c -lm
 * 运行: ./bnn-MNIST-CNN model-cnn.bnn t10k-images-idx3-ubyte t10k-labels-idx1-ubyte [ref_logits.bin]
 *       ./bnn-MNIST-CNN --soft ...    # 走软件 golden 内核做对照
 *
 * 与 MLP 版同一量化约定：每层激活按 absmax 对称量化为 int8，组内整数累加，
 * 组外每组乘一次 FP16 scale。conv 用 im2col 展开成同一套分组二值点积。
 *
 * 加速器绑定：设备组大小固定为 group_size（本后端为 128），且要求
 * in_features 是它的整数倍。而 conv 的一行权重只有 in_ch*k*k 个（9 / 72），
 * BNN2 文件里的 group 也各不相同。因此这里在加载后用“补齐”适配：
 * 把原始的第 g 组(长度 group)放进第 g 个 128 元素 block 的前 group 位，
 * 其余位补 0（激活侧也补 0，0*±1 = 0 不改变任何点积）。
 * 于是每个 block 的组内点积与原组完全一致，scale 数组仍是一组一个，
 * 加速器结果与软件内核逐位相同。
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bnn-accelerator/bnn-accelerator.h"

#define MAX_LAYERS 16
#define MAX_DIM 8192
#define MAX_TOTAL 8192              /* 每行权重个数上限 in_ch*k*k 或 in_f */
#define MAX_FMAP 65536              /* 特征图元素上限 */
#define IMG 28
#define IMG_PIXELS 784
#define N_CLASSES 10

typedef struct {
    uint32_t type;                  /* 0=conv 1=fc 2=pool */
    uint32_t in_ch, out_ch;         /* conv 通道数 */
    uint32_t k, group, relu;        /* conv 卷积核 / 分组 / 末层 relu */
    uint32_t in_f, out_f;           /* fc 维度（conv 时 out_f=out_ch） */
    uint32_t pool;                  /* pool 窗口 */
    uint32_t total, ng, row_bytes, out_rows;
    uint8_t* bits;                  /* [out_rows][row_bytes] */
    uint16_t* scale_bits;           /* [out_rows][ng] FP16 位模式，上传加速器 */
    float* scales;                  /* [out_rows][ng] */
    float* post_a;                  /* [out_rows] 折叠后 BN 缩放 */
    float* post_b;                  /* [out_rows] 折叠后 BN 偏置 */
    uint32_t pad_k;                 /* 加速器行宽 = ng * 设备组大小 */
    uint8_t* pad_bits;              /* [out_rows][pad_k/8] 补齐后的权重位 */
    BNN_Matrix matrix;              /* 加速器上的只读权重（软件模式为 NULL） */
} Layer;

typedef struct {
    uint32_t n_layers;
    Layer layers[MAX_LAYERS];
} Model;

static void die(const char* msg) {
    fprintf(stderr, "error: %s\n", msg);
    exit(1);
}

static void* xmalloc(size_t n) {
    void* p = malloc(n);
    if (!p)
        die("out of memory");
    return p;
}

static void read_exact(FILE* f, void* buf, size_t n) {
    if (fread(buf, 1, n, f) != n)
        die("unexpected end of file");
}

static uint16_t read_le16(FILE* f) {
    uint8_t b[2];
    read_exact(f, b, 2);
    return (uint16_t)(b[0] | (b[1] << 8));
}

static uint32_t read_le32(FILE* f) {
    uint8_t b[4];
    read_exact(f, b, 4);
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

static uint32_t read_be32(FILE* f) { /* MNIST IDX 头是大端 */
    uint8_t b[4];
    read_exact(f, b, 4);
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | (uint32_t)b[3];
}

static float read_le_f32(FILE* f) {
    uint32_t u = read_le32(f);
    float v;
    memcpy(&v, &u, sizeof v);
    return v;
}

/* IEEE754 half -> float */
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FFu;
            bits = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
        }
    } else {
        bits = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

/* ---------- 模型加载（BNN2，带尺寸校验） ---------- */
static void load_model(const char* path, Model* m) {
    FILE* f = fopen(path, "rb");
    if (!f)
        die("cannot open model file");
    char magic[4];
    read_exact(f, magic, 4);
    if (memcmp(magic, "BNN2", 4) != 0)
        die("bad model magic (expect BNN2)");
    m->n_layers = read_le32(f);
    if (m->n_layers == 0 || m->n_layers > MAX_LAYERS)
        die("bad layer count");

    for (uint32_t l = 0; l < m->n_layers; l++) {
        Layer* L = &m->layers[l];
        memset(L, 0, sizeof *L);
        uint32_t type = read_le32(f);
        L->type = type;
        if (type == 2) {
            L->pool = read_le32(f);
            if (L->pool < 1 || L->pool > 4)
                die("bad pool size");
            printf("layer %2u: pool %u\n", l, L->pool);
            continue;
        }
        if (type == 0) {
            L->in_ch = read_le32(f);
            L->out_ch = read_le32(f);
            L->k = read_le32(f);
            L->group = read_le32(f);
            L->relu = read_le32(f);
            if (L->in_ch == 0 || L->out_ch == 0 || L->k == 0 || L->k > 7 || L->k % 2 == 0)
                die("bad conv dims (k 需为奇数)");
            L->total = L->in_ch * L->k * L->k;
            L->out_rows = L->out_ch;
            L->out_f = L->out_ch;
        } else if (type == 1) {
            L->in_f = read_le32(f);
            L->out_f = read_le32(f);
            L->group = read_le32(f);
            L->relu = read_le32(f);
            if (L->in_f == 0 || L->in_f > MAX_TOTAL || L->out_f == 0 || L->out_f > MAX_DIM)
                die("bad fc dims");
            L->total = L->in_f;
            L->out_rows = L->out_f;
        } else {
            die("bad layer type");
        }
        if (L->group == 0 || L->total % L->group != 0)
            die("bad group size");
        L->ng = L->total / L->group;
        L->row_bytes = (L->total + 7) / 8;
        if (L->out_rows > MAX_DIM || L->ng > MAX_DIM)
            die("layer too large");

        size_t nb = (size_t)L->out_rows * L->row_bytes;
        L->bits = xmalloc(nb);
        read_exact(f, L->bits, nb);
        size_t n_scales = (size_t)L->out_rows * L->ng;
        L->scales = xmalloc(n_scales * sizeof(float));
        L->scale_bits = xmalloc(n_scales * sizeof(uint16_t));
        for (size_t i = 0; i < n_scales; i++) {
            uint16_t h = read_le16(f);
            L->scale_bits[i] = h;
            L->scales[i] = fp16_to_fp32(h);
        }
        L->pad_k = 0;
        L->pad_bits = NULL;
        L->matrix = NULL;
        L->post_a = xmalloc(L->out_rows * sizeof(float));
        L->post_b = xmalloc(L->out_rows * sizeof(float));
        for (uint32_t i = 0; i < L->out_rows; i++)
            L->post_a[i] = read_le_f32(f);
        for (uint32_t i = 0; i < L->out_rows; i++)
            L->post_b[i] = read_le_f32(f);

        printf("layer %2u: %s in %u out %u group %u relu %u | bits %zu B, scales %zu B\n", l,
               type == 0 ? "conv" : " fc ", L->total, L->out_rows, L->group, L->relu, nb,
               (size_t)L->out_rows * L->ng * 2);
    }
    if (fgetc(f) != EOF)
        die("trailing bytes in model file");
    fclose(f);
}

static void free_model(Model* m) {
    for (uint32_t l = 0; l < m->n_layers; l++) {
        free(m->layers[l].bits);
        free(m->layers[l].scale_bits);
        free(m->layers[l].scales);
        free(m->layers[l].pad_bits);
        free(m->layers[l].post_a);
        free(m->layers[l].post_b);
    }
}

/* ---------- MNIST IDX 读取 ---------- */
static uint8_t* load_idx(const char* path, uint32_t magic, uint32_t* count, size_t* item_size) {
    FILE* f = fopen(path, "rb");
    if (!f)
        die("cannot open IDX file");
    if (read_be32(f) != magic)
        die("bad IDX magic");
    uint32_t ndims = magic & 0xFF;
    *count = read_be32(f);
    if (*count == 0 || *count > 100000)
        die("unexpected IDX item count");
    size_t item = 1;
    for (uint32_t d = 1; d < ndims; d++)
        item *= read_be32(f);
    if (item == 0 || item > 4096)
        die("unexpected IDX item size");
    uint8_t* data = xmalloc((size_t)*count * item);
    read_exact(f, data, (size_t)*count * item);
    fclose(f);
    *item_size = item;
    return data;
}

/* ---------- 激活量化：对称 int8，scale = absmax/127，对齐 numpy round half away from zero */
static float quantize_act(const float* x, uint32_t n, int8_t* xq) {
    float amax = 0.0f;
    for (uint32_t i = 0; i < n; i++)
        if (fabsf(x[i]) > amax)
            amax = fabsf(x[i]);
    float xs = amax > 0.0f ? amax / 127.0f : 1.0f;
    for (uint32_t i = 0; i < n; i++) {
        float q = roundf(x[i] / xs);
        xq[i] = (int8_t)(q > 127.0f ? 127 : (q < -127.0f ? -127 : q));
    }
    return xs;
}

/* ---------- conv 3x3 same（im2col + 分组二值点积），输入 [in_ch][H][W] -> [out_ch][H][W] */
static void conv_same(const Layer* L, const float* x, uint32_t H, uint32_t W, float* y) {
    uint32_t in_ch = L->in_ch, out_ch = L->out_ch, k = L->k, pad = L->k / 2;
    uint32_t ng = L->ng, group = L->group, total = L->total;
    uint32_t Hp = H + 2 * pad, Wp = W + 2 * pad;
    static int8_t xp[MAX_FMAP];              /* padding 后的量化激活 */
    static uint32_t poff[MAX_TOTAL];         /* 每个权重索引在 patch 内的偏移 */
    static int32_t gtot[MAX_DIM];

    float xs = quantize_act(x, in_ch * H * W, xp);   /* 先量化到 xp 前段 */
    static int8_t xq[MAX_FMAP];
    memcpy(xq, xp, in_ch * H * W);

    memset(xp, 0, (size_t)in_ch * Hp * Wp);
    for (uint32_t c = 0; c < in_ch; c++)
        for (uint32_t h = 0; h < H; h++)
            for (uint32_t w = 0; w < W; w++)
                xp[c * Hp * Wp + (h + pad) * Wp + (w + pad)] = xq[c * H * W + h * W + w];

    for (uint32_t kk = 0; kk < total; kk++) {
        uint32_t ic = kk / (k * k), rem = kk % (k * k);
        poff[kk] = ic * Hp * Wp + (rem / k) * Wp + (rem % k);
    }

    for (uint32_t oh = 0; oh < H; oh++) {
        for (uint32_t ow = 0; ow < W; ow++) {
            uint32_t base = oh * Wp + ow;
            for (uint32_t g = 0; g < ng; g++) {
                int32_t s = 0;
                for (uint32_t t = 0; t < group; t++)
                    s += xp[poff[g * group + t] + base];
                gtot[g] = s;
            }
            for (uint32_t oc = 0; oc < out_ch; oc++) {
                const uint8_t* wrow = L->bits + (size_t)oc * L->row_bytes;
                const float* srow = L->scales + (size_t)oc * ng;
                float acc = 0.0f;
                for (uint32_t g = 0; g < ng; g++) {
                    int32_t pos = 0;
                    for (uint32_t t = 0; t < group; t++) {
                        uint32_t kk = g * group + t;
                        if ((wrow[kk >> 3] >> (kk & 7)) & 1u)
                            pos += xp[poff[kk] + base];
                    }
                    acc += srow[g] * (float)(2 * pos - gtot[g]);
                }
                float v = L->post_a[oc] * (acc * xs) + L->post_b[oc];
                y[(size_t)oc * H * W + oh * W + ow] = (L->relu && v < 0.0f) ? 0.0f : v;
            }
        }
    }
}

/* ---------- 二值全连接 ---------- */
static void binary_fc(const Layer* L, const float* x, float* y) {
    uint32_t total = L->total, ng = L->ng, group = L->group, out = L->out_rows;
    static int8_t xq[MAX_TOTAL];
    static int32_t gtot[MAX_DIM];

    float xs = quantize_act(x, total, xq);
    for (uint32_t g = 0; g < ng; g++) {
        int32_t s = 0;
        for (uint32_t t = 0; t < group; t++)
            s += xq[g * group + t];
        gtot[g] = s;
    }
    for (uint32_t o = 0; o < out; o++) {
        const uint8_t* wrow = L->bits + (size_t)o * L->row_bytes;
        const float* srow = L->scales + (size_t)o * ng;
        float acc = 0.0f;
        for (uint32_t g = 0; g < ng; g++) {
            int32_t pos = 0;
            for (uint32_t t = 0; t < group; t++) {
                uint32_t kk = g * group + t;
                if ((wrow[kk >> 3] >> (kk & 7)) & 1u)
                    pos += xq[kk];
            }
            acc += srow[g] * (float)(2 * pos - gtot[g]);
        }
        float v = L->post_a[o] * (acc * xs) + L->post_b[o];
        y[o] = (L->relu && v < 0.0f) ? 0.0f : v;
    }
}

/* ---------- maxpool ---------- */
static void maxpool(const Layer* L, const float* x, uint32_t C, uint32_t H, uint32_t W, float* y) {
    uint32_t p = L->pool, Ho = H / p, Wo = W / p;
    for (uint32_t c = 0; c < C; c++)
        for (uint32_t ho = 0; ho < Ho; ho++)
            for (uint32_t wo = 0; wo < Wo; wo++) {
                float best = -1e30f;
                for (uint32_t i = 0; i < p; i++)
                    for (uint32_t j = 0; j < p; j++) {
                        float v = x[c * H * W + (ho * p + i) * W + (wo * p + j)];
                        if (v > best)
                            best = v;
                    }
                y[c * Ho * Wo + ho * Wo + wo] = best;
            }
}

/* ---------- 硬件加速后端：conv/fc 的二值点积交给 BNN 加速器 ---------- */
static BNN_Device g_dev = NULL;
static BNN_Buffer g_in = NULL, g_out = NULL;
static int g_use_accel = 1;
static int8_t* g_pad;           /* 补齐后的 int8 激活行（im2col / fc 输入） */
static float* g_res;            /* 加速器输出读回缓冲 */

static void acc_status(BNN_Status st, const char* what) {
    if (st != BNN_OK) {
        fprintf(stderr, "error: %s: %s\n", what, BNN_StatusString(st));
        exit(1);
    }
}

/* 把 [out_rows][total] 的 group 布局重排为 [out_rows][ng*G]：
 * 第 g 组（长度 group）放进第 g 个 G 长 block 的前 group 位，其余位补 0。 */
static uint8_t* pad_weight_bits(const Layer* L, uint32_t G) {
    uint32_t row_bytes = L->pad_k / 8;
    uint8_t* out = xmalloc((size_t)L->out_rows * row_bytes);
    memset(out, 0, (size_t)L->out_rows * row_bytes);
    for (uint32_t o = 0; o < L->out_rows; o++) {
        const uint8_t* src = L->bits + (size_t)o * L->row_bytes;
        uint8_t* dst = out + (size_t)o * row_bytes;
        for (uint32_t g = 0; g < L->ng; g++)
            for (uint32_t t = 0; t < L->group; t++) {
                uint32_t kk = g * L->group + t;
                if ((src[kk >> 3] >> (kk & 7)) & 1u) {
                    uint32_t p = g * G + t;
                    dst[p >> 3] |= (uint8_t)(1u << (p & 7));
                }
            }
    }
    return out;
}

static void accel_init(Model* m) {
    BNN_Caps caps;
    uint32_t G, H = IMG, W = IMG;
    size_t max_pad = 0, max_res = 0;

    acc_status(BNN_DeviceOpen(0, &g_dev), "BNN_DeviceOpen");
    acc_status(BNN_DeviceGetCaps(g_dev, &caps), "BNN_DeviceGetCaps");
    G = caps.group_size;
    printf("accelerator: api %u, group %u, max_in %u, max_out %u\n", caps.api_version, G,
           caps.max_in_features, caps.max_out_features);

    for (uint32_t l = 0; l < m->n_layers; l++) {
        Layer* L = &m->layers[l];
        if (L->type == 2) {
            H /= L->pool;
            W /= L->pool;
            continue;
        }
        uint32_t rows = (L->type == 0) ? H * W : 1; /* conv: 输出像素数; fc: 单行 */
        L->pad_k = L->ng * G;
        if (L->pad_k > caps.max_in_features || L->out_rows > caps.max_out_features)
            die("layer exceeds device feature limits");
        L->pad_bits = pad_weight_bits(L, G);
        BNN_MatrixDesc d = {
            .in_features = L->pad_k,
            .out_features = L->out_rows,
            .group_size = G,
            .weight_bits = L->pad_bits,
            .weight_bytes = (size_t)L->out_rows * (L->pad_k / 8),
            .scale_fp16 = L->scale_bits,
            .scale_count = (size_t)L->out_rows * L->ng,
        };
        acc_status(BNN_MatrixCreate(g_dev, &d, &L->matrix), "BNN_MatrixCreate");
        if ((size_t)rows * L->pad_k > max_pad)
            max_pad = (size_t)rows * L->pad_k;
        if ((size_t)rows * L->out_rows * sizeof(float) > max_res)
            max_res = (size_t)rows * L->out_rows * sizeof(float);
        if (L->type == 1)
            H = W = 1;
        printf("  layer %2u -> matrix in %u (pad %u) out %u, %u row(s)\n", l, L->total, L->pad_k, L->out_rows,
               rows);
    }
    acc_status(BNN_BufferAlloc(g_dev, max_pad, &g_in), "BNN_BufferAlloc(input)");
    acc_status(BNN_BufferAlloc(g_dev, max_res, &g_out), "BNN_BufferAlloc(output)");
    g_pad = xmalloc(max_pad);
    g_res = xmalloc(max_res);
    printf("accelerator buffers: input %zu B, output %zu B\n", max_pad, max_res);
}

static void accel_shutdown(Model* m) {
    for (uint32_t l = 0; l < m->n_layers; l++) {
        BNN_MatrixDestroy(m->layers[l].matrix);
        m->layers[l].matrix = NULL;
    }
    BNN_BufferFree(g_in);
    g_in = NULL;
    BNN_BufferFree(g_out);
    g_out = NULL;
    if (g_dev)
        BNN_DeviceClose(g_dev);
    g_dev = NULL;
    free(g_pad);
    g_pad = NULL;
    free(g_res);
    g_res = NULL;
}

/* conv：主机侧量化 + im2col 补齐 -> 加速器 GEMM（rows = 输出像素数）-> BN + ReLU */
static void accel_conv(const Layer* L, const float* x, uint32_t H, uint32_t W, float* y) {
    uint32_t in_ch = L->in_ch, out_ch = L->out_ch, k = L->k, pad = k / 2;
    uint32_t group = L->group, ng = L->ng, total = L->total, pad_k = L->pad_k, G = pad_k / ng;
    uint32_t Hp = H + 2 * pad, Wp = W + 2 * pad, rows = H * W;
    static int8_t xq[MAX_FMAP], xp[MAX_FMAP];
    static uint32_t poff[MAX_TOTAL];

    float xs = quantize_act(x, in_ch * H * W, xq);
    memset(xp, 0, (size_t)in_ch * Hp * Wp);
    for (uint32_t c = 0; c < in_ch; c++)
        for (uint32_t h = 0; h < H; h++)
            for (uint32_t w = 0; w < W; w++)
                xp[c * Hp * Wp + (h + pad) * Wp + (w + pad)] = xq[c * H * W + h * W + w];
    for (uint32_t kk = 0; kk < total; kk++) {
        uint32_t ic = kk / (k * k), rem = kk % (k * k);
        poff[kk] = ic * Hp * Wp + (rem / k) * Wp + (rem % k);
    }

    memset(g_pad, 0, (size_t)rows * pad_k);
    for (uint32_t oh = 0; oh < H; oh++)
        for (uint32_t ow = 0; ow < W; ow++) {
            int8_t* row = g_pad + (size_t)(oh * W + ow) * pad_k;
            uint32_t base = oh * Wp + ow;
            for (uint32_t g = 0; g < ng; g++)
                for (uint32_t t = 0; t < group; t++)
                    row[g * G + t] = (int8_t)xp[poff[g * group + t] + base];
        }

    BNN_Job job;
    memset(&job, 0, sizeof job);
    job.operation = BNN_OP_GEMM;
    job.matrix = L->matrix;
    job.input = g_in;
    job.output = g_out;
    job.rows = rows;
    acc_status(BNN_BufferWrite(g_in, 0, g_pad, (size_t)rows * pad_k), "BNN_BufferWrite");
    acc_status(BNN_Execute(g_dev, &job), "BNN_Execute(GEMM)");
    acc_status(BNN_BufferRead(g_out, 0, g_res, (size_t)rows * out_ch * sizeof(float)), "BNN_BufferRead");

    for (uint32_t p = 0; p < rows; p++)
        for (uint32_t oc = 0; oc < out_ch; oc++) {
            float v = L->post_a[oc] * (g_res[(size_t)p * out_ch + oc] * xs) + L->post_b[oc];
            y[(size_t)oc * rows + p] = (L->relu && v < 0.0f) ? 0.0f : v;
        }
}

/* fc：主机侧量化 + 补齐 -> 加速器 GEMV -> BN + ReLU */
static void accel_fc(const Layer* L, const float* x, float* y) {
    uint32_t total = L->total, group = L->group, ng = L->ng, out = L->out_rows;
    uint32_t pad_k = L->pad_k, G = pad_k / ng;
    static int8_t xq[MAX_TOTAL];

    float xs = quantize_act(x, total, xq);
    memset(g_pad, 0, pad_k);
    for (uint32_t g = 0; g < ng; g++)
        for (uint32_t t = 0; t < group; t++)
            g_pad[g * G + t] = xq[g * group + t];

    BNN_Job job;
    memset(&job, 0, sizeof job);
    job.operation = BNN_OP_GEMV;
    job.matrix = L->matrix;
    job.input = g_in;
    job.output = g_out;
    job.rows = 1;
    acc_status(BNN_BufferWrite(g_in, 0, g_pad, pad_k), "BNN_BufferWrite");
    acc_status(BNN_Execute(g_dev, &job), "BNN_Execute(GEMV)");
    acc_status(BNN_BufferRead(g_out, 0, g_res, (size_t)out * sizeof(float)), "BNN_BufferRead");

    for (uint32_t o = 0; o < out; o++) {
        float v = L->post_a[o] * (g_res[o] * xs) + L->post_b[o];
        y[o] = (L->relu && v < 0.0f) ? 0.0f : v;
    }
}

static void forward_with(const Model* m, const float* input, float* out, int use_accel) {
    static float bufA[MAX_FMAP], bufB[MAX_FMAP];
    const float* x = input;
    float* y = bufA;
    uint32_t C = 1, H = IMG, W = IMG;
    for (uint32_t l = 0; l < m->n_layers; l++) {
        const Layer* L = &m->layers[l];
        if (L->type == 2) {
            maxpool(L, x, C, H, W, y);
            H /= L->pool;
            W /= L->pool;
        } else if (L->type == 0) {
            if (use_accel)
                accel_conv(L, x, H, W, y);
            else
                conv_same(L, x, H, W, y);
            C = L->out_ch;
        } else {
            if (use_accel)
                accel_fc(L, x, y);
            else
                binary_fc(L, x, y);
            C = L->out_f;
            H = W = 1;
        }
        x = y;
        y = (y == bufA) ? bufB : bufA;
    }
    memcpy(out, x, m->layers[m->n_layers - 1].out_f * sizeof(float));
}

static void forward(const Model* m, const float* input, float* out) {
    forward_with(m, input, out, g_use_accel);
}

static void preprocess(const uint8_t* px, float* x) {
    for (uint32_t i = 0; i < IMG_PIXELS; i++)
        x[i] = ((float)px[i] / 255.0f - 0.1307f) / 0.3081f;
}

static uint32_t argmax(const float* v, uint32_t n) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; i++)
        if (v[i] > v[best])
            best = i;
    return best;
}

int main(int argc, char** argv) {
    int ai = 1;
    uint32_t limit = 0; /* 0 = 用全部样本 */
    while (ai < argc && strncmp(argv[ai], "--", 2) == 0) {
        if (strcmp(argv[ai], "--soft") == 0) {
            g_use_accel = 0;
            ai += 1;
        } else if (strcmp(argv[ai], "--limit") == 0 && ai + 1 < argc) {
            limit = (uint32_t)strtoul(argv[ai + 1], NULL, 10);
            ai += 2;
        } else {
            break;
        }
    }
    if (argc - ai < 3 || argc - ai > 4) {
        fprintf(stderr,
                "usage: %s [--soft] [--limit N] model-cnn.bnn images-idx3 labels-idx1 [ref_logits.bin]\n",
                argv[0]);
        return 1;
    }
    Model m;
    load_model(argv[ai], &m);
    if (m.layers[0].type != 0 || m.layers[0].in_ch != 1)
        die("first layer must be conv 1 -> N (MNIST 单通道)");
    if (m.layers[m.n_layers - 1].out_f != N_CLASSES)
        die("last layer must output 10 classes");
    if (g_use_accel)
        accel_init(&m);

    uint32_t n_img, n_lab;
    size_t img_sz, lab_sz;
    uint8_t* images = load_idx(argv[ai + 1], 0x00000803u, &n_img, &img_sz);
    uint8_t* labels = load_idx(argv[ai + 2], 0x00000801u, &n_lab, &lab_sz);
    if (img_sz != IMG_PIXELS || lab_sz != 1 || n_img != n_lab)
        die("image/label files mismatch");

    uint32_t n_use = (limit > 0 && limit < n_img) ? limit : n_img;
    float x[IMG_PIXELS], logits[N_CLASSES];

    /* 1) 精度 + 耗时 */
    uint32_t correct = 0;
    clock_t t0 = clock();
    for (uint32_t i = 0; i < n_use; i++) {
        preprocess(images + (size_t)i * IMG_PIXELS, x);
        forward(&m, x, logits);
        if (argmax(logits, N_CLASSES) == labels[i])
            correct++;
    }
    double sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("C inference accuracy: %.4f (%u/%u), %.1f us/image\n", (double)correct / n_use, correct, n_use,
           sec * 1e6 / n_use);

    /* 1.5) 加速器 vs 软件 golden 内核：逐样本交叉验证 */
    if (g_use_accel) {
        uint32_t K = n_use < 200 ? n_use : 200;
        float maxdiff = 0.0f, logits_soft[N_CLASSES];
        uint32_t agree = 0, soft_correct = 0;
        for (uint32_t i = 0; i < K; i++) {
            preprocess(images + (size_t)i * IMG_PIXELS, x);
            forward_with(&m, x, logits, 1);
            uint32_t a = argmax(logits, N_CLASSES);
            forward_with(&m, x, logits_soft, 0);
            uint32_t b = argmax(logits_soft, N_CLASSES);
            agree += (a == b);
            soft_correct += (b == labels[i]);
            for (uint32_t c = 0; c < N_CLASSES; c++) {
                float d = fabsf(logits[c] - logits_soft[c]);
                if (d > maxdiff)
                    maxdiff = d;
            }
        }
        printf("accel vs soft kernel: argmax agree %u/%u, soft acc %.4f, max abs logit diff %.2e\n", agree, K,
               (double)soft_correct / K, maxdiff);
    }

    /* 2) 与 numpy 参考对比（同样的 int8 量化与整数累加） */
    uint32_t n_ref = 0;
    float* ref = NULL;
    if (argc - ai == 4) {
        FILE* f = fopen(argv[ai + 3], "rb");
        if (!f)
            die("cannot open ref logits");
        n_ref = read_le32(f);
        if (read_le32(f) != N_CLASSES || n_ref == 0 || n_ref > n_img)
            die("bad ref logits header");
        ref = xmalloc((size_t)n_ref * N_CLASSES * sizeof(float));
        for (size_t i = 0; i < (size_t)n_ref * N_CLASSES; i++)
            ref[i] = read_le_f32(f);
        fclose(f);
    }
    if (ref) {
        uint32_t n_cmp = n_ref < n_use ? n_ref : n_use;
        float err_ref = 0.0f;
        for (uint32_t i = 0; i < n_cmp; i++) {
            preprocess(images + (size_t)i * IMG_PIXELS, x);
            forward(&m, x, logits);
            for (uint32_t c = 0; c < N_CLASSES; c++) {
                float r = ref[(size_t)i * N_CLASSES + c];
                float d = fabsf(logits[c] - r) / (1.0f + fabsf(r));
                if (d > err_ref)
                    err_ref = d;
            }
        }
        printf("C vs numpy ref  max rel err: %.2e %s\n", err_ref, err_ref < 1e-3f ? "OK" : "MISMATCH");
    }

    free(ref);
    free(images);
    free(labels);
    if (g_use_accel)
        accel_shutdown(&m);
    free_model(&m);
    return 0;
}
