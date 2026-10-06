/* bnn-MNIST-MLP.c — W1A8 MNIST MLP 纯 C 推理器（±1 权重，int8 激活，g128 FP16
 * scale） 编译: gcc -O2 -std=c99 -Wall -Wextra -o bnn-MNIST-MLP bnn-MNIST-MLP.c
 * -lm 运行: ./bnn-MNIST-MLP model.bnn t10k-images-idx3-ubyte
 * t10k-labels-idx1-ubyte [ref_logits.bin]
 *
 * 唯一的推理工作模式：每层激活按 absmax 对称量化为 int8，
 * 组内（group=128）纯整数累加，组外每组乘一次 FP16 scale，
 * 即硬件 PE 的 golden reference。
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bnn-accelerator/bnn-accelerator.h"

#define MAX_LAYERS 8
#define MAX_DIM 4096
#define IMG_PIXELS 784
#define N_CLASSES 10

typedef struct {
    uint32_t in_f, out_f, group, relu;
    uint8_t* bits; /* [out_f][in_f/8]，第 k 个权重 = 字节 k>>3 的第 k&7 位，1 => +1 */
    uint16_t* scale_bits; /* [out_f][in_f/group]，FP16 位模式，上传给加速器 */
    float* scales; /* [out_f][in_f/group]，加载时由 fp16 转为 fp32（软件参考用） */
    float* post_a; /* [out_f] 折叠后的 BN 缩放 */
    float* post_b; /* [out_f] 折叠后的 BN 偏置 */
    BNN_Matrix matrix; /* 加速器上的只读权重（软件模式下为 NULL） */
} Layer;

typedef struct {
    uint32_t n_layers;
    Layer layers[MAX_LAYERS];
} Model;

/* ---------- 基础 I/O（显式按字节序解析，不依赖主机字节序） ---------- */
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

static uint32_t read_be32(FILE* f) { /* MNIST IDX 文件头是大端 */
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

/* IEEE754 half -> float：符号 1 位、指数 5 位（偏置 15）、尾数 10 位 */
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0x1F) { /* Inf / NaN */
        bits = sign | 0x7F800000u | (mant << 13);
    } else if (exp == 0) {
        if (mant == 0) {
            bits = sign; /* ±0 */
        } else {         /* 次正规数：先规格化 */
            exp = 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FFu;
            bits = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
        }
    } else {
        bits = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13); /* 112 = 127 - 15 */
    }
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

/* ---------- 模型加载（带尺寸校验） ---------- */
static void load_model(const char* path, Model* m) {
    FILE* f = fopen(path, "rb");
    if (!f)
        die("cannot open model file");
    char magic[4];
    read_exact(f, magic, 4);
    if (memcmp(magic, "BNN1", 4) != 0)
        die("bad model magic");
    m->n_layers = read_le32(f);
    if (m->n_layers == 0 || m->n_layers > MAX_LAYERS)
        die("bad layer count");

    for (uint32_t l = 0; l < m->n_layers; l++) {
        Layer* L = &m->layers[l];
        L->in_f = read_le32(f);
        L->out_f = read_le32(f);
        L->group = read_le32(f);
        L->relu = read_le32(f);
        if (L->in_f == 0 || L->in_f > MAX_DIM || L->out_f == 0 || L->out_f > MAX_DIM)
            die("bad layer dims");
        if (L->group == 0 || L->group % 8 != 0 || L->in_f % L->group != 0)
            die("bad group size");
        if (l > 0 && L->in_f != m->layers[l - 1].out_f)
            die("layer shape mismatch");

        size_t n_bytes = (size_t)L->out_f * L->in_f / 8;
        size_t n_scales = (size_t)L->out_f * (L->in_f / L->group);
        L->bits = xmalloc(n_bytes);
        read_exact(f, L->bits, n_bytes);
        L->scales = xmalloc(n_scales * sizeof(float));
        L->scale_bits = xmalloc(n_scales * sizeof(uint16_t));
        for (size_t i = 0; i < n_scales; i++) {
            uint16_t h = read_le16(f);
            L->scale_bits[i] = h;
            L->scales[i] = fp16_to_fp32(h);
        }
        L->matrix = NULL;
        L->post_a = xmalloc(L->out_f * sizeof(float));
        L->post_b = xmalloc(L->out_f * sizeof(float));
        for (uint32_t i = 0; i < L->out_f; i++)
            L->post_a[i] = read_le_f32(f);
        for (uint32_t i = 0; i < L->out_f; i++)
            L->post_b[i] = read_le_f32(f);

        printf("layer %u: %4u -> %4u, group %u, relu %u | sign bits %zu B, fp16 "
               "scales %zu B\n",
               l, L->in_f, L->out_f, L->group, L->relu, n_bytes, n_scales * 2);
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

/* ---------- 核心算子 ---------- */
static inline float finish(const Layer* L, uint32_t o, float acc) {
    float v = L->post_a[o] * acc + L->post_b[o]; /* 折叠后的 BN（最后一层是 1*acc + bias） */
    return (L->relu && v < 0.0f) ? 0.0f : v;
}

/* 激活量化：对称 int8，scale = absmax/127，对齐 numpy round half away from zero */
static float quantize_int8(const float* x, uint32_t n, int8_t* xq) {
    float amax = 0.0f;
    for (uint32_t k = 0; k < n; k++)
        if (fabsf(x[k]) > amax)
            amax = fabsf(x[k]);
    float xs = amax > 0.0f ? amax / 127.0f : 1.0f;
    for (uint32_t k = 0; k < n; k++) {
        float q = roundf(x[k] / xs);
        xq[k] = (int8_t)(q > 127.0f ? 127 : (q < -127.0f ? -127 : q));
    }
    return xs;
}

/* W1A8：每层输入按 absmax 量化为 int8，组内纯整数累加，组外一次浮点乘
 * 这就是硬件 PE 的 golden reference */
static void binary_linear(const Layer* L, const float* x, float* y) {
    uint32_t in = L->in_f, ng = in / L->group, row_bytes = in / 8;
    int8_t xq[MAX_DIM];
    int32_t gtotal[MAX_DIM / 8];

    /* 1) 激活量化：对称 int8，scale = absmax / 127 */
    float xs = quantize_int8(x, in, xq);

    /* 2) 每组激活和：所有输出行共享（硬件上每组只算一次） */
    for (uint32_t g = 0; g < ng; g++) {
        int32_t s = 0;
        for (uint32_t k = g * L->group; k < (g + 1) * L->group; k++)
            s += xq[k];
        gtotal[g] = s;
    }

    /* 3) 每行：bit 掩码整数累加，组内结果 2*pos - total 在 int16 范围内 */
    for (uint32_t o = 0; o < L->out_f; o++) {
        const uint8_t* wrow = L->bits + (size_t)o * row_bytes;
        const float* srow = L->scales + (size_t)o * ng;
        float acc = 0.0f;
        for (uint32_t g = 0; g < ng; g++) {
            int32_t pos = 0;
            for (uint32_t k = g * L->group; k < (g + 1) * L->group; k++)
                if ((wrow[k >> 3] >> (k & 7)) & 1u)
                    pos += xq[k];
            acc += srow[g] * (float)(2 * pos - gtotal[g]); /* 每组一次浮点乘 */
        }
        y[o] = finish(L, o, acc * xs); /* 乘回激活 scale */
    }
}

/* ---------- 硬件加速后端：把每层二值点积交给 BNN 加速器 ---------- */
static BNN_Device g_dev = NULL;
static BNN_Buffer g_in = NULL, g_out = NULL;
static int g_use_accel = 1;

static void acc_status(BNN_Status st, const char* what) {
    if (st != BNN_OK) {
        fprintf(stderr, "error: %s: %s\n", what, BNN_StatusString(st));
        exit(1);
    }
}

static void accel_init(Model* m) {
    BNN_Caps caps;
    acc_status(BNN_DeviceOpen(0, &g_dev), "BNN_DeviceOpen");
    acc_status(BNN_DeviceGetCaps(g_dev, &caps), "BNN_DeviceGetCaps");
    printf("accelerator: api %u, group %u, max_in %u, max_out %u\n", caps.api_version, caps.group_size,
           caps.max_in_features, caps.max_out_features);

    uint32_t max_in = 0, max_out = 0;
    for (uint32_t l = 0; l < m->n_layers; l++) {
        Layer* L = &m->layers[l];
        if (L->group != caps.group_size)
            die("model group_size != device group_size");
        if (L->in_f > caps.max_in_features || L->out_f > caps.max_out_features)
            die("layer exceeds device feature limits");
        BNN_MatrixDesc d = {
            .in_features = L->in_f,
            .out_features = L->out_f,
            .group_size = L->group,
            .weight_bits = L->bits,
            .weight_bytes = (size_t)L->out_f * (L->in_f / 8),
            .scale_fp16 = L->scale_bits,
            .scale_count = (size_t)L->out_f * (L->in_f / L->group),
        };
        acc_status(BNN_MatrixCreate(g_dev, &d, &L->matrix), "BNN_MatrixCreate");
        if (L->in_f > max_in)
            max_in = L->in_f;
        if (L->out_f > max_out)
            max_out = L->out_f;
    }
    acc_status(BNN_BufferAlloc(g_dev, max_in, &g_in), "BNN_BufferAlloc(input)");
    acc_status(BNN_BufferAlloc(g_dev, (size_t)max_out * sizeof(float), &g_out), "BNN_BufferAlloc(output)");
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
}

/* 一层 W1A8：量化激活 -> 加速器 GEMV（组内整数点积 + 组外 FP16 scale）-> BN + ReLU */
static void accel_linear(const Layer* L, const float* x, float* y) {
    int8_t xq[MAX_DIM];
    float acc[MAX_DIM];
    float xs = quantize_int8(x, L->in_f, xq);
    acc_status(BNN_BufferWrite(g_in, 0, xq, L->in_f), "BNN_BufferWrite");
    BNN_Job job;
    memset(&job, 0, sizeof job);
    job.operation = BNN_OP_GEMV;
    job.matrix = L->matrix;
    job.input = g_in;
    job.output = g_out;
    job.rows = 1;
    acc_status(BNN_Execute(g_dev, &job), "BNN_Execute(GEMV)");
    acc_status(BNN_BufferRead(g_out, 0, acc, (size_t)L->out_f * sizeof(float)), "BNN_BufferRead");
    for (uint32_t o = 0; o < L->out_f; o++) {
        float v = L->post_a[o] * (acc[o] * xs) + L->post_b[o];
        y[o] = (L->relu && v < 0.0f) ? 0.0f : v;
    }
}

static void forward_with(const Model* m, const float* input, float* out, int use_accel) {
    static float buf[2][MAX_DIM]; /* 层间 ping-pong 缓冲 */
    const float* x = input;
    for (uint32_t l = 0; l < m->n_layers; l++) {
        float* y = buf[l & 1];
        if (use_accel)
            accel_linear(&m->layers[l], x, y);
        else
            binary_linear(&m->layers[l], x, y);
        x = y;
    }
    memcpy(out, x, m->layers[m->n_layers - 1].out_f * sizeof(float));
}

static void forward(const Model* m, const float* input, float* out) {
    forward_with(m, input, out, g_use_accel);
}

static void preprocess(const uint8_t* px, float* x, uint32_t in_f) {
    for (uint32_t i = 0; i < IMG_PIXELS; i++)
        x[i] = ((float)px[i] / 255.0f - 0.1307f) / 0.3081f; /* 与训练时 Normalize 一致 */
    for (uint32_t i = IMG_PIXELS; i < in_f; i++)
        x[i] = 0.0f; /* 补零到 896 */
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
    if (argc > 1 && strcmp(argv[1], "--soft") == 0) {
        g_use_accel = 0;
        ai = 2;
    }
    if (argc - ai < 3 || argc - ai > 4) {
        fprintf(stderr, "usage: %s [--soft] model.bnn images-idx3 labels-idx1 [ref_logits.bin]\n", argv[0]);
        return 1;
    }
    Model m;
    load_model(argv[ai], &m);
    if (m.layers[0].in_f < IMG_PIXELS)
        die("first layer narrower than 784");
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

    float x[MAX_DIM], logits[N_CLASSES];

    /* 1) 全测试集精度 + 耗时（W1A8 内核） */
    uint32_t correct = 0;
    clock_t t0 = clock();
    for (uint32_t i = 0; i < n_img; i++) {
        preprocess(images + (size_t)i * IMG_PIXELS, x, m.layers[0].in_f);
        forward(&m, x, logits);
        if (argmax(logits, N_CLASSES) == labels[i])
            correct++;
    }
    double sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("C inference accuracy: %.4f (%u/%u), %.1f us/image\n", (double)correct / n_img, correct, n_img,
           sec * 1e6 / n_img);

    /* 1.5) 加速器 vs 软件 golden 内核：逐样本交叉验证 */
    if (g_use_accel) {
        uint32_t K = n_img < 200 ? n_img : 200;
        float maxdiff = 0.0f, logits_soft[N_CLASSES];
        uint32_t agree = 0, soft_correct = 0;
        for (uint32_t i = 0; i < K; i++) {
            preprocess(images + (size_t)i * IMG_PIXELS, x, m.layers[0].in_f);
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

    /* 2) 与 numpy 参考对比（同样的 int8 激活量化） */
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
        uint32_t n_check = n_ref;
        float err_ref = 0.0f;
        for (uint32_t i = 0; i < n_check; i++) {
            preprocess(images + (size_t)i * IMG_PIXELS, x, m.layers[0].in_f);
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