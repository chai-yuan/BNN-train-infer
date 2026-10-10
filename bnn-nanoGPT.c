/* bnn-nanoGPT.c — 最小 nanoGPT（中文 + 单汉字分词）纯 C 推理器
 *
 * 唯一的推理工作模式 W1A8，与 bnn-accelerator.h 的计算契约一致：
 *   权重  ±1，每 group(128) 个输入共享一个 FP16 scale（组内整数点积，组外一次浮点乘）
 *   激活  逐 token 对称量化为 int8，scale = absmax / 127
 * 加速器负责 Q/K/V/输出投影、注意力分数（X=int8 的 key，W=±1 的 query）、MLP、lm_head；
 * softmax / attn@v / 残差加 / RMSNorm / embedding 查表在 FP32 里做。
 *
 * 编译（软件后端）：
 *   gcc -O2 -std=c99 -Wall -Wextra -o bnn-nanoGPT bnn-nanoGPT.c \
 *       bnn-accelerator/soft-bnn-accelerator.c -lm
 * 运行：
 *   ./bnn-nanoGPT [--soft] [--temp 0.8] [--topk 10] [--steps 40] [--prompt 春天] \
 *       nanogpt.bnn [ref_logits.bin] [val_tokens.bin]
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
#define MAX_GROUPS (MAX_DIM / 128)
#define MAX_BLOCK 256
#define MAX_VOCAB 4096

/* ------------------------------ 基础工具 ------------------------------ */
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

static char* xrealloc_char(char* p, size_t n) {
    char* q = realloc(p, n);
    if (!q)
        die("out of memory");
    return q;
}

static void read_exact(FILE* f, void* buf, size_t n) {
    if (fread(buf, 1, n, f) != n)
        die("unexpected end of file");
}

static uint32_t read_le32(FILE* f) {
    uint8_t b[4];
    read_exact(f, b, 4);
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

static uint16_t read_le16(FILE* f) {
    uint8_t b[2];
    read_exact(f, b, 2);
    return (uint16_t)(b[0] | ((uint16_t)b[1] << 8));
}

static float read_le_f32(FILE* f) {
    uint32_t u = read_le32(f);
    float v;
    memcpy(&v, &u, sizeof v);
    return v;
}

/* IEEE754 binary16 -> binary32 */
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    int32_t exp = (int32_t)((h >> 10) & 0x1Fu);
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

/* binary32 -> binary16（round to nearest even，与 numpy 的 float16 一致） */
static uint16_t fp32_to_fp16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    uint32_t sign = (u >> 16) & 0x8000u;
    uint32_t biased = (u >> 23) & 0xFFu;
    int32_t exp = (int32_t)biased - 127 + 15;
    uint32_t mant = u & 0x7FFFFFu;
    uint32_t half;
    if (biased == 0xFFu)
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u)); /* Inf / NaN */
    if (exp >= 31)
        return (uint16_t)(sign | 0x7C00u);                         /* 溢出 */
    if (exp <= 0) {                                                /* 次正规或零 */
        if (exp < -10)
            return (uint16_t)sign;
        mant |= 0x800000u; /* 补上隐含位 */
        half = mant >> (14 - exp);
        if ((mant >> (13 - exp)) & 1u) { /* 舍入位 */
            uint32_t rest = mant & ((((uint32_t)1) << (14 - exp)) - 1u);
            uint32_t halfway = ((uint32_t)1) << (13 - exp);
            if (rest > halfway || (rest == halfway && (half & 1u)))
                half++;
        }
        return (uint16_t)(sign | half);
    }
    half = ((uint32_t)exp << 10) | (mant >> 13);
    if ((mant >> 12) & 1u) {
        uint32_t rest = mant & 0x1FFFu;
        if (rest > 0x1000u || (rest == 0x1000u && (half & 1u)))
            half++;
    }
    return (uint16_t)(sign | half);
}

/* ------------------------------ 模型 ------------------------------ */
typedef struct {
    uint32_t in_f, out_f, group, relu;
    uint8_t* bits;        /* [out_f][in_f/8]，第 k 个权重 = 字节 k>>3 的第 k&7 位，1 => +1 */
    uint16_t* scale_bits; /* [out_f][in_f/group]，FP16 位模式，上传给加速器 */
    float* scales;        /* 同上，FP32（软件内核用） */
    BNN_Matrix matrix;    /* 加速器上的只读权重（软件模式下为 NULL） */
} W1Layer;

typedef struct {
    uint32_t V, T, C, H, L, G; /* vocab / block / embd / head / layer / group */
    float eps;                 /* RMSNorm eps，必须与训练端一致 */
    float* tok_emb;            /* [V][C] */
    float* pos_emb;            /* [T][C] */
    float* ln1;                /* [L][C] */
    float* ln2;                /* [L][C] */
    float* lnf;                /* [C] */
    W1Layer q[MAX_LAYERS], k[MAX_LAYERS], v[MAX_LAYERS], o[MAX_LAYERS];
    W1Layer fc[MAX_LAYERS], proj[MAX_LAYERS];
    W1Layer head;
    char* vocab;     /* 拼接的 UTF-8 词表 */
    uint32_t* voff;  /* [V+1] 偏移 */
} Model;

static void read_w1(FILE* f, W1Layer* L, uint32_t group, uint32_t in_expect) {
    L->in_f = read_le32(f);
    L->out_f = read_le32(f);
    L->group = read_le32(f);
    L->relu = read_le32(f);
    if (L->in_f == 0 || L->in_f > MAX_DIM || L->out_f == 0 || L->out_f > MAX_DIM)
        die("bad W1 layer dims");
    if (L->group != group || L->in_f % group != 0 || L->group % 8 != 0)
        die("bad W1 group size");
    if (in_expect && L->in_f != in_expect)
        die("unexpected W1 in_features");
    size_t nb = (size_t)L->out_f * L->in_f / 8;
    size_t ns = (size_t)L->out_f * (L->in_f / L->group);
    L->bits = xmalloc(nb);
    read_exact(f, L->bits, nb);
    L->scales = xmalloc(ns * sizeof(float));
    L->scale_bits = xmalloc(ns * sizeof(uint16_t));
    for (size_t i = 0; i < ns; i++) {
        uint16_t h = read_le16(f);
        L->scale_bits[i] = h;
        L->scales[i] = fp16_to_fp32(h);
    }
    L->matrix = NULL;
}

static float* read_f32_array(FILE* f, size_t n) {
    float* p = xmalloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++)
        p[i] = read_le_f32(f);
    return p;
}

static void load_model(const char* path, Model* m) {
    FILE* f = fopen(path, "rb");
    if (!f)
        die("cannot open model file");
    char magic[4];
    read_exact(f, magic, 4);
    if (memcmp(magic, "GPT1", 4) != 0)
        die("bad model magic (expected GPT1)");
    m->V = read_le32(f);
    m->T = read_le32(f);
    m->C = read_le32(f);
    m->H = read_le32(f);
    m->L = read_le32(f);
    m->G = read_le32(f);
    m->eps = read_le_f32(f);
    if (m->V == 0 || m->V > MAX_VOCAB || m->T == 0 || m->T > MAX_BLOCK)
        die("bad vocab/block size");
    if (m->C == 0 || m->C > MAX_DIM || m->C % m->G != 0 || m->H == 0 || m->C / m->H != m->G)
        die("bad n_embd/n_head (head_size 必须等于 group_size)");
    if (m->L == 0 || m->L > MAX_LAYERS)
        die("bad layer count");
    size_t C = m->C, T = m->T, V = m->V, L = m->L;

    m->tok_emb = read_f32_array(f, V * C);
    m->pos_emb = read_f32_array(f, T * C);
    m->ln1 = xmalloc(L * C * sizeof(float));
    m->ln2 = xmalloc(L * C * sizeof(float));
    for (uint32_t l = 0; l < L; l++) {
        float* g1 = read_f32_array(f, C);
        memcpy(m->ln1 + (size_t)l * C, g1, C * sizeof(float));
        free(g1);
        read_w1(f, &m->q[l], m->G, C);
        read_w1(f, &m->k[l], m->G, C);
        read_w1(f, &m->v[l], m->G, C);
        read_w1(f, &m->o[l], m->G, C);
        float* g2 = read_f32_array(f, C);
        memcpy(m->ln2 + (size_t)l * C, g2, C * sizeof(float));
        free(g2);
        read_w1(f, &m->fc[l], m->G, C);
        read_w1(f, &m->proj[l], m->G, 4 * C);
    }
    m->lnf = read_f32_array(f, C);
    read_w1(f, &m->head, m->G, C);
    if (m->head.out_f != V)
        die("lm_head out_features != vocab_size");

    m->voff = xmalloc((V + 1) * sizeof(uint32_t));
    size_t cap = 4 * V + 16, len = 0;
    m->vocab = xmalloc(cap);
    for (uint32_t i = 0; i < V; i++) {
        uint8_t l;
        read_exact(f, &l, 1);
        while (len + l + 1 > cap) {
            cap *= 2;
            m->vocab = xrealloc_char(m->vocab, cap);
        }
        if (l)
            read_exact(f, m->vocab + len, l);
        m->voff[i] = (uint32_t)len;
        len += l;
    }
    m->voff[V] = (uint32_t)len;
    if (fgetc(f) != EOF)
        die("trailing bytes in model file");
    fclose(f);

    size_t bytes = 0;
    W1Layer* all[] = {m->q, m->k, m->v, m->o, m->fc, m->proj, &m->head};
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) {
        size_t cnt = (all[i] == &m->head) ? 1 : L;
        for (size_t j = 0; j < cnt; j++) {
            W1Layer* w = &all[i][j];
            bytes += (size_t)w->out_f * w->in_f / 8 + (size_t)w->out_f * (w->in_f / w->group) * 2;
        }
    }
    printf("loaded %s: layer %u head %u embd %u block %u vocab %u | W1 权重 %.1f KB（FP32 等价 %.1f KB）\n", path,
           m->L, m->H, m->C, m->T, m->V, bytes / 1024.0,
           ((V * C + T * C + 2 * L * C + C) * 4 + bytes * 32) / 1024.0);
}

static void free_model(Model* m) {
    free(m->tok_emb);
    free(m->pos_emb);
    free(m->ln1);
    free(m->ln2);
    free(m->lnf);
    free(m->vocab);
    free(m->voff);
    W1Layer* all[] = {m->q, m->k, m->v, m->o, m->fc, m->proj, &m->head};
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) {
        size_t cnt = (all[i] == &m->head) ? 1 : m->L;
        for (size_t j = 0; j < cnt; j++) {
            free(all[i][j].bits);
            free(all[i][j].scale_bits);
            free(all[i][j].scales);
        }
    }
}

/* ------------------------------ 加速器后端 ------------------------------ */
static BNN_Device g_dev = NULL;
static BNN_Buffer g_in = NULL, g_out = NULL;
static int g_use_accel = 1;

/* 性能统计：把一次前向拆成"可卸载到加速器的部分"和"主机 FP32 胶水" */
static int g_profile = 0;
static double g_t_exec_attn = 0.0; /* 注意力分数的矩阵乘（BNN_Execute / 软件点积） */
static double g_t_dma = 0.0;    /* 与加速器打交道的数据搬运：BufferWrite/Read/MatrixCreate */
static double g_t_quant = 0.0;  /* 主机侧：激活 int8 量化 + query 打 sign bit */
static double g_t_total = 0.0;  /* 前向总时间 */
static uint64_t g_mac = 0;      /* 矩阵乘的乘加次数 */
static uint64_t g_mac_attn = 0;
static uint32_t g_calls = 0;
/* 矩阵乘分类：0=QKV+输出投影 1=MLP 2=lm_head（由 forward 在调用前设置） */
static int g_cat = 0;
static double g_t_cat[3] = {0, 0, 0};
static uint64_t g_mac_cat[3] = {0, 0, 0};
static const char* kCatName[3] = {"QKV+输出投影", "MLP(fc+proj)", "lm_head"};

static double now_s(void) {
    return (double)clock() / CLOCKS_PER_SEC;
}

static void profile_report(uint32_t nforward) {
    if (!g_profile) {
        printf("profile: 用 --profile 打开性能统计\n");
        return;
    }
    double exec = g_t_cat[0] + g_t_cat[1] + g_t_cat[2] + g_t_exec_attn;
    double glue = g_t_total - exec - g_t_dma - g_t_quant;
    if (glue < 0.0)
        glue = 0.0;
    printf("\n=== profile（%u 次前向，总 %.1f ms，每次 %.0f us）===\n", nforward, g_t_total * 1e3,
           g_t_total * 1e6 / (nforward ? nforward : 1));
    for (int c = 0; c < 3; c++)
        printf("  加速器 GEMM %-12s %8.1f ms  %5.1f%%   (%5.1f%% of MAC, %.0f MMAC/s)\n", kCatName[c],
               g_t_cat[c] * 1e3, 100.0 * g_t_cat[c] / g_t_total, 100.0 * (double)g_mac_cat[c] / g_mac,
               g_t_cat[c] > 0 ? (double)g_mac_cat[c] / g_t_cat[c] / 1e6 : 0.0);
    printf("  加速器 GEMM %-12s %8.1f ms  %5.1f%%   (%5.1f%% of MAC, %.0f MMAC/s)\n", "注意力分数",
           g_t_exec_attn * 1e3, 100.0 * g_t_exec_attn / g_t_total, 100.0 * (double)g_mac_attn / g_mac,
           g_t_exec_attn > 0 ? (double)g_mac_attn / g_t_exec_attn / 1e6 : 0.0);
    printf("  ------------------------------------------------------------\n");
    printf("  矩阵乘合计（可卸载）  %8.1f ms  %5.1f%%\n", exec * 1e3, 100.0 * exec / g_t_total);
    printf("  加速器 DMA/建矩阵     %8.1f ms  %5.1f%%\n", g_t_dma * 1e3, 100.0 * g_t_dma / g_t_total);
    printf("  激活 int8 量化/打包   %8.1f ms  %5.1f%%\n", g_t_quant * 1e3, 100.0 * g_t_quant / g_t_total);
    printf("  FP32 胶水（其余）     %8.1f ms  %5.1f%%   [softmax/attn@v/RMSNorm/残差/embedding]\n", glue * 1e3,
           100.0 * glue / g_t_total);
    printf("  ==> 与加速器相关的部分（矩阵乘+DMA）占总时间 %.1f%%；"
           "Amdahl 上限（加速器算得无限快）%.1fx\n",
           100.0 * (exec + g_t_dma) / g_t_total, g_t_total / (g_t_total - exec - g_t_dma));
    printf("  GEMM 调用 %u 次，%.2f M MAC/前向，软件后端等效 %.0f MMAC/s\n", g_calls,
           (double)g_mac / (nforward ? nforward : 1) / 1e6, (double)g_mac / exec / 1e6);
}

static void acc_status(BNN_Status st, const char* what) {
    if (st != BNN_OK) {
        fprintf(stderr, "error: %s: %s\n", what, BNN_StatusString(st));
        exit(1);
    }
}

/* ------------------------------ 核心算子 ------------------------------ */
/* 逐行对称 int8：xs = absmax/127，round half away from zero（对齐 numpy 的量化） */
static float quantize_row(const float* x, uint32_t n, int8_t* xq) {
    float amax = 0.0f;
    for (uint32_t j = 0; j < n; j++) {
        float a = fabsf(x[j]);
        if (a > amax)
            amax = a;
    }
    float xs = amax > 0.0f ? amax / 127.0f : 1.0f;
    for (uint32_t j = 0; j < n; j++) {
        float q = roundf(x[j] / xs);
        xq[j] = (int8_t)(q > 127.0f ? 127.0f : (q < -127.0f ? -127.0f : q));
    }
    return xs;
}

/* 软件 golden：组内整数累加 + 组外 FP32 scale，逐位等价于加速器的 GEMM */
static void soft_gemm(const W1Layer* L, const int8_t* xq, const float* xs, uint32_t rows, float* out) {
    uint32_t in = L->in_f, out_f = L->out_f, ng = in / L->group, rb = in / 8;
    int32_t gtotal[MAX_GROUPS];
    for (uint32_t r = 0; r < rows; r++) {
        const int8_t* xr = xq + (size_t)r * in;
        for (uint32_t g = 0; g < ng; g++) {
            int32_t s = 0;
            for (uint32_t j = g * L->group; j < (g + 1) * L->group; j++)
                s += xr[j];
            gtotal[g] = s;
        }
        for (uint32_t o = 0; o < out_f; o++) {
            const uint8_t* wrow = L->bits + (size_t)o * rb;
            const float* srow = L->scales + (size_t)o * ng;
            float acc = 0.0f;
            for (uint32_t g = 0; g < ng; g++) {
                int32_t pos = 0;
                for (uint32_t j = g * L->group; j < (g + 1) * L->group; j++)
                    if ((wrow[j >> 3] >> (j & 7)) & 1u)
                        pos += xr[j];
                acc += srow[g] * (float)(2 * pos - gtotal[g]);
            }
            float v = acc * xs[r];
            out[(size_t)r * out_f + o] = (L->relu && v < 0.0f) ? 0.0f : v;
        }
    }
}

/* 一层 W1A8：量化激活 -> GEMM（组内整数点积 + 组外 FP16 scale）-> 乘回逐行激活 scale -> 可选 ReLU */
static void w1_forward(const W1Layer* L, const float* x, uint32_t rows, float* out, int8_t* xq, float* xs,
                       float* zbuf) {
    double t0 = g_profile ? now_s() : 0.0;
    for (uint32_t r = 0; r < rows; r++)
        xs[r] = quantize_row(x + (size_t)r * L->in_f, L->in_f, xq + (size_t)r * L->in_f);
    if (g_profile)
        g_t_quant += now_s() - t0;
    if (g_profile) {
        g_mac += (uint64_t)rows * L->in_f * L->out_f;
        g_mac_cat[g_cat] += (uint64_t)rows * L->in_f * L->out_f;
        g_calls++;
        t0 = now_s();
    }
    if (!g_use_accel) {
        soft_gemm(L, xq, xs, rows, out);
        if (g_profile)
            g_t_cat[g_cat] += now_s() - t0;
        return;
    }
    double d0 = g_profile ? now_s() : 0.0;
    acc_status(BNN_BufferWrite(g_in, 0, xq, (size_t)rows * L->in_f), "BNN_BufferWrite");
    if (g_profile)
        g_t_dma += now_s() - d0;
    BNN_Job job;
    memset(&job, 0, sizeof job);
    job.operation = BNN_OP_GEMM;
    job.matrix = L->matrix;
    job.input = g_in;
    job.output = g_out;
    job.rows = rows;
    double e0 = g_profile ? now_s() : 0.0;
    acc_status(BNN_Execute(g_dev, &job), "BNN_Execute(GEMM)");
    if (g_profile)
        g_t_cat[g_cat] += now_s() - e0;
    d0 = g_profile ? now_s() : 0.0;
    acc_status(BNN_BufferRead(g_out, 0, zbuf, (size_t)rows * L->out_f * sizeof(float)), "BNN_BufferRead");
    if (g_profile)
        g_t_dma += now_s() - d0;
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t o = 0; o < L->out_f; o++) {
            float v = zbuf[(size_t)r * L->out_f + o] * xs[r];
            out[(size_t)r * L->out_f + o] = (L->relu && v < 0.0f) ? 0.0f : v;
        }
    }
}

/* 注意力分数：X = 逐行 int8 的 key，W = ±1(+FP16 组 scale) 的 query，输出 scores[tq][tk] */
static void attn_scores(uint32_t hs, uint32_t n, const float* qh, const float* kh, float* scores, float sqrt_hs,
                        int8_t* kq, float* kxs, uint8_t* bits, uint16_t* sbits, float* alpha, float* zbuf) {
    double t0 = g_profile ? now_s() : 0.0;
    for (uint32_t r = 0; r < n; r++)
        kxs[r] = quantize_row(kh + (size_t)r * hs, hs, kq + (size_t)r * hs);
    /* query -> sign bit + FP16 组 scale（组内 |q| 均值，与训练端一致） */
    for (uint32_t r = 0; r < n; r++) {
        const float* qr = qh + (size_t)r * hs;
        float s = 0.0f;
        for (uint32_t j = 0; j < hs; j++)
            s += fabsf(qr[j]);
        uint16_t h = fp32_to_fp16(s / (float)hs);
        sbits[r] = h;
        alpha[r] = fp16_to_fp32(h);
        uint8_t* brow = bits + (size_t)r * (hs / 8);
        memset(brow, 0, hs / 8);
        for (uint32_t j = 0; j < hs; j++)
            if (qr[j] >= 0.0f)
                brow[j >> 3] |= (uint8_t)(1u << (j & 7));
    }
    if (g_profile) {
        g_t_quant += now_s() - t0;
        g_mac_attn += (uint64_t)n * n * hs;
        g_calls++;
        t0 = now_s();
    }
    if (g_use_accel) {
        BNN_MatrixDesc d;
        memset(&d, 0, sizeof d);
        d.in_features = hs;
        d.out_features = n;
        d.group_size = hs;
        d.weight_bits = bits;
        d.weight_bytes = (size_t)n * hs / 8;
        d.scale_fp16 = sbits;
        d.scale_count = n;
        BNN_Matrix mat = NULL;
        double d0 = g_profile ? now_s() : 0.0;
        acc_status(BNN_MatrixCreate(g_dev, &d, &mat), "BNN_MatrixCreate(attn)");
        acc_status(BNN_BufferWrite(g_in, 0, kq, (size_t)n * hs), "BNN_BufferWrite(attn)");
        if (g_profile)
            g_t_dma += now_s() - d0;
        BNN_Job job;
        memset(&job, 0, sizeof job);
        job.operation = BNN_OP_GEMM;
        job.matrix = mat;
        job.input = g_in;
        job.output = g_out;
        job.rows = n; /* M = key 行数 */
        double e0 = g_profile ? now_s() : 0.0;
        acc_status(BNN_Execute(g_dev, &job), "BNN_Execute(attn GEMM)");
        if (g_profile)
            g_t_exec_attn += now_s() - e0;
        d0 = g_profile ? now_s() : 0.0;
        acc_status(BNN_BufferRead(g_out, 0, zbuf, (size_t)n * n * sizeof(float)), "BNN_BufferRead(attn)");
        BNN_MatrixDestroy(mat);
        if (g_profile)
            g_t_dma += now_s() - d0;
        /* 输出是 [M=key][N=query]，转置回 scores[query][key] 并乘回 key 的激活 scale */
        for (uint32_t tq = 0; tq < n; tq++)
            for (uint32_t tk = 0; tk < n; tk++)
                scores[(size_t)tq * n + tk] = zbuf[(size_t)tk * n + tq] * kxs[tk] / sqrt_hs;
        return;
    }
    for (uint32_t tq = 0; tq < n; tq++) {
        const uint8_t* brow = bits + (size_t)tq * (hs / 8);
        for (uint32_t tk = 0; tk < n; tk++) {
            int32_t dot = 0;
            for (uint32_t j = 0; j < hs; j++)
                dot += kq[(size_t)tk * hs + j] * (((brow[j >> 3] >> (j & 7)) & 1u) ? 1 : -1);
            scores[(size_t)tq * n + tk] = alpha[tq] * (float)dot * kxs[tk] / sqrt_hs;
        }
    }
    if (g_profile)
        g_t_exec_attn += now_s() - t0;
}

static void rmsnorm(const float* x, const float* gain, uint32_t n, uint32_t C, float eps, float* out) {
    for (uint32_t t = 0; t < n; t++) {
        const float* xr = x + (size_t)t * C;
        float ss = 0.0f;
        for (uint32_t c = 0; c < C; c++)
            ss += xr[c] * xr[c];
        float rms = sqrtf(ss / (float)C + eps);
        for (uint32_t c = 0; c < C; c++)
            out[(size_t)t * C + c] = xr[c] / rms * gain[c];
    }
}

/* ------------------------------ 前向 ------------------------------ */
typedef struct {
    float *x, *xb, *qb, *kb, *vb, *yb, *ob, *hb;
    float *qh, *kh, *sc, *att;
    int8_t *xq, *kq;
    float *xs, *kxs, *alpha, *z;
    uint8_t* bits;
    uint16_t* sbits;
} Scratch;

static Scratch* scratch_new(const Model* m) {
    uint32_t T = m->T, C = m->C, hs = C / m->H;
    uint32_t max_in = 4 * C, max_out = (m->V > 4 * C) ? m->V : 4 * C;
    if (T > max_out)
        max_out = T;
    Scratch* s = xmalloc(sizeof *s);
    s->x = xmalloc((size_t)T * C * sizeof(float));
    s->xb = xmalloc((size_t)T * C * sizeof(float));
    s->qb = xmalloc((size_t)T * C * sizeof(float));
    s->kb = xmalloc((size_t)T * C * sizeof(float));
    s->vb = xmalloc((size_t)T * C * sizeof(float));
    s->yb = xmalloc((size_t)T * C * sizeof(float));
    s->ob = xmalloc((size_t)T * C * sizeof(float));
    s->hb = xmalloc((size_t)T * 4 * C * sizeof(float));
    s->qh = xmalloc((size_t)T * hs * sizeof(float));
    s->kh = xmalloc((size_t)T * hs * sizeof(float));
    s->sc = xmalloc((size_t)T * T * sizeof(float));
    s->att = xmalloc((size_t)T * T * sizeof(float));
    s->xq = xmalloc((size_t)T * max_in);
    s->kq = xmalloc((size_t)T * hs);
    s->xs = xmalloc(T * sizeof(float));
    s->kxs = xmalloc(T * sizeof(float));
    s->alpha = xmalloc(T * sizeof(float));
    s->z = xmalloc((size_t)T * max_out * sizeof(float));
    s->bits = xmalloc((size_t)T * hs / 8);
    s->sbits = xmalloc(T * sizeof(uint16_t));
    return s;
}

static void scratch_free(Scratch* s) {
    free(s->x);
    free(s->xb);
    free(s->qb);
    free(s->kb);
    free(s->vb);
    free(s->yb);
    free(s->ob);
    free(s->hb);
    free(s->qh);
    free(s->kh);
    free(s->sc);
    free(s->att);
    free(s->xq);
    free(s->kq);
    free(s->xs);
    free(s->kxs);
    free(s->alpha);
    free(s->z);
    free(s->bits);
    free(s->sbits);
    free(s);
}

static void forward(const Model* m, const int32_t* ids, uint32_t n, float* logits, Scratch* s) {
    uint32_t C = m->C, H = m->H, hs = C / H, T = m->T;
    if (n == 0 || n > T)
        die("sequence length out of range");
    float sqrt_hs = sqrtf((float)hs);

    for (uint32_t t = 0; t < n; t++) {
        const float* te = m->tok_emb + (size_t)ids[t] * C;
        const float* pe = m->pos_emb + (size_t)t * C;
        for (uint32_t c = 0; c < C; c++)
            s->x[(size_t)t * C + c] = te[c] + pe[c];
    }

    for (uint32_t l = 0; l < m->L; l++) {
        rmsnorm(s->x, m->ln1 + (size_t)l * C, n, C, m->eps, s->xb);
        g_cat = 0; /* QKV 与输出投影（同在本层末尾的 out_proj 也归这一类） */
        w1_forward(&m->q[l], s->xb, n, s->qb, s->xq, s->xs, s->z);
        w1_forward(&m->k[l], s->xb, n, s->kb, s->xq, s->xs, s->z);
        w1_forward(&m->v[l], s->xb, n, s->vb, s->xq, s->xs, s->z);

        for (uint32_t h = 0; h < H; h++) {
            for (uint32_t t = 0; t < n; t++) /* head 切片在行内不连续，先聚拢 */
                for (uint32_t j = 0; j < hs; j++) {
                    s->qh[(size_t)t * hs + j] = s->qb[(size_t)t * C + h * hs + j];
                    s->kh[(size_t)t * hs + j] = s->kb[(size_t)t * C + h * hs + j];
                }
            attn_scores(hs, n, s->qh, s->kh, s->sc, sqrt_hs, s->kq, s->kxs, s->bits, s->sbits, s->alpha, s->z);
            for (uint32_t tq = 0; tq < n; tq++) { /* 因果 mask + softmax */
                float mx = -INFINITY;
                for (uint32_t tk = 0; tk <= tq; tk++)
                    if (s->sc[(size_t)tq * n + tk] > mx)
                        mx = s->sc[(size_t)tq * n + tk];
                float sum = 0.0f;
                for (uint32_t tk = 0; tk <= tq; tk++) {
                    float e = expf(s->sc[(size_t)tq * n + tk] - mx);
                    s->att[(size_t)tq * n + tk] = e;
                    sum += e;
                }
                for (uint32_t tk = 0; tk <= tq; tk++)
                    s->att[(size_t)tq * n + tk] /= sum;
                for (uint32_t tk = tq + 1; tk < n; tk++)
                    s->att[(size_t)tq * n + tk] = 0.0f;
            }
            for (uint32_t tq = 0; tq < n; tq++) /* attn @ V（FP32） */
                for (uint32_t j = 0; j < hs; j++) {
                    float acc = 0.0f;
                    for (uint32_t tk = 0; tk <= tq; tk++)
                        acc += s->att[(size_t)tq * n + tk] * s->vb[(size_t)tk * C + h * hs + j];
                    s->yb[(size_t)tq * C + h * hs + j] = acc;
                }
        }
        w1_forward(&m->o[l], s->yb, n, s->ob, s->xq, s->xs, s->z);
        for (size_t i = 0; i < (size_t)n * C; i++)
            s->x[i] += s->ob[i];

        rmsnorm(s->x, m->ln2 + (size_t)l * C, n, C, m->eps, s->xb);
        g_cat = 1; /* MLP */
        w1_forward(&m->fc[l], s->xb, n, s->hb, s->xq, s->xs, s->z);
        w1_forward(&m->proj[l], s->hb, n, s->ob, s->xq, s->xs, s->z);
        for (size_t i = 0; i < (size_t)n * C; i++)
            s->x[i] += s->ob[i];
    }

    rmsnorm(s->x, m->lnf, n, C, m->eps, s->xb);
    g_cat = 2; /* lm_head */
    w1_forward(&m->head, s->xb, n, logits, s->xq, s->xs, s->z);
}

/* 计时的前向包装：g_profile 打开时累计总时间 */
static uint32_t g_nforward = 0;

static void forward_timed(const Model* m, const int32_t* ids, uint32_t n, float* logits, Scratch* s) {
    double t0 = g_profile ? now_s() : 0.0;
    forward(m, ids, n, logits, s);
    if (g_profile) {
        g_t_total += now_s() - t0;
        g_nforward++;
    }
}

/* ------------------------------ 采样与词表 ------------------------------ */
static uint64_t rng_next(uint64_t* st) {
    uint64_t x = *st;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *st = x;
    return x * 0x2545F4914F6CDD1Dull;
}

static uint32_t argmax(const float* v, uint32_t n) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; i++)
        if (v[i] > v[best])
            best = i;
    return best;
}

/* work 是和 prob 等长的暂存区：阈值搜索在副本上做，prob 只被过滤不被破坏 */
static uint32_t sample_token(const float* logits, uint32_t V, float temp, uint32_t topk, uint64_t* rs, float* prob,
                             float* work) {
    if (temp <= 0.0f)
        return argmax(logits, V);
    float mx = logits[0];
    for (uint32_t i = 1; i < V; i++)
        if (logits[i] > mx)
            mx = logits[i];
    for (uint32_t i = 0; i < V; i++)
        prob[i] = expf((logits[i] - mx) / temp);
    uint32_t k = (topk == 0 || topk > V) ? V : topk;
    memcpy(work, prob, V * sizeof(float));
    float thresh = 0.0f;
    for (uint32_t pass = 0; pass < k; pass++) { /* 在副本上找第 k 大的概率作为阈值 */
        uint32_t bi = argmax(work, V);
        thresh = work[bi];
        work[bi] = -1.0f;
    }
    double sum = 0.0;
    for (uint32_t i = 0; i < V; i++) {
        if (prob[i] < thresh)
            prob[i] = 0.0f;
        sum += prob[i];
    }
    float r = (float)((double)(rng_next(rs) >> 11) / 9007199254740992.0 * sum);
    float acc = 0.0f;
    for (uint32_t i = 0; i < V; i++) {
        acc += prob[i];
        if (acc > r)
            return i;
    }
    return V - 1;
}

static uint32_t encode_prompt(const Model* m, const char* text, int32_t* ids, uint32_t cap) {
    uint32_t n = 0;
    const char* p = text;
    while (*p && n < cap) {
        int best = -1;
        size_t bl = 0;
        for (uint32_t i = 0; i < m->V; i++) { /* 最长匹配 */
            size_t len = m->voff[i + 1] - m->voff[i];
            if (len > bl && strncmp(p, m->vocab + m->voff[i], len) == 0) {
                best = (int)i;
                bl = len;
            }
        }
        if (best < 0) {
            p++; /* 词表外的字符跳过 */
            continue;
        }
        ids[n++] = best;
        p += bl;
    }
    return n;
}

static void print_ids(const Model* m, const int32_t* ids, uint32_t n) {
    for (uint32_t i = 0; i < n; i++)
        fwrite(m->vocab + m->voff[ids[i]], 1, m->voff[ids[i] + 1] - m->voff[ids[i]], stdout);
}

/* top-k 命中判定：把 logits 拷进 tmp 后逐次剔除最大值 */
static int in_topk(const float* lg, uint32_t V, uint32_t y, uint32_t k, float* tmp) {
    memcpy(tmp, lg, V * sizeof(float));
    for (uint32_t pass = 0; pass < k; pass++) {
        uint32_t bi = argmax(tmp, V);
        if (bi == y)
            return 1;
        tmp[bi] = -INFINITY;
    }
    return 0;
}

int main(int argc, char** argv) {
    const char* model_path = NULL, *ref_path = NULL, *tok_path = NULL;
    const char* prompt = "春天";
    float temp = 0.8f;
    uint32_t topk = 10, steps = 40;
    int use_accel = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--soft") == 0) {
            use_accel = 0;
        } else if (strcmp(argv[i], "--profile") == 0) {
            g_profile = 1;
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            temp = strtof(argv[++i], NULL);
        } else if (strcmp(argv[i], "--topk") == 0 && i + 1 < argc) {
            topk = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (!model_path) {
            model_path = argv[i];
        } else if (!ref_path && strstr(argv[i], "logits")) { /* 按名字识别，顺序就不敏感了 */
            ref_path = argv[i];
        } else if (!tok_path) {
            tok_path = argv[i];
        } else {
            fprintf(stderr, "usage: %s [--soft] [--temp F] [--topk K] [--steps N] [--prompt S] "
                            "model.bnn [ref_logits.bin] [val_tokens.bin]\n", argv[0]);
            return 1;
        }
    }
    if (!model_path) {
        fprintf(stderr, "usage: %s [--soft] model.bnn [ref_logits.bin] [val_tokens.bin]\n", argv[0]);
        return 1;
    }

    Model m;
    memset(&m, 0, sizeof m);
    load_model(model_path, &m);
    Scratch* s = scratch_new(&m);
    float* logits = xmalloc((size_t)m.T * m.V * sizeof(float));
    float* prob = xmalloc(m.V * sizeof(float));
    float* work = xmalloc(m.V * sizeof(float));
    float* tmp = xmalloc(m.V * sizeof(float));

    if (use_accel) {
        BNN_Caps caps;
        acc_status(BNN_DeviceOpen(0, &g_dev), "BNN_DeviceOpen");
        acc_status(BNN_DeviceGetCaps(g_dev, &caps), "BNN_DeviceGetCaps");
        printf("accelerator: api %u, group %u, max_in %u, max_out %u\n", caps.api_version, caps.group_size,
               caps.max_in_features, caps.max_out_features);
        if (caps.group_size != m.G)
            die("model group_size != device group_size");
        W1Layer* all[] = {m.q, m.k, m.v, m.o, m.fc, m.proj, &m.head};
        size_t max_in_bytes = 0, max_out_bytes = 0;
        for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) {
            size_t cnt = (all[i] == &m.head) ? 1 : m.L;
            for (size_t j = 0; j < cnt; j++) {
                W1Layer* L = &all[i][j];
                if (L->in_f > caps.max_in_features || L->out_f > caps.max_out_features)
                    die("layer exceeds device feature limits");
                BNN_MatrixDesc d;
                memset(&d, 0, sizeof d);
                d.in_features = L->in_f;
                d.out_features = L->out_f;
                d.group_size = L->group;
                d.weight_bits = L->bits;
                d.weight_bytes = (size_t)L->out_f * (L->in_f / 8);
                d.scale_fp16 = L->scale_bits;
                d.scale_count = (size_t)L->out_f * (L->in_f / L->group);
                acc_status(BNN_MatrixCreate(g_dev, &d, &L->matrix), "BNN_MatrixCreate");
                if ((size_t)m.T * L->in_f > max_in_bytes)
                    max_in_bytes = (size_t)m.T * L->in_f;
                if ((size_t)m.T * L->out_f * sizeof(float) > max_out_bytes)
                    max_out_bytes = (size_t)m.T * L->out_f * sizeof(float);
            }
        }
        size_t attn_in = (size_t)m.T * (m.C / m.H);
        size_t attn_out = (size_t)m.T * m.T * sizeof(float);
        if (attn_in > max_in_bytes)
            max_in_bytes = attn_in;
        if (attn_out > max_out_bytes)
            max_out_bytes = attn_out;
        acc_status(BNN_BufferAlloc(g_dev, max_in_bytes, &g_in), "BNN_BufferAlloc(input)");
        acc_status(BNN_BufferAlloc(g_dev, max_out_bytes, &g_out), "BNN_BufferAlloc(output)");
    }
    g_use_accel = use_accel;

    /* 1) 与 Python 端参考 logits 对比 */
    if (ref_path) {
        FILE* f = fopen(ref_path, "rb");
        if (!f)
            die("cannot open ref logits");
        uint32_t n = read_le32(f), V = read_le32(f);
        if (V != m.V || n == 0 || n > m.T)
            die("bad ref logits header");
        int32_t* ids = xmalloc(n * sizeof(int32_t));
        float* ref = xmalloc((size_t)n * V * sizeof(float));
        for (uint32_t i = 0; i < n; i++)
            ids[i] = (int32_t)read_le32(f);
        for (size_t i = 0; i < (size_t)n * V; i++)
            ref[i] = read_le_f32(f);
        fclose(f);

        forward_timed(&m, ids, n, logits, s);
        float maxrel = 0.0f, maxabs = 0.0f;
        uint32_t agree = 0, n_exact = 0;
        for (uint32_t t = 0; t < n; t++) {
            const float* lg = logits + (size_t)t * V;
            const float* rf = ref + (size_t)t * V;
            float dpos = 0.0f;
            agree += (argmax(lg, V) == argmax(rf, V));
            for (uint32_t i = 0; i < V; i++) {
                float d = fabsf(lg[i] - rf[i]);
                float r = d / (1.0f + fabsf(rf[i]));
                if (d > dpos)
                    dpos = d;
                if (d > maxabs)
                    maxabs = d;
                if (r > maxrel)
                    maxrel = r;
            }
            n_exact += (dpos < 1e-4f);
        }
        /* 判据：argmax 必须逐位置一致。误差本身允许 1 个 int8 量化台阶——
           C 与 torch 的 FP32 累加次序不同，个别激活会落在 round 的 .5 边界两侧。 */
        printf("C vs python ref: argmax agree %u/%u, max abs err %.3e, max rel err %.3e, "
               "positions within 1e-4 %u/%u %s\n",
               agree, n, (double)maxabs, (double)maxrel, n_exact, n,
               (agree == n && maxrel < 1e-2f) ? "OK" : "MISMATCH");
        printf("  ref text: ");
        print_ids(&m, ids, n);
        printf("\n");
        free(ids);
        free(ref);
    }

    /* 2) 加速器 vs 软件 golden 内核交叉验证 */
    if (use_accel) {
        int32_t ids[MAX_BLOCK];
        for (uint32_t i = 0; i < m.T; i++)
            ids[i] = (int32_t)(i % m.V);
        float* soft = xmalloc((size_t)m.T * m.V * sizeof(float));
        g_use_accel = 1;
        forward_timed(&m, ids, m.T, logits, s);
        g_use_accel = 0;
        forward_timed(&m, ids, m.T, soft, s);
        g_use_accel = 1;
        float maxdiff = 0.0f;
        uint32_t agree = 0;
        for (uint32_t t = 0; t < m.T; t++) {
            agree += (argmax(logits + (size_t)t * m.V, m.V) == argmax(soft + (size_t)t * m.V, m.V));
            for (uint32_t i = 0; i < m.V; i++) {
                float d = fabsf(logits[(size_t)t * m.V + i] - soft[(size_t)t * m.V + i]);
                if (d > maxdiff)
                    maxdiff = d;
            }
        }
        printf("accel vs soft kernel: argmax agree %u/%u, max abs logit diff %.3e\n", agree, m.T, (double)maxdiff);
        free(soft);
    }

    /* 3) 验证集 token 流上评估下一汉字预测 */
    if (tok_path) {
        FILE* f = fopen(tok_path, "rb");
        if (!f)
            die("cannot open val tokens");
        uint32_t n = read_le32(f);
        if (n < m.T + 1)
            die("val token stream too short");
        int32_t* ids = xmalloc(n * sizeof(int32_t));
        for (uint32_t i = 0; i < n; i++)
            ids[i] = (int32_t)read_le32(f);
        fclose(f);

        /* 与 Python 端同一套协议：滑窗取 T 个 token，窗口内每个位置都预测下一个汉字。
           stride = T 保证每个位置恰好被预测一次。 */
        uint32_t wins = n - m.T;
        uint32_t stride = m.T;
        double loss_sum = 0.0;
        uint32_t top1 = 0, top5 = 0, cnt = 0, nwin = 0;
        clock_t t0 = clock();
        for (uint32_t st = 0; st < wins; st += stride) {
            forward_timed(&m, ids + st, m.T, logits, s);
            for (uint32_t t = 0; t < m.T; t++) {
                const float* lg = logits + (size_t)t * m.V;
                uint32_t y = (uint32_t)ids[st + t + 1];
                float mx = lg[0];
                for (uint32_t i = 1; i < m.V; i++)
                    if (lg[i] > mx)
                        mx = lg[i];
                float sum = 0.0f;
                for (uint32_t i = 0; i < m.V; i++)
                    sum += expf(lg[i] - mx);
                loss_sum += (double)mx + logf(sum) - (double)lg[y];
                top1 += (argmax(lg, m.V) == y);
                top5 += in_topk(lg, m.V, y, 5, tmp);
                cnt++;
            }
            nwin++;
        }
        double sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
        printf("C val eval: %u windows × %u positions = %u, loss %.4f (ppl %.2f), top-1 %.4f, top-5 %.4f, "
               "%.0f us/window\n",
               nwin, m.T, cnt, loss_sum / cnt, exp(loss_sum / cnt), (double)top1 / cnt, (double)top5 / cnt,
               sec * 1e6 / nwin);
        free(ids);
    }

    /* 4) 生成一段中文 */
    {
        int32_t ids[MAX_BLOCK + 1024];
        uint32_t total = encode_prompt(&m, prompt, ids, m.T);
        if (total == 0)
            die("prompt 里没有词表内的汉字");
        uint64_t rs = 0x9E3779B97F4A7C15ull;
        for (uint32_t step = 0; step < steps; step++) {
            uint32_t ctx = total < m.T ? total : m.T;
            forward_timed(&m, ids + (total - ctx), ctx, logits, s);
            uint32_t nxt = sample_token(logits + (size_t)(ctx - 1) * m.V, m.V, temp, topk, &rs, prob, work);
            if (total < MAX_BLOCK + 1024)
                ids[total++] = (int32_t)nxt;
        }
        printf("generate [%s] temp %.2f topk %u: ", prompt, (double)temp, topk);
        print_ids(&m, ids, total);
        printf("\n");
    }

    profile_report(g_nforward);

    free(logits);
    free(prob);
    free(work);
    free(tmp);
    scratch_free(s);
    if (use_accel) {
        W1Layer* all[] = {m.q, m.k, m.v, m.o, m.fc, m.proj, &m.head};
        for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) {
            size_t cnt = (all[i] == &m.head) ? 1 : m.L;
            for (size_t j = 0; j < cnt; j++) {
                BNN_MatrixDestroy(all[i][j].matrix);
                all[i][j].matrix = NULL;
            }
        }
        BNN_BufferFree(g_in);
        BNN_BufferFree(g_out);
        BNN_DeviceClose(g_dev);
    }
    free_model(&m);
    return 0;
}
