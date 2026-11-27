/*
 * Copyright (c) 2026 teenygrad (https://teenygrad.org).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * y = x W^T on the SpacemiT K3 A100 cores' int8 matrix engine.
 *
 * smt.vmadot vd, vs1, vs2, i8 (A100: VLEN 1024, lambda 4) accumulates an 8x8 int32
 * tile C (vd, vd+1, row-major) += A (vs1: 8 rows x 16 int8, row-major) times
 * B^T (vs2: 8 columns x 16 int8, each column's 16 values contiguous).
 *
 *  - B is the weights: 8 output rows' 16 values of one K chunk, 128 bytes, chunks
 *    back to back: [N / 8][K / 16][8][16] int8, then N f32 row scales.
 *  - A is 8 tokens' 16 activations of the same chunk, from quant_pack_i8:
 *    [ceil(M / 8)][K / 16][8][16] int8, then one f32 scale per token.
 *  - One program is BLOCKS * 8 outputs over every tile of 8 tokens. A batch of one token
 *    uses one row of the tile; the A100 cores are only fast through these
 *    instructions, and decode is bound by the weight bytes, not the multiplies.
 *
 * The X100 cores cannot run these instructions: the launcher sends
 * *_a100_entry_point kernels to a pool on the A100 cores.
 *
 * The matrix instructions are emitted as their encodings, which GNU as does not
 * know; the comment beside each gives the mnemonic.
 */
#include <stdint.h>

/* Four chunks in flight; K / 16 must be a multiple of 4. */
static void tile_dot(const int8_t *a, const int8_t *b, long chunks, int32_t c[64]) {
    __asm__ volatile(
        "vsetvli    t0, zero, e32, m2, ta, ma\n"
        "vmv.v.i    v16, 0\n"
        "vmv.v.i    v18, 0\n"
        "vmv.v.i    v20, 0\n"
        "vmv.v.i    v22, 0\n"
        "vsetvli    t0, zero, e8, m1, ta, ma\n"
        "1:\n"
        "vle8.v     v1, (%[b])\n"
        "addi       t1, %[b], 128\n"
        "vle8.v     v3, (t1)\n"
        "addi       t1, %[b], 256\n"
        "vle8.v     v5, (t1)\n"
        "addi       t1, %[b], 384\n"
        "vle8.v     v7, (t1)\n"
        "vle8.v     v0, (%[a])\n"
        "addi       t2, %[a], 128\n"
        "vle8.v     v2, (t2)\n"
        "addi       t2, %[a], 256\n"
        "vle8.v     v4, (t2)\n"
        "addi       t2, %[a], 384\n"
        "vle8.v     v6, (t2)\n"
        ".4byte     0xe210382b\n" /* smt.vmadot v16, v0, v1, i8 */
        ".4byte     0xe231392b\n" /* smt.vmadot v18, v2, v3, i8 */
        ".4byte     0xe2523a2b\n" /* smt.vmadot v20, v4, v5, i8 */
        ".4byte     0xe2733b2b\n" /* smt.vmadot v22, v6, v7, i8 */
        "addi       %[b], %[b], 512\n"
        "addi       %[a], %[a], 512\n"
        "addi       %[n], %[n], -4\n"
        "bnez       %[n], 1b\n"
        "vsetvli    t0, zero, e32, m2, ta, ma\n"
        "vadd.vv    v16, v16, v18\n"
        "vadd.vv    v20, v20, v22\n"
        "vadd.vv    v16, v16, v20\n"
        "vse32.v    v16, (%[c])\n"
        : [a] "+r"(a), [b] "+r"(b), [n] "+r"(chunks)
        : [c] "r"(c)
        : "t0", "t1", "t2", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
}

/* Row blocks of 8 outputs per program: enough weight bytes per program that its
   setup and the scalar epilogue on these slow cores stay small. */
#define BLOCKS 4

void linear_ime_a100_entry_point(const int8_t *xq, const int8_t *w, float *y, int32_t m,
                                 int32_t n, int32_t k, int32_t m_pad, uint32_t pid,
                                 uint32_t pid_y, uint32_t pid_z, uint32_t grid_x,
                                 uint32_t grid_y, uint32_t grid_z) {
    (void)pid_y; (void)pid_z; (void)grid_x; (void)grid_y; (void)grid_z;
    const long chunks = k / 16;
    const float *wscale = (const float *)(w + (long)n * k);
    const float *xscale = (const float *)(xq + (long)m_pad * k);
    int32_t c[64] __attribute__((aligned(64)));
    for (int32_t blk = (int32_t)pid * BLOCKS; blk < (int32_t)pid * BLOCKS + BLOCKS && blk * 8 < n; blk++) {
        const int8_t *wb = w + (long)blk * chunks * 128;
        for (int32_t t0 = 0; t0 < m; t0 += 8) {
            tile_dot(xq + (long)t0 * k, wb, chunks, c);
            int32_t rows = m - t0 < 8 ? m - t0 : 8;
            for (int32_t r = 0; r < rows; r++) {
                float xs = xscale[t0 + r];
                float *out = y + (long)(t0 + r) * n + (long)blk * 8;
                const float *ws = wscale + blk * 8;
                for (int32_t j = 0; j < 8; j++)
                    out[j] = (float)c[r * 8 + j] * xs * ws[j];
            }
        }
    }
}
