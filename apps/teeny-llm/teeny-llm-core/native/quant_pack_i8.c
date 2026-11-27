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
 * Quantises each token's activations to int8 with one scale per token, into the
 * matrix-engine tiles linear_ime reads: token t goes to tile t / 8, row t % 8,
 * [ceil(M / 8)][K / 16][8][16] int8, then the M_pad scales amax / 127 as f32.
 * One program per padded token; those past the M real ones return.
 */
#include <stdint.h>
#include <riscv_vector.h>

void quant_pack_i8_entry_point(const float *x, int8_t *out, int32_t k, int32_t m_pad,
                               int32_t m, uint32_t t, uint32_t pid_y, uint32_t pid_z,
                               uint32_t grid_x, uint32_t grid_y, uint32_t grid_z) {
    (void)pid_y; (void)pid_z; (void)grid_x; (void)grid_y; (void)grid_z;
    if ((int32_t)t >= m) return; /* a padding row: nothing to read */
    const float *row = x + (long)t * k;
    size_t vl;
    vfloat32m8_t vmax = __riscv_vfmv_v_f_f32m8(0.0f, __riscv_vsetvlmax_e32m8());
    for (int32_t i = 0; i < k; i += vl) {
        vl = __riscv_vsetvl_e32m8(k - i);
        vmax = __riscv_vfmax_vv_f32m8_tu(vmax, vmax, __riscv_vfabs_v_f32m8(__riscv_vle32_v_f32m8(row + i, vl), vl), vl);
    }
    size_t vlmax = __riscv_vsetvlmax_e32m8();
    float amax = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmax_vs_f32m8_f32m1(vmax, __riscv_vfmv_s_f_f32m1(0.0f, 1), vlmax));
    float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    float inv = 1.0f / scale;
    int8_t *tile = out + (long)(t / 8) * k * 8 + (t % 8) * 16;
    /* One 16-value chunk at a time: round to nearest (the default rounding mode),
       saturate to [-127, 127], narrow to int8, store into the chunk's tile row. */
    for (int32_t c = 0; c < k / 16; c++) {
        vfloat32m4_t v = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(row + c * 16, 16), inv, 16);
        vint32m4_t q = __riscv_vfcvt_x_f_v_i32m4(v, 16);
        q = __riscv_vmax_vx_i32m4(__riscv_vmin_vx_i32m4(q, 127, 16), -127, 16);
        vint8m1_t b = __riscv_vncvt_x_x_w_i8m1(__riscv_vncvt_x_x_w_i16m2(q, 16), 16);
        __riscv_vse8_v_i8m1(tile + (long)c * 128, b, 16);
    }
    ((float *)(out + (long)m_pad * k))[t] = amax > 0.0f ? scale : 0.0f;
}
