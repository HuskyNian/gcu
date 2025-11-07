#include <tops/tops_runtime.h>
#include <tops.h>
#include <math.h>

#ifndef GCU_MLP_K_STEP
#define GCU_MLP_K_STEP 48
#endif
#ifndef GCU_MLP_J_TILE
#define GCU_MLP_J_TILE 384
#endif
#ifndef GCU_MLP_H_TILE
#define GCU_MLP_H_TILE 384
#endif
#ifndef GCU_MLP_H_CHUNK
#define GCU_MLP_H_CHUNK 192
#endif

#if GCU_MLP_K_STEP <= 0
#error "GCU_MLP_K_STEP must be positive"
#endif
#if GCU_MLP_J_TILE <= 0
#error "GCU_MLP_J_TILE must be positive"
#endif
#if GCU_MLP_H_TILE <= 0
#error "GCU_MLP_H_TILE must be positive"
#endif
#if GCU_MLP_H_CHUNK <= 0
#error "GCU_MLP_H_CHUNK must be positive"
#endif

__device__ __forceinline__ int gcu_min_int(int a, int b) {
    return a < b ? a : b;
}

__device__ __forceinline__ float silu_scalar(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ vfloat silu_vector(vfloat vx) {
    auto vs = tops::vsigmoid(vx);
    return tops::vmul<vfloat>(vx, vs);
}

__global__ void kernel_gcu_mlp(
    float * __restrict gate_w,
    float * __restrict up_w,
    float * __restrict down_w,
    float * __restrict input,
    float * __restrict output,
    const int seq_len,
    const int hidden_size,
    const int intermediate_size)
{
    const int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    const int thread_lane = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
    const int block_lane = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
    const size_t global_thread = (size_t)block_lane * (size_t)threads_per_block + (size_t)thread_lane;
    const size_t total_threads = (size_t)gridDim.x * gridDim.y * gridDim.z * (size_t)threads_per_block;

    if (total_threads == 0 || global_thread >= total_threads) {
        return;
    }

    const size_t base_rows = seq_len / total_threads;
    const size_t remain_rows = seq_len - base_rows * total_threads;
    const size_t row_begin = global_thread * base_rows + (global_thread < remain_rows ? global_thread : remain_rows);
    const size_t row_count = base_rows + (global_thread < remain_rows ? 1u : 0u);

    if (row_begin >= (size_t)seq_len || row_count == 0) {
        return;
    }

    const int VLEN = tops::vlength<vfloat>();

    tops_dte_ctx_t ctx_in;
    tops_dte_ctx_t ctx_gate[2];
    tops_dte_ctx_t ctx_up[2];
    tops_dte_ctx_t ctx_down;
    tops_dte_ctx_t ctx_out;
    tops::dte_scope scope_in(ctx_in);
    tops::dte_scope scope_gate0(ctx_gate[0]);
    tops::dte_scope scope_gate1(ctx_gate[1]);
    tops::dte_scope scope_up0(ctx_up[0]);
    tops::dte_scope scope_up1(ctx_up[1]);
    tops::dte_scope scope_down(ctx_down);
    tops::dte_scope scope_out(ctx_out);

    __valigned__ float input_tile[GCU_MLP_K_STEP];
    __valigned__ float gate_buffer[2][GCU_MLP_J_TILE];
    __valigned__ float up_buffer[2][GCU_MLP_J_TILE];
    __valigned__ float gate_acc[GCU_MLP_J_TILE];
    __valigned__ float up_acc[GCU_MLP_J_TILE];
    __valigned__ float act_tile[GCU_MLP_J_TILE];
    __valigned__ float out_tile[GCU_MLP_H_TILE];
    __valigned__ float down_tile[GCU_MLP_J_TILE * GCU_MLP_H_CHUNK];

    for (size_t row_offset = 0; row_offset < row_count; ++row_offset) {
        const int row = (int)(row_begin + row_offset);

        for (int h0 = 0; h0 < hidden_size; h0 += GCU_MLP_H_TILE) {
            const int hb = gcu_min_int(GCU_MLP_H_TILE, hidden_size - h0);
            for (int i = 0; i < hb; ++i) {
                out_tile[i] = 0.0f;
            }

            for (int j0 = 0; j0 < intermediate_size; j0 += GCU_MLP_J_TILE) {
                const int jb = gcu_min_int(GCU_MLP_J_TILE, intermediate_size - j0);
                for (int j = 0; j < jb; ++j) {
                    gate_acc[j] = 0.0f;
                    up_acc[j] = 0.0f;
                }

                for (int k0 = 0; k0 < hidden_size; k0 += GCU_MLP_K_STEP) {
                    const int kb = gcu_min_int(GCU_MLP_K_STEP, hidden_size - k0);
                    tops::memcpy(
                        ctx_in,
                        tops::mdspan(tops::Private, input_tile, kb),
                        tops::mdspan(tops::Global, input + (size_t)row * hidden_size + k0, kb));

                    tops::event gate_events[2];
                    tops::event up_events[2];
                    bool gate_inflight[2] = {false, false};
                    bool up_inflight[2] = {false, false};

                    int cur = 0;
                    int nxt = 1;

                    if (kb > 0) {
                        gate_events[cur] = tops::memcpy_async(
                            ctx_gate[cur],
                            tops::mdspan(tops::Private, gate_buffer[cur], jb),
                            tops::mdspan(tops::Global, gate_w + (size_t)(k0) * intermediate_size + j0, jb));
                        up_events[cur] = tops::memcpy_async(
                            ctx_up[cur],
                            tops::mdspan(tops::Private, up_buffer[cur], jb),
                            tops::mdspan(tops::Global, up_w + (size_t)(k0) * intermediate_size + j0, jb));
                        gate_inflight[cur] = true;
                        up_inflight[cur] = true;
                    }
                    if (kb > 1) {
                        gate_events[nxt] = tops::memcpy_async(
                            ctx_gate[nxt],
                            tops::mdspan(tops::Private, gate_buffer[nxt], jb),
                            tops::mdspan(tops::Global, gate_w + (size_t)(k0 + 1) * intermediate_size + j0, jb));
                        up_events[nxt] = tops::memcpy_async(
                            ctx_up[nxt],
                            tops::mdspan(tops::Private, up_buffer[nxt], jb),
                            tops::mdspan(tops::Global, up_w + (size_t)(k0 + 1) * intermediate_size + j0, jb));
                        gate_inflight[nxt] = true;
                        up_inflight[nxt] = true;
                    }

                    for (int kk = 0; kk < kb; ++kk) {
                        if (gate_inflight[cur]) {
                            tops::wait(gate_events[cur]);
                            gate_inflight[cur] = false;
                        }
                        if (up_inflight[cur]) {
                            tops::wait(up_events[cur]);
                            up_inflight[cur] = false;
                        }

                        const float x = input_tile[kk];
                        const float * gate_row = gate_buffer[cur];
                        const float * up_row = up_buffer[cur];

                        const int vec = (jb / VLEN) * VLEN;
                        const vfloat vx = tops::vbroadcast<vfloat>(x);
                        int j = 0;
                        for (; j < vec; j += VLEN) {
                            vfloat vacc_g = tops::vload<vfloat>(gate_acc + j);
                            vfloat vacc_u = tops::vload<vfloat>(up_acc + j);
                            const vfloat vg = tops::vload<vfloat>(gate_row + j);
                            const vfloat vu = tops::vload<vfloat>(up_row + j);
                            vacc_g = tops::vadd<vfloat>(vacc_g, tops::vmul<vfloat>(vg, vx));
                            vacc_u = tops::vadd<vfloat>(vacc_u, tops::vmul<vfloat>(vu, vx));
                            tops::vstore(vacc_g, gate_acc + j);
                            tops::vstore(vacc_u, up_acc + j);
                        }
                        for (; j < jb; ++j) {
                            gate_acc[j] += gate_row[j] * x;
                            up_acc[j] += up_row[j] * x;
                        }

                        const int next_row = kk + 2;
                        if (next_row < kb) {
                            gate_events[cur] = tops::memcpy_async(
                                ctx_gate[cur],
                                tops::mdspan(tops::Private, gate_buffer[cur], jb),
                                tops::mdspan(tops::Global, gate_w + (size_t)(k0 + next_row) * intermediate_size + j0, jb));
                            up_events[cur] = tops::memcpy_async(
                                ctx_up[cur],
                                tops::mdspan(tops::Private, up_buffer[cur], jb),
                                tops::mdspan(tops::Global, up_w + (size_t)(k0 + next_row) * intermediate_size + j0, jb));
                            gate_inflight[cur] = true;
                            up_inflight[cur] = true;
                        }

                        const int tmp = cur;
                        cur = nxt;
                        nxt = tmp;
                    }
                }

                int j = 0;
                const int vec_act = (jb / VLEN) * VLEN;
                for (; j + 8 * VLEN <= vec_act; j += 8 * VLEN) {
                    vfloat g0 = tops::vload<vfloat>(gate_acc + j + 0 * VLEN);
                    vfloat g1 = tops::vload<vfloat>(gate_acc + j + 1 * VLEN);
                    vfloat g2 = tops::vload<vfloat>(gate_acc + j + 2 * VLEN);
                    vfloat g3 = tops::vload<vfloat>(gate_acc + j + 3 * VLEN);
                    vfloat g4 = tops::vload<vfloat>(gate_acc + j + 4 * VLEN);
                    vfloat g5 = tops::vload<vfloat>(gate_acc + j + 5 * VLEN);
                    vfloat g6 = tops::vload<vfloat>(gate_acc + j + 6 * VLEN);
                    vfloat g7 = tops::vload<vfloat>(gate_acc + j + 7 * VLEN);
                    vfloat u0 = tops::vload<vfloat>(up_acc + j + 0 * VLEN);
                    vfloat u1 = tops::vload<vfloat>(up_acc + j + 1 * VLEN);
                    vfloat u2 = tops::vload<vfloat>(up_acc + j + 2 * VLEN);
                    vfloat u3 = tops::vload<vfloat>(up_acc + j + 3 * VLEN);
                    vfloat u4 = tops::vload<vfloat>(up_acc + j + 4 * VLEN);
                    vfloat u5 = tops::vload<vfloat>(up_acc + j + 5 * VLEN);
                    vfloat u6 = tops::vload<vfloat>(up_acc + j + 6 * VLEN);
                    vfloat u7 = tops::vload<vfloat>(up_acc + j + 7 * VLEN);
                    tops::vstore(tops::vmul<vfloat>(silu_vector(g0), u0), act_tile + j + 0 * VLEN);
                    tops::vstore(tops::vmul<vfloat>(silu_vector(g1), u1), act_tile + j + 1 * VLEN);
                    tops::vstore(tops::vmul<vfloat>(silu_vector(g2), u2), act_tile + j + 2 * VLEN);
                    tops::vstore(tops::vmul<vfloat>(silu_vector(g3), u3), act_tile + j + 3 * VLEN);
                    tops::vstore(tops::vmul<vfloat>(silu_vector(g4), u4), act_tile + j + 4 * VLEN);
                    tops::vstore(tops::vmul<vfloat>(silu_vector(g5), u5), act_tile + j + 5 * VLEN);
                    tops::vstore(tops::vmul<vfloat>(silu_vector(g6), u6), act_tile + j + 6 * VLEN);
                    tops::vstore(tops::vmul<vfloat>(silu_vector(g7), u7), act_tile + j + 7 * VLEN);
                }
                for (; j + 4 * VLEN <= vec_act; j += 4 * VLEN) {
                    vfloat g0 = tops::vload<vfloat>(gate_acc + j + 0 * VLEN);
                    vfloat g1 = tops::vload<vfloat>(gate_acc + j + 1 * VLEN);
                    vfloat g2 = tops::vload<vfloat>(gate_acc + j + 2 * VLEN);
                    vfloat g3 = tops::vload<vfloat>(gate_acc + j + 3 * VLEN);
                    vfloat u0 = tops::vload<vfloat>(up_acc + j + 0 * VLEN);
                    vfloat u1 = tops::vload<vfloat>(up_acc + j + 1 * VLEN);
                    vfloat u2 = tops::vload<vfloat>(up_acc + j + 2 * VLEN);
                    vfloat u3 = tops::vload<vfloat>(up_acc + j + 3 * VLEN);
                    tops::vstore(tops::vmul<vfloat>(silu_vector(g0), u0), act_tile + j + 0 * VLEN);
                    tops::vstore(tops::vmul<vfloat>(silu_vector(g1), u1), act_tile + j + 1 * VLEN);
                    tops::vstore(tops::vmul<vfloat>(silu_vector(g2), u2), act_tile + j + 2 * VLEN);
                    tops::vstore(tops::vmul<vfloat>(silu_vector(g3), u3), act_tile + j + 3 * VLEN);
                }
                for (; j < jb; ++j) {
                    act_tile[j] = silu_scalar(gate_acc[j]) * up_acc[j];
                }

                for (int h_chunk = 0; h_chunk < hb; h_chunk += GCU_MLP_H_CHUNK) {
                    const int hc = gcu_min_int(GCU_MLP_H_CHUNK, hb - h_chunk);
                    float * out_ptr = out_tile + h_chunk;

                    for (int jj = 0; jj < jb; ++jj) {
                        tops::memcpy(
                            ctx_down,
                            tops::mdspan(tops::Private, down_tile + (size_t)jj * hc, hc),
                            tops::mdspan(tops::Global, down_w + (size_t)(j0 + jj) * hidden_size + (h0 + h_chunk), hc));
                        const float a = act_tile[jj];
                        float * weight_row = down_tile + (size_t)jj * hc;
                        const vfloat va = tops::vbroadcast<vfloat>(a);
                        int c = 0;
                        const int vec_c = (hc / VLEN) * VLEN;
                        for (; c < vec_c; c += VLEN) {
                            vfloat vo = tops::vload<vfloat>(out_ptr + c);
                            vfloat vw = tops::vload<vfloat>(weight_row + c);
                            vo = tops::vadd<vfloat>(vo, tops::vmul<vfloat>(vw, va));
                            tops::vstore(vo, out_ptr + c);
                        }
                        for (; c < hc; ++c) {
                            out_ptr[c] += weight_row[c] * a;
                        }
                    }
                }
            }

            tops::memcpy(
                ctx_out,
                tops::mdspan(tops::Global, output + (size_t)row * hidden_size + h0, hb),
                tops::mdspan(tops::Private, out_tile, hb));
        }
    }
}

void GCU_MLP(float * __restrict gate_proj_weight,
             float * __restrict up_proj_weight,
             float * __restrict down_proj_weight,
             float * __restrict input,
             float * __restrict output,
             const int seq_len,
             const int hidden_size,
             const int intermediate_size)
{
    if (seq_len <= 0 || hidden_size <= 0 || intermediate_size <= 0) {
        return;
    }

    int threads = 12;
    if (seq_len < 4) {
        threads = seq_len;
        if (threads <= 0) {
            threads = 1;
        }
    }

    size_t target_rows_per_thread = 1;
    if (seq_len >= 2048) {
        target_rows_per_thread = 4;
    }
    if (seq_len >= 8192) {
        target_rows_per_thread = 8;
    }
    const size_t total_threads_needed = (seq_len + target_rows_per_thread - 1) / target_rows_per_thread;
    int blocks = (int)((total_threads_needed + threads - 1) / threads);
    if (blocks < 1) {
        blocks = 1;
    }
    if (blocks > 65535) {
        blocks = 65535;
    }

    kernel_gcu_mlp<<<dim3(blocks, 1, 1), dim3(threads, 1, 1)>>>(
        gate_proj_weight,
        up_proj_weight,
        down_proj_weight,
        input,
        output,
        seq_len,
        hidden_size,
        intermediate_size);
}

