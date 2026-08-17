#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include <string>
#include <cassert>

#define CHECK_HIP(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "[HIP ERROR] " << #call << " failed: " << hipGetErrorString(err) << std::endl; \
            return; \
        } \
    } while (0)

struct TestCase {
    int B, H, D, P, S, num_splits;
    std::string name;
};

void initialize_qkv(std::vector<float>& q, std::vector<float>& kv, bool use_random) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (int i = 0; i < q.size(); ++i)
        q[i] = use_random ? dist(rng) : float(i + 1);
    for (int i = 0; i < kv.size(); ++i) {
        kv[i] = use_random ? dist(rng) : float(i + 1);
    }
}

void init_kv_layout(int B, int P, int S, std::vector<int>& h_indptr, std::vector<int>& h_indices) {
    h_indptr.resize(B + 1, 0);
    h_indices.clear();
    int kv_per_batch = P * S;
    for (int b = 0; b < B; ++b) {
        h_indptr[b + 1] = h_indptr[b] + kv_per_batch;
        for (int i = 0; i < kv_per_batch; ++i)
            h_indices.push_back(b * kv_per_batch + i);
    }
}

bool check_allclose_cpu_vs_gpu(const std::vector<float>& ref,
                               const std::vector<float>& out,
                               float rtol = 1e-2f, float atol = 1e-2f,
                               const std::string& name = "checkAllclose",
                               int printNum = 8) {
    assert(ref.size() == out.size());
    int numel = ref.size();
    int mismatch_count = 0;
    float max_delta = 0.0f;
    std::vector<int> mismatch_indices;

    for (int i = 0; i < numel; ++i) {
        float diff = std::abs(ref[i] - out[i]);
        float tol = atol + rtol * std::abs(ref[i]);
        if (diff > tol) {
            mismatch_count++;
            if (mismatch_indices.size() < (size_t)printNum)
                mismatch_indices.push_back(i);
            max_delta = std::max(max_delta, diff);
        }
    }

    if (mismatch_count == 0) {
        std::cout << name << " [PASSED] Allclose passed. (rtol=" << rtol << ", atol=" << atol << ")\n";
        return true;
    }

    float percent = float(mismatch_count) / numel * 100.0f;
    std::cout << name << " [FAILED] " << mismatch_count << " / " << numel
              << " (" << percent << "%) elements mismatch\n";
    std::cout << "Max delta = " << max_delta << "\n";

    for (int i = 0; i < mismatch_indices.size(); ++i) {
        int idx = mismatch_indices[i];
        std::cout << "  idx=" << idx
                  << "  ref=" << ref[idx]
                  << "  out=" << out[idx]
                  << "  delta=" << std::abs(ref[idx] - out[idx]) << "\n";
    }
    return false;
}

__global__ void mla_decode_hip_kernel(
    const __hip_bfloat16* __restrict__ q,
    const __hip_bfloat16* __restrict__ kv,
    __hip_bfloat16* __restrict__ out,
    const int* __restrict__ kv_indptr,
    const int* __restrict__ kv_indices,
    int D, int H, int B,
    int S,
    int max_tile
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    int idx = b * H + h;

    const __hip_bfloat16* q_ptr = q + idx * D;
    __hip_bfloat16* out_ptr = out + idx * D;

    int start = kv_indptr[b];
    int end = kv_indptr[b + 1];
    int len = end - start;

    extern __shared__ float shared_mem[];
    float* scores = shared_mem;
    float* acc = shared_mem + max_tile;
    float* max_buf = shared_mem + max_tile + D;

    for (int d = tid; d < D; d += blockDim.x)
        acc[d] = 0.0f;
    __syncthreads();

    float local_max = -1e9f;
    for (int tile_start = 0; tile_start < len; tile_start += max_tile) {
        int tile_len = min(max_tile, len - tile_start);
        for (int i = tid; i < tile_len; i += blockDim.x) {
            int global_idx = start + tile_start + i;
            int kv_page = kv_indices[global_idx];
            int slot_id = global_idx % S;
            int flat_idx = ((kv_page * S + slot_id) * H + h) * D;
            const __hip_bfloat16* k_ptr = kv + flat_idx;

            float dot = 0.0f;
            for (int d = 0; d < D; ++d)
                dot += __bfloat162float(q_ptr[d]) * __bfloat162float(k_ptr[d]);
            scores[i] = dot;
            local_max = fmaxf(local_max, dot);
        }
        __syncthreads();
    }

    if (tid < blockDim.x)
        max_buf[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            max_buf[tid] = fmaxf(max_buf[tid], max_buf[tid + stride]);
        __syncthreads();
    }
    float e_max = max_buf[0];

    __shared__ float e_sum_shared;
    if (tid == 0) e_sum_shared = 0.0f;
    __syncthreads();

    for (int tile_start = 0; tile_start < len; tile_start += max_tile) {
        int tile_len = min(max_tile, len - tile_start);
        for (int i = tid; i < tile_len; i += blockDim.x) {
            int global_idx = start + tile_start + i;
            int kv_page = kv_indices[global_idx];
            int slot_id = global_idx % S;
            int flat_idx = ((kv_page * S + slot_id) * H + h) * D;
            const __hip_bfloat16* k_ptr = kv + flat_idx;
            const __hip_bfloat16* v_ptr = k_ptr;

            float dot = 0.0f;
            for (int d = 0; d < D; ++d)
                dot += __bfloat162float(q_ptr[d]) * __bfloat162float(k_ptr[d]);
            float p = expf(dot - e_max);
            atomicAdd(&e_sum_shared, p);

            for (int d = 0; d < D; ++d) {
                float weighted = p * __bfloat162float(v_ptr[d]);
                atomicAdd(&acc[d], weighted);
            }
        }
        __syncthreads();
    }

    float e_sum = e_sum_shared;
    for (int d = tid; d < D; d += blockDim.x) {
        float result = acc[d] / (e_sum + 1e-6f);
        out_ptr[d] = __float2bfloat16(result);
    }
}

void run_test(const TestCase& tc) {
    int B = tc.B, H = tc.H, D = tc.D, P = tc.P, S = tc.S;
    size_t size_q = B * H * D;
    size_t size_k = B * P * S * H * D;
    size_t size_out = size_q;
    size_t size_indptr = B + 1;

    std::vector<float> h_q(size_q), h_kv(size_k), h_ref(size_out), h_out(size_out);
    std::vector<__hip_bfloat16> h_q_bf16(size_q), h_k_bf16(size_k), h_out_bf16(size_out);
    std::vector<int> h_indptr(size_indptr), h_indices;
    initialize_qkv(h_q, h_kv, true);
    init_kv_layout(B, P, S, h_indptr, h_indices);
    for (int i = 0; i < size_q; ++i) h_q_bf16[i] = __float2bfloat16(h_q[i]);
    for (int i = 0; i < size_k; ++i) h_k_bf16[i] = __float2bfloat16(h_kv[i]);

    __hip_bfloat16 *d_q, *d_kv, *d_out;
    int *d_indptr, *d_indices;
    CHECK_HIP(hipMalloc(&d_q, size_q * sizeof(__hip_bfloat16)));
    CHECK_HIP(hipMalloc(&d_kv, size_k * sizeof(__hip_bfloat16)));
    CHECK_HIP(hipMalloc(&d_out, size_out * sizeof(__hip_bfloat16)));
    CHECK_HIP(hipMalloc(&d_indptr, size_indptr * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_indices, h_indices.size() * sizeof(int)));

    CHECK_HIP(hipMemcpy(d_q, h_q_bf16.data(), size_q * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_kv, h_k_bf16.data(), size_k * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_indptr, h_indptr.data(), size_indptr * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_indices, h_indices.data(), h_indices.size() * sizeof(int), hipMemcpyHostToDevice));

    int tile_max = std::min(P * S, 1024);
    size_t shared_mem = (tile_max + D + tile_max) * sizeof(float);
    dim3 grid(B, H);
    dim3 block(tile_max);

    hipLaunchKernelGGL(mla_decode_hip_kernel, grid, block, shared_mem, 0,
        d_q, d_kv, d_out, d_indptr, d_indices, D, H, B, S, tile_max);
    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipMemcpy(h_out_bf16.data(), d_out, size_out * sizeof(__hip_bfloat16), hipMemcpyDeviceToHost));
    for (int i = 0; i < size_out; ++i)
        h_out[i] = __bfloat162float(h_out_bf16[i]);

    // compute reference
    std::vector<float> scores, probs;
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            int idx = b * H + h;
            const float* q_ptr = h_q.data() + idx * D;
            float* ref_ptr = h_ref.data() + idx * D;
            std::fill(ref_ptr, ref_ptr + D, 0.0f);

            int start = h_indptr[b];
            int end = h_indptr[b + 1];
            int len = end - start;

            scores.resize(len);
            float e_max = -1e9f;
            for (int i = 0; i < len; ++i) {
                int global_idx = start + i;
                int kv_page = h_indices[global_idx];
                int slot_id = global_idx % S;
                int flat_idx = ((kv_page * S + slot_id) * H + h) * D;
                const float* k_ptr = h_kv.data() + flat_idx;
                float dot = 0.0f;
                for (int d = 0; d < D; ++d)
                    dot += q_ptr[d] * k_ptr[d];
                scores[i] = dot;
                e_max = std::max(e_max, dot);
            }

            probs.resize(len);
            float e_sum = 0.0f;
            for (int i = 0; i < len; ++i) {
                probs[i] = std::exp(scores[i] - e_max);
                e_sum += probs[i];
            }

            for (int i = 0; i < len; ++i) {
                float p = probs[i] / (e_sum + 1e-6f);
                int global_idx = start + i;
                int kv_page = h_indices[global_idx];
                int slot_id = global_idx % S;
                int flat_idx = ((kv_page * S + slot_id) * H + h) * D;
                const float* v_ptr = h_kv.data() + flat_idx;
                for (int d = 0; d < D; ++d)
                    ref_ptr[d] += p * v_ptr[d];
            }
        }
    }

    check_allclose_cpu_vs_gpu(h_ref, h_out, 1e-2f, 1e-2f, tc.name);

    CHECK_HIP(hipFree(d_q));
    CHECK_HIP(hipFree(d_kv));
    CHECK_HIP(hipFree(d_out));
    CHECK_HIP(hipFree(d_indptr));
    CHECK_HIP(hipFree(d_indices));
}

int main() {
    std::vector<TestCase> test_cases = {
        {1, 1, 1, 1, 1, 1, "B1_H1_D1_P1_S1"},
        {1, 1, 1, 5, 1, 1, "B1_H1_D1_P5_S1"},
        {2, 1, 1, 5, 1, 1, "B2_H1_D1_P5_S1"},
        {1, 1, 2, 3, 1, 1, "B1_H1_D2_P3_S1"},
        {1, 2, 1, 5, 1, 1, "B1_H2_D1_P5_S1"},
        {256, 16, 64, 1, 1, 1, "B256_H16_D64_P1_S1"},
        {256, 16, 576, 1, 1, 1, "B256_H16_D576_P1_S1"},
        {256, 16, 576, 2, 1, 1, "B256_H16_D576_P2_S1"},
        {256, 16, 576, 256, 1, 1, "B256_H16_D576_P256_S1"}
    };

    for (const auto& tc : test_cases)
        run_test(tc);

    return 0;
}
