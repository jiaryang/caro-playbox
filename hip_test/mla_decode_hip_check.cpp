#include <hip/hip_runtime.h>
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

// HIP kernel for QKV attention with sparse index indirection
__global__ void mla_decode_hip_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    const int* __restrict__ kv_indptr,
    const int* __restrict__ kv_indices,
    int D, int H, int B
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;

    int idx = b * H + h;
    const float* q_ptr = q + idx * D;
    float* out_ptr = out + idx * D;

    int start = kv_indptr[b];
    int end = kv_indptr[b + 1];
    int len = end - start;

    extern __shared__ float shared_mem[];
    float* scores = shared_mem;           // size: len
    float* acc = shared_mem + len;        // size: D

    // Initialize shared memory safely
    for (int d = tid; d < D; d += blockDim.x)
        acc[d] = 0.0f;
    if (tid == 0)
        for (int i = 0; i < len; ++i)
            scores[i] = 0.0f;
    __syncthreads();

    float e_max = -1e9f;
    for (int i = tid; i < len; i += blockDim.x) {
        int kv_idx = kv_indices[start + i];
        const float* k_ptr = k + kv_idx * D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d)
            dot += q_ptr[d] * k_ptr[d];
        scores[i] = dot;
        e_max = fmaxf(e_max, dot);
    }
    __syncthreads();

    __shared__ float block_max;
    if (tid == 0) {
        float local_max = scores[0];
        for (int i = 1; i < len; ++i)
            local_max = fmaxf(local_max, scores[i]);
        block_max = local_max;
    }
    __syncthreads();
    e_max = block_max;

    __shared__ float e_sum_shared;
    if (tid == 0) e_sum_shared = 0.0f;
    __syncthreads();

    for (int i = tid; i < len; i += blockDim.x) {
        float p = expf(scores[i] - e_max);
        atomicAdd(&e_sum_shared, p);

        int kv_idx = kv_indices[start + i];
        const float* v_ptr = v + kv_idx * D;
        for (int d = 0; d < D; ++d) {
            float weighted = p * v_ptr[d];
            atomicAdd(&acc[d], weighted);
        }
    }
    __syncthreads();
    float e_sum = e_sum_shared;

    for (int d = tid; d < D; d += blockDim.x) {
        out_ptr[d] = acc[d] / (e_sum + 1e-6f);
    }

    // Debug output for each block
    //if (tid == 0) {
    //    printf("[DEBUG] b=%d h=%d | out[0]=%.6f acc[0]=%.6f e_sum=%.6f\n", b, h, out_ptr[0], acc[0], e_sum);
    //}
}

struct TestCase {
    int B, H, D, P, S, num_splits;
    std::string name;
};

void initialize_qkv(std::vector<float>& q, std::vector<float>& k, std::vector<float>& v, bool use_random) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (int i = 0; i < q.size(); ++i)
        q[i] = use_random ? dist(rng) : float(i + 1);
    for (int i = 0; i < k.size(); ++i) {
        k[i] = use_random ? dist(rng) : float(i + 1);
        v[i] = use_random ? dist(rng) : float(i + 1);
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

void run_test(const TestCase& tc) {
    int B = tc.B, H = tc.H, D = tc.D, P = tc.P, S = tc.S, num_splits = tc.num_splits;
    std::cout << "\n========== Running Test: " << tc.name << " ==========" << std::endl;

    size_t size_q = B * H * D;
    size_t size_k = B * P * S * H * D;
    size_t size_v = size_k;
    size_t size_out = size_q;
    size_t size_indptr = B + 1;

    std::vector<float> h_q(size_q), h_k(size_k), h_v(size_v), h_out(size_out), h_ref(size_out);
    std::vector<int> h_indptr(size_indptr), h_indices;
    std::vector<int> h_lastpage(B, S);

    initialize_qkv(h_q, h_k, h_v, true);
    init_kv_layout(B, P, S, h_indptr, h_indices);
    int size_indices = h_indices.size();

    float *d_q, *d_k, *d_v, *d_out;
    int *d_indptr, *d_indices;
    CHECK_HIP(hipMalloc(&d_q, size_q * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_k, size_k * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_v, size_v * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_out, size_out * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_indptr, size_indptr * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_indices, size_indices * sizeof(int)));

    CHECK_HIP(hipMemcpy(d_q, h_q.data(), size_q * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_k, h_k.data(), size_k * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_v, h_v.data(), size_v * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_indptr, h_indptr.data(), size_indptr * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_indices, h_indices.data(), size_indices * sizeof(int), hipMemcpyHostToDevice));

    dim3 grid(B, H);
    dim3 block(std::max(P * S, D));
    size_t shared_size = (P * S + D) * sizeof(float);

    hipLaunchKernelGGL(mla_decode_hip_kernel, grid, block, shared_size, 0,
        d_q, d_k, d_v, d_out, d_indptr, d_indices, D, H, B);
    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipMemcpy(h_out.data(), d_out, size_out * sizeof(float), hipMemcpyDeviceToHost));

    // Host-side reference computation for verification
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            int idx = b * H + h;
            const float* q_ptr = h_q.data() + idx * D;
            float* ref_ptr = h_ref.data() + idx * D;
            std::fill(ref_ptr, ref_ptr + D, 0.0f);

            int start = h_indptr[b];
            int end = h_indptr[b + 1];
            int len = end - start;

            std::vector<float> scores(len);
            float e_max = -1e9f;
            for (int i = 0; i < len; ++i) {
                int kv_idx = h_indices[start + i];
                const float* k_ptr = h_k.data() + kv_idx * D;
                float dot = 0.0f;
                for (int d = 0; d < D; ++d)
                    dot += q_ptr[d] * k_ptr[d];
                scores[i] = dot;
                e_max = std::max(e_max, dot);
            }

            float e_sum = 0.0f;
            std::vector<float> probs(len);
            for (int i = 0; i < len; ++i) {
                probs[i] = std::exp(scores[i] - e_max);
                e_sum += probs[i];
            }

            for (int i = 0; i < len; ++i) {
                float p = probs[i] / (e_sum + 1e-6f);
                int kv_idx = h_indices[start + i];
                const float* v_ptr = h_v.data() + kv_idx * D;
                for (int d = 0; d < D; ++d)
                    ref_ptr[d] += p * v_ptr[d];
            }
        }
    }

    check_allclose_cpu_vs_gpu(h_ref, h_out, 1e-2f, 1e-2f, tc.name);

    CHECK_HIP(hipFree(d_q); hipFree(d_k); hipFree(d_v); hipFree(d_out));
    CHECK_HIP(hipFree(d_indptr); hipFree(d_indices));
}

int main() {
    std::vector<TestCase> test_cases = {
        {1, 1, 1, 1, 1, 1, "B1_H1_D1_P1_S1"},
        {1, 1, 1, 5, 1, 1, "B1_H1_D1_P5_S1"},
        {2, 1, 1, 5, 1, 1, "B2_H1_D1_P5_S1"},
        {1, 1, 2, 3, 1, 1, "B1_H1_D2_P3_S1"},
        {1, 2, 1, 5, 1, 1, "B1_H2_D1_P5_S1"},
        {1, 1, 1, 8, 1, 2, "B1_H1_D1_P8_S1_split2"},
        {256, 16, 576, 1, 1, 1, "B256_H16_D576_P1_S1"},
        {256, 16, 576, 2, 1, 1, "B256_H16_D576_P2_S1"},
        {256, 16, 576, 8, 1, 1, "B256_H16_D576_P8_S1"},
        {256, 16, 576, 16, 1, 1, "B256_H16_D576_P16_S1"}
    };

    for (const auto& tc : test_cases)
        run_test(tc);

    return 0;
}
