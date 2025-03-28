// mla_decode_hip_single.cpp（改用 HIP 實作 reference attention）
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <cstring>
#include <cassert>

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
    

    printf("b = %d, h = %d, q[0] = %.6f\\n", b, h, q_ptr[0]);

    float* out_ptr = out + idx * D;

    int start = kv_indptr[b];
    int end = kv_indptr[b + 1];
    int len = end - start;
    printf("start = %d, end = %d\\n", start, end);

    extern __shared__ float shared_mem[];
    float* scores = shared_mem;           // size: len
    float* acc = shared_mem + len;        // size: D

    for (int d = tid; d < D; d += blockDim.x)
        acc[d] = 0.0f;
    __syncthreads();

    float e_max = -1e9f;
    for (int i = tid; i < len; i += blockDim.x) {
        int kv_idx = kv_indices[start + i];
        const float* k_ptr = k + (kv_idx * H + h) * D;
        printf("kv_idx=%d  k[%d]=%.6f  \\n", kv_idx, i, k_ptr[i]);
        float dot = 0.0f;
        for (int d = 0; d < D; ++d)
            dot += q_ptr[d] * k_ptr[d];
        scores[i] = dot;
        e_max = fmaxf(e_max, dot);
        printf("scores[%d] = %f, \\n", i, scores[i]);
    }
    __syncthreads();

    // Reduce max across threads
    __shared__ float block_max;
    if (tid == 0) {
        float local_max = scores[0];
        for (int i = 1; i < len; ++i)
            local_max = fmaxf(local_max, scores[i]);
        block_max = local_max;
        printf("block_max = %f, \\n", block_max);
    }
    __syncthreads();
    e_max = block_max;

    __shared__ float e_sum_shared;
    if (tid == 0) e_sum_shared = 0.0f;
        __syncthreads();

    for (int i = tid; i < len; i += blockDim.x) {
        float p = expf(scores[i] - e_max);
        atomicAdd(&e_sum_shared, p);
        printf("e_sum_shared = %f, p[%d] = %f\\n", e_sum_shared, i, p);

        int kv_idx = kv_indices[start + i];
        const float* v_ptr = v + (kv_idx * H + h) * D;
        printf("kv_idx=%d  v[%d]=%.6f  \\n", kv_idx, i, v_ptr[i]);
        for (int d = 0; d < D; ++d) {
            float weighted = p * v_ptr[d];
            atomicAdd(&acc[d], weighted);
            printf("acc[%d] = %f, \\n", d, acc[d]);
        }
    }
    __syncthreads();
    float e_sum = e_sum_shared;

    for (int d = tid; d < D; d += blockDim.x) {
        out_ptr[d] = acc[d] / (e_sum + 1e-6f);
    }

    if (tid == 0) {
        printf("b=%d h=%d out=%.6f (acc=%.6f / sum=%.6f)\\n", b, h, out_ptr[0], acc[0], e_sum);
    }
}

// ===================== Main ======================= //

void init_kv_layout(int B, int P, int S,
    std::vector<int>& h_indptr,
    std::vector<int>& h_indices) {

    h_indptr.resize(B + 1, 0);
    h_indices.clear();

    int kv_per_batch = P * S;
    for (int b = 0; b < B; ++b) {
        h_indptr[b + 1] = h_indptr[b] + kv_per_batch;
        for (int i = 0; i < kv_per_batch; ++i) {
            h_indices.push_back(b * kv_per_batch + i);
        }
    }
    std::cout << "h_indptr = { ";
    for (int i = 0; i < h_indptr.size(); ++i)
        std::cout << h_indptr[i] << (i + 1 < h_indptr.size() ? ", " : " ");
    std::cout << "}" << std::endl;
    
    std::cout << "h_indices = { ";
    for (int i = 0; i < h_indices.size(); ++i)
        std::cout << h_indices[i] << (i + 1 < h_indices.size() ? ", " : " ");
    std::cout << "}" << std::endl;
}

void initialize_qkv(std::vector<float>& q, std::vector<float>& k, std::vector<float>& v, bool use_random) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);

    for (int i = 0; i < q.size(); ++i) {
        q[i] = use_random ? dist(rng) : float(i + 1);
        std::cout << "q[" << i <<"] = " << q[i] << "\n";
    }
    for (int i = 0; i < k.size(); ++i) {
        k[i] = use_random ? dist(rng) : float(i + 1);
        v[i] = use_random ? dist(rng) : float(i + 1);
        std::cout 
            << "k[" << i <<"] = " << k[i]
            << ", v[" << i <<"] = " << v[i] << "\n";
    }
}

int main() {
    //const int B = 1, H = 1, D = 1, P = 1, S = 1, num_splits = 1;
    //const int B = 1, H = 1, D = 1, P = 5, S = 1, num_splits = 1;
    const int B = 2, H = 1, D = 1, P = 5, S = 1, num_splits = 1;
    //const int B = 1, H = 1, D = 2, P = 3, S = 1, num_splits = 1;
    //const int B = 1, H = 2, D = 1, P = 5, S = 1, num_splits = 1;
    //const int B = 1, H = 1, D = 1, P = 8, S = 1, num_splits = 2;

    using scalar_t = float;

    size_t size_q = B * H * D;
    size_t size_k = P * S * H * D;
    size_t size_v = P * S * H * D;
    size_t size_out = size_q;
    size_t size_lse = B * H * num_splits;
    size_t size_indptr = B + 1;

    std::vector<scalar_t> h_q(size_q), h_k(size_k), h_v(size_v);
    std::vector<scalar_t> h_out(size_out);
    std::vector<float> h_lse(size_lse);
    std::vector<scalar_t> h_ref(size_out);
    std::vector<int> h_indptr(size_indptr, 0);
    std::vector<int> h_indices;
    std::iota(h_indices.begin(), h_indices.end(), 0);
    std::vector<int> h_lastpage = {S, S};

    const bool use_random = false;
    initialize_qkv(h_q, h_k, h_v, use_random);
    init_kv_layout(B, P, S, h_indptr, h_indices);
    int size_indices = h_indices.size();

    float *d_q, *d_k, *d_v, *d_out;
    hipMalloc(&d_q, size_q * sizeof(scalar_t));
    hipMalloc(&d_k, size_k * sizeof(scalar_t));
    hipMalloc(&d_v, size_v * sizeof(scalar_t));
    hipMalloc(&d_out, size_out * sizeof(scalar_t));
    
    int *d_indptr, *d_indices;
    hipMalloc(&d_indptr, size_indptr * sizeof(int));
    hipMalloc(&d_indices, size_indices * sizeof(int));

    float *d_lse;
    int *d_lastpage;
    hipMalloc(&d_lse, size_lse * sizeof(float));
    hipMalloc(&d_lastpage, B * sizeof(int));

    hipMemcpy(d_q, h_q.data(), size_q * sizeof(scalar_t), hipMemcpyHostToDevice);
    hipMemcpy(d_k, h_k.data(), size_k * sizeof(scalar_t), hipMemcpyHostToDevice);
    hipMemcpy(d_v, h_v.data(), size_v * sizeof(scalar_t), hipMemcpyHostToDevice);
    hipMemcpy(d_indptr, h_indptr.data(), size_indptr * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_indices, h_indices.data(), size_indices * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_lastpage, h_lastpage.data(), B * sizeof(int), hipMemcpyHostToDevice);

    float scale = 1.0f / std::sqrt((float)D);
    int stride_qh = D, stride_kbs = S * H * D, stride_kh = D;
    int stride_vbs = stride_kbs, stride_vh = D;

    size_t shared_size = (P + D) * sizeof(float);  // scores + acc
    dim3 grid(B, H);
    dim3 block(std::max(P, D));

    auto start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(mla_decode_hip_kernel, grid, block, shared_size, 0,
        d_q, d_k, d_v, d_out,
        d_indptr, d_indices,
        D, H, B);
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration<double, std::micro>(end - start).count();

    hipMemcpy(h_out.data(), d_out, size_out * sizeof(scalar_t), hipMemcpyDeviceToHost);
    hipMemcpy(h_lse.data(), d_lse, size_lse * sizeof(float), hipMemcpyDeviceToHost);

    for (int i = 0; i < size_out; ++i) {
        float test_val = h_out[i];
        std::cout << "i=" << i
              << "  out=" << test_val
              << std::endl;
    }

    std::cout << "HIP MLA time (us): " << dur << "\n";

    hipFree(d_q); hipFree(d_k); hipFree(d_v);hipFree(d_out);
    hipFree(d_indptr); hipFree(d_indices);
    return 0;
}
