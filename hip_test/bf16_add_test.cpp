#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CHECK(cmd) \
    { \
        hipError_t err = cmd; \
        if (err != hipSuccess) { \
            std::cerr << "Error: " << hipGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    }

__global__ void bf16_add(const hip_bfloat16* a, const hip_bfloat16* b, hip_bfloat16* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 16;
    std::vector<hip_bfloat16> h_a(N), h_b(N), h_o(N);
    for (int i = 0; i < N; ++i) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 0.5f;
    }

    // Allocate device memory
    hip_bfloat16 *d_a, *d_b, *d_o;
    CHECK(hipMalloc(&d_a, N * sizeof(hip_bfloat16)));
    CHECK(hipMalloc(&d_b, N * sizeof(hip_bfloat16)));
    CHECK(hipMalloc(&d_o, N * sizeof(hip_bfloat16)));

    // Copy to device
    CHECK(hipMemcpy(d_a, h_a.data(), N * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_b, h_b.data(), N * sizeof(hip_bfloat16), hipMemcpyHostToDevice));

    // Launch kernel
    dim3 block(64);
    dim3 grid((N + block.x - 1) / block.x);
    hipLaunchKernelGGL(bf16_add, grid, block, 0, 0, d_a, d_b, d_o, N);

    // Copy result back
    CHECK(hipMemcpy(h_o.data(), d_o, N * sizeof(hip_bfloat16), hipMemcpyDeviceToHost));

    // Convert to float and print
    std::cout << "Result of a + b (bf16):\n";
    for (int i = 0; i < N; ++i) {
        std::cout << h_a[i] << " + " << h_b[i] << " = " << h_o[i] << std::endl;
    }

    // Free memory
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_o);

    return 0;
}

