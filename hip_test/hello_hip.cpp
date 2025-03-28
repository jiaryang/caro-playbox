#include "hip/hip_runtime.h"
#include <cstdlib>
#include <iostream>

#define HIP_CHECK(expression) {                             \
    const hipError_t err = expression;                      \
    if (err != hipSuccess){                                 \
      std::cout << "HIP Error: " << hipGetErrorString(err)  \
                << " at line " << __LINE__ << std::endl;    \
      std::exit(EXIT_FAILURE);                              \
    }                                                       \
  }

__host__ __device__ void call_func(){
  #ifdef __HIP_DEVICE_COMPILE__
    printf("device\n");
  #else
    std::cout << "host" << std::endl;
  #endif
}

__global__ void test_kernel(){
  call_func();
}

int main(int argc, char** argv) {

  // Identifying Host or Device Compilation Pass
  test_kernel<<<1, 1, 0, 0>>>();
  HIP_CHECK(hipDeviceSynchronize());
  call_func();

  // Host Code Feature Identification
  int deviceCount;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  std::cout << "Get device count:" << deviceCount << std::endl;

  int device = 0; // Query first available GPU. Can be replaced with any
                  // integer up to, not including, deviceCount
  hipDeviceProp_t deviceProp;
  HIP_CHECK(hipGetDeviceProperties(&deviceProp, device));

  std::cout << "The queried device ";
  if (deviceProp.arch.hasSharedInt32Atomics) // portable HIP feature query
    std::cout << "supports";
  else
    std::cout << "does not support";
  std::cout << " shared int32 atomic operations" << std::endl;

}