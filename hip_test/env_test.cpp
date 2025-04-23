#include <iostream>

int main() {
#if defined(__HIPCC__) || defined(__HIP__)
    std::cout << "HIPCC || HIP Platform" << std::endl;	
#elif defined(__HIP_PLATFORM_AMD__)
    std::cout << "HIP AMD Platform" << std::endl;
#elif defined(__HIP_PLATFORM_HCC__)
    std::cout << "HIP HCC Platform (deprecated)" << std::endl;
#elif defined(__CUDACC__) || defined(__NVCC__)
    std::cout << "CUDA Platform" << std::endl;
#else
    std::cout << "Unknown GPU Platform" << std::endl;
#endif
}

