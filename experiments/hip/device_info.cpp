#if defined(__HIPCC__) || defined(__HIP__)
  #include <hip/hip_runtime.h>
  #define GET_DEVICE_PROPERTIES hipGetDeviceProperties
  #define DEVICE_PROP_TYPE hipDeviceProp_t
#else
  #include <cuda_runtime.h>
  #define GET_DEVICE_PROPERTIES cudaGetDeviceProperties
  #define DEVICE_PROP_TYPE cudaDeviceProp
#endif

#include <iostream>

void print_device_info() {
    DEVICE_PROP_TYPE props;
    GET_DEVICE_PROPERTIES(&props, 0);  // device 0

    std::cout << "Device name: " << props.name << " \n\n";

    std::cout << "MultiProcessorCount: " << props.multiProcessorCount << std::endl;
    std::cout << "Max threads per multi processor: " << props.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Max blocks per multi processor: " << props.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "Shared memory per multi processor: " << props.sharedMemPerMultiprocessor << " bytes \n\n";

    std::cout << "Max threads per block: " << props.maxThreadsPerBlock << std::endl;
    std::cout << "Shared memory per block: " << props.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "Shared memory per block optin: " << props.sharedMemPerBlockOptin  << " bytes \n\n";

    std::cout << "Max grid dim (x): " << props.maxGridSize[0] << std::endl;
    std::cout << "Max grid dim (y): " << props.maxGridSize[1] << std::endl;
    std::cout << "Max grid dim (z): " << props.maxGridSize[2] << std::endl;
    std::cout << "Max block dim (x): " << props.maxThreadsDim[0] << std::endl;
    std::cout << "Max block dim (y): " << props.maxThreadsDim[1] << std::endl;
    std::cout << "Max block dim (z): " << props.maxThreadsDim[2] << std::endl;
}

int main(){
	print_device_info();
	return 0;
}
