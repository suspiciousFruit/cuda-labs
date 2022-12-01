#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

void print_device_properties(cudaDeviceProp& prop) {

}

constexpr const size_t N = 1024 * 1024;

__global__ void kernel(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = 2.0f * 3.1415926f * (float)idx / (float)N;
    data[idx] = sinf (sqrtf (x));
}

void run_hello_world(dim3 grid_dim, dim3 block_dim) {
    float* a;
    float* dev = nullptr;

    a = (float*)malloc(N * sizeof(float));
    cudaMalloc((void**)&dev, N * sizeof(float));


    auto start = std::chrono::high_resolution_clock::now();
    kernel<<<grid_dim, block_dim>>> (dev);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << "ns\n";

    cudaMemcpy(a, dev, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev);
    free(a);
}

int main(int argc, char* argv[]) {

    int deviceCount;
    cudaDeviceProp prop;

    cudaGetDeviceCount(&deviceCount);
    printf("Found %d devices\n", deviceCount);
    for (int device = 0; device < deviceCount; device++) {
        cudaGetDeviceProperties (&prop, device);
        printf("Device %d\n", device);
        printf("Compute capability     : %d.%d\n", prop.major, prop.minor);
        printf("Name                   : %s\n", prop.name);
        printf("Total Global Memory    : %u\n", prop.totalGlobalMem);
        printf("Shared memory per block: %d\n", prop.sharedMemPerBlock);
        printf("Registers per block    : %d\n", prop.regsPerBlock);
        printf("Warp size              : %d\n", prop.warpSize);
        printf("Max threads per block  : %d\n", prop.maxThreadsPerBlock);
        printf("Total constant memory  : %d\n", prop.totalConstMem);
        printf("MaxThreadsDim          : [%d, %d, %d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("MaxGridSize            : [%d, %d, %d]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }

    std::cout << "Base GPU execution time: ";
    run_hello_world({(N/512),1}, {512, 1});

//    start = std::chrono::high_resolution_clock::now();
    std::cout << "Full GPU execution time: ";
    run_hello_world({(N/prop.maxThreadsDim[0]),1}, {(uint32_t)prop.maxThreadsDim[0], 1});
//    end = std::chrono::high_resolution_clock::now();
//    std::cout << "Full GPU execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";

    return 0;
}
