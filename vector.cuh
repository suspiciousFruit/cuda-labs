#include <cuda_runtime.h>


void calculate_gpu(float* a, float* b, float* r, size_t size) {
    float *device_a, *device_b, *device_r;
    cudaError error;

    auto assert = [](cudaError error) {
        if (error != cudaSuccess) {
            std::cerr << cudaGetErrorString(error) << std::endl;
            exit(1);
        }
    };

    error = cudaMalloc(&device_a, size * sizeof(float));
    assert(error);

    error = cudaMalloc(&device_b, size * sizeof(float));
    assert(error);

    error = cudaMalloc(&device_r, size * sizeof(float));
    assert(error);

    error = cudaMemcpy(device_a, a, size * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
    assert(error);

    error = cudaMemcpy(device_b, b, size * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
    assert(error);

    unsigned int threads_per_block = 256;
    unsigned int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    /// Run on one dimensional grid
    /// Run one dimensional block
//    kernels::vector_add<<<blocks_per_grid, threads_per_block>>>(device_a, device_b, device_r, size);
    kernels::huge_add<<<256, 1>>>(device_a, device_b, device_r, size);

    error = cudaMemcpy(r, device_r, size *  sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    assert(error);

    error = cudaFree(device_a);
    assert(error);
    error = cudaFree(device_b);
    assert(error);
    error = cudaFree(device_r);
    assert(error);
}