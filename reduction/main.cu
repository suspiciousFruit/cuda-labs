#include <iostream>
#include <chrono>


void fill_range(unsigned int* A, size_t N) {
    for (size_t i = 0; i < N; ++i)
            A[i] = rand();
}


/// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
/// reduction kernel
/// A : array of natural numbers
/// R : array of reduction results, each block produce one number in results
/// Launch on grid sizes == N / block_size / 2 due to first reducing
/// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

/*

    |--------------- A size ----------------|

    |---- BLOCK_SIZE ---|

    -----------------------------------------
    |    |    |    |    |    |    |    |    |
    -----------------------------------------

    s == 2
    ---------------------
    |    |    |    |    |
    ---------------------

    s == 1
    -----------
    |    |    |
    -----------

 */

static constexpr const uint64_t BLOCK_SIZE = 16;

__global__ void min_reduce_kernel(const unsigned int* A, unsigned int* R) {
    __shared__ unsigned int data[BLOCK_SIZE];

    size_t tid = threadIdx.x;
    data[tid] = min(A[blockIdx.x * BLOCK_SIZE * 2 + tid], /// As launch on grid with arr_size / 2
                    A[blockIdx.x * BLOCK_SIZE * 2 + tid + BLOCK_SIZE]);

    for (size_t s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            data[tid] = min(data[tid], data[tid + s]);
        }

        __syncthreads();
    }

    if (tid == 0) R[tid] = data[0];
}

/// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
/// main section
/// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

void cuda_assert(cudaError error) {
    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

void run_min_reduce(unsigned int N, dim3 grid_dim, dim3 block_dim) {
    std::unique_ptr<unsigned int[]> a = std::make_unique<unsigned int[]>(N);
    std::unique_ptr<unsigned int[]> r = std::make_unique<unsigned int[]>(N);

    fill_range(a.get(), N);

    unsigned int *device_a, *device_r;
    cudaError error;

    error = cudaMalloc(&device_a, N * sizeof(unsigned int));
    cuda_assert(error);

    error = cudaMalloc(&device_r, N * sizeof(unsigned int));
    cuda_assert(error);

    error = cudaMemcpy(device_a, a.get(), N * sizeof(unsigned int), cudaMemcpyKind::cudaMemcpyHostToDevice);
    cuda_assert(error);

    /// <<<blocks count, threads count per block>>>>
    min_reduce_kernel<<<grid_dim, block_dim>>>(device_a, device_r);

    error = cudaMemcpy(r.get(), device_r, N * sizeof(unsigned int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cuda_assert(error);

    error = cudaFree(device_a); cuda_assert(error);
    error = cudaFree(device_r); cuda_assert(error);

    if (std::min(r.get(), r.get() + N) != std::min(a.get(), a.get() + N)) {
        std::cout << "TEST [PASSED]" << std::endl;
    }
    else {
        std::cout << "TEST [REJECTED]" << std::endl;
    }
}

int main() {
    unsigned int N = 1024;

    size_t grid_size = N / BLOCK_SIZE / 2;
    run_min_reduce(N, grid_size, BLOCK_SIZE);

    return 0;
}
