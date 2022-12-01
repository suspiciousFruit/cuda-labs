#include <iostream>


void fill_matrix(unsigned int* A, size_t N) {
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            A[i * N + j] = i + j;
}


void print_matrix(unsigned int* A, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j)
            std::cout << A[i * N + j] << ' ';
        std::cout << '\n';
    }
}


void multiply_matrix(const unsigned int* A, const unsigned int* B, unsigned int* R, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0;
            for (size_t k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            R[N * i + j] = sum;
        }
    }
}

void test_cpu() {
    const size_t N = 4;
    std::unique_ptr<unsigned int[]> a = std::make_unique<unsigned int[]>(N * N);
    std::unique_ptr<unsigned int[]> b = std::make_unique<unsigned int[]>(N * N);
    std::unique_ptr<unsigned int[]> r = std::make_unique<unsigned int[]>(N * N);

    fill_matrix(a.get(), N);
    fill_matrix(b.get(), N);


    multiply_matrix(a.get(), b.get(), r.get(), N);

//    print_matrix(a.get(), N);
//    print_matrix(b.get(), N);
    print_matrix(r.get(), N);

}

/// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
/// GPU section
/// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
/// size, count of parts
__device__ size_t get_range_size(unsigned int count, unsigned int parts) {
    return (count + parts - 1) / parts;
}


__global__ void transpose_kernel(const unsigned int* A, const unsigned int* B, unsigned int* R, size_t N) {
    /// Each thread get a part of result matrix (square) and calculate values

    /// Part of full blocks range that handled by current thread
    unsigned int block_range_x = get_range_size(N, gridDim.x);
    /// Part of full threads range that handled by current thread
    unsigned int thread_range_x = get_range_size(block_range_x, blockDim.x);

    unsigned int block_range_y = get_range_size(N, gridDim.y);
    unsigned int thread_range_y = get_range_size(block_range_y, blockDim.y);

    auto start_x = blockIdx.x * block_range_x + threadIdx.x * thread_range_x; /// Start index
    auto stop_x = start_x + thread_range_x > N ? N : start_x + thread_range_x; /// Stop index

    auto start_y = blockIdx.y * block_range_y + threadIdx.y * thread_range_y;
    auto stop_y = start_y + thread_range_y > N ? N : start_y + thread_range_y;

    /// A[i][j] -> A[i * size + j]
    for (unsigned int i = start_x; i < stop_x; ++i) {
        for (unsigned int j = start_y; j < stop_y; ++j) {
            unsigned int sum = 0;
            for (unsigned int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j]; /// TODO can be cache optimized with transpose(B)
            R[i * N + j] = sum;
        }
    }
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

int main() {
    test_cpu();
    const size_t N = 4;
    std::unique_ptr<unsigned int[]> a = std::make_unique<unsigned int[]>(N * N);
    std::unique_ptr<unsigned int[]> b = std::make_unique<unsigned int[]>(N * N);
    std::unique_ptr<unsigned int[]> r = std::make_unique<unsigned int[]>(N * N);

    fill_matrix(a.get(), N);
    fill_matrix(b.get(), N);

    unsigned int *device_a, *device_b, *device_r;
    cudaError error;

    error = cudaMalloc(&device_a, N * N * sizeof(unsigned int));
    cuda_assert(error);

    error = cudaMalloc(&device_b, N * N * sizeof(unsigned int));
    cuda_assert(error);

    error = cudaMalloc(&device_r, N * N * sizeof(unsigned int));
    cuda_assert(error);

    error = cudaMemcpy(device_a, a.get(), N * N * sizeof(unsigned int), cudaMemcpyKind::cudaMemcpyHostToDevice);
    cuda_assert(error);

    error = cudaMemcpy(device_b, b.get(), N * N * sizeof(unsigned int), cudaMemcpyKind::cudaMemcpyHostToDevice);
    cuda_assert(error);

    /// Run on two dimensional grid
    /// Run two dimensional blocks
    /// <<<blocks count, threads count per block>>>>
    kernel<<<dim3(1, 3), dim3(2, 3)>>>(device_a, device_b, device_r, N);

    error = cudaMemcpy(r.get(), device_r, N * N *  sizeof(unsigned int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cuda_assert(error);

    error = cudaFree(device_a);
    cuda_assert(error);
    error = cudaFree(device_b);
    cuda_assert(error);
    error = cudaFree(device_r);
    cuda_assert(error);

    print_matrix(r.get(), N);
    return 0;
}
