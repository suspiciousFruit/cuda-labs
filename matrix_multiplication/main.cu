#include <iostream>
#include <chrono>



void fill_matrix(unsigned int* A, size_t N) {
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            A[i * N + j] = rand();
//            A[i * N + j] = (i + j) % 1024;
//            A[i * N + j] = 1;

}


void print_matrix(unsigned int* A, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j)
            std::cout << A[i * N + j] << ' ';
        std::cout << '\n';
    }
}

bool check_multiplication(const unsigned int* a, const unsigned int* b, const unsigned int* r, unsigned int N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            unsigned int sum = 0;
            for (size_t k = 0; k < N; ++k)
                sum += a[i * N + k] * b[k * N + j];
            if (r[i * N + j] != sum) {
                printf("r[%llu][%llu] target: %u, value: %u\n", i, j, sum, r[N * i + j]);
                return false;
            }
        }
    }

    return true;
//    print_matrix(a.get(), N);
//    print_matrix(b.get(), N);
//    print_matrix(r.get(), N);
}


/// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
/// matrix multiplication without shared memory usage
/// can be launched on all threads_size == N * N
/// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
__global__ void kernel(const unsigned int* A, const unsigned int* B, unsigned int* R, size_t N) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int sum = 0;
    for (unsigned int k = 0; k < N; ++k)
        sum += A[row * N + k] * B[k * N + col];
    R[row * N + col] = sum;
}

/// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
/// matrix multiplication using shared memory
/// N is k * BLOCK_SIZE
/// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
/*
    Each block is one of result matrix sub matrix
    Go through rows

                                     ----------------
                                     |    |    |    |
                                     ----------------
                                     |    |    |    |
                                     ----------------
                                     |    |    |    |
                                     ----------------
                                     |    |    |    |
                                     ----------------

            ---------------------    ----------------
            |    |    |    |    |    |    |    |    |
            ---------------------    ----------------
            |    |    |    |    |    |    |    |    |
            ---------------------    ----------------

    A with fixed row
    B with fixed column
 */
static constexpr const uint64_t BLOCK_SIZE = 16;

__global__ void multiply_shared_kernel(const unsigned int* A, const unsigned int* B, unsigned int* R, size_t N) {
    __shared__ unsigned int a_sub_matrix[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int b_sub_matrix[BLOCK_SIZE][BLOCK_SIZE];

    const size_t sub_matrices_count = N / BLOCK_SIZE;

    const size_t row = threadIdx.y;
    const size_t col = threadIdx.x;

    /// Coordinates in result matrix
    const size_t a_row_idx = blockIdx.y * BLOCK_SIZE + row; /// Fixed A row (row index in A)
    const size_t b_col_idx = blockIdx.x * BLOCK_SIZE + col; /// Fixed B column (col index in B)

    unsigned int r = 0;
    for (size_t n = 0; n < sub_matrices_count; ++n) {

        /// Load sub matrix of A, B
        a_sub_matrix[row][col] = A[a_row_idx              * N + n * BLOCK_SIZE + col];
        b_sub_matrix[row][col] = B[(BLOCK_SIZE * n + row) * N + b_col_idx];

        __syncthreads();

        for (size_t k = 0; k < BLOCK_SIZE; ++k)
            r += a_sub_matrix[row][k] * b_sub_matrix[k][col];

        __syncthreads();
    }


    R[a_row_idx * N + b_col_idx] = r;
}


/// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
/// main section
/// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

void cuda_assert(cudaError error) {
    if (error != cudaSuccess) {
        std::cout << "CUDA ERROR: " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

void run_matrix_multiply(unsigned int N, dim3 grid_dim, dim3 block_dim) {
    std::unique_ptr<unsigned int[]> a = std::make_unique<unsigned int[]>(N * N);
    std::unique_ptr<unsigned int[]> b = std::make_unique<unsigned int[]>(N * N);
    std::unique_ptr<unsigned int[]> r = std::make_unique<unsigned int[]>(N * N);

    fill_matrix(a.get(), N);
    fill_matrix(b.get(), N);

    unsigned int *device_a, *device_b, *device_r;
    cudaError error;

    error = cudaMalloc(&device_a, N * N * sizeof(unsigned int)); cuda_assert(error);
    error = cudaMalloc(&device_b, N * N * sizeof(unsigned int)); cuda_assert(error);
    error = cudaMalloc(&device_r, N * N * sizeof(unsigned int)); cuda_assert(error);

    error = cudaMemcpy(device_a, a.get(), N * N * sizeof(unsigned int), cudaMemcpyKind::cudaMemcpyHostToDevice);
    cuda_assert(error);

    error = cudaMemcpy(device_b, b.get(), N * N * sizeof(unsigned int), cudaMemcpyKind::cudaMemcpyHostToDevice);
    cuda_assert(error);

    /// Run on two dimensional grid
    /// Run two dimensional blocks
    /// <<<blocks count, threads count per block>>>>
//    kernel<<<grid_dim, block_dim>>>(device_a, device_b, device_r, N);
    multiply_shared_kernel<<<grid_dim, block_dim>>>(device_a, device_b, device_r, N);

    error = cudaMemcpy(r.get(), device_r, N * N *  sizeof(unsigned int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cuda_assert(error);

    error = cudaFree(device_a); cuda_assert(error);
    error = cudaFree(device_b); cuda_assert(error);
    error = cudaFree(device_r); cuda_assert(error);


    if (check_multiplication(a.get(), b.get(), r.get(), N)) {
        std::cout << "TEST [PASSED]" << std::endl;
    }
    else {
        std::cout << "TEST [REJECTED]" << std::endl;
    }
}

int main() {
//    const unsigned int N = 1024;
    const unsigned int N = 32;
    const unsigned int block_size = 16;

    run_matrix_multiply(N, dim3{(N/block_size),(N/block_size)}, dim3{block_size, block_size});
}
