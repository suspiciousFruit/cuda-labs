#include <iostream>
#include <chrono>


void fill_data(unsigned int* A, size_t N) {
    for (size_t i = 0; i < N; ++i)
            A[i] = i % 256;
}

void cuda_assert(cudaError error) {
    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

bool check_sum(unsigned int* a, unsigned int* b, unsigned int* r,unsigned int N) {
    for (size_t i = 0; i < N; ++i) {
        if ((a[i] + b[i]) != r[i])
            return false;
    }
    return true;
}

class Timer {
    std::chrono::high_resolution_clock::time_point start_;
public:
    Timer() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    size_t elapsed() {
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_).count();
    }
};

const unsigned int BLOCK_SIZE = 128;
__global__ void kernel(unsigned int* a, unsigned int* b, unsigned int* r,unsigned int N) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    r[tid] = a[tid] + b[tid];
}

__global__ void wrong_kernel(unsigned int* a, unsigned int* b, unsigned int* r,unsigned int N) {
    unsigned int tid = threadIdx.x * blockDim.x + blockIdx.x;
    r[tid] = a[tid] + b[tid];
}



void run_vector_add(unsigned int N) {
    std::unique_ptr<unsigned int[]> a = std::make_unique<unsigned int[]>(N);
    std::unique_ptr<unsigned int[]> b = std::make_unique<unsigned int[]>(N);
    std::unique_ptr<unsigned int[]> r = std::make_unique<unsigned int[]>(N);

    fill_data(a.get(), N);
    fill_data(b.get(), N);

    unsigned int *d_a, *d_b, *d_r;

    cudaError error;
    error = cudaMalloc(&d_a, N * sizeof(unsigned int)); cuda_assert(error);
    error = cudaMalloc(&d_b, N * sizeof(unsigned int)); cuda_assert(error);
    error = cudaMalloc(&d_r, N * sizeof(unsigned int)); cuda_assert(error);

    error = cudaMemcpy(d_a, a.get(), N * sizeof(unsigned int), cudaMemcpyKind::cudaMemcpyHostToDevice);
    cuda_assert(error);

    error = cudaMemcpy(d_b, b.get(), N * sizeof(unsigned int), cudaMemcpyKind::cudaMemcpyHostToDevice);
    cuda_assert(error);

//    error = cudaMemcpy(d_r, r.get(), N * sizeof(unsigned int), cudaMemcpyKind::cudaMemcpyHostToDevice);
//    cuda_assert(error);

    {
        Timer t;
        kernel<<<BLOCK_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_r, N);
        std::cout << "kernel execution time: " << t.elapsed() << " microseconds" << std::endl;
    }
    {
        Timer t;
        wrong_kernel<<<BLOCK_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_r, N);
        std::cout << "wrong kernel execution time: " << t.elapsed() << " microseconds" << std::endl;
    }

    error = cudaMemcpy(r.get(), d_r, N * sizeof(unsigned int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cuda_assert(error);

    error = cudaFree(d_a); cuda_assert(error);
    error = cudaFree(d_b); cuda_assert(error);
    error = cudaFree(d_r); cuda_assert(error);

    if (check_sum(a.get(), b.get(),r.get(), N)) {
        std::cout << "TEST [PASSED]" << std::endl;
    }
    else {
        std::cout << "TEST [REJECTED]" << std::endl;
    }
}



int main() {
    unsigned int N = BLOCK_SIZE * BLOCK_SIZE;

    run_vector_add(N);

    return 0;
}
