#include <iostream>
#include <chrono>


void fill_data(unsigned int* A, size_t N) {
    for (size_t i = 0; i < N; ++i)
//            A[i] = rand() % 256;
            A[i] = i % 256;
}


/// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
/// histogram kernel
/// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

__device__ inline void add_byte(volatile unsigned* warp_hist, unsigned data, unsigned tag) {
    unsigned count;
    do {
        count = warp_hist[data] & 0x07FFFFFFU;
        count = tag | (count + 1);
        warp_hist[data] = count;
    } while (warp_hist[data] != count);
}


static constexpr const uint64_t BLOCK_SIZE = 32;
static constexpr const uint64_t BINS_COUNT = 256;
static constexpr const uint64_t WARP_SIZE = 32;
static constexpr const uint64_t LOG2_WARP_SIZE = 5;
static constexpr const uint64_t BLOCK_WARP_COUNT = 6; /// Warp count in each block
/// For each warp make shared buffer for local histogram
///
//__global__ void histogram_kernel(unsigned int* result, const unsigned int* data, unsigned int N) {
//    /// Each block has BLOCK_WARP_COUNT * WARP_SIZE threads
//    __shared__ unsigned int histogram[BLOCK_WARP_COUNT * BINS_COUNT]; /// Block local histogram
//    /// Initializing histogram
//    for (size_t i = 0; i < BINS_COUNT / WARP_SIZE; ++i) /// Each thread should set BINS_COUNT / WARP_SIZE
//        histogram[i * BLOCK_WARP_COUNT * WARP_SIZE + threadIdx.x] = 0;
//
//    __syncthreads();
//
//    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//    size_t threads_count = blockDim.x * gridDim.x;
//    size_t warp_base = (threadIdx.x >> LOG2_WARP_SIZE) * BINS_COUNT;
//    unsigned int tag = threadIdx.x << (32 - LOG2_WARP_SIZE);
//    for (size_t i = tid; i < N; i += threads_count) {
//        unsigned int data4 = data [i];
//        add_byte(histogram + warp_base, (data4 >> 0) & 0xFFU, tag);
//        add_byte(histogram + warp_base, (data4 >> 8) & 0xFFU, tag);
//        add_byte(histogram + warp_base, (data4 >> 16) & 0xFFU, tag);
//        add_byte(histogram + warp_base, (data4 >> 24) & 0xFFU, tag);
//    }
//
//
//
//    __syncthreads();
//
//    for (size_t bin = threadIdx.x; bin < BINS_COUNT; bin += (BLOCK_WARP_COUNT * WARP_SIZE)) {
//        unsigned sum = 0;
//        for (int i = 0; i < BLOCK_WARP_COUNT; i++)
//            sum += histogram[bin + i * BINS_COUNT] & 0x07FFFFFFU;
//        result[blockIdx.x * BINS_COUNT + bin] = sum;
//    }
//}


/// Launch: for each data element has a thread, block_size is BLOCK_WARP_COUNT * BINS_COUNT
/// Result: array of histograms [ BINS_COUNT ][ BINS_COUNT ]...[ BINS_COUNT ] x blocks_count
/// My version
//__global__ void histogram_kernel(unsigned int* result, const unsigned int* data, unsigned int N) {
//    /// Each block has BLOCK_WARP_COUNT * WARP_SIZE threads
//    __shared__ unsigned int histogram[BLOCK_WARP_COUNT * BINS_COUNT]; /// Block local histogram
//    /// Initializing histogram
//    for (size_t i = 0; i < BINS_COUNT / WARP_SIZE; ++i) /// Each thread should set BINS_COUNT / WARP_SIZE
//        histogram[i * BLOCK_WARP_COUNT * WARP_SIZE + threadIdx.x] = 0;
//
//    __syncthreads();
//
//    /// Each warp has local bins
//    /// warp_base: pointer on local bin
//    size_t warp_idx = threadIdx.x / WARP_SIZE;
//    unsigned int* local_bins = histogram + warp_idx * BINS_COUNT;
//
//
//    /// data_base: pointer to warp handle data
//    const unsigned int* block_handle_data = data + blockIdx.x * blockDim.x;
//    const unsigned int* warp_handle_data = block_handle_data + warp_idx * WARP_SIZE;
//
//    unsigned int bin_idx = warp_handle_data[threadIdx.x];
//
//    __syncthreads();
//
////    atomicAdd(local_bins + bin_idx, 1);
//
//    __syncthreads();
//
//    for (size_t bin = threadIdx.x; bin < BINS_COUNT; bin += (BLOCK_WARP_COUNT * WARP_SIZE)) {
//        unsigned sum = 0;
//        for (int i = 0; i < BLOCK_WARP_COUNT; i++)
//            sum += histogram[bin + i * BINS_COUNT];
//        result[blockIdx.x * BINS_COUNT + bin] = sum;
//    }
//}

/// BINS_COUNT threads in each block
__global__ void histogram_kernel_debug(unsigned int* result, const unsigned int* data, unsigned int N) {
    /// Initialize

    const unsigned int* block_data = data + blockIdx.x * blockDim.x;
    unsigned int* block_histogram = result + blockIdx.x * BINS_COUNT;

    /// threads in block is 192
    /// but block_size is 256
    for (unsigned int i = threadIdx.x; i < BINS_COUNT; ++i)
        block_histogram[threadIdx.x] = 0;

    unsigned int value = block_data[threadIdx.x];
    atomicAdd(block_histogram + value, 1);
//    block_histogram[value]++;
}

/// Block for each bin
__global__ void merge_histogram_kernel(unsigned int* out_histogram,
                                        const unsigned int* partial_histograms,
                                        unsigned int histogram_count) {
//    unsigned int sum = 0;
//    for (size_t i = threadIdx.x; i < histogram_count; i += 256) /// ! ++i
//        sum += partial_histograms[i * BINS_COUNT + blockIdx.x];
//
//    __shared__ unsigned int data[BINS_COUNT];
//    data[threadIdx.x] = sum;
//
//    for (unsigned int stride = BINS_COUNT / 2; stride > 0; stride >>= 1) {
//        __syncthreads ();
//        if (threadIdx.x < stride) data[threadIdx.x] +=  data[threadIdx.x + stride];
//    }
//
//    if (threadIdx.x == 0)
//        out_histogram[blockIdx.x] = data[0]; /// Each block sum single bin with idx blockIdx.x

    unsigned int sum;
    for (size_t i = threadIdx.x; i < histogram_count; ++i)
        sum += partial_histograms[i * BINS_COUNT + blockIdx.x];
    out_histogram[threadIdx.x] = sum;
}

__global__ void merge_histogram_kernel_debug(unsigned int* out_histogram,
                                       const unsigned int* partial_histograms,
                                       unsigned int histogram_count) {
    unsigned int sum;
    for (size_t i = threadIdx.x; i < histogram_count; ++i) {
        __syncthreads();
//        unsigned int value = partial_histograms[i * BINS_COUNT + threadIdx.x];
        unsigned int value = *(partial_histograms + i * BINS_COUNT + threadIdx.x);
        sum += value;
    }
    __syncthreads();
    out_histogram[threadIdx.x] = sum;
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


//__host__ void evaluate_histogram(unsigned int* d_histogram, unsigned int* d_data, unsigned int N) {
//    unsigned int blocks_count = N / (BLOCK_WARP_COUNT * WARP_SIZE);
////    int partials_count = 240;
////    int partials_count = blocks_count;
//    unsigned* d_partial_histograms = nullptr;
//
//    cudaError error;
//    error = cudaMalloc((void**)&d_partial_histograms, blocks_count * BINS_COUNT * sizeof(unsigned int));
//    cuda_assert(error);
//
//    histogram_kernel<<<blocks_count, BLOCK_WARP_COUNT * WARP_SIZE>>>
//        (d_partial_histograms, d_data, N);
//
//    merge_histogram_kernel<<<dim3(BINS_COUNT), dim3(256)>>>
//        (d_histogram, d_partial_histograms, blocks_count);
//
//    cudaFree(d_partial_histograms);
//}

__host__ std::unique_ptr<unsigned int[]> evaluate_histogram_debug(unsigned int* d_histogram, unsigned int* d_data, unsigned int N) {
    unsigned int blocks_count = N / (BLOCK_WARP_COUNT * WARP_SIZE);
    unsigned int* d_partial_histograms = nullptr;

    cudaError error;
    error = cudaMalloc((void**)&d_partial_histograms, blocks_count * BINS_COUNT * sizeof(unsigned int));
    cuda_assert(error);

    histogram_kernel_debug<<<blocks_count, BLOCK_WARP_COUNT * WARP_SIZE>>>
    (d_partial_histograms, d_data, N);

    std::unique_ptr<unsigned int[]> res = std::make_unique<unsigned int[]>(blocks_count * BINS_COUNT * sizeof(unsigned int));

    cudaMemcpy(res.get(), d_partial_histograms, blocks_count * BINS_COUNT * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    merge_histogram_kernel_debug<<<1, BINS_COUNT>>>
    (d_histogram, d_partial_histograms, blocks_count);

    cudaFree(d_partial_histograms);

    return res;
}

/// result -> BINS_COUNT
__global__ void custom_histogram_kernel(unsigned int* result, const unsigned int* data, unsigned int N) {
//    for (unsigned int i = 0; i < BINS_COUNT; ++i)
//        result[i] = 0;

    __syncthreads();

    unsigned int value = data[blockIdx.x * blockDim.x + threadIdx.x];

    __syncthreads();

    atomicAdd(result + value, 1);
}

bool check_histogram(const unsigned int* data, unsigned int N, const unsigned int* histogram) {
    std::unique_ptr<unsigned int[]> hist = std::make_unique<unsigned int[]>(BINS_COUNT);

    std::memset(hist.get(), 0, BINS_COUNT * sizeof(unsigned int));
    for (size_t i = 0; i < N; ++i)
        ++hist[data[i]];

    for (size_t i = 0; i < BINS_COUNT; ++i) {
        if (histogram[i] != hist[i]) {
            printf("[%d] target: %d, value: %d\n", i, hist[i], histogram[i]);
            return false;
        }
    }

    return true;
}

void run_histogram(unsigned int N) {
    std::unique_ptr<unsigned int[]> data = std::make_unique<unsigned int[]>(N);
    std::unique_ptr<unsigned int[]> histogram = std::make_unique<unsigned int[]>(BINS_COUNT);

    std::memset(histogram.get(), 0, BINS_COUNT * sizeof(unsigned int));

    fill_data(data.get(), N);

    unsigned int *d_data, *d_histogram;

    cudaError error;
    error = cudaMalloc(&d_data, N * sizeof(unsigned int)); cuda_assert(error);
    error = cudaMalloc(&d_histogram, BINS_COUNT * sizeof(unsigned int)); cuda_assert(error);

    error = cudaMemcpy(d_data, data.get(), N * sizeof(unsigned int), cudaMemcpyKind::cudaMemcpyHostToDevice);
    cuda_assert(error);

    error = cudaMemcpy(d_histogram, histogram.get(), BINS_COUNT * sizeof(unsigned int), cudaMemcpyKind::cudaMemcpyHostToDevice);
    cuda_assert(error);



    custom_histogram_kernel<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(d_histogram, d_data, N);

    error = cudaMemcpy(histogram.get(), d_histogram, BINS_COUNT * sizeof(unsigned int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cuda_assert(error);

    error = cudaFree(d_data); cuda_assert(error);
    error = cudaFree(d_histogram); cuda_assert(error);

    if (check_histogram(data.get(), N, histogram.get())) {
        std::cout << "TEST [PASSED]" << std::endl;
    }
    else {
        std::cout << "TEST [REJECTED]" << std::endl;
    }

//    for (size_t i = 0; i < BINS_COUNT; ++i) {
//        unsigned int sum = 0;
//        for (size_t j = 0; j < N / (BLOCK_WARP_COUNT * WARP_SIZE); ++j)
//            sum += res[j * BINS_COUNT];
//        if (histogram[i] != sum) {
//            printf("[%d] TEST [REJECTED] target: %d, value: %d\n", i, sum, histogram[i]);
//            return;
//        }
//    }
}

int main() {
    unsigned int N = 1024 * 1024;

    run_histogram(N);

//    unsigned int* d_var;
//    unsigned int var;
//    cudaMalloc(&d_var, sizeof(unsigned int));
//
//    ker<<<100, 100>>>(d_var);
//
//    cudaMemcpy(&var, d_var, sizeof(unsigned int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
//
//    std::cout << "var: " << var << std::endl;
//
//    cudaFree(d_var);
    return 0;
}
