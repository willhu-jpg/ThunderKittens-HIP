#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include "example_utils.hpp"

#include "kittens.cuh"
using namespace kittens;
#define NUM_THREADS (kittens::WARP_THREADS) // use 1 warp

#define _row 16
#define _col 32

struct micro_globals {
    using _gl  = gl<float, -1, -1, -1, -1, st_fl<_row, _col>>;
    _gl x, o;
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {

    // shared memory
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_fl<_row, _col> (&x_s) = al.allocate<st_fl<_row, _col>>();
    st_fl<_row, _col> (&o_s) = al.allocate<st_fl<_row, _col>>();

    // register memory 
    rt_fl<_row, _col> x_reg_fl;

    // load from HBM to shared
    load(x_s, g.x, {0, 0, 0, 0});
    __syncthreads();

    // load from shared to register
    load(x_reg_fl, x_s);
    __syncthreads();

    // x (dst) = x (src b) + x (src a)
    add(x_reg_fl, x_reg_fl, x_reg_fl);
    __syncthreads();

    // store from register to shared
    store(o_s, x_reg_fl);
    __syncthreads();

    // store from shared to HBM
    store(g.o, o_s, {0, 0, 0, 0});
    __syncthreads();
}


// HIP Kernel: C = A + B
__global__ void vector_add(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1000000;  // Number of elements
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    std::vector<float> h_A(N, 1.0f);  // Initialize vectors on host
    std::vector<float> h_B(N, 2.0f);
    std::vector<float> h_C(N, 0.0f);

    float *d_A, *d_B, *d_C;
    size_t bytes = N * sizeof(float);

    // Allocate memory on GPU
    HIP_CHECK(hipMalloc(&d_A, bytes));
    HIP_CHECK(hipMalloc(&d_B, bytes));
    HIP_CHECK(hipMalloc(&d_C, bytes));

    // Copy data from host to device
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), bytes, hipMemcpyHostToDevice));

    // Launch Kernel
    hipLaunchKernelGGL(vector_add, dim3(gridSize), dim3(blockSize), 0, 0, d_A, d_B, d_C, N);

    // Copy result back to host
    HIP_CHECK(hipMemcpy(h_C.data(), d_C, bytes, hipMemcpyDeviceToHost));

    // Verify results
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != 3.0f) {  // Expecting A[i] + B[i] = 1.0 + 2.0 = 3.0
            success = false;
            break;
        }
    }

    std::cout << (success ? "✅ Vector Addition Successful!" : "❌ Error in computation!") << std::endl;

    // Free device memory
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    return 0;
}
