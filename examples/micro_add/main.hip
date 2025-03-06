#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

#include "kittens.cuh"
#include "example_utils.hpp"

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
