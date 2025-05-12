#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include "example_utils.hpp"

#include "kittens.cuh"
using namespace kittens;
#define NUM_THREADS (kittens::WARP_THREADS) // use 1 warp

#define M 16
#define N 16
#define K 16

struct micro_globals {
    using _gl_A  = gl<float, -1, -1, -1, -1, st_fl<M, K>>;
    using _gl_B  = gl<float, -1, -1, -1, -1, st_fl<K, N>>;
    using _gl_C  = gl<float, -1, -1, -1, -1, st_fl<M, N>>;
    _gl_A a;
    _gl_B b;
    _gl_C c;
};

__global__
void micro_tk(const micro_globals g) {

    // register memory 
    rt_bf<M, K> a;
    rt_bf<K, N, ducks::rt_layout::col> b;
    rt_fl<M, N, ducks::rt_layout::col> c;

    load(a, g.a, {});
    load(b, g.b, {});
    zero(c);
    __syncthreads();
    mma_AB(c, a, b, c);
    store(g.c, c, {});
}

void dispatch_micro( float *d_x, float *d_y, float *d_o ) {
    using _gl_A  = gl<float, -1, -1, -1, -1, st_fl<M, K>>;
    using _gl_B  = gl<float, -1, -1, -1, -1, st_fl<K, N>>;
    using _gl_C  = gl<float, -1, -1, -1, -1, st_fl<M, N>>;
    using globals = micro_globals;
    _gl_A  x_arg{d_x, 1, 1, M, K};
    _gl_B  y_arg{d_y, 1, 1, K, N};
    _gl_C  o_arg{d_o, 1, 1, M, N};
    globals g{x_arg, y_arg, o_arg};
    unsigned long mem_size = 50480; 
    HIP_CHECK(hipFuncSetAttribute(reinterpret_cast<const void*>(micro_tk), 
                hipFuncAttributeMaxDynamicSharedMemorySize, mem_size));
    micro_tk<<<1,64,mem_size>>>(g);
    hipDeviceSynchronize();
}

#include "harness.impl"
