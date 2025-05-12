#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include "example_utils.hpp"

#include "kittens.cuh"
using namespace kittens;
#define NUM_THREADS (kittens::WARP_THREADS) // use 1 warp

#define BATCH_SIZE 16
#define IN_CHANNELS 32
#define HIDDEN_CHANNELS 16

struct micro_globals {
    using _gl_DATA  = gl<float, -1, -1, -1, -1, st_fl<BATCH_SIZE, IN_CHANNELS>>;
    using _gl_W1  = gl<float, -1, -1, -1, -1, st_fl<IN_CHANNELS, HIDDEN_CHANNELS>>;
    using _gl_O  = gl<float, -1, -1, -1, -1, st_fl<BATCH_SIZE, HIDDEN_CHANNELS>>;

    _gl_DATA data;
    _gl_W1 W1;
    _gl_O o;
};

__global__
void micro_tk(const micro_globals g) {

    // register memory 
    rt_bf<BATCH_SIZE, IN_CHANNELS> data;
    rt_bf<IN_CHANNELS, HIDDEN_CHANNELS, ducks::rt_layout::col> W1;
    rt_fl<BATCH_SIZE, HIDDEN_CHANNELS, ducks::rt_layout::col> o;

    load(data, g.data, {});
    load(W1, g.W1, {});
    zero(o);
    __syncthreads();

    mma_AB(o, data, W1, o);
    relu(o, o);

    store(g.o, o, {});
}

void dispatch_micro( float *d_x, float *d_W1, float *d_o ) {
    using _gl_DATA  = gl<float, -1, -1, -1, -1, st_fl<BATCH_SIZE, IN_CHANNELS>>;
    using _gl_W1  = gl<float, -1, -1, -1, -1, st_fl<IN_CHANNELS, HIDDEN_CHANNELS>>;
    using _gl_O  = gl<float, -1, -1, -1, -1, st_fl<BATCH_SIZE, HIDDEN_CHANNELS>>;
    using globals = micro_globals;
    _gl_DATA  x_arg{d_x, 1, 1, BATCH_SIZE, IN_CHANNELS};
    _gl_W1  W1_arg{d_W1, 1, 1, IN_CHANNELS, HIDDEN_CHANNELS};
    _gl_O  o_arg{d_o, 1, 1, BATCH_SIZE, HIDDEN_CHANNELS};
    globals g{x_arg, W1_arg, o_arg};
    unsigned long mem_size = 50480; 
    HIP_CHECK(hipFuncSetAttribute(reinterpret_cast<const void*>(micro_tk), 
                hipFuncAttributeMaxDynamicSharedMemorySize, mem_size));
    micro_tk<<<1,64,mem_size>>>(g);
    hipDeviceSynchronize();
}

#include "harness.impl"
