#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

#define NUM_THREADS (kittens::WARP_THREADS) // use 1 warp

#define _row 16
#define _col 32

// define global layout
using _gl  = gl<float, -1, -1, -1, -1, st_fl<_row, _col>>;

struct micro_globals {
    _gl x, o;
    // grid - number of thread blocks we are launching
    dim3 grid()  { return dim3(x.batch, x.depth, x.rows); } 
    // block - number of threads in a thread block
    dim3 block() { return dim3(x.cols); } 
    // Safe shared memory size for H100
    size_t dynamic_shared_memory() { return 224000; } 
};

// define kernel
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

// Launch Kernel
void dispatch_micro(micro_globals g) {
    // need to recompute for MI300
    unsigned long mem_size = 50480; 
    hipFuncSetAttribute(
        (void*)micro_tk,
        hipFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}


PYBIND11_MODULE(simple_tk, m) {
    m.doc() = "simple_tk python module";
    // For wrapping kernels directly.
    // Old: BIND_KERNEL(m, "micro_tk", micro_tk, micro_globals, x, o); 
    py::bind_kernel<micro_tk>(m, "micro_tk", &micro_globals::x, &micro_globals::o); 

    // For host functions that wrap the kernel, this will be called from Python
    // Old: BIND_FUNCTION(m, "dispatch_micro", dispatch_micro, micro_globals, x, o); 
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::x, &micro_globals::o); 
}