/**
 * @file
 * @brief Functions for transferring data directly between global and shared memory and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

__device__ inline float4 load_global_vec(const float4* gptr) {
    float4 v;
    // Use global_load_dwordx4 which is more cache-friendly than flat_load
    asm volatile(
        "global_load_dwordx4 %0, %1, off\n"
        : "=v"(v) 
        : "v"(gptr)
        : "memory"
    );
    return v;   
}

// Store function using ds_write_b128 - proper float handling
__device__ inline void store_shared_vec(uint32_t lds_off, float4 val) {
    float *f = reinterpret_cast<float*>(&val);
    asm volatile(
        "ds_write_b128 %4, [%0, %1, %2, %3]\n"
        :
        : "v"(f[0]), "v"(f[1]), "v"(f[2]), "v"(f[3]), "v"(lds_off)
        : "memory"
    );
}

// template< int  axis, bool assume_aligned,
//           ducks::st::all ST, ducks::gl::all GL,
//           ducks::coord::tile COORD = coord<ST>,
//           int  N_THREADS = WARP_THREADS >
// __device__ inline void load(ST& dst, const GL& src, const COORD& idx)
// {
//     using T = typename ST::dtype;
//     const int row_stride = src.template stride<axis>();
//     // we can handle this many rows each time we run a memcpy_async
//     constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
//     constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
//     constexpr int total_calls = (ST::cols * ST::rows + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy); // round up

//     coord<> unit_coord = idx.template unit_coord<axis, 3>();
//     typename GL::dtype *src_ptr = (typename GL::dtype*)&src[unit_coord];

//     uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);
//     int laneid = threadIdx.x % N_THREADS;

//     // printf("elem_per_memcpy: %d, memcpy_per_row: %d, total_calls: %d\n", elem_per_memcpy, memcpy_per_row, total_calls);

//     #pragma unroll
//     for(int i = 0; i < total_calls; i++) {

//         int load_idx = i * N_THREADS + laneid;
        
//         int row = load_idx / memcpy_per_row;

//         int col = (load_idx*elem_per_memcpy) % dst.cols;

//         if (row < dst.rows) {
//             float4 tmp = load_global_vec((float4*) (src_ptr + (row * row_stride + col)));
//             asm volatile("s_waitcnt vmcnt(0)"); 
//             asm volatile("s_mov_b32 m0, 0");
//             store_shared_vec(dst.idx(dst_ptr, {row, col}), tmp);
//             asm volatile("s_waitcnt lgkmcnt(0)"); 
//         }
//     }
// }

template< int  axis, bool assume_aligned,
          ducks::st::all ST, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<ST>,
          int  N_THREADS = WARP_THREADS >
__device__ inline void load(ST& dst, const GL& src, const COORD& idx)
{
    using T = typename ST::dtype;
    const int row_stride = src.template stride<axis>();
    // we can handle this many rows each time we run a memcpy_async
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total_calls = (ST::cols * ST::rows + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy); // round up

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[unit_coord];

    uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);
    const int laneid = threadIdx.x % N_THREADS;

    // TODO: This is a hack to avoid the issue of too many VGPRs.
    // We should find a better way to do this.
    const int small_calls = 8;
    const int big_calls = (total_calls + small_calls - 1) / small_calls;
    float4    buf[small_calls];
    uint32_t  ofs[small_calls];

    // TODO: This is a hack to avoid the issue of too many VGPRs.
    // We should find a better way to do this.
    for (int i = 0; i < big_calls; i++) {
        #pragma unroll
        for(int j = i * small_calls; j < (i + 1) * small_calls; j++) {

            int load_idx = j * N_THREADS + laneid;

            int row = load_idx / memcpy_per_row;

            int col = (load_idx % memcpy_per_row) * elem_per_memcpy;

            if (row < dst.rows) {
                buf[j - i * small_calls] = load_global_vec((float4*) (src_ptr + (row * row_stride + col)));
                ofs[j - i * small_calls] = dst.idx(dst_ptr, {row, col});
            }
        }

        asm volatile("s_waitcnt vmcnt(0)"); 

        #pragma unroll
        for(int j = i * small_calls; j < (i + 1) * small_calls; j++) {
            int load_idx = j * N_THREADS + laneid;
            int row = load_idx / memcpy_per_row;
            int col = (load_idx % memcpy_per_row) * elem_per_memcpy;

            if (row < dst.rows) {
                store_shared_vec(ofs[j - i * small_calls], buf[j - i * small_calls]);
            }
        }
        asm volatile("s_waitcnt lgkmcnt(0)");
    } 
}

// template< int  axis, bool assume_aligned,
//           ducks::st::all ST, ducks::gl::all GL,
//           ducks::coord::tile COORD = coord<ST>,
//           int  N_THREADS = WARP_THREADS >
// __device__ inline void load(ST& dst, const GL& src, const COORD& idx)
// {
//     using DType = typename ST::dtype;
//     static constexpr int ELEM_PER_VEC = sizeof(float4) / sizeof(DType);
//     static constexpr int VEC_PER_ROW  = ST::cols / ELEM_PER_VEC;
//     static constexpr int TOTAL_VEC    = ST::rows * VEC_PER_ROW;
//     static constexpr int VEC_PER_THD  = (TOTAL_VEC + N_THREADS - 1) / N_THREADS;

//     // Small per-thread scratch
//     float4   buf[VEC_PER_THD];
//     uint32_t ofs[VEC_PER_THD];

//     const int lane   = threadIdx.x % N_THREADS;
//     const int stride = src.template stride<axis>();

//     coord<> base     = idx.template unit_coord<axis,3>(); // take tile-level coord passed into load and create 3d coord. Axis is the dim we treat as contiguous
//     auto*   gptr0    = &src[base];                        // global memory pointer to the start of the tile in DRAM
//     uint32_t lds0    = reinterpret_cast<uintptr_t>(&dst.data[0]); // where we will write the tile in LDS

//     // Queue global loads
//     #pragma unroll
//     for (int i = 0; i < VEC_PER_THD; ++i) {
//         int vec = i * N_THREADS + lane;          // flat vector index
//         if (vec >= TOTAL_VEC) break;
//         int r   = vec / VEC_PER_ROW;
//         int c   = (vec % VEC_PER_ROW) * ELEM_PER_VEC;

//         buf[i]  = load_global_vec(reinterpret_cast<const float4*>(gptr0 + r*stride + c));
//         ofs[i]  = dst.idx(lds0, {r, c}); 
//     }

//     // One waitcnt + m0 per warp (SA: maybe need sync here?)
//     __syncthreads(); // ensure all threads have loaded their data
//     asm volatile("s_waitcnt vmcnt(0)"); 
//     asm volatile("s_mov_b32 m0, 0");

//     // commit VGPRs to LDS
//     #pragma unroll
//     for (int i = 0; i < VEC_PER_THD; ++i) {
//         int vec = i * N_THREADS + lane;
//         if (vec >= TOTAL_VEC) break;
//         store_shared_vec(ofs[i], buf[i]);
//     }
// }
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx) {
    load<2, false, ST, GL, COORD, WARP_THREADS>(dst, src, idx);
}

/**
 * @brief Stores data from a shared memory tile into global memory.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory tile.
 * @param row_stride[in] The stride between rows in the destination array.
 */
template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>, int N_THREADS=WARP_THREADS>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    using T = typename ST::dtype;
    const int row_stride = dst.template stride<axis>();
    // we can handle this many rows each time we run a memcpy_async
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total_calls = (ST::cols * ST::rows + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy); // round up

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    typename GL::dtype *dst_ptr = (typename GL::dtype*)&dst[unit_coord];

    uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);
    int laneid = threadIdx.x % N_THREADS;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int load_idx = i * N_THREADS + laneid;
        
        int row = load_idx / memcpy_per_row;

        int col = (load_idx*elem_per_memcpy) % src.cols;

        if (row < src.rows) 
            *(float4*) &dst_ptr[row * row_stride + col] = *(float4*)(&src[{row, col}]);
    }
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    store<2, false, ST, GL, COORD, WARP_THREADS>(dst, src, idx);
}

}