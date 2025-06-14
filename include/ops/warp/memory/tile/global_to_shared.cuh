/**
 * @file
 * @brief Functions for transferring data directly between global and shared memory and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

__device__ static inline void global_to_shared_b128(uint32_t lds_offset_bytes, float4* src_ptr) {
    float4 temp;
    uintptr_t ptr = reinterpret_cast<uintptr_t>(src_ptr);  // float4 indexing
    uint32_t addr_lo = static_cast<uint32_t>(ptr & 0xFFFFFFFF);
    uint32_t addr_hi = static_cast<uint32_t>(ptr >> 32);

    asm volatile (
        "flat_load_dwordx4 %0, [%1, %2]\n"
        "s_waitcnt vmcnt(0)\n"
        "s_mov_b32 m0, 0\n"
        "ds_write_b128 %3, %0\n"
        : "=v"(temp)                     // %0: output float4 (VGPR)
        : "s"(addr_lo),                  // %1: 32-bit low address (SGPR)
          "s"(addr_hi),                  // %2: 32-bit high address (SGPR)
          "v"(lds_offset_bytes)         // %3: LDS offset
        : "memory", "m0"
    );
}

/**
 * @brief Loads data from global memory into a shared memory tile.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination shared memory tile.
 * @param[in] src The source global memory array.
 * @param[in] idx The coordinate of the tile in the global memory array.
 */
template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>, int N_THREADS=WARP_THREADS>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx) {
    using T = typename ST::dtype;
    const int row_stride = src.template stride<axis>();
    // we can handle this many rows each time we run a memcpy_async
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total_calls = (ST::cols * ST::rows + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy); // round up

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[unit_coord];

    uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);
    int laneid = threadIdx.x % N_THREADS;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {
        int load_idx = i * N_THREADS + laneid;
        int row = load_idx / memcpy_per_row;
        int col = (load_idx*elem_per_memcpy) % dst.cols;

        // if (row < dst.rows) 
        //     *(float4*)(&dst[{row, col}]) = *(float4*)&src_ptr[row * row_stride + col];
        
        if (row < dst.rows)
            global_to_shared_b128(dst.idx(dst_ptr, {row, col}), (float4*)&src_ptr[row * row_stride + col]); 
    }
}
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