/**
 * @file
 * @brief Functions for transferring data directly between global and shared memory and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {
// Store function using ds_write_b128 - proper float handling
// __device__ inline void store_shared_vec(uint32_t lds_off, float4 val) {
//     float *f = reinterpret_cast<float*>(&val);
//     asm volatile(
//         "ds_write_b128 %4, [%0, %1, %2, %3]\n"
//         :
//         : "v"(f[0]), "v"(f[1]), "v"(f[2]), "v"(f[3]), "v"(lds_off)
//         : "memory"
//     );
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
    constexpr int elem_per_half_memcpy = sizeof(float2)/sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total_calls = (ST::cols * ST::rows + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy); // round up

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[unit_coord];

    uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);
    const int laneid = threadIdx.x % N_THREADS;

    // TODO: This is a hack to avoid the issue of too many VGPRs.
    // We should find a better way to do this.
    const int small_calls = 16;
    const int big_calls = (total_calls + small_calls - 1) / small_calls;
    float4    buf[small_calls];

    for (int i = 0; i < big_calls; i++) {
        const int offset = i * small_calls;
        #pragma unroll
        for(int j = 0; j < small_calls; j++) {
            int load_idx = (offset + j) * N_THREADS + laneid;
            int row = load_idx / memcpy_per_row;
            int col = (load_idx % memcpy_per_row) * elem_per_memcpy;

            if (row < dst.rows) {
                buf[j] = load_global_vec4_async((float4*) (src_ptr + (row * row_stride + col)));
            }
        }

        asm volatile("s_waitcnt vmcnt(0)"); 

        #pragma unroll
        for(int j = 0; j < small_calls; j++) {
            int load_idx = (offset + j) * N_THREADS + laneid;
            int row = load_idx / memcpy_per_row;
            int col = (load_idx % memcpy_per_row) * elem_per_memcpy;

            if (row < dst.rows) {
                store_shared_vec(dst.idx(dst_ptr, {row, col}), {buf[j].x, buf[j].y});
                store_shared_vec(dst.idx(dst_ptr, {row, col + elem_per_half_memcpy}), {buf[j].z, buf[j].w});
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
//     using T = typename ST::dtype;
//     const int row_stride = src.template stride<axis>();
//     // we can handle this many rows each time we run a memcpy_async
//     constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
//     constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
//     constexpr int total_calls = (ST::cols * ST::rows + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy); // round up

//     coord<> unit_coord = idx.template unit_coord<axis, 3>();
//     typename GL::dtype *src_ptr = (typename GL::dtype*)&src[unit_coord];

//     uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);
//     const int laneid = threadIdx.x % N_THREADS;

//     // TODO: This is a hack to avoid the issue of too many VGPRs.
//     // We should find a better way to do this.
//     const int small_calls = 8;
//     const int big_calls = (total_calls + small_calls - 1) / small_calls;
//     float4    buf[small_calls];
//     uint32_t  ofs[small_calls];

//     // TODO: This is a hack to avoid the issue of too many VGPRs.
//     // We should find a better way to do this.
//     for (int i = 0; i < big_calls; i++) {
//         #pragma unroll
//         for(int j = i * small_calls; j < (i + 1) * small_calls; j++) {

//             int load_idx = j * N_THREADS + laneid;

//             int row = load_idx / memcpy_per_row;

//             int col = (load_idx % memcpy_per_row) * elem_per_memcpy;

//             if (row < dst.rows) {
//                 buf[j - i * small_calls] = load_global_vec((float4*) (src_ptr + (row * row_stride + col)));
//                 ofs[j - i * small_calls] = dst.idx(dst_ptr, {row, col});
                
//             }
//         }

//         asm volatile("s_waitcnt vmcnt(0)"); 

//         #pragma unroll
//         for(int j = i * small_calls; j < (i + 1) * small_calls; j++) {
//             int load_idx = j * N_THREADS + laneid;
//             int row = load_idx / memcpy_per_row;
//             int col = (load_idx % memcpy_per_row) * elem_per_memcpy;

//             if (row < dst.rows) {
//                 store_shared_vec(ofs[j - i * small_calls], buf[j - i * small_calls]);
//             }
//         }
//         asm volatile("s_waitcnt lgkmcnt(0)");
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
    constexpr int elem_per_float = sizeof(float)/sizeof(typename ST::dtype);
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

        if (row < src.rows) {
            *(float*) &dst_ptr[row * row_stride + col] = *(float*)(&src[{row, col}]);
            *(float*) &dst_ptr[row * row_stride + col + elem_per_float] = *(float*)(&src[{row, col + elem_per_float}]);
            *(float*) &dst_ptr[row * row_stride + col + elem_per_float * 2] = *(float*)(&src[{row, col + elem_per_float * 2}]);
            *(float*) &dst_ptr[row * row_stride + col + elem_per_float * 3] = *(float*)(&src[{row, col + elem_per_float * 3}]);
        }
    }
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    store<2, false, ST, GL, COORD, WARP_THREADS>(dst, src, idx);
}

}