/**
 * @file
 * @brief Matrix multiply-accumulate operations for tiles stored in registers.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

__device__ static inline void mfma161616(      float2 (&D)[2],
                                         const bf16_2 (&A)[2],
                                         const bf16_2 (&B)[2],
                                         const float2 (&C)[2]) {
    (*(float4*)D).data = {__builtin_amdgcn_mfma_f32_16x16x16f16(
        (*(short4*)A).data,
        (*(short4*)B).data,
        (*(float4*)C).data,
        0, 0, 0
    )};
}

/**
 * @brief Base matrix multiply-accumulate operation for row layout.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, row_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::col> &d,
                                    const rt_base<bf16, ducks::rt_layout::row> &a,
                                    const rt_base<bf16, ducks::rt_layout::col> &b, // in col-major mode
                                    const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
/**
 * @brief Base dot product operation for row layout.
 *
 * This function performs the base dot product operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, row_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, row_layout> matrix in row-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<bf16, ducks::rt_layout::row> &a,
                                     const rt_base<bf16, ducks::rt_layout::row> &b, // in row-major mode
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, col_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<bf16, ducks::rt_layout::col> &a,
                                     const rt_base<bf16, ducks::rt_layout::col> &b, // in col-major mode
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A and B.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, col_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::col> &d,
                                      const rt_base<bf16, ducks::rt_layout::col> &a,
                                      const rt_base<bf16, ducks::rt_layout::row> &b, // in col-major mode
                                      const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}

/**
 * @brief Matrix multiply-accumulate operation.
 *
 * This function performs the matrix multiply-accumulate operation
 * using the `hmma16816` function.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_bf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
 */
template<int N, int K, int M>
__device__ static inline void mma_AB(rt_fl<N, M, ducks::rt_layout::col> &d,
                               const rt_bf<N, K, ducks::rt_layout::row> &a,
                               const rt_bf<K, M, ducks::rt_layout::col> &b,
                               const rt_fl<N, M, ducks::rt_layout::col> &c) {
    #pragma unroll
    for(int n = 0; n < N; n++) {
        #pragma unroll
        for(int m = 0; m < M; m++) {
            mma_AB_base(
                d.tiles[n][m],
                a.tiles[n][0],
                b.tiles[0][m],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < K; k++) {
                mma_AB_base(
                    d.tiles[n][m],
                    a.tiles[n][k],
                    b.tiles[k][m],
                    d.tiles[n][m]
                );
            }
        }
    }
}
/**
 * @brief Dot product operation for row layout.
 *
 * This function performs the dot product operation
 * using the `hmma16816` function.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_bf<M, K, row_layout> matrix in row-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
 */
template<int N, int K, int M>
__device__ static inline void mma_ABt(rt_fl<N, M, ducks::rt_layout::col> &d,
                                const rt_bf<N, K, ducks::rt_layout::row> &a,
                                const rt_bf<M, K, ducks::rt_layout::row> &b, // notice row and (M, K) instead of col and (K, M)
                                const rt_fl<N, M, ducks::rt_layout::col> &c) {
    #pragma unroll
    for(int n = 0; n < N; n++) {
        #pragma unroll
        for(int m = 0; m < M; m++) {
            mma_ABt_base(
                d.tiles[n][m],
                a.tiles[n][0],
                b.tiles[m][0],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < K; k++) {
                mma_ABt_base(
                    d.tiles[n][m],
                    a.tiles[n][k],
                    b.tiles[m][k],
                    d.tiles[n][m]
                );
            }
        }
    }
}
/**
 * @brief Matrix multiply-accumulate operation with transposed A.
 *
 * This function performs the matrix multiply-accumulate operation
 * using the `hmma16816` instruction.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<K, N, row_layout> matrix.
 * @param[in] b The second input rt_bf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
 */
template<int N, int K, int M>
__device__ static inline void mma_AtB(rt_fl<N, M, ducks::rt_layout::col> &d,
                                const rt_bf<K, N, ducks::rt_layout::col> &a,
                                const rt_bf<K, M, ducks::rt_layout::col> &b,
                                const rt_fl<N, M, ducks::rt_layout::col> &c) {
    #pragma unroll
    for(int n = 0; n < N; n++) {
        #pragma unroll
        for(int m = 0; m < M; m++) {
            mma_AtB_base(
                d.tiles[n][m],
                a.tiles[0][n],
                b.tiles[0][m],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < K; k++) {
                mma_AtB_base(
                    d.tiles[n][m],
                    a.tiles[k][n],
                    b.tiles[k][m],
                    d.tiles[n][m]
                );
            }
        }
    }
}
/**
 * @brief Matrix multiply-accumulate operation with transposed A and B.
 *
 * This function performs the matrix multiply-accumulate operation
 * using the `hmma16816` instruction.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<K, N, col_layout> matrix.
 * @param[in] b The second input rt_bf<M, K, row_layout> matrix in column-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
 */
template<int N, int K, int M>
__device__ static inline void mma_AtBt(rt_fl<N, M, ducks::rt_layout::col> &d,
                                 const rt_bf<K, N, ducks::rt_layout::col> &a,
                                 const rt_bf<M, K, ducks::rt_layout::row> &b,
                                 const rt_fl<N, M, ducks::rt_layout::col> &c) {
    #pragma unroll
    for(int n = 0; n < N; n++) {
        #pragma unroll
        for(int m = 0; m < M; m++) {
            mma_AtBt_base(
                d.tiles[n][m],
                a.tiles[0][n],
                b.tiles[m][0],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < K; k++) {
                mma_AtBt_base(
                    d.tiles[n][m],
                    a.tiles[k][n],
                    b.tiles[m][k],
                    d.tiles[n][m]
                );
            }
        }
    }
}

}