#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <random>
#include <type_traits>
#include "kittens.cuh"
#include <hip/hip_bfloat16.h>

/* --------------------  hip ERROR UTILS  -------------------- */

#include <stdio.h>
#define HipCheckError()    __hipCheckError( __FILE__, __LINE__ )
inline void __hipCheckError( const char *file, const int line ) {
    hipError_t err = hipGetLastError();
    if ( hipSuccess != err )
    {
        fprintf( stderr, "hipCheckError() failed at %s:%i : %s\n",
                 file, line, hipGetErrorString( err ) );
        exit( -1 );
    }
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = hipDeviceSynchronize();
    if( hipSuccess != err )
    {
        fprintf( stderr, "hipCheckError() with sync failed at %s:%i : %s\n",
                 file, line, hipGetErrorString( err ) );
        exit( -1 );
    }
}

/* --------------------  TEST STRUCTS  -------------------- */

/*
General Unit Test Struct
struct test {
    template<typename... args> using valid = true; // Set this invalid if you don't want the test to be compiled and run.
    static inline const std::string test_identifier; ("block::load" as an example.)
    template<typename... args> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref);
    template<typename... args> __global__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output);
};
*/
enum test_result {
    PASSED = 0,
    FAILED = 1,
    INVALID = 2 // This is a useful one for tests that are only defined for certain template specializations, but we still want to sweep.
};
struct test_info {
    std::string label;
    test_result result;
};
using test_data = std::vector<test_info>;


/* --------------------  TEMPLATE METAPROGRAMMING UTILS  -------------------- */

// 1D wrapper
template<template<typename,int,int,typename...> typename base, typename test, int MAX_S, int NUM_WORKERS, int S, typename... args>
struct loop_s {
    static void run(test_data& results) {
        if constexpr (S > 1) {
            loop_s<base, test, MAX_S, NUM_WORKERS, S-1, args...>::run(results);
        }
        base<test, S, NUM_WORKERS, args...>::run(results);
    }
};

// 2D wrappers
template<template<typename,int,int,int,typename...> typename base, typename test, int MAX_H, int MAX_W, int NUM_WORKERS, int H, int W, typename... args>
struct loop_w {
    static void run(test_data& results) {
        if constexpr (W > 1) {
            loop_w<base, test, MAX_H, MAX_W, NUM_WORKERS, H, W-1, args...>::run(results);
        }
        base<test, H, W, NUM_WORKERS, args...>::run(results);
    }
};
template<template<typename,int,int,int,typename...> typename base, typename test, int MAX_H, int MAX_W, int NUM_WORKERS, int H, typename... args>
struct loop_h {
    static void run(test_data& results) {
        if constexpr (H > 1) {
            loop_h<base, test, MAX_H, MAX_W, NUM_WORKERS, H-1, args...>::run(results);
        }
        loop_w<base, test, MAX_H, MAX_W, NUM_WORKERS, H, MAX_W, args...>::run(results);
    }
};


/* --------------------  TEST INITIALIZE+VALIDATE FUNCS  -------------------- */

enum initializers {
    RANDOM = 0, // uniform random init. useful for confirming correctness.
    ARANGE = 1, // write an increasing sequence into i_ref and d_i arrays. useful for debugging memory movement.
    NONE   = 2  // use whatever values were already in i_ref. useful for detailed debugging.
};
template<typename T, initializers initializer=initializers::RANDOM, int SEED=42>
void initialize(T **d_i, T **d_o, std::vector<float> &i_ref, std::vector<float> &o_ref) {
    using namespace kittens;

    const int input_size  = i_ref.size();
    const int output_size = o_ref.size();

    // Initialize matrices
    std::vector<T> i_t(input_size);

    std::mt19937 gen(SEED); // Standard mersenne_twister_engine
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    for(int idx = 0; idx < input_size; idx++) {
        float f;
        if constexpr (initializer == initializers::RANDOM) {
            f = dis(gen);
        }
        else if constexpr (initializer == initializers::ARANGE) {
            f = float(idx);
        }
        else {
            f = i_ref[idx];
        }
        if constexpr (std::is_same_v<T, bf16>) {
            i_t[idx] = __float2bfloat16(f); // fill in for transfer to device
            i_ref[idx] = __bfloat162float(i_t[idx]); // ensure lossiness of fp16 is captured on cpu
        }
        else if constexpr (std::is_same_v<T, float>) {
            i_t[idx] = f;
            i_ref[idx] = f;
        }
        else if constexpr (std::is_same_v<T, half>) {
            i_t[idx] = __float2half(f);
            i_ref[idx] = __half2float(i_t[idx]);
        }
        else {
            assert(false && "Unsupported data type");
        }
    }

    hipMalloc(d_i, input_size  * sizeof(T));
    hipMalloc(d_o, output_size * sizeof(T));
    HipCheckError();

    hipMemcpy(*d_i, i_t.data(), input_size * sizeof(T), hipMemcpyHostToDevice);
    HipCheckError();
}
extern int should_write_outputs;
template<typename T>
test_result validate(T *d_i, T *d_o, const std::vector<float> &i_ref, std::vector<float> &o_ref, std::string test_name, int cols, float eps=5e-2) { // default eps has to be fairly high due to lots of different types
    using namespace kittens;
    const int input_size  = i_ref.size();
    const int output_size = o_ref.size();
    // copy back
    T* o_t = new T[output_size];
    float *o = new float[output_size];
    hipDeviceSynchronize();
    HipCheckError();
    hipMemcpy(o_t, d_o, output_size * sizeof(T), hipMemcpyDeviceToHost);
    HipCheckError();
    for(int idx = 0; idx < output_size; idx++) {
        if constexpr (std::is_same_v<T, bf16>) {
            o[idx] = __bfloat162float(o_t[idx]);
            o_ref[idx] = __bfloat162float(__float2bfloat16(o_ref[idx]));
        }
        else if constexpr (std::is_same_v<T, half>) {
            o[idx] = __half2float(o_t[idx]);
            o_ref[idx] = __half2float(__float2half(o_ref[idx]));
        }
        else if constexpr(std::is_same_v<T, float>) {
            o[idx] = o_t[idx];
            o_ref[idx] = o_ref[idx];
        }
        else {
            assert(false && "Unsupported data type");
        }
    }
    // check
    std::cout << "test `" << test_name << "`";
    bool good = true;
    for(int i = 0; i < output_size; i++) {
        if(abs(o_ref[i] - o[i]) > eps) {
            good = false;
            break;
        }
    }
    if(good) std::cout << " -- PASSED" << std::endl;
    else std::cout << " ----- ALERT! FAILED test `" << test_name << "` -----" << std::endl;
    if(should_write_outputs) {
        std::ofstream reffile("outputs/"+test_name+"_ref.txt");
        std::ofstream outfile("outputs/"+test_name+"_out.txt");
        for(int i = 0; i < output_size; i++) {
            reffile << o_ref[i] << ' ';
            outfile << o[i] << ' ';
            if(i%cols == cols-1) {
                reffile << '\n';
                outfile << '\n';
            }
        }
        reffile << "\n\n\nINPUTS:\n\n";
        for(int i = 0; i < input_size; i++) {
            reffile << i_ref[i] << ' ';
            if(i%cols == cols-1) {
                reffile << '\n';
            }
        }
        reffile.close();
        outfile.close();
    }
    hipFree(d_i);
    hipFree(d_o);
    delete[] o_t, o;
    HipCheckError();
    return good ? test_result::PASSED : test_result::FAILED;
}