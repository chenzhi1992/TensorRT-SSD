

#ifndef __MATH_FUNCTINS_H__
#define __MATH_FUNCTINS_H__

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <cblas.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types
#include <algorithm>

#include <glog/logging.h>
#define PERMUTELAYER_ORDERNUM 4
#define BLOCK 512
//
// CUDA macros
//

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << cublasGetErrorString(status); \
  } while (0)

#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " \
      << curandGetErrorString(status); \
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())


// CUDA: library error reporting.
const char* cublasGetErrorString(cublasStatus_t error);
const char* curandGetErrorString(curandStatus_t error);

// CUDA: use 512 threads per block
const int TENSORRT_CUDA_NUM_THREADS = 256;

// CUDA: number of blocks for threads.
inline int TENSORRT_GET_BLOCKS(const int N) {
  return (N + TENSORRT_CUDA_NUM_THREADS - 1) / TENSORRT_CUDA_NUM_THREADS;
}


/* 
 * function: X[i] = alpha,initialize X with constant alpha
 * 
 */
template <typename Dtype>
void tensorrt_gpu_set(const int N, const Dtype alpha, Dtype *X);

/*
 * function: y[index] = pow(a[index], alpha)
 *@params n: the dims of matrix a
 *@params a: matrix
 *@params y: vector
 */
template <typename Dtype>
void tensorrt_gpu_powx(const int n, const Dtype* a, const Dtype alpha, Dtype* y);


/*
 *function:y = alpha*A*x + beta*y;
 *@params handle: handle
 *@params TransA: transpose flag
 *@params M: the rows of A
 *@params N: the cols of A
 *@params alpha: the coefficient of A*x
 *@params A: matrix [M x N]
 *@params x: vector x
 *@params beta: the coefficient of y
 *@params y: vector y
 */
template <typename Dtype>
void tensorrt_gpu_gemv(cublasHandle_t handle,const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* y);



template <typename Dtype>
void tensorrt_gpu_divbsx(const int nthreads, const Dtype* A,
    const Dtype* v, const int rows, const int cols, const CBLAS_TRANSPOSE trans,
    Dtype* B);

template <typename Dtype>
void tensorrt_gpu_mulbsx(const int nthreads, const Dtype* A,
    const Dtype* v, const int rows, const int cols, const CBLAS_TRANSPOSE trans,
    Dtype* B);
cudaError_t tensorrt_gpu_permute(const int nthreads,float* const  bottom_data,const bool forward,
	const int* permute_order,const int* old_steps,const int* new_steps,const int num_axes,float* const top_data,cudaStream_t stream);

cudaError_t SoftmaxLayer(const float *bottom_data, int count, int channels, int outer_num_, int inner_num_, float *scale_data, float *top_data, cudaStream_t stream);

cudaError_t ConcatLayer(int nthreads, const float *bottom_data, bool kForward, int num_concats_, int concat_input_size_, int top_concat_axis, int bottom_concat_axis, int offset_concat_axis, float *top_data, cudaStream_t stream);


#endif
