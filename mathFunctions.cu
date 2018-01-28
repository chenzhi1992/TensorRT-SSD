#include "mathFunctions.h"
#include <iostream>
#include "cudaUtility.h"


//concatlayer
template <typename Dtype>
__global__ void Concat(const int nthreads, const Dtype* in_data,
                       const bool forward, const int num_concats, const int concat_size,
                       const int top_concat_axis, const int bottom_concat_axis,
                       const int offset_concat_axis, Dtype* out_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int total_concat_size = concat_size * bottom_concat_axis;
        const int concat_num = index / total_concat_size;
        const int concat_index = index % total_concat_size;
        const int top_index = concat_index +
                              (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
        if (forward) {
            out_data[top_index] = in_data[index];
        } else {
            out_data[index] = in_data[top_index];
        }
    }
}
cudaError_t ConcatLayer(int nthreads, const float *bottom_data, bool kForward, int num_concats_, int concat_input_size_,
                        int top_concat_axis, int bottom_concat_axis, int offset_concat_axis, float *top_data, cudaStream_t stream)
{
    Concat<float><<<TENSORRT_GET_BLOCKS(nthreads), TENSORRT_CUDA_NUM_THREADS,0,stream>>>(nthreads, bottom_data,
    kForward, num_concats_, concat_input_size_, top_concat_axis, bottom_concat_axis, offset_concat_axis, top_data);
    return cudaPeekAtLastError();
}
