#include <vector>

#include "caffe/layers/reverse_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// infer the sequence length from the 0/1 indicator blob
// indi_data - [T x N], seq_len - [N]
template <typename Dtype>
__global__ void SequenceLengthGPU(const int T, 
  const int N, const Dtype* indi_data, int* seq_len)
{
  CUDA_KERNEL_LOOP(n, N)
  {
    // infer the sequence length backward
    int length = T;
    while (length > 0 && indi_data[(--length)*N + n] == 0);
    seq_len[n] = length; // actual_length - 1
    // more convinient for index calculation: seq_lex - t
  }
}

template <typename Dtype>
__global__ void ReverseWithIndicatorGPU(const int nthreads, const int N, const int C, 
  const Dtype* src_data, const int* seq_len, Dtype* dest_data)
{
  CUDA_KERNEL_LOOP(index, nthreads)
  {
    const int t = index / (N*C);
    int tmp_index = index % (N*C);
    const int n = tmp_index / C;
    if (t > seq_len[n])
    {
      continue;
    }
    const int dest_index = (seq_len[n] - t)*N*C + tmp_index;
    dest_data[dest_index] = src_data[index];
  }
}

template <typename Dtype>
__global__ void ReverseAlongAxisGPU(const int nthreads, const int outer_step, 
  const int axis_len, const int axis_inner, const Dtype* src_data, Dtype* dest_data)
{
  CUDA_KERNEL_LOOP(index, nthreads)
  {
    const int i = index / outer_step;
    int tmp_index = index % outer_step;
    const int j = tmp_index / axis_inner;
    const int dest_index = i*outer_step + (axis_len - j)*axis_inner + (tmp_index % axis_inner);
    dest_data[dest_index] = src_data[index];
  }
}

template <typename Dtype>
void ReverseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* src = bottom[0]->gpu_data();
  Dtype* const dest = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  if (bottom.size() == 2)
  {
    const int N = bottom[1]->shape(1);
    SequenceLengthGPU<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      bottom[1]->shape(0), N, bottom[1]->gpu_data(), seq_len_blob_.mutable_gpu_data());
    ReverseWithIndicatorGPU<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, N, bottom[0]->shape(2), src, seq_len_blob_.gpu_data(), dest);
  }
  else
  {
    ReverseAlongAxisGPU<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, outer_step_, axis_length_, axis_inner_, src, dest);
  }
}

template <typename Dtype>
void ReverseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }

  const Dtype* src = top[0]->gpu_diff();
  Dtype* const dest = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  if (bottom.size() == 2)
  {
    ReverseWithIndicatorGPU<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom[1]->shape(1), bottom[0]->shape(2), src, seq_len_blob_.gpu_data(), dest);
  }
  else
  {
    ReverseAlongAxisGPU<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, outer_step_, axis_length_, axis_inner_, src, dest);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ReverseLayer);


}  // namespace caffe
