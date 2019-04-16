#include <algorithm>
#include <cfloat>

#include "caffe/layers/conv_to_recur_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ConvertDataC2R_Kernel(const int nthreads,
    Dtype* const bottom_data, const bool forward, const int seq_len, 
    const int batch_size, const int channels, Dtype* const top_data)
{
    const int batch_step = seq_len * channels;
    CUDA_KERNEL_LOOP(index, nthreads) {
        // index for convolution data blob [N x C x T]
        const int n = index / batch_step;
        int tmp_idx = index % batch_step;
        const int c = tmp_idx / seq_len;
        const int t = tmp_idx % seq_len;
        int recur_idx = (t*batch_size + n)*channels + c;
        if (forward) {
            top_data[recur_idx] = bottom_data[index];
        } else {
            bottom_data[index] = top_data[recur_idx];
        }
    }
}
template <typename Dtype>
__global__ void ConvertLabelC2R_Kernel(const int nthreads,
    const Dtype* bottom_label, const int label_size, 
	const int batch_size, Dtype* const top_label)
{
    CUDA_KERNEL_LOOP(index, nthreads) {
        // index for convolution label blob [N x L]
        const int n = index / label_size;
        const int l = index % label_size;
        int recur_idx = l*batch_size + n;
        top_label[recur_idx] = bottom_label[index];
    }
}

template <typename Dtype>
void ConvToRecurLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    Dtype* bottom_data = bottom[0]->mutable_gpu_data();
    Dtype* top_data = top[data_index_]->mutable_gpu_data();
    int count = bottom[0]->count();
    bool foward = true;
    // NOLINT_NEXT_LINE(whitespace/operators)
    ConvertDataC2R_Kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, foward, seq_len_, N_, C_, top_data);
    if (indi_index_ >= 0)
    {
        // generate the indicator blob
        top_data = top[indi_index_]->mutable_gpu_data();
		caffe_gpu_set<Dtype>(N_, 0, top_data);
        count = N_*seq_len_;
		caffe_gpu_set<Dtype>(count - N_, 1, top_data + N_);
		caffe_gpu_set<Dtype>(top[indi_index_]->count() - count, 0, top_data + count);
    }
    if (label_index_ >= 0)
    {
        count = bottom[1]->count();
        ConvertLabelC2R_Kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom[1]->gpu_data(), bottom[1]->channels(), N_, 
            top[label_index_]->mutable_gpu_data());
    }
    CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
void ConvToRecurLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
	if (!propagate_down[0])
	{
		return;
	}
    Dtype* top_diff = top[data_index_]->mutable_gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    bool foward = false;
    // NOLINT_NEXT_LINE(whitespace/operators)
    ConvertDataC2R_Kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_diff, foward, seq_len_, N_, C_, top_diff);
    CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvToRecurLayer);

}  // namespace caffe
