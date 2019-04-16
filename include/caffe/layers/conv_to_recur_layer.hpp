#ifndef CAFFE_CONV_TO_RECUR_LAYER_HPP_
#define CAFFE_CONV_TO_RECUR_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
* @brief Convert the formate of data blob from CNN to RNN
*        CNN data shape: N * C * H * W (batch_size * channels * height * width)
*        RNN data shape: T * N * C (time_steps * batch_size * channels)
*        And create the corresponding "indicator" blob for RNN
*
*        If there are two bottom blobs
*        Also convert the formate of label blob from CNN to RNN
*        CNN label shape: N * L (batch_size * label_size)
*        RNN label shape: L * N (label_size * batch_size)
*
* @param The time steps can be specified with
*        recurrent_param { time_steps = 10 },
*        or it is default to 0 and will be inferred from the data blob shape
*        Allow the user to specify the time steps (as large value as needed) to build the following recurrent structure
*        in case that the change of input size lead to longer input sequences and cause errors
*/

template <typename Dtype>
class ConvToRecurLayer : public Layer<Dtype> {
public:
	explicit ConvToRecurLayer(const LayerParameter& param);
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "ConvToRecur"; }
	virtual inline int MinBottomBlobs() const { return 1; }
	virtual inline int MaxBottomBlobs() const { return 2; }
	virtual inline int MinTopBlobs() const { return 1; }
	virtual inline int MaxTopBlobs() const { return 3; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	// T for the time steps of the RNN
	int N_, C_, T_, seq_len_;
	int data_index_, indi_index_, label_index_;
};

}  // namespace caffe

#endif  // CAFFE_CONV_TO_RECUR_LAYER_HPP_
