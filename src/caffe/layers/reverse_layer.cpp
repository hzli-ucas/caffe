#include "caffe/layers/reverse_layer.hpp"

namespace caffe {

template <typename Dtype>
ReverseLayer<Dtype>::ReverseLayer(const LayerParameter& param)
  : Layer<Dtype>(param) {
}

template <typename Dtype>
void ReverseLayer<Dtype>::LayerSetUp(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	CHECK_NE(top[0], bottom[0]) << this->type()
		<< " Layer does not allow in-place computation.";
	const ReverseParameter &reverse_param = this->layer_param_.reverse_param();
	if (bottom.size() == 2)
	{
		// cannot specify reverse axis for recurrent data blob with indicator
		CHECK(!reverse_param.has_axis()) << "Cannot specify reverse axis "
			<< "while there is an indicator blob, the data within the indicated scope "
			<< "will be reversed along the first axis, i.e. the time axis for recurrent data blob [T x N x C]";
		reverse_axis_ = 0;
	}
	else
	{
		// in case of misuse the reverse layer
		// must specify reverse axis for data blob without indicator
		CHECK(reverse_param.has_axis()) << "Must specify the reverse axis "
			<< "for data blob without indicator, in case of misuse the reverse layer.";
		reverse_axis_ = bottom[0]->CanonicalAxisIndex(reverse_param.axis());
	}
}

template <typename Dtype>
void ReverseLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	top[0]->ReshapeLike(*bottom[0]);
	if (bottom.size() == 2)
	{
		// the indicator blob (bottom[1]) must has shape [T x N]
		// as the recurrent data blob (bottom[0]) shape is [T x N x C]
		CHECK_EQ(bottom[0]->num_axes(), 3);
		CHECK_EQ(bottom[1]->num_axes(), 2);
		CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
		CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1));
		seq_len_blob_.Reshape(vector<int>(1, bottom[0]->shape(1)));
	}
	else
	{
		// axis_outer * axis_length * axis_inner = N * C * H * W
		//             |<----- outer_step ----->|
		axis_length_ = bottom[0]->shape(reverse_axis_);
		outer_step_ = bottom[0]->count(reverse_axis_);
		axis_outer_ = bottom[0]->count() / outer_step_;
		axis_inner_ = outer_step_ / axis_length_;
	}
}

template <typename Dtype>
void ReverseLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	const Dtype* src = bottom[0]->cpu_data();
	Dtype* const top_data = top[0]->mutable_cpu_data();
	if (bottom.size() == 2)
	{
		const int N = bottom[0]->shape(1);
		const int C = bottom[0]->shape(2);
		const Dtype* indicator = bottom[1]->cpu_data();
		int* seq_len = seq_len_blob_.mutable_cpu_data();
		for (int n = 0; n < N; ++n)
		{
			axis_length_ = bottom[0]->shape(0);
			while (axis_length_ > 0 && indicator[(--axis_length_)*N + n] == 0);
			const Dtype* src_n = src + n*C;
			Dtype* dest_n = top_data + n*C + axis_length_*N*C;
			seq_len[n] = ++axis_length_;
			for (int t = 0; t < axis_length_; ++t)
			{
				caffe_copy(C, src_n, dest_n);
				src_n += N*C;
				dest_n -= N*C;
			}
		}
	}
	else
	{
		for (int i = 0; i < axis_outer_; ++i)
		{
			Dtype* dest = top_data + (i + 1) * outer_step_ - axis_inner_;
			// invert along the reverse axis
			for (int j = 0; j < axis_length_; ++j) {
				caffe_copy(axis_inner_, src, dest);
				src += axis_inner_;
				dest -= axis_inner_;
			}
		}
	}
}

template <typename Dtype>
void ReverseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
	if (!propagate_down[0]) { return; }

	const Dtype* src = top[0]->cpu_diff();
	Dtype* const bottom_diff = bottom[0]->mutable_cpu_diff();
	if (bottom.size() == 2)
	{
		const int N = bottom[0]->shape(1);
		const int C = bottom[0]->shape(2);
		const Dtype* indicator = bottom[1]->cpu_data();
		const int* seq_len = seq_len_blob_.cpu_data();
		for (int n = 0; n < N; ++n)
		{
			const Dtype* src_n = src + n*C;
			Dtype* dest_n = bottom_diff + n*C + (seq_len[n]-1)*N*C;
			for (int t = 0; t < seq_len[n]; ++t)
			{
				caffe_copy(C, src_n, dest_n);
				src_n += N*C;
				dest_n -= N*C;
			}
		}
	}
	else
	{
		for (int i = 0; i < axis_outer_; ++i)
		{
			Dtype* dest = bottom_diff + (i + 1) * outer_step_ - axis_inner_;
			// invert along the reverse axis
			for (int j = 0; j < axis_length_; ++j) {
				caffe_copy(axis_inner_, src, dest);
				src += axis_inner_;
				dest -= axis_inner_;
			}
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(ReverseLayer);
#endif

INSTANTIATE_CLASS(ReverseLayer);
REGISTER_LAYER_CLASS(Reverse);

}  // namespace caffe
