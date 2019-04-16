#include <vector>

#include "caffe/layers/conv_to_recur_layer.hpp"

namespace caffe {
	
// deal with the elements in bottom_data priority
// which may have less parameters than top_data
template <typename Dtype>
void ConvertDataC2R(Dtype* bottom_data, const bool forward, const int seq_len, 
	const int batch_size, const int channels, Dtype* top_data)
{
    for (int i = 0; i < seq_len; ++i) {
		for (int j = 0; j < batch_size; ++j) {
			for (int k = 0; k < channels; ++k) {
				int conv_idx = (j*channels + k)*seq_len + i; // [N x C x T]
				int recur_idx = (i*batch_size + j)*channels + k; // [T x N x C]
				if (forward) {
					top_data[recur_idx] = bottom_data[conv_idx];
				}
				else {
					bottom_data[conv_idx] = top_data[recur_idx];
				}
			}
		}
    }
}
template <typename Dtype>
void ConvertLabelC2R(const Dtype* bottom_label, const int label_size, 
	const int batch_size, Dtype* top_label)
{
    for (int i = 0; i < label_size; ++i) {
		for (int j = 0; j < batch_size; ++j) {
            int conv_idx = j*label_size + i;
            int recur_idx = i*batch_size + j;
            top_label[recur_idx] = bottom_label[conv_idx];
		}
    }
}

template <typename Dtype>
ConvToRecurLayer<Dtype>::ConvToRecurLayer(const LayerParameter& param)
	: Layer<Dtype>(param) {
}

template <typename Dtype>
void ConvToRecurLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
    CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
        "allow in-place computation.";
    // infer the top index from the blob number
    switch (bottom.size())
    {
    case 1:
        switch(top.size())
        {
        case 1:
            indi_index_ = -1;
            break;
        case 2:
            indi_index_ = 1;
            break;
        default:
            LOG(ERROR) << "\nThe bottom blob:\n"
                << "\t(1) CNN data\nThe top blob should be:\n"
                << "\t(1) RNN data, or\n\t(1) RNN data (2) RNN indicator\n";
            break;
        }
        label_index_ = -1;
        break;
    case 2:
        switch(top.size())
        {
        case 2:
            indi_index_ = -1;
            label_index_ = 1;
            break;
        case 3:
            indi_index_ = 1;
            label_index_ = 2;
            break;
        default:
            LOG(ERROR) << "\nThe bottom blob:\n"
                << "\t(1) CNN data (2) CNN label\nThe top blob should be:\n"
                << "\t(1) RNN data (2) RNN label, or\n\t(1) RNN data (2) RNN indicator (3) RNN label\n";
            break;
        }
        break;
    default:
        LOG(ERROR) << "\nInvalid bottom blob num: " << bottom.size()
            << "\nThe bottom blob should be:\n"
            << "\t(1) CNN data, or\n\t(1) CNN data (2) CNN label\n";
        break;
    }
    data_index_ = 0;
}

template <typename Dtype>
void ConvToRecurLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    const int data_num_axes = bottom[0]->num_axes();
    CHECK(data_num_axes == 3 || data_num_axes == 4)
        << "\nThe convolutional data blob should have 3 or 4 axes.\n";
    N_ = bottom[0]->shape(0);
    C_ = bottom[0]->shape(1);
    seq_len_ = bottom[0]->shape(2);
    if (data_num_axes == 4)
    {
        if (seq_len_ == 1)
            seq_len_ = bottom[0]->shape(3);
        else
        {
            CHECK(bottom[0]->shape(3) == 1) << "The height (or width) of the CNN feature maps "
                "should be 1 pixel to convert into one-dimension recurrent sequence. "
                "Invalid height = " << bottom[0]->height() << ", width = " << bottom[0]->width();
        }
    }
    // if time step is specified, check if it fits the data shape
    // or infer the time step from the data shape
	const int time_steps = this->layer_param_.recurrent_param().time_steps();
    if (time_steps == 0)
	{
		T_ = seq_len_;
	}
	else
    {
        CHECK_LE(seq_len_, time_steps) << "The sequence length = max(height, width) "
            "should not be larger than the predefined time step. Invalid "
            "sequence length = " << seq_len_ << ", time step = " << T_;
		T_ = time_steps;
    }
	vector<int> top_shape; // [T x N x C]
	top_shape.push_back(T_);
	top_shape.push_back(N_);
	top_shape.push_back(C_);
	top[data_index_]->Reshape(top_shape);
	if (indi_index_ >= 0) {
		top_shape.pop_back();
		top[indi_index_]->Reshape(top_shape);
	}
    if (label_index_ >= 0)
    {
        // bottom[1] shape: [N x L]
        CHECK_EQ(N_, bottom[1]->shape(0));
        // CHECK_GE(T_, bottom[1]->shape(1)) << "\nThe label size should not "
        //     << "be larger than the time steps, in case that the output "
        //     << "sequences cannot map to the label sequences.\n";
		top_shape.clear();
        top_shape.push_back(bottom[1]->shape(1));
        top_shape.push_back(N_);
        top[label_index_]->Reshape(top_shape);
    }
}

template <typename Dtype>
void ConvToRecurLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
    Dtype* bottom_data = bottom[0]->mutable_cpu_data();
    Dtype* top_data = top[data_index_]->mutable_cpu_data();
    ConvertDataC2R(bottom_data, true, seq_len_, N_, C_, top_data);
    // in fact, no need to fill 0 into the rest memory
    // 'cause those won't be used in following calculation
    int count = bottom[0]->count();
    caffe_set<Dtype>(top[data_index_]->count() - count, 0, top_data + count);
    if (indi_index_ != -1)
    {
        // generate the indicator blob
        top_data = top[indi_index_]->mutable_cpu_data();
        caffe_set<Dtype>(N_, 0, top_data);
        count = N_*seq_len_;
        caffe_set<Dtype>(count - N_, 1, top_data + N_);
        caffe_set<Dtype>(top[indi_index_]->count() - count, 0, top_data + count);
    }
    if (label_index_ != -1)
    {
        ConvertLabelC2R(bottom[1]->cpu_data(), bottom[1]->channels(), N_, 
            top[label_index_]->mutable_cpu_data());
    }
}

template <typename Dtype>
void ConvToRecurLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    Dtype* top_diff = top[data_index_]->mutable_cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    ConvertDataC2R(bottom_diff, false, seq_len_, N_, C_, top_diff);
}

#ifdef CPU_ONLY
STUB_GPU(ConvToRecurLayer);
#endif

INSTANTIATE_CLASS(ConvToRecurLayer);
REGISTER_LAYER_CLASS(ConvToRecur);

}  // namespace caffe
