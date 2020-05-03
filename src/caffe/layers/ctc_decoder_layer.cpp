#include "caffe/layers/ctc_decoder_layer.hpp"

#include <algorithm>

namespace caffe {

template <typename Dtype>
CTCDecoderLayer<Dtype>::CTCDecoderLayer(const LayerParameter& param)
	: Layer<Dtype>(param)
	, blank_index_(param.ctc_param().blank_index())
	, ctc_param_(param.ctc_param())
{
}

template <typename Dtype>
void CTCDecoderLayer<Dtype>::LayerSetUp(
	const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
	switch (bottom.size())
	{
	case 1:
		switch (ctc_param_.nct_axis_size())
		{
		case 2:
			CHECK_NE(ctc_param_.nct_axis(0), ctc_param_.nct_axis(1));
		case 0:
			break;
		default:
			LOG(FATAL) << "\nInvalid nct_axis number: "
				<< ctc_param_.nct_axis_size()
				<< "\nThere should be 2 or NONE nct_axis.";
			break;
		}
		break;
	case 2:
		// cannot specify axis order for recurrent data blob with indicator
		CHECK_EQ(ctc_param_.nct_axis_size(), 0) << "\nCannot specify axis "
			"order while there is an indicator blob, the axis order for "
			"recurrent data blob is fixed to be time-num-channel.";
		break;
	default:
		LOG(FATAL) << "\nInvalid bottom blob number: " << bottom.size()
			<< "\nThe bottom blobs should be:\n"
			<< "\t(1) data , or\n"
			<< "\t(1) RNN data (2) RNN indicator\n";
		break;
	}
}

template <typename Dtype>
void CTCDecoderLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	const Blob<Dtype>* data_blob = bottom[0];
	const int data_num_axes = data_blob->num_axes();
	int n_axis, c_axis, t_axis;
	if (bottom.size() == 2)
	{
		// check the data_blob shape
		CHECK_EQ(data_num_axes, 3)
			<< "\nThe RNN data blob should have 3 axes.";
		n_axis = 1;
		c_axis = 2;
		t_axis = 0;
	}
	else
	{
		// check the data_blob shape
		CHECK(data_num_axes == 3 || data_num_axes == 4)
			<< "\nThe data blob should have 3 or 4 axes.";
		if (ctc_param_.nct_axis_size() == 0)
		{
			n_axis = 0;
			c_axis = 1;
		}
		else
		{
			n_axis = ctc_param_.nct_axis(0);
			c_axis = ctc_param_.nct_axis(1);
			CHECK_LT(n_axis, data_num_axes);
			CHECK_LT(c_axis, data_num_axes);
		}
		t_axis = -1;
		for (int i = 0; i < data_num_axes; ++i)
		{
			if (i == n_axis || i == c_axis)
				continue;
			if (t_axis == -1 || data_blob->shape(t_axis) == 1)
			{
				t_axis = i;
			}
			else
			{
				CHECK(data_blob->shape(i) == 1) << "The height or width of the CNN feature maps "
					<< "should be 1 pixel to convert into one-dimension recurrent sequence. "
					<< "Invalid shape = [" << data_blob->num() << ", " << data_blob->channels()
					<< ", " << data_blob->height() << ", " << data_blob->width() << "].";
			}
		} // get t_axis
	}
	N_ = data_blob->shape(n_axis);
	C_ = data_blob->shape(c_axis);
	CHECK_GT(C_, blank_index_);
	CHECK_GE(C_, -blank_index_);
	T_ = data_blob->shape(t_axis);
	if (bottom.size() == 2)
	{
		const Blob<Dtype>* indi_blob = bottom[1];
		CHECK_EQ(indi_blob->num_axes(), 2);
		CHECK_EQ(T_, indi_blob->shape(0));
		CHECK_EQ(N_, indi_blob->shape(1));
	}
	// data blob step for each axis
	N_step_ = data_blob->count(n_axis + 1);
	C_step_ = data_blob->count(c_axis + 1);
	T_step_ = data_blob->count(t_axis + 1);

	switch (ctc_param_.decode()) {
	case CTCParameter_DecodeOpt_IN_PLACE:
	case CTCParameter_DecodeOpt_BEST_PATH:
		CHECK(!ctc_param_.has_beam_width());
		break;
	case CTCParameter_DecodeOpt_BEAM_SEARCH:
		beam_width_ = ctc_param_.beam_width();
		CHECK_LT(beam_width_, 0);
		break;
	default:
		LOG(FATAL) << "Undefined CTC decode algorithm: "
			<< ctc_param_.decode();
		break;
	}
	top[0]->Reshape(N_, T_, 1, 1);
}

template <typename Dtype>
void CTCDecoderLayer<Dtype>::Forward_cpu(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	const Dtype *indicator = (bottom.size() == 2) ? bottom[1]->cpu_data() : nullptr;
	Decode(bottom[0], indicator, top[0]);
}

template <typename Dtype>
void CTCDecoderLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
	// nothing to do
}

template <typename Dtype>
inline bool larger_prob(const pair<vector<int>, pair<Dtype, Dtype> > &p1, const pair<vector<int>, pair<Dtype, Dtype> > &p2) {
    return (p1.second.first + p1.second.second) > (p2.second.first + p2.second.second);
}

template <typename Dtype>
void CTCDecoderLayer<Dtype>::Decode(
	Blob<Dtype> *data_blob, const Dtype *indicator, Blob<Dtype> *sequence_blob)
{
	Dtype *sequence = sequence_blob->mutable_cpu_data();
	caffe_set<Dtype>(sequence_blob->count(), -1, sequence);
	// default blank_index = C - 1
	const int blank_index = (blank_index_ < 0) ? C_ + blank_index_ : blank_index_;
	switch (ctc_param_.decode()) {
	case CTCParameter_DecodeOpt_IN_PLACE:
	{
		// in-place decoding
		for (int n = 0; n < N_; ++n) {
			for (int t = 0; t < T_; ++t) {
				if (indicator && t > 0 && indicator[t*N_ + n] == 0)
				{
					break; // the output sequence ended here
				}
				// get maximum probability and its index
				const Dtype* data_nt = data_blob->cpu_data() + n*N_step_ + t*T_step_;
				int max_class_idx = 0;
				Dtype max_prob = *data_nt;
				data_nt += C_step_;
				for (int c = 1; c < C_; ++c, data_nt += C_step_) {
					if (*data_nt > max_prob) {
						max_class_idx = c;
						max_prob = *data_nt;
					}
				}
				sequence[n*T_ + t] = max_class_idx;
			} // T_
		} // N_
		break;
	}
	case CTCParameter_DecodeOpt_BEST_PATH:
	{
		for (int n = 0; n < N_; ++n) {
			int prev_class_idx = -1;
			int seq_index = -1;
			for (int t = 0; t < T_; ++t) {
				if (indicator && t > 0 && indicator[t*N_ + n] == 0)
				{
					break; // the output sequence ended here
				}
				// get maximum probability and its index
				const Dtype *data_nt = data_blob->cpu_data() + n*N_step_ + t*T_step_;
				int max_class_idx = 0;
				Dtype max_prob = data_nt[0];
				data_nt += C_step_;
				for (int c = 1; c < C_; ++c, data_nt += C_step_) {
					if (*data_nt > max_prob) {
						max_class_idx = c;
						max_prob = *data_nt;
					}
				}
				if (max_class_idx != blank_index && max_class_idx != prev_class_idx) {
					sequence[n*T_ + (++seq_index)] = max_class_idx;
				}
				prev_class_idx = max_class_idx;
			} // T_
		} // N_
		break;
	}
	case CTCParameter_DecodeOpt_BEAM_SEARCH:
	{
		SequenceSoftmax(data_blob, indicator);
		const int blank_offset = blank_index * C_step_;
		// map< y, pair< Pr+, Pr-> >, y is the sequence in set B and B_hat
		// Pr(y, t) corresponds to B, Pr(y, t-1) corresponds to B_hat
		typedef map<vector<int>, pair<Dtype, Dtype> > BeamType;
		for (int n = 0; n < N_; ++n) {
			BeamType beam, beam_prefix; // B and B_hat
			// t = 0, the first time step
			{
				// deal with the only sequence: an empty sequence { }
				vector<int> tmp_seq = vector<int>();
				const Dtype *data_nt = data_blob->cpu_data() + n*N_step_;
				beam[tmp_seq].second = data_nt[blank_offset];
				// the uninitialized probabilities default to 0
				for (int c = 0; c < C_; ++c) {
					if (c == blank_index)
						continue;
					tmp_seq.push_back(c);
					beam[tmp_seq].first = data_nt[c*C_step_];
					tmp_seq.pop_back();
				}
			}
			for (int t = 1; t < T_; ++t) {
				if (indicator && indicator[t*N_ + n] == 0)
				{
					break; // the output sequence ended here
				}
				// select the W most probable sequences in B to B_hat
				if (beam.size() <= beam_width_) {
					beam_prefix.swap(beam);
				} else {
					vector<pair<vector<int>, pair<Dtype, Dtype> > > tmp_vec(beam.begin(), beam.end());
					std::sort(tmp_vec.begin(), tmp_vec.end(), larger_prob<Dtype>);
					beam_prefix.clear();
					for (int w = 0; w < beam_width_; ++w) {
						beam_prefix.insert(tmp_vec[w]);
					}
				}
				// and empty the beam
				beam.clear();
				const Dtype *data_nt = data_blob->cpu_data() + n*N_step_ + t*T_step_;
				for (BeamType::iterator iter = beam_prefix.begin(); iter != beam_prefix.end(); ++iter) {
					// deal with sequence y
					vector<int> tmp_seq = iter->first;
					const pair<Dtype, Dtype> &prob_y_t1 = iter->second;
					int seq_end = blank_index;
					if (!tmp_seq.empty()) {
						seq_end = tmp_seq[tmp_seq.size() - 1];
						beam[tmp_seq].first += prob_y_t1.first * data_nt[seq_end*C_step_];
						beam[tmp_seq].second = data_nt[blank_offset] * (prob_y_t1.first + prob_y_t1.second);
					}
					else {
						beam[tmp_seq].second = data_nt[blank_offset] * prob_y_t1.second;
					} // end if the sequence is empty
					for (int c = 0; c < C_; ++c) {
						if (c == blank_index)
							continue;
						tmp_seq.push_back(c);
						// add sequence (y+k) to B
						if (c == seq_end) {
							beam[tmp_seq].first += data_nt[c*C_step_] * prob_y_t1.second;
						}
						else {
							beam[tmp_seq].first += data_nt[c*C_step_] * (prob_y_t1.first + prob_y_t1.second);
						}
						tmp_seq.pop_back();
					}
				} // end for each sequence in B_hat
			} // end for each time step
			// find the sequence with the max probability
			Dtype max_prob = 0;
			const vector<int> *max_seq_pointer = nullptr;
			for (BeamType::iterator iter = beam.begin(); iter != beam.end(); ++iter) {
				const Dtype tmp_prob = iter->second.first + iter->second.second;
				if (tmp_prob > max_prob) {
					max_prob = tmp_prob;
					max_seq_pointer = &(iter->first);
				}
			}
			// copy the sequence into the blob
			for (int seq_index = 0; seq_index < max_seq_pointer->size(); ++seq_index) {
				sequence[sequence_blob->offset(n, seq_index)] = (*max_seq_pointer)[seq_index];
			}
		} // end for each sample within the mini-batch
		break;
	} // end beam search decoding
	} // end swith decode option
}

template <typename Dtype>
void CTCDecoderLayer<Dtype>::SequenceSoftmax(Blob<Dtype> *data_blob, const Dtype *indicator)
{
	Dtype *data = data_blob->mutable_cpu_data();
	// compute softmax (until sequence length is sufficient)
	for (int n = 0; n < N_; ++n) {
		for (int t = 0; t < T_; ++t) {
			if (indicator && t > 0 && indicator[t*N_ + n] == 0)
			{
				break; // the output sequence ended here
			}
			Dtype *data_nt = data + n*N_step_ + t*T_step_;
			Dtype max_coeff = data_nt[0];
			const int C_end = C_*C_step_;
			// get max coeff
			for (int i = C_step_; i < C_end; i += C_step_) {
				if (max_coeff < data_nt[i])
					max_coeff = data_nt[i];
			}
			// calc exp and its sum
			Dtype sum = 0;
			for (int i = 0; i < C_end; i += C_step_) {
				data_nt[i] = exp(data_nt[i] - max_coeff);
				sum += data_nt[i];
			}
			// division by sum
			for (int i = 0; i < C_end; i += C_step_) {
				data_nt[i] /= sum;
			}
		}
	}
}

INSTANTIATE_CLASS(CTCDecoderLayer);
REGISTER_LAYER_CLASS(CTCDecoder);

}  // namespace caffe
