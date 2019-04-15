#include "caffe/layers/ctc_loss_layer.hpp"

#include <algorithm>
#include <limits>
#include <sstream>
#include <string>

namespace caffe {

// Zero probability in log space: negtive infinity
#define kLogZero -INFINITY 

/**
* @brief Adds two probabilities in log space.
* @returns log_(prob_1 + prob_2)
*/
template <typename Dtype>
inline Dtype LogSumExp(Dtype log_prob_1, Dtype log_prob_2) {
	if (log_prob_1 == kLogZero) {
		return log_prob_2;
	}
	if (log_prob_2 == kLogZero) {
		return log_prob_1;
	}
	// log(a+b) = log(b*(a/b+1)) = log(b) + log(exp(log(a)-log(b)) + 1)
	// Always have 'b' be the smaller number to
	// prevent the exponential from blowing up.
	return (log_prob_1 > log_prob_2)
		? log_prob_1 + log1p(exp(log_prob_2 - log_prob_1))
		: log_prob_2 + log1p(exp(log_prob_1 - log_prob_2));
}

template <typename Dtype>
inline Dtype min(Dtype lhs, Dtype rhs)
{
	return lhs < rhs ? lhs : rhs;
}

template <typename Dtype>
inline Dtype max(Dtype lhs, Dtype rhs)
{
	return lhs > rhs ? lhs : rhs;
}

template <typename Dtype>
CTCLossLayer<Dtype>::CTCLossLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
	blank_index_(param.ctc_param().blank_index()),
	already_softmax_(param.ctc_param().already_softmax())
{
}

template <typename Dtype>
CTCLossLayer<Dtype>::~CTCLossLayer() {}

template <typename Dtype>
void CTCLossLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top)
{
	// LossLayers have a non-zero (1) loss by default.
	if (this->layer_param_.loss_weight_size() == 0) {
		this->layer_param_.add_loss_weight(Dtype(1));
	}
	const CTCParameter &ctc_param = this->layer_param_.ctc_param();
	switch (bottom.size())
	{
	case 2:
		if (ctc_param.nct_axis_size() == 0)
			break; // The axis order defaults to be num-channel-time
		CHECK_EQ(ctc_param.nct_axis_size(), 2) << "\nMust specify exactly two values "
			<< "to indicate the num and channel axis, and the rest axes should be unidimensional to be considered as the time axis.\n";
		CHECK_NE(ctc_param.nct_axis(0), ctc_param.nct_axis(1))
			<< "\nThe num axis and channel axis cannot be the same axis.\n";
		CHECK_LT(ctc_param.nct_axis(0), 2) << "\nThe num axis must be smaller than 2, "
			<< "for it also indicates the num axis for the label blob, which has only 2 axes.\n";
		break;
	case 3:
		// cannot specify axis order for recurrent data blob with indicator
		CHECK_EQ(ctc_param.nct_axis_size(), 0) << "\nCannot specify axis order while "
			<< "there is an indicator blob, the axis order for recurrent data blob is fixed to be time-num-channel.\n";
		break;
	default:
		LOG(FATAL) << "\nInvalid bottom blob number: " << bottom.size()
			<< "\nThe bottom blobs should be:\n"
			<< "\t(1) data (2) label, or\n"
			<< "\t(1) RNN data (2) RNN indicator (3) RNN label\n";
		break;
	}
}

template <typename Dtype>
void CTCLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
	const Blob<Dtype>* data_blob = bottom[0];
	const Blob<Dtype>* label_blob = (bottom.size() == 2) ? bottom[1] : bottom [2];
	CHECK_EQ(label_blob->num_axes(), 2);
	const int data_num_axes = data_blob->num_axes();
	int n_axis, c_axis, t_axis;
	const CTCParameter &ctc_param = this->layer_param_.ctc_param();
	if (bottom.size() == 3) // the recurrent data
	{
		// check the data_blob shape
		CHECK_EQ(data_num_axes, 3)
			<< "\nThe RNN data blob should have 3 axes.\n";
		// the axis order for recurrent data blob
		// is fixed to be time-num-channel
		n_axis = 1;
		c_axis = 2;
		t_axis = 0;
	}
	else
	{
		// check the data_blob shape
		CHECK(data_num_axes == 3 || data_num_axes == 4)
			<< "\nThe data blob should have 3 or 4 axes.\n";
		if (ctc_param.nct_axis_size() == 0)
		{
			// The axis order defaults
			// to be num-channel-time(s)
			n_axis = 0;
			c_axis = 1;
		}
		else
		{
			n_axis = ctc_param.nct_axis(0);
			c_axis = ctc_param.nct_axis(1);
			//CHECK_LT(n_axis, data_num_axes); // already check n_axis < 2 above
			CHECK_LT(c_axis, data_num_axes);
		}
		t_axis = -1;
		for (int i = 0; i < data_num_axes; ++i)
		{
			if (i == n_axis || i == c_axis)
				continue;
			if (t_axis < 0 || data_blob->shape(t_axis) == 1)
			{
				t_axis = i;
			}
			else
			{
				CHECK(data_blob->shape(i) == 1) << "The height (or width) of the CNN feature maps "
					"should be 1 pixel to convert into a unidimensional sequence. "
					"Invalid height = " << data_blob->shape(t_axis) << ", width = " << data_blob->shape(i);
			}
		} // get t_axis
	}
	N_ = data_blob->shape(n_axis);
	C_ = data_blob->shape(c_axis);
	CHECK_GT(C_, blank_index_);
	CHECK_GE(C_, -blank_index_);
	T_ = data_blob->shape(t_axis);
	// check the label_blob size
	CHECK_EQ(N_, label_blob->shape(n_axis));
	// the other axis of label blob
	L_ = label_blob->shape(1 - n_axis);
	CHECK_GE(T_, L_) << "\nThe label_size = " << L_
		<< " should be less than or equivalent to time_steps = " << T_
		<< ", because the label length cannot exceed the output sequence length.\n";
	// data blob step for each axis
	N_step_ = data_blob->count(n_axis + 1);
	C_step_ = data_blob->count(c_axis + 1);
	T_step_ = data_blob->count(t_axis + 1);
	// label blob step for each axis
	label_N_step_ = label_blob->count(n_axis + 1);
	label_L_step_ = label_blob->count(2 - n_axis);
	// Reshape the blob to be 4-d blob with unnecessary dimensions
	// 'cause the Reshape fuction cannot take a variable number of
	// arguments, while using vector<int> is troublesome.
	// resize data storage blobs for each sequence
	prime_len_blob_.Reshape(N_, 1, 1, 1);
	log_pzx_blob_.Reshape(N_, 1, 1, 1);
	// the max possible capacity
	l_primes_blob_.Reshape(N_, 2 * L_ + 1, 1, 1);
	log_alpha_beta_blob_.Reshape(N_, 2 * L_ + 1, T_, 1);
	if (bottom.size() == 3) // the recurrent data
	{
		const Blob<Dtype>* indi_blob = bottom[1];
		CHECK_EQ(indi_blob->num_axes(), 2);
		CHECK_EQ(T_, indi_blob->shape(0));
		CHECK_EQ(N_, indi_blob->shape(1));
		// only used in CPU mode
		seq_len_ = prime_len_blob_.mutable_cpu_diff();
	}
	else
	{
		seq_len_ = nullptr;
	}
	vector<int> loss_shape(0);  // Loss layers output a scalar, 0 axes.
	top[0]->Reshape(loss_shape);
}

template <typename Dtype>
void CTCLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
										const vector<Blob<Dtype>*>& top)
{
	Dtype &loss = top[0]->mutable_cpu_data()[0];
	if (seq_len_) // the recurrent data
	{
		// calculate the sequence length according to indicator
		const Dtype *const indicator = bottom[1]->cpu_data();
		for (int n = 0; n < N_; ++n)
		{
			// e.g. indicator: 0 1 1 1 1 0 0 0 0 0, T = 10
			//      seq_len = T; // seq_len == 10
			//      while (seq_len > 0 && indicator[--seq_len] == 0);
			//      // break at indicator[4] == 0, therefore seq_len == 4
			//      ++seq_len; // seq_len == 5
			seq_len_[n] = T_;
			while (seq_len_[n] > 0 && indicator[(--seq_len_[n])*N_ + n] == 0);
			++seq_len_[n]; // plus one
		}
		CalculateLoss(loss, bottom[0], bottom[2]);
	}
	else {
		CalculateLoss(loss, bottom[0], bottom[1]);
	}
	// normalize log probabilities by number of parallel batches
	loss /= N_;
}

template <typename Dtype>
void CTCLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
	if (!propagate_down[0])
	{
		LOG(INFO) << "Skip CTC Loss BackwardCPU";
		return;
	}
	Blob<Dtype> *const data_blob = bottom[0];
	CalculateGradient(data_blob);
	// Scale gradient
	Dtype loss_coefficient = 0;
	if (seq_len_) // the recurrent data
	{
		for (int n = 0; n < N_; ++n)
		{
			loss_coefficient += seq_len_[n];
		}
	}
	else {
		loss_coefficient = N_ * T_;
	}
	// loss weight: top[0]->cpu_diff()[0], defalut to be 1.0
	Dtype loss_weight = top[0]->cpu_diff()[0] / loss_coefficient;
	caffe_scal<Dtype>(data_blob->count(), loss_weight, data_blob->mutable_cpu_diff());
}

template <typename Dtype>
void CTCLossLayer<Dtype>::CalculateLoss(
	Dtype &loss,
	Blob<Dtype>* data_blob,
	const Blob<Dtype>* label_blob
	)
{
	// set loss to 0
	loss = 0;
	if (!already_softmax_){
		SequenceSoftmax(data_blob);
	}
	// calculate the modified label sequence for each batch element,
	// and the length of the modified label sequence.
	LabelPrimes(label_blob);

	Dtype *log_pzx = log_pzx_blob_.mutable_cpu_data(); // [N]
	Dtype *log_alpha = log_alpha_beta_blob_.mutable_cpu_data(); // [N, 2L+1, T]
	Dtype *log_beta = log_alpha_beta_blob_.mutable_cpu_diff(); // [N, 2L+1, T]
	const int *l_primes = l_primes_blob_.cpu_data(); // [N, 2L+1]
	const int *prime_len = prime_len_blob_.cpu_data(); // [N]
	const Dtype *probability = data_blob->cpu_data(); // [data_blob_shape]
	for (int n = 0; n < N_; ++n) {
		Dtype *log_alpha_n = log_alpha + log_alpha_beta_blob_.offset(n);
		Dtype *log_beta_n = log_beta + log_alpha_beta_blob_.offset(n);
		const int* l_prime_n = l_primes + l_primes_blob_.offset(n);
		const Dtype *prob_n = probability + n*N_step_;
		const int U = prime_len[n];
		const int seq_len_n = seq_len_ ? seq_len_[n] : T_;
		// start with p(z|x) = 0 before sum
		Dtype& log_pzx_n = log_pzx[n];
		// the output sequence is not long enough to map to label sequence
		if (U > seq_len_n * 2 + 1)
		{
			log_pzx_n = kLogZero;
		}
		else
		{
			CalculateForwardVariables(U, log_alpha_n, prob_n, l_prime_n, seq_len_n);
			CalculateBackwardVariables(U, log_beta_n, prob_n, l_prime_n, seq_len_n);
			// The loss is computed as the log(p(z|x)), p(z|x) is the probability of mapping the
			// the predicted sequence to the target label sequence, which can be calculated by 
			// summing alpha*beta over u\in[0, U) at any time. Do lazy evaluation of log_prob here.
			// [N, 2L+1, T], UT_offset = offset(0, U-1, seq_len-1)
			int UT_offset = (U - 1) * T_ + seq_len_n - 1;
			// Given p(z|x) = sum_{u=0}^{U-1}(alpha*beta)[u, t], \forall t\in[0, seq_len),
			// we choose t = seq_len-1, where beta[u, t] = 0 \forall u < U-2
			// Therefore p(z|x) = (alpha*beta)[U-1, seq_len-1] + (alpha*beta)[U-2, seq_len-1]
			//                  = alpha[U-1, seq_len-1] + alpha[U-2, seq_len-1],
			// for beta[U-1, seq_len-1] = 1 and beta[U-2, seq_len-1] = 1
			log_pzx_n = LogSumExp(log_alpha_n[UT_offset], log_alpha_n[UT_offset - T_]);
		} // or the p(z|x) remains 0
		// use negative loss for display
		loss -= log_pzx_n;
	} // for N
}

template <typename Dtype>
void CTCLossLayer<Dtype>::SequenceSoftmax(Blob<Dtype> *data_blob)
{
	Dtype *data = data_blob->mutable_cpu_data();
	// compute softmax (until sequence length is sufficient)
	for (int n = 0; n < N_; ++n) {
		const int seq_len_n = seq_len_ ? seq_len_[n] : T_;
		for (int t = 0; t < seq_len_n; ++t) {
			Dtype *data_nt = data + n*N_step_ + t*T_step_;
			Dtype max_coeff = data_nt[0];
			const int C_end = C_*C_step_;
			// get max coeff
			for (int i = C_step_; i < C_end; i += C_step_) {
				max_coeff = max(max_coeff, data_nt[i]);
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

template <typename Dtype>
void CTCLossLayer<Dtype>::LabelPrimes(const Blob<Dtype> *label_blob)
{
	const Dtype *const label = label_blob->cpu_data();
	int *const l_primes = l_primes_blob_.mutable_cpu_data(); // [N, 2L+1]
	int *const prime_len = prime_len_blob_.mutable_cpu_data(); // [N]
	// default blank_index = C - 1
	const int blank_index = (blank_index_ < 0) ? C_ + blank_index_ : blank_index_;
	for (int n = 0; n < N_; ++n) {
		// Target indices with blanks before each index and a blank at the end.
		const Dtype *label_nl = label + n*label_N_step_;
		int *const l_prime_n = l_primes + l_primes_blob_.offset(n);
		// calculate l_prime length
		int &U = prime_len[n];
		U = 0;
		// label indicators are negative if the sequence has ended
		for (int l = 0; l < L_; ++l) {
			if (*label_nl < 0)
			{
				break;
			}
			int i_label = static_cast<int>(*label_nl + 0.5);  // integer label (round)
			if (i_label < 0 || i_label >= C_ || i_label == blank_index) { // illegal index or blank_index (C - 1)
				// saw an invalid sequence with non-null following null labels.
				LOG(FATAL) << "The " << l << "-th label element " << i_label
						<< " is illegal while the number of classes = " << C_
						<< ", sequence number " << n;
			}
			l_prime_n[U++] = blank_index;
			l_prime_n[U++] = i_label;
			label_nl += label_L_step_;
		}
		l_prime_n[U++] = blank_index;
	}
}

template <typename Dtype>
void CTCLossLayer<Dtype>::CalculateForwardVariables(
	const int U,
	Dtype *const log_alpha_n,	// [N, 2L+1, T]
	const Dtype *const prob_n,	// [data_blob_shape]
	const int *const l_prime_n,	// [N, 2L+1]
	const int seq_len_n			// [N]
	)
{
	// default blank_index = C - 1
	const int blank_index = (blank_index_ < 0) ? C_ + blank_index_ : blank_index_;
	caffe_set<Dtype>(U * T_, kLogZero, log_alpha_n);
	// Calculate forward variables
	// Initialize alpha values in Graves Eq (7.5) and Eq (7.6).
	log_alpha_n[0] // log_alpha[0, 0]
		= log(prob_n[blank_index * C_step_]); // l_prime_n[0] must be blank
	// Below, l_prime_n[1] == label[0]
	if (U > 1)
	{
		log_alpha_n[T_] // log_alpha[1, 0]
			= log(prob_n[l_prime_n[1] * C_step_]);
	}
	for (int t = 1; t < seq_len_n; ++t) {
		// If there is not enough time steps for the sequence to map to
		// the previous or remaining labels, leave log_alpha[u, t] continue
		// to be kLogZero.
		for (int u = max(0, U - (2 * (seq_len_n - t)));
			u < min(U, 2 * (t + 1));
			++u) {
			Dtype &log_alpha_ut = log_alpha_n[u * T_ + t]; // log_alpha[u, t]
			// Begin Graves Eq (7.9)
			// Add in the u, t - 1 term.
			log_alpha_ut = log_alpha_n[u * T_ + t - 1]; // log_alpha[u, t-1]
			// Add in the u - 1, t - 1 term.
			if (u > 0) {
				log_alpha_ut = LogSumExp(log_alpha_ut,
					log_alpha_n[(u - 1) * T_ + t - 1]); // log_alpha[u-1, t-1]
			}
			// Add in the u - 2, t - 1 term if l_prime[u] != blank or l_prime[u-2].
			if (u > 1 && l_prime_n[u] != blank_index && l_prime_n[u] != l_prime_n[u - 2]) {
				log_alpha_ut = LogSumExp(log_alpha_ut,
					log_alpha_n[(u - 2) * T_ + t - 1]); // log_alpha[u-2, t-1]
			}
			// Multiply the summed alphas with the activation log probability.
			log_alpha_ut += log(prob_n[l_prime_n[u] * C_step_ + t * T_step_]);
		}   // End Graves Eq (7.9)
	}
}

template <typename Dtype>
void CTCLossLayer<Dtype>::CalculateBackwardVariables(
	const int U,
	Dtype *const log_beta_n,	// [N, 2L+1, T]
	const Dtype *const prob_n,	// [data_blob_shape]
	const int *const l_prime_n,	// [N, 2L+1]
	const int seq_len_n			// [N]
	)
{
	// default blank_index = C - 1
	const int blank_index = (blank_index_ < 0) ? C_ + blank_index_ : blank_index_;
	caffe_set<Dtype>(U * T_, kLogZero, log_beta_n);
	// Calculate backward varibles
	// Initial beta blaues in Graves Eq (7.13): log of probability 1.
	for (int u = U - 2; u < U; ++u) { // U-2 <= u <= U-1
		log_beta_n[(u + 1) * T_ - 1] = 0; // log_beta[u, T-1]
	}
	for (int t = seq_len_n - 2; t >= 0; --t) {
		// If there is not enough time steps for the sequence to map to
		// the previous or remaining labels, leave log_beta[u, t] continue
		// to be kLogZero.
		for (int u = max(0, U - (2 * (seq_len_n - t)));
			u < min(U, 2 * (t + 1));
			++u) {
			Dtype &log_beta_ut = log_beta_n[u * T_ + t]; // log_beta[u, t]
			// Begin Graves Eq (7.15)
			// Add in the u, t + 1 term.
			log_beta_ut = log_beta_n[u * T_ + t + 1] + // log_beta[u, t+1]
				log(prob_n[l_prime_n[u] * C_step_ + (t + 1) * T_step_]); // prob[l_prime_n[u], t+1]
			// Add in the u + 1, t + 1 term.
			if (u + 1 < U) {
				log_beta_ut = LogSumExp(log_beta_ut, 
					log_beta_n[(u + 1) * T_ + t + 1] + // log_beta[u+1, t+1]
					log(prob_n[l_prime_n[u + 1] * C_step_ + (t + 1) * T_step_])); // prob[l_prime_n[u+1], t+1]
			}
			// Add in the u + 2, t + 1 term if l_prime_n[u] != blank or l_prime_n[u+2]
			if (u + 2 < U && l_prime_n[u] != blank_index && l_prime_n[u] != l_prime_n[u + 2]) {
				// Add in u + 2 term.
				log_beta_ut = LogSumExp(log_beta_ut, 
					log_beta_n[(u + 2) * T_ + t + 1] + // log_beta[u+2, t+1]
					log(prob_n[l_prime_n[u + 2] * C_step_ + (t + 1) * T_step_])); // prob[l_prime_n[u+2], t+1]
			}
		}   // End Graves Eq. (7.15)
	}
}

string output_primes(int prime_len, const int *l_prime, int blank_index)
{
	stringstream ss;
	for (int i = 0; i < prime_len; ++i) {
		if (l_prime[i] == blank_index)
			ss << ' ';
		else
			ss << l_prime[i];
	}
	return ss.str();
}

template <typename Dtype>
void CTCLossLayer<Dtype>::CalculateGradient(Blob<Dtype> *data_blob)
{
	const int *const prime_len = prime_len_blob_.cpu_data();
	const int *const l_primes = l_primes_blob_.cpu_data();
	const Dtype *const log_pzx = log_pzx_blob_.cpu_data();
	const Dtype *const log_alpha = log_alpha_beta_blob_.cpu_data();
	const Dtype *const log_beta = log_alpha_beta_blob_.cpu_diff();
	const Dtype *const predict_prob = data_blob->cpu_data();
	Dtype *const data_diff = data_blob->mutable_cpu_diff();
	// the data_diff is first used as log_prob_sum
	// before the final gradient calculation
	caffe_set<Dtype>(data_blob->count(), kLogZero, data_diff);
	for (int n = 0; n < N_; ++n) {
		const int U = prime_len[n];
		const int *const l_prime_n = l_primes + l_primes_blob_.offset(n);
		const Dtype log_pzx_n = log_pzx[n];
		const int seq_len_n = seq_len_ ? seq_len_[n] : T_;
		// It is possible that no valid path is found if the probability
		// of mapping data sequence to label are zero.
		if (log_pzx_n == kLogZero) {
			// default blank_index = C - 1
			const int blank_index = (blank_index_ < 0) ? C_ + blank_index_ : blank_index_;
			LOG(WARNING) << "No valid path for labels ["
				<< output_primes(U, l_prime_n, blank_index) << "]\n";
			// set the target probability of each channel to zero
			if (already_softmax_) {
				// gradient = -target_prob / prob = 0
				Dtype *const data_diff_n = data_diff + n*N_step_;
				for (int t = 0; t < seq_len_n; ++t)
				{
					Dtype *data_diff_nct = data_diff_n + t*T_step_;
					for (int c = 0; c < C_; ++c, data_diff_nct += C_step_)
					{
						*data_diff_nct = 0;
					}
				}
			}
			else {
				// gradient = prob - target_prob = prob
				const int n_offset = n*N_step_;
				for (int t = 0; t < seq_len_n; ++t)
				{
					int nct_offset = n_offset + t*T_step_;
					for (int c = 0; c < C_; ++c, nct_offset += C_step_)
					{
						data_diff[nct_offset] = predict_prob[nct_offset];
					}
				}
			}
			continue;
		}
		for (int t = 0; t < seq_len_n; ++t)
		{
			const Dtype *const log_alpha_nt = log_alpha + log_alpha_beta_blob_.offset(n, 0, t);
			const Dtype *const log_beta_nt = log_beta + log_alpha_beta_blob_.offset(n, 0, t);
			int nct_offset = n*N_step_ + t*T_step_;
			// calculate target_prob(label, t) = sum_{l_prime(u) == label}{target_prob(u, t)},
			// where target_prob(u, t) = exp(log_alpha(u,t) + log_beta(u,t) - log_pzx)
			for (int u = 0; u < U; ++u) {
				Dtype &log_prob_sum_ntl= data_diff[nct_offset + l_prime_n[u]*C_step_];
				log_prob_sum_ntl = LogSumExp(log_prob_sum_ntl,
					log_alpha_nt[u * T_] + log_beta_nt[u * T_]);
			}
			if (already_softmax_){
				// gradient = -target_prob / predict_prob
				for (int offset_end = nct_offset + C_*C_step_; 
					nct_offset < offset_end; nct_offset += C_step_)
				{
					Dtype &diff_nct = data_diff[nct_offset];
					if (diff_nct == kLogZero){
						diff_nct = 0;
					}
					else if (diff_nct >= log_pzx_n)
					{
						// where precision allowed
						diff_nct = -1 / predict_prob[nct_offset];
					}
					else
					{
						diff_nct = -exp(diff_nct - log_pzx_n) / predict_prob[nct_offset];
					}
				} // loop C
			}
			else{
				// gradient = predict_prob - target_prob
				for (int offset_end = nct_offset + C_*C_step_; 
					nct_offset < offset_end; nct_offset += C_step_)
				{
					Dtype &diff_nct = data_diff[nct_offset];
					if (diff_nct == kLogZero)
					{
						diff_nct = predict_prob[nct_offset];
					}
					else if (diff_nct >= log_pzx_n)
					{
						// where precision allowed
						diff_nct = predict_prob[nct_offset] - 1;
					}
					else
					{
						diff_nct = predict_prob[nct_offset] - exp(diff_nct - log_pzx_n);
					}
				} // loop C
			} // already softmax normalized or not
		} // t < seq_len_n
	} // n < N_
}

INSTANTIATE_CLASS(CTCLossLayer);
REGISTER_LAYER_CLASS(CTCLoss);

}  // namespace caffe
