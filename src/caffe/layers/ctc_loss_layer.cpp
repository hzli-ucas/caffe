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
	blank_index_(param.ctc_param().blank_index())
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
	const int data_num_axes = bottom[0]->num_axes();
	switch (bottom.size())
	{
	case 2:
		// check the data_blob shape
		CHECK(data_num_axes == 3 || data_num_axes == 4)
			<< "\nThe data blob should have 3 or 4 axes.\n";
		if (ctc_param.nct_axis_size() == 0)
		{
			// The axis order defaults to be [N, C, T, 1] or [N, C, 1, T]
			N_axis_ = 0;
			C_axis_ = 1;
		}
		else
		{
			CHECK_EQ(ctc_param.nct_axis_size(), 2) << "\nMust specify exactly two values "
				<< "to indicate the num and channel axis, and the rest axes should be unidimensional to be considered as the time axis.\n";
			N_axis_ = ctc_param.nct_axis(0);
			C_axis_ = ctc_param.nct_axis(1);
			CHECK_NE(N_axis_, C_axis_)
				<< "\nThe num axis and channel axis cannot be the same axis.\n";
			CHECK_LT(N_axis_, 2) << "\nThe num axis must be smaller than 2, "
				<< "for it also indicates the num axis for the label blob, which has only 2 axes.\n";
			CHECK_LT(C_axis_, data_num_axes);
		}
		T_axis_ = -1;
		for (int i = 0; i < data_num_axes; ++i)
		{
			if (i == N_axis_ || i == C_axis_)
				continue;
			if (T_axis_ < 0 || bottom[0]->shape(T_axis_) == 1)
			{
				T_axis_ = i;
			}
			else
			{
				CHECK_EQ(bottom[0]->shape(i), 1) << "The height (or width) of the CNN feature maps "
					"should be 1 pixel to convert into a unidimensional sequence. "
					"Invalid height = " << bottom[0]->shape(T_axis_) << ", width = " << bottom[0]->shape(i);
			}
		} // get t_axis
		break;
	case 3:
		// cannot specify axis order for recurrent data blob with indicator
		CHECK_EQ(ctc_param.nct_axis_size(), 0) << "\nCannot specify axis order while "
			<< "there is an indicator blob, the axis order for recurrent data blob is fixed to be time-num-channel.\n";
		// check the data_blob shape
		CHECK_EQ(data_num_axes, 3)
			<< "\nThe RNN data blob should have 3 axes [T, N, C].\n";
		// the axis order for recurrent data blob
		// is fixed to be time-num-channel
		N_axis_ = 1;
		C_axis_ = 2;
		T_axis_ = 0;
		break;
	default:
		LOG(FATAL) << "\nInvalid bottom blob number: " << bottom.size()
			<< "\nThe bottom blobs should be:\n"
			<< "\t(1) data (2) label, or\n"
			<< "\t(1) RNN data (2) RNN indicator (3) RNN label\n";
		break;
	}

	int param_size = this->layer_param_.ctc_param().gamma_size();
	for (int i = 0; i < param_size; ++i)
		gammas_.push_back(this->layer_param_.ctc_param().gamma(i));
	param_size = this->layer_param_.ctc_param().alpha_size();
	for (int i = 0; i < param_size; ++i) {
		alphas_.push_back(this->layer_param_.ctc_param().alpha(i));
		CHECK(alphas_.back() >= 0 && alphas_.back() <= 1);
	}
	param_size = this->layer_param_.ctc_param().stepvalue_size();
	if (param_size > 0) {
		CHECK(gammas_.size() <= 1 || gammas_.size() - 1 == param_size);
		CHECK(alphas_.size() <= 1 || alphas_.size() - 1 == param_size);
		stepvalues_.push_back(this->layer_param_.ctc_param().stepvalue(0));
		CHECK_GT(stepvalues_[0], 0);
		for (int i = 1; i < param_size; ++i) {
			stepvalues_.push_back(this->layer_param_.ctc_param().stepvalue(i));
			CHECK_GT(stepvalues_[i], stepvalues_[i - 1]);
		}
		tmp_iter_ = 0;
	}
	if (!alphas_.empty() || this->layer_param_.ctc_param().label_width())
		label_count_.Reshape({ bottom[0]->shape(C_axis_) });
}

template <typename Dtype>
void CTCLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
	const Blob<Dtype>* data_blob = bottom[0];
	N_ = data_blob->shape(N_axis_);
	C_ = data_blob->shape(C_axis_);
	CHECK_GT(C_, blank_index_);
	CHECK_GE(C_, -blank_index_);
	T_ = data_blob->shape(T_axis_);
	// data blob step for each axis
	N_step_ = data_blob->count(N_axis_ + 1);
	C_step_ = data_blob->count(C_axis_ + 1);
	T_step_ = data_blob->count(T_axis_ + 1);

	// check the label_blob size
	const Blob<Dtype>* label_blob = (bottom.size() == 2) ? bottom[1] : bottom [2];
	CHECK_EQ(label_blob->num_axes(), 2);
	CHECK_EQ(N_, label_blob->shape(N_axis_));
	L_ = label_blob->shape(1 - N_axis_);
	CHECK_GE(T_, L_) << "\nThe label_size = " << L_
		<< " should be less than or equivalent to time_steps = " << T_
		<< ", because the label length cannot exceed the output sequence length.\n";
	// label blob step for each axis
	label_N_step_ = label_blob->count(N_axis_ + 1);
	label_L_step_ = label_blob->count(2 - N_axis_);

	if (bottom.size() == 3) // the recurrent data
	{
		const Blob<Dtype>* indi_blob = bottom[1];
		CHECK_EQ(indi_blob->num_axes(), 2);
		CHECK_EQ(T_, indi_blob->shape(0));
		CHECK_EQ(N_, indi_blob->shape(1));
	}

	// resize data storage blobs for each sequence
	prob_.ReshapeLike(*bottom[0]);
	prime_len_blob_.Reshape({ N_ });
	log_pzx_blob_.Reshape({ N_ });
	// the max possible capacity
	l_primes_blob_.Reshape({ N_, 2 * L_ + 1 });
	log_alpha_beta_blob_.Reshape({ N_, 2 * L_ + 1, T_ });
	if (label_count_.count() && label_count_.count() != C_)
		label_count_.Reshape({ C_ });
	vector<int> loss_shape(0);  // Loss layers output a scalar, 0 axes.
	top[0]->Reshape(loss_shape);
}

template <typename Dtype>
void CTCLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
										const vector<Blob<Dtype>*>& top)
{
	int *seq_len = prime_len_blob_.mutable_cpu_diff();
	if (bottom.size() == 2)
	{
		caffe_set(N_, T_, seq_len);
	}
	else // the recurrent data
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
			seq_len[n] = 0;
			while (seq_len[n] < T_ && indicator[(++seq_len[n])*N_ + n]);
		}
	}
	const Dtype *label = (bottom.size() == 2) ? bottom[1]->cpu_data() : bottom[2]->cpu_data();
	Dtype *predict_prob = prob_.mutable_cpu_data();
	Dtype *target_prob = prob_.mutable_cpu_diff();
	SequenceSoftmax(bottom[0]->cpu_data(), seq_len, predict_prob);
	GetTarget_cpu(predict_prob, label, target_prob);
	
	// normalize log probabilities by number of parallel batches
	top[0]->mutable_cpu_data()[0] =
		caffe_cpu_asum<Dtype>(N_, log_pzx_blob_.cpu_data())  / N_;
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
	
	caffe_sub(prob_.count(), prob_.cpu_data(), prob_.cpu_diff(), bottom[0]->mutable_cpu_diff());
	// loss weight: top[0]->cpu_diff()[0], defalut to be 1.0
	Dtype loss_weight = top[0]->cpu_diff()[0] / N_;
	caffe_scal<Dtype>(prob_.count(), loss_weight, bottom[0]->mutable_cpu_diff());
}

std::string output_primes(int prime_len, const int *l_prime, int blank_index)
{
	std::stringstream ss;
	for (int i = 0; i < prime_len; ++i) {
		if (l_prime[i] == blank_index)
			ss << ' ';
		else
			ss << l_prime[i];
	}
	return ss.str();
}

template <typename Dtype>
void CTCLossLayer<Dtype>::GetTarget_cpu(const Dtype *predict_prob, const Dtype *label, Dtype *target_prob)
{
	const int *seq_len = prime_len_blob_.cpu_diff();				// [N]
	int *l_primes = l_primes_blob_.mutable_cpu_data();			// [N, 2L+1]
	int *prime_len = prime_len_blob_.mutable_cpu_data();		// [N]
	Dtype *log_alpha = log_alpha_beta_blob_.mutable_cpu_data();	// [N, 2L+1, T]
	Dtype *log_beta = log_alpha_beta_blob_.mutable_cpu_diff();	// [N, 2L+1, T]
	Dtype *log_pzx = log_pzx_blob_.mutable_cpu_data();			// [N]
	// default blank_index = C - 1
	const int blank_index = (blank_index_ < 0) ? C_ + blank_index_ : blank_index_;

	caffe_set<Dtype>(log_alpha_beta_blob_.count(), kLogZero, log_alpha);
	caffe_set<Dtype>(log_alpha_beta_blob_.count(), kLogZero, log_beta);
	caffe_set<Dtype>(prob_.count(), kLogZero, target_prob);
	for (int n = 0; n < N_; ++n) {
		// Calculate the modified label sequence for each batch element,
		// and the length of the modified label sequence.
		// Target indices with blanks before each index and a blank at the end.
		const Dtype *label_nl = label + n * label_N_step_;
		int *l_prime_n = l_primes + l_primes_blob_.offset(n);
		// calculate l_prime length
		int &U = prime_len[n];
		U = 0;
		// label indicators are negative if the sequence has ended
		for (int l = 0; l < L_; ++l) {
			if (*label_nl < 0)
			{
				break;
			}
			const int i_label = static_cast<int>(*label_nl + 0.5);  // integer label (round)
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

		Dtype *log_alpha_n = log_alpha + log_alpha_beta_blob_.offset(n);
		Dtype *log_beta_n = log_beta + log_alpha_beta_blob_.offset(n);
		const Dtype *prob_n = predict_prob + n * N_step_;
		Dtype *target_n = target_prob + n * N_step_;
		const int seq_len_n = seq_len[n];
		// the output sequence is long enough to map to label sequence or not
		if (U > seq_len_n * 2 + 1)
		{
			log_pzx[n] = kLogZero;
			LOG(WARNING) << "No valid path for label sequence: ["
				<< output_primes(U, l_prime_n, blank_index) << "]\n";
			// set the target probability of each channel to zero
			for (int t = 0; t < seq_len_n; ++t)
			{
				Dtype *target_nt = target_n + t * T_step_;
				for (int i = C_step_ * (C_ - 1); i >= 0; i -= C_step_)
				{
					target_nt[i] = 0;
				}
			}
			continue;
		}
		
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
		// Calculate backward varibles
		// Initial beta blaues in Graves Eq (7.13): log of probability 1.
		for (int u = U - 2; u < U; ++u) { // U-2 <= u <= U-1
			log_beta_n[u * T_ + seq_len_n - 1] = 0; // log_beta[u, T-1]
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
		log_pzx[n] = LogSumExp(log_alpha_n[UT_offset], log_alpha_n[UT_offset - T_]);
		
		for (int t = 0; t < seq_len_n; ++t)
		{
			const Dtype *log_alpha_nt = log_alpha_n + t;
			const Dtype *log_beta_nt = log_beta_n + t;
			Dtype *target_nt = target_n + t * T_step_;
			// calculate target_prob(label, t) = sum_{l_prime(u) == label}{target_prob(u, t)},
			// where target_prob(u, t) = exp(log_alpha(u,t) + log_beta(u,t) - log_pzx)
			for (int u = 0; u < U; ++u) {
				Dtype &log_prob_sum_ntl = target_nt[l_prime_n[u] * C_step_];
				log_prob_sum_ntl = LogSumExp(log_prob_sum_ntl,
					log_alpha_nt[u * T_] + log_beta_nt[u * T_]);
			}
			for (int i = (C_ - 1) * C_step_; i >= 0; i -= C_step_)
			{
				if (target_nt[i] == kLogZero)
				{
					target_nt[i] = 0;
				}
				else if (target_nt[i] >= log_pzx[n])
				{
					target_nt[i] = 1;
				}
				else
				{
					target_nt[i] = exp(target_nt[i] - log_pzx[n]);
				}
			} // loop C
		} // t < seq_len_n
	} // for N
}

template <typename Dtype>
void CTCLossLayer<Dtype>::SequenceSoftmax(const Dtype *data, const int *seq_len, Dtype *probability)
{
	// compute softmax (until sequence length is sufficient)
	for (int n = 0; n < N_; ++n) {
		const int seq_len_n = seq_len[n];
		for (int t = 0; t < seq_len_n; ++t) {
			const int nt_offset = n * N_step_ + t * T_step_;
			const Dtype *data_nt = data + nt_offset;
			Dtype *prob_nt = probability + nt_offset;
			Dtype max_coeff = data_nt[0];
			const int C_end = C_*C_step_;
			// get max coeff
			for (int i = C_step_; i < C_end; i += C_step_) {
				max_coeff = max(max_coeff, data_nt[i]);
			}
			// calc exp and its sum
			Dtype sum = 0;
			for (int i = 0; i < C_end; i += C_step_) {
				prob_nt[i] = exp(data_nt[i] - max_coeff);
				sum += prob_nt[i];
			}
			// division by sum
			for (int i = 0; i < C_end; i += C_step_) {
				prob_nt[i] /= sum;
			}
		}
	}
}

INSTANTIATE_CLASS(CTCLossLayer);
REGISTER_LAYER_CLASS(CTCLoss);

}  // namespace caffe
