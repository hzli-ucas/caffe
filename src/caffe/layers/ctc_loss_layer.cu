#include "caffe/layers/ctc_loss_layer.hpp"
#include <iostream>
#include <iomanip>

namespace caffe {

// negtive infinity
#define kLogZero -INFINITY 

/**
* @brief Adds two probabilities in log.
* @returns log_(prob_1 + prob_2)
*/
template <typename Dtype>
__device__ Dtype LogSumExp_kernel(Dtype log_prob_1, Dtype log_prob_2) {
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
__global__ void CTCSequenceSoftmaxGPU(
	const int nthreads			// T * N
	, const int N, const int C
	, const int N_step, const int C_step, const int T_step
	, const Dtype *data			// data blob shape
	, const Dtype *seq_len		// [N]
	, Dtype *probability		// data blob shape
)
{
	const int C_end = C * C_step;
	// compute softmax (until each sequence ends)
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int t = index / N;
		const int n = index % N;
		if (t >= static_cast<int>(seq_len[n] + 0.5))
		{
			continue;
		}
		const int nt_offset = n * N_step + t * T_step;
		const Dtype *data_nt = data + nt_offset;
		Dtype *prob_nt = probability + nt_offset;
		Dtype max_coeff = data_nt[0];
		// get max coeff
		for (int i = C_step; i < C_end; i += C_step) {
			max_coeff = max(max_coeff, data_nt[i]);
		}
		// calc exp and its sum
		Dtype sum = 0;
		for (int i = 0; i < C_end; i += C_step) {
			prob_nt[i] = exp(data_nt[i] - max_coeff);
			sum += prob_nt[i];
		}
		// division by sum
		for (int i = 0; i < C_end; i += C_step) {
			prob_nt[i] /= sum;
		}
	}
}

template <typename Dtype>
__global__ void CTCCalculateLossGPU(
	const int N, const int C, const int T
	, const int N_step, const int C_step, const int T_step
	, const Dtype *probability		// data blob shape
	, const Dtype *seq_len			// [N]
	, const int L, const int label_N_step, const int label_L_step
	, const int blank_index
	, const Dtype *label			// [N, L] or [L, N], decided by the n-axis
	, int *l_primes					// [N, 2L+1]
	, int *prime_len				// [N]
	, Dtype *log_alpha				// [N, 2L+1, T], initialized to kLogZero
	, Dtype *log_beta				// [N, 2L+1, T], initialized to kLogZero
	, Dtype *log_pzx				// [N]
	)
{
	CUDA_KERNEL_LOOP(n, N) {
		// Insert a blank before each element and a blank at the end.
		const Dtype *label_nl = label + n * label_N_step;
		int *l_prime_n = l_primes + n * (2 * L + 1);
		// Calculate l_prime length
		int &U = prime_len[n];
		U = 0;
		for (int l = 0; l < L; ++l) {
			if (*label_nl < 0) // label indicators are negative if the sequence has ended
			{
				break;
			}
			int i_label = static_cast<int>(*label_nl + 0.5);  // integer label (round)
			if (i_label < 0 || i_label >= C || i_label == blank_index) { // illegal index or blank_index
				break; // treate the illegal label as an end indicator
			}
			l_prime_n[U++] = blank_index;
			l_prime_n[U++] = i_label;
			label_nl += label_L_step;
		}
		l_prime_n[U++] = blank_index;
		// get the sequence length according to the indicator
		const int seq_len_n = static_cast<int>(seq_len[n] + 0.5);
		// although T >= L, it is possible that seq_len < label_len for RNN data,
		// in which case, set p(z|x) = 0 and skip the calculation of edit distance
		if (seq_len_n * 2 + 1 < U)
		{
			log_pzx[n] = kLogZero;
			continue;
		}
		// [N, 2L+1, T] alpha and beta at offset(n, 0, 0)
		Dtype *log_alpha_n = log_alpha + n * (2 * L + 1) * T;
		Dtype *log_beta_n = log_beta + n * (2 * L + 1) * T;
		// data blob shape, probability at offset n*N_step
		const Dtype *prob_n = probability + n * N_step;
		// Calculate forward variables
		// Initialize alpha values in Graves Eq (7.5) and Eq (7.6).
		log_alpha_n[0] // log_alpha[0, 0]
			= log(prob_n[blank_index * C_step]);
		// l_prime[0] == blank, l_prime[1] == label[0]
		if (U > 1)
		{
			log_alpha_n[T] // log_alpha[1, 0]
				= log(prob_n[l_prime_n[1] * C_step]);
		}
		for (int t = 1; t < seq_len_n; ++t) {
			// If there is not enough time steps for the sequence to map to
			// the previous or remaining labels, leave log_alpha[u, t] continue
			// to be kLogZero.
			for (int u = max(0, U - (2 * (seq_len_n - t)));
				u < min(U, 2 * (t + 1));
				++u) {
				Dtype &log_alpha_ut = log_alpha_n[u * T + t]; // log_alpha[u, t]
				// Begin Graves Eq (7.9)
				// Add in the u, t - 1 term.
				log_alpha_ut = log_alpha_n[u * T + t - 1]; // log_alpha[u, t-1]
				// Add in the u - 1, t - 1 term.
				if (u > 0) {
					log_alpha_ut = LogSumExp_kernel(log_alpha_ut,
						log_alpha_n[(u - 1) * T + t - 1]); // log_alpha[u-1, t-1]
				}
				// Add in the u - 2, t - 1 term if l_prime[u] != blank or l_prime[u-2].
				if (u > 1 && l_prime_n[u] != blank_index && l_prime_n[u] != l_prime_n[u - 2]) {
					log_alpha_ut = LogSumExp_kernel(log_alpha_ut,
						log_alpha_n[(u - 2) * T + t - 1]); // log_alpha[u-2, t-1]
				}
				// Multiply the summed alphas with the activation log probability.
				log_alpha_ut += log(prob_n[l_prime_n[u] * C_step + t * T_step]);
			}   // End Graves Eq (7.9)
		}
		// Calculate backward varibles
		// Initial beta blaues in Graves Eq (7.13): log of probability 1.
		for (int u = U - 2; u < U; ++u) {
			log_beta_n[u * T + seq_len_n - 1] = 0; // log_beta[u, T-1]
		}
		for (int t = seq_len_n - 2; t >= 0; --t) {
			// If there is not enough time steps for the sequence to map to
			// the previous or remaining labels, leave log_beta[u, t] continue
			// to be kLogZero.
			for (int u = max(0, U - (2 * (seq_len_n - t)));
				u < min(U, 2 * (t + 1));
				++u) {
				Dtype &log_beta_ut = log_beta_n[u * T + t];
				// Begin Graves Eq (7.15)
				// Add in the u, t + 1 term.
				log_beta_ut = log_beta_n[u * T + t + 1] + // log_beta[u, t+1]
					log(prob_n[l_prime_n[u] * C_step + (t + 1) * T_step]); // prob[l_prime_n[u], t+1]
				// Add in the u + 1, t + 1 term.
				if (u + 1 < U) {
					log_beta_ut = LogSumExp_kernel(log_beta_ut, 
						log_beta_n[(u + 1) * T + t + 1] + // log_beta[u+1, t+1]
						log(prob_n[l_prime_n[u + 1] * C_step + (t + 1) * T_step])); // prob[l_prime_n[u+1], t+1]
				}

				// Add in the u + 2, t + 1 term if l_prime[u] != blank or l_prime[u+2]
				if (u + 2 < U && l_prime_n[u] != blank_index && l_prime_n[u] != l_prime_n[u + 2]) {
					// Add in u + 2 term.
					log_beta_ut = LogSumExp_kernel(log_beta_ut, 
						log_beta_n[(u + 2) * T + t + 1] + // log_beta[u+2, t+1]
						log(prob_n[l_prime_n[u + 2] * C_step + (t + 1) * T_step])); // prob[l_prime_n[u+2], t+1]
				}
			}  // End Graves Eq. (7.15)
		}
		// The loss is computed as the log(p(z|x)), p(z|x) is the probability of mapping the
		// the predicted sequence to the target label sequence, which can be calculated by 
		// summing alpha*beta over u\in[0, U) at any time. Do lazy evaluation of log_prob here.
		// [N, 2L+1, T], UT_offset = offset(0, U-1, seq_len-1)
		int UT_offset = (U - 1) * T + seq_len_n - 1;
		// Given p(z|x) = sum_{u=0}^{U-1}(alpha*beta)[u, t], \forall t\in[0, seq_len),
		// we choose t = seq_len-1, where beta[u, t] = 0 \forall u < U-2
		// Therefore p(z|x) = (alpha*beta)[U-1, seq_len-1] + (alpha*beta)[U-2, seq_len-1]
		//                  = alpha[U-1, seq_len-1] + alpha[U-2, seq_len-1],
		// for beta[U-1, seq_len-1] = 1 and beta[U-2, seq_len-1] = 1
		log_pzx[n] = LogSumExp_kernel(log_alpha_n[UT_offset], log_alpha_n[UT_offset - T]);
	} // CUDA_KERNEL_LOOP(n, N)
}

template <typename Dtype>
__global__ void CTCLogSumProbGPU(
	const int nthreads				// N * T
	, const int T, const int Lm2p1	// 2 * L + 1
	, const int N_step, const int C_step, const int T_step
	, const Dtype *seq_len		// [N]
	, const int *prime_len		// [N]
	, const int *l_primes		// [N, 2L+1]
	, const Dtype *log_alpha	// [N, 2L+1, T]
	, const Dtype *log_beta		// [N, 2L+1, T]
	, const Dtype *log_pzx		// [N]
	, Dtype *log_sum_prob		// data blob shape, initialized with kLogZero
)
{
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int n = index / T;
		// It is possible that no valid path is found
		// if the activations for the targets are zero.
		if (log_pzx[n] == kLogZero) {
			// set the target probability of each dimension to zero
			// in this case, let it be the default kLogZero
			continue;
		}
		const int t = index % T;
		if (t >= static_cast<int>(seq_len[n] + 0.5))
		{
			continue;
		}
		const int U = prime_len[n];
		const int *const l_prime_n = l_primes + n * Lm2p1;
		const Dtype *const log_alpha_nt = log_alpha + n * Lm2p1 * T + t;
		const Dtype *const log_beta_nt = log_beta + n * Lm2p1 * T + t;
		// Calculate log_prob_sum
		//     = LogSum_{l_prime(u) == label}{log_alpha(u,t) + log_beta(u,t)}
		//     = log(sum_{l_prime(u) == label}{(alpha*beta)[u,t] / p(z|x) * p(z|x)})
		//     = log(target_prob(label, t)) + log_pzx
		Dtype *const log_sum_prob_nt = log_sum_prob + n * N_step + t * T_step;
		for (int u = 0; u < U; ++u) {
			Dtype &log_sum_prob_nct = log_sum_prob_nt[l_prime_n[u] * C_step];
			log_sum_prob_nct = LogSumExp_kernel(log_sum_prob_nct,
				log_alpha_nt[u * T] + log_beta_nt[u * T]);
		}
		// the operation target_prob = exp(log_prob_sum - log_pzx)
		// will be N*C*T-parallelly performed in function TargetProbGPU
	}
}

template <typename Dtype>
__global__ void CTCTargetProbGPU(
	const int nthreads // N * C * T
	, const int N, const int N_step
	, const Dtype *log_pzx			// [N]
	, Dtype *target_prob			// data blob shape, still log_prob_sum now
)
{
	CUDA_KERNEL_LOOP(index, nthreads) {
		// if log_pzx_n == kLogZero or the sequence ends already
		// there must be target_prob_nct == kLogZero
		Dtype &prob_nct = target_prob[index];
		if (prob_nct == kLogZero) {
			prob_nct = 0;
			continue;
		}
		const int n = (index % (N_step * N)) / N_step;
		const Dtype log_pzx_n = log_pzx[n];
		// log_pzx_n cannot be kLogZero then
		if (prob_nct >= log_pzx_n) {
			prob_nct = 1;
		}
		else {
			prob_nct = exp(prob_nct - log_pzx_n);
		}
	}
}

template <typename Dtype>
void CTCLossLayer<Dtype>::GetTarget_gpu(const Dtype *predict_prob,
	const Dtype *label, Dtype *target_prob)
{
	const int nt_count = N_ * T_;
	const int nct_count = nt_count * C_;
	// default blank_index = C - 1
	const int blank_index = (blank_index_ < 0) ? C_ + blank_index_ : blank_index_;
	Dtype *log_alpha = log_alpha_beta_blob_.mutable_gpu_data();
	Dtype *log_beta = log_alpha_beta_blob_.mutable_gpu_diff();
	// Initialize the log_alpha and log_beta blob with kLogZero
	caffe_gpu_set<Dtype>(log_alpha_beta_blob_.count(), kLogZero, log_alpha);
	caffe_gpu_set<Dtype>(log_alpha_beta_blob_.count(), kLogZero, log_beta);
	CTCCalculateLossGPU<Dtype><<<CAFFE_GET_BLOCKS(N_), CAFFE_CUDA_NUM_THREADS>>>(
		N_, C_, T_, N_step_, C_step_, T_step_, predict_prob,
		log_pzx_blob_.gpu_diff(), // length of each sequence
		L_, label_N_step_, label_L_step_, blank_index, label,
		l_primes_blob_.mutable_gpu_data(), prime_len_blob_.mutable_gpu_data(),
		log_alpha, log_beta, log_pzx_blob_.mutable_gpu_data());
	caffe_gpu_set<Dtype>(nct_count, kLogZero, target_prob);
	CTCLogSumProbGPU<Dtype><<<CAFFE_GET_BLOCKS(nt_count), CAFFE_CUDA_NUM_THREADS>>>(
		nt_count, T_, 2 * L_ + 1, N_step_, C_step_, T_step_, log_pzx_blob_.gpu_diff(),
		prime_len_blob_.gpu_data(), l_primes_blob_.gpu_data(),
		log_alpha, log_beta, log_pzx_blob_.gpu_data(), target_prob);
	CTCTargetProbGPU<Dtype><<<CAFFE_GET_BLOCKS(nct_count), CAFFE_CUDA_NUM_THREADS>>>(
		nct_count, N_, N_step_, log_pzx_blob_.gpu_data(), target_prob);
}

template <typename Dtype>
void CTCLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
	// calculate the length of each sequence
	if (bottom.size() == 2) {
		caffe_gpu_set(N_, (Dtype)T_, log_pzx_blob_.mutable_gpu_diff());
	}
	else {
		// [T, N] -> [N]
		stride_sum(N_, T_, bottom[1]->gpu_data(), log_pzx_blob_.mutable_gpu_diff());
		caffe_gpu_add_scalar(N_, (Dtype)1, log_pzx_blob_.mutable_gpu_diff());
	}
	const Dtype *label = (bottom.size() == 2) ? bottom[1]->gpu_data() : bottom[2]->gpu_data();
	Dtype *predict_prob = prob_.mutable_gpu_data();
	Dtype *target_prob = prob_.mutable_gpu_diff();
	CTCSequenceSoftmaxGPU<Dtype><<<CAFFE_GET_BLOCKS(N_ * T_), CAFFE_CUDA_NUM_THREADS>>>(
		N_ * T_, N_, C_, N_step_, C_step_, T_step_, bottom[0]->gpu_data(), log_pzx_blob_.gpu_diff(), predict_prob);
	GetTarget_gpu(predict_prob, label, target_prob);

	Dtype loss;
	caffe_gpu_asum<Dtype>(N_, log_pzx_blob_.gpu_data(), &loss);
	// normalize log probabilities by number of parallel batches
	top[0]->mutable_cpu_data()[0] = loss / N_;
}


template <typename Dtype>
__global__ void CTCReweightGPU(
	const int nthreads				// T * N
	, const int dim, const int step
	, const Dtype *weight			// [N]
	, Dtype *data					// [T, N]
)
{
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int i = (index % (step * dim)) / step;
		data[index] *= weight[i];
	}
}

template <typename Dtype>
__global__ void CTCReweightGPU(
	const int nthreads				// N * C * T
	, const int dim1, const int step1
	, const int dim2, const int step2
	, const Dtype *weight			// [T, N]
	, Dtype *data					// data blob shape
)
{
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int i = (index % (step1 * dim1)) / step1;
		const int j = (index % (step2 * dim2)) / step2;
		data[index] *= weight[i * dim2 + j];
	}
}

template <typename Dtype>
__global__ void CTCMaxNegChannelGPU(
	const int nthreads // T * N
	, const int N, const int C
	, const int N_step, const int C_step, const int T_step
	, const Dtype *data				// data blob shape
	, const Dtype *seq_len			// [N]
	, Dtype *weight					// [T, N], the weight for each timestep
)
{
	const int C_end = C * C_step;
	CUDA_KERNEL_LOOP(index, nthreads) {
		Dtype &weight_nt = weight[index];
		weight_nt = 0;
		const int t = index / N;
		const int n = index % N;
		if (t >= static_cast<int>(seq_len[n] + 0.5))
		{
			continue;
		}
		const Dtype *data_nt = data + n * N_step + t * T_step;
		for (int i = 0; i < C_end; i += C_step) {
			if (data_nt[i] < weight_nt)
				weight_nt = data_nt[i];
		}
		weight_nt = -weight_nt;
	}
}

template <typename Dtype>
__global__ void permute_kernel(const int nthreads,
	const int dim1_outer, const int dim2_outer, const int dim3_outer,
	const int dim1_step, const int dim2_step, const int dim3_step,
	const int dst_dim1_step, const int dst_dim2_step, const int dst_dim3_step,
	const Dtype *src, Dtype *dst)
{
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int d1 = (index % dim1_outer) / dim1_step;
		const int d2 = (index % dim2_outer) / dim2_step;
		const int d3 = (index % dim3_outer) / dim3_step;
		dst[d1 * dst_dim1_step + d2 * dst_dim2_step + d3 * dst_dim3_step] = src[index];
	}
}

template <typename Dtype>
__global__ void inverse_kernel(const int n, const Dtype* a, Dtype* b, const Dtype when_zero = 1 / (Dtype)FLT_MIN) {
	CUDA_KERNEL_LOOP(index, n) {
		b[index] = a[index] ? (1 / a[index]) : when_zero;
	}
}

// Defined in softmax_FL_layer.cu
template <typename Dtype>
void stride_sum(const int stride, const int count, const Dtype *src, Dtype *dst);

template <typename Dtype>
Dtype get_tmp_parameter(const vector<Dtype> &parameters,
	const vector<int> &stepvalues, const int tmp_step);


template <typename Dtype>
void CTCLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
	if (!propagate_down[0]) {
		LOG(INFO) << "Skip CTC Loss BackwardCPU";
		return;
	}
	const int nt_count = N_ * T_;
	const int nct_count = nt_count * C_;
	Dtype *diff = bottom[0]->mutable_gpu_diff();
	const Dtype *predict_prob = prob_.gpu_data();
	Dtype *target_prob = prob_.mutable_gpu_diff(); // may be modified later
	// default blank_index = C - 1
	const int blank_index = (blank_index_ < 0) ? C_ + blank_index_ : blank_index_;

	tmp_iter_++;
	const Dtype gamma = gammas_.empty() ? 0 :
		get_tmp_parameter<Dtype>(gammas_, stepvalues_, tmp_iter_);
	const Dtype alpha = alphas_.empty() ? 0 :
		get_tmp_parameter<Dtype>(alphas_, stepvalues_, tmp_iter_);
	const Dtype label_width = this->layer_param_.ctc_param().label_width();

	if (alpha || label_width)
	{
		// get the positive-negative ratio
		permute_kernel<Dtype><<<CAFFE_GET_BLOCKS(nct_count), CAFFE_CUDA_NUM_THREADS>>>(
			nct_count, N_ * N_step_, C_ * C_step_, T_ * T_step_, N_step_, C_step_, T_step_,
			C_, 1, N_ * C_, target_prob, diff);
		stride_sum(C_, nt_count, diff, diff); // [T, N, C] -> [C]
		
		const bool balanced = this->layer_param_.ctc_param().balanced();
		Dtype *count_cpu = label_count_.mutable_cpu_data();
		if (balanced || label_width) {
			caffe_set<Dtype>(C_, 0, count_cpu);
			const Dtype *label_cpu = (bottom.size() == 2) ? bottom[1]->cpu_data() : bottom[2]->cpu_data();
			for (int n = 0; n < N_; ++n) {
				for (int l = 0; l < L_; ++l) {
					Dtype label_val = label_cpu[n*label_N_step_ + l * label_L_step_];
					if (label_val < 0)
						break;
					++count_cpu[static_cast<int>(label_val + 0.5)];
				}
			}
			count_cpu[blank_index] = caffe_cpu_asum<Dtype>(C_, count_cpu);
		}

		if (balanced) {
			if (alpha) {
				count_cpu[blank_index] *= (1 - alpha) / alpha;
			}
			else // must be label_width
			{
				Dtype amount_all;
				caffe_gpu_asum<Dtype>(C_, diff, &amount_all);
				for (int c = 0; c < C_; ++c) {
					count_cpu[c] *= label_width;
				}
				count_cpu[blank_index] = max(amount_all - count_cpu[blank_index], (Dtype)0);
				if (count_cpu[blank_index])
					LOG(WARNING) << "The label may be too wide to be contained in the sequence.";
			}
			inverse_kernel<Dtype><<<CAFFE_GET_BLOCKS(C_), CAFFE_CUDA_NUM_THREADS>>>(C_, diff, diff);
			caffe_gpu_mul<Dtype>(C_, label_count_.gpu_data(), diff, diff);
		}
		else {
			Dtype amount_blank, amount_all;
			caffe_gpu_asum<Dtype>(C_, diff, &amount_all);
			caffe_gpu_asum<Dtype>(1, diff + blank_index, &amount_blank);
			if (alpha) {
				amount_all = alpha / amount_all;
				amount_blank = (1 - alpha) / amount_blank;
			}
			else // must be label_width
			{
				count_cpu[blank_index] *= label_width;
				amount_blank = max(amount_all - count_cpu[blank_index], (Dtype)0) / amount_blank;
				amount_all = count_cpu[blank_index] / amount_all;
			}
			caffe_gpu_set<Dtype>(C_, amount_all, diff);
			caffe_gpu_set<Dtype>(1, amount_blank, diff + blank_index);
		}
		
		// adjust the target class ratio
		target_prob = prob_.mutable_gpu_diff();
		CTCReweightGPU<Dtype><<<CAFFE_GET_BLOCKS(nct_count), CAFFE_CUDA_NUM_THREADS>>>(
			nct_count, C_, C_step_, diff, target_prob);
		// normalize to ensure probabilites of all classes sum to one for each sample
		permute_kernel<Dtype><<<CAFFE_GET_BLOCKS(nct_count), CAFFE_CUDA_NUM_THREADS>>>(
			nct_count, N_ * N_step_, C_ * C_step_, T_ * T_step_, N_step_, C_step_, T_step_,
			1, nt_count, N_, target_prob, diff);
		stride_sum(nt_count, C_, diff, diff); // [C, T, N] -> [T, N]
		inverse_kernel<Dtype><<<CAFFE_GET_BLOCKS(nt_count), CAFFE_CUDA_NUM_THREADS>>>(nt_count, diff, diff);
		CTCReweightGPU<Dtype><<<CAFFE_GET_BLOCKS(nct_count), CAFFE_CUDA_NUM_THREADS>>>(
			nct_count, T_, T_step_, N_, N_step_, diff, target_prob);
	}
	
	caffe_gpu_sub<Dtype>(nct_count, predict_prob, target_prob, diff);

	if (gamma) {
		// unused space [N, 2L+1, T], we use it for gamma-reweighting
		Dtype *log_alpha = log_alpha_beta_blob_.mutable_gpu_data();
		Dtype *log_beta = log_alpha_beta_blob_.mutable_gpu_diff();
		// \max_k{(target - predict)_k} = \max_k{-(predict - target)_k} = -\min_k{(predict - target)_k}
		CTCMaxNegChannelGPU<Dtype><<<CAFFE_GET_BLOCKS(nt_count), CAFFE_CUDA_NUM_THREADS>>>(
			nt_count, N_, C_, N_step_, C_step_, T_step_, diff, log_pzx_blob_.gpu_diff(), log_alpha);
		Dtype dtype_val;
		caffe_gpu_amax<Dtype>(nt_count, log_alpha, &dtype_val);
		caffe_gpu_scal<Dtype>(nt_count, 1 / dtype_val, log_alpha);
		caffe_gpu_powx<Dtype>(nt_count, log_alpha, gamma, log_alpha);
		stride_sum(N_, T_, log_alpha, log_beta); // [T, N] -> [N]
		inverse_kernel<Dtype><<<CAFFE_GET_BLOCKS(N_), CAFFE_CUDA_NUM_THREADS>>>(N_, log_beta, log_beta, 0);
		caffe_gpu_mul<Dtype>(N_, log_pzx_blob_.gpu_diff(), log_beta, log_beta);
		CTCReweightGPU<Dtype><<<CAFFE_GET_BLOCKS(nt_count), CAFFE_CUDA_NUM_THREADS>>>(
			nt_count, N_, 1, log_beta, log_alpha);
		CTCReweightGPU<Dtype><<<CAFFE_GET_BLOCKS(nct_count), CAFFE_CUDA_NUM_THREADS>>>(
			nct_count, T_, T_step_, N_, N_step_, log_alpha, diff);
	}
	// Scale gradient
	Dtype loss_weight = top[0]->cpu_diff()[0] / N_;
	caffe_gpu_scal<Dtype>(nct_count, loss_weight, diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(CTCLossLayer);

}  // namespace caffe
