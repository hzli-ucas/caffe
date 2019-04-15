#include "caffe/layers/ctc_loss_layer.hpp"
#include <math.h>

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
__device__ Dtype max(Dtype lhs, Dtype rhs)
{
	return lhs > rhs ? lhs : rhs;
}

template <typename Dtype>
__device__ Dtype min(Dtype lhs, Dtype rhs)
{
	return lhs < rhs ? lhs : rhs;
}

template <typename Dtype>
__global__ void SequenceSoftmaxGPU(
	const int nthreads			// T * N
	, const int N, const int C
	, const int N_step, const int C_step, const int T_step
	, const Dtype *indicator	// [T, N] if not nullptr
	, Dtype *const data			// data blob shape
	)
{
	// compute softmax (until each sequence ends)
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int t = index / N;
		if (indicator)
		{
			// if exceeds the sequence length
			if (t != 0 && indicator[index] == 0)
			{
				continue;
			}
		}
		const int n = index % N;
		Dtype *const data_nt = data + n*N_step + t*T_step;
		Dtype max_coeff = data_nt[0];
		const int C_end = C*C_step;
		// get max coeff
		for (int i = C_step; i < C_end; i += C_step) {
			max_coeff = max(max_coeff, data_nt[i]);
		}
		// calc exp and its sum
		Dtype sum = 0;
		for (int i = 0; i < C_end; i += C_step) {
			data_nt[i] = exp(data_nt[i] - max_coeff);
			sum += data_nt[i];
		}
		// division by sum
		for (int i = 0; i < C_end; i += C_step) {
			data_nt[i] /= sum;
		}
	}
}

template <typename Dtype>
__global__ void CalculateCTCLossGPU(
	const int N, const int C, const int T
	, const int N_step, const int C_step, const int T_step
	, const Dtype *const probability	// data blob shape
	, const Dtype *const indicator		// [T, N] if not nullptr
	, const int L, const int label_N_step, const int label_L_step
	, const int blank_index
	, const Dtype *const label			// [N, L] or [L, N], decided by the n-axis
	, int *const prime_len				// [N]
	, int *const l_primes				// [N, 2L+1]
	, Dtype *const log_alpha			// [N, 2L+1, T], already initialized to kLogZero
	, Dtype *const log_beta				// [N, 2L+1, T], already initialized to kLogZero
	, Dtype *const log_pzx				// [N]
	)
{
	CUDA_KERNEL_LOOP(n, N) {
		// Insert a blank before each element and a blank at the end.
		const Dtype *label_nl = label + n * label_N_step;
		int *const l_prime_n = l_primes + n * (2 * L + 1);
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
		int seq_len_n = T;
		if (indicator)
		{
			const Dtype *const indicator_n = indicator + n;
			while (seq_len_n > 0 && indicator_n[(--seq_len_n)*N] == 0);
			++seq_len_n;
			// although T >= L, it is possible that seq_len < label_len
			// in which case, set p(z|x) = 0 and skip the calculation of edit distance
			if (seq_len_n * 2 + 1 < U)
			{
				log_pzx[n] = kLogZero;
				continue;
			}
		}
		// [N, 2L+1, T] alpha and beta at offset(n, 0, 0)
		Dtype *const log_alpha_n = log_alpha + n * (2 * L + 1) * T;
		Dtype *const log_beta_n = log_beta + n * (2 * L + 1) * T;
		// data blob shape, probability at offset n*N_step
		const Dtype *const prob_n = probability + n * N_step;
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
			log_beta_n[(u + 1) * T - 1] = 0; // log_beta[u, T-1]
		}
		for (int t = seq_len_n - 1 - 1; t >= 0; --t) {
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
void CTCLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
	Blob<Dtype> *const data_blob = bottom[0];
	const Blob<Dtype> *const label_blob = (bottom.size() == 2) ? bottom[1] : bottom[2];
	const Dtype *const indicator = (bottom.size() == 2) ? nullptr : bottom[1]->gpu_data();
	Dtype *const data = data_blob->mutable_gpu_data();
	if (!already_softmax_){
		const int nt_count = N_*T_;
		SequenceSoftmaxGPU<Dtype><<<CAFFE_GET_BLOCKS(nt_count), CAFFE_CUDA_NUM_THREADS>>>(
			nt_count, N_, C_, N_step_, C_step_, T_step_, indicator, data);
	}
	// default blank_index = C - 1
	const int blank_index = (blank_index_ < 0) ? C_ + blank_index_ : blank_index_;
	Dtype *const log_alpha = log_alpha_beta_blob_.mutable_gpu_data();
	Dtype *const log_beta = log_alpha_beta_blob_.mutable_gpu_diff();
	// Initialize the log_alpha and log_beta blob with kLogZero
	caffe_gpu_set<Dtype>(log_alpha_beta_blob_.count(), kLogZero, log_alpha);
	caffe_gpu_set<Dtype>(log_alpha_beta_blob_.count(), kLogZero, log_beta);
	CalculateCTCLossGPU<Dtype><<<CAFFE_GET_BLOCKS(N_), CAFFE_CUDA_NUM_THREADS>>>(
		N_, C_, T_, N_step_, C_step_, T_step_, data, indicator,
		L_, label_N_step_, label_L_step_, blank_index, label_blob->gpu_data(),
		prime_len_blob_.mutable_gpu_data(), l_primes_blob_.mutable_gpu_data(),
		log_alpha, log_beta, log_pzx_blob_.mutable_gpu_data());
	Dtype loss;
	caffe_gpu_asum<Dtype>(N_, log_pzx_blob_.gpu_data(), &loss);
	// normalize log probabilities by number of parallel batches
	top[0]->mutable_cpu_data()[0] = loss / N_;
}

template <typename Dtype>
__global__ void LogSumProbGPU(
	const int nthreads				// N * T
	, const int T, const int Lm2p1	// 2 * L + 1
	, const int N_step, const int C_step, const int T_step
	, const int *const prime_len	// [N]
	, const int *const l_primes		// [N, 2L+1]
	, const Dtype *const log_alpha	// [N, 2L+1, T]
	, const Dtype *const log_beta	// [N, 2L+1, T]
	, const Dtype *const log_pzx	// [N]
	, const Dtype *const indicator	// [T, N] if not nullptr
	, Dtype *const log_sum_prob		// data blob shape, already initialized to kLogZero
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
		if (indicator)
		{
			// [T, N, C], N = (N*C) / C
			const int N = T_step / N_step;
			if (t != 0 && indicator[t*N + n] == 0)
			{
				// sequence ends
				continue;
			}
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
		// will be parallelly performed in function CalculateGradient<<<N*C*T>>>
	}
}

template <typename Dtype>
__global__ void CalculateCTCGradientGPU(
	const int nthreads // N * C * T
	, const int N, const int N_step
	, const bool already_softmax
	, const Dtype *log_pzx			// [N]
	, const Dtype *predict_prob		// data blob shape
	, Dtype *diff					// data blob shape, still log_prob_sum now
	)
{
	CUDA_KERNEL_LOOP(index, nthreads) {
		// if log_pzx_n == kLogZero or the sequence ends already
		// there must be target_prob_nct == kLogZero
		Dtype &diff_nct = diff[index];
		if (diff_nct == kLogZero){
			diff_nct = already_softmax ? 0 : predict_prob[index];
			continue;
		}
		const int n = (index % (N_step * N)) / N_step;
		const Dtype log_pzx_n = log_pzx[n];
		// log_pzx_n cannot be kLogZero then
		if (diff_nct >= log_pzx_n) {
			diff_nct = already_softmax ? 
				(-1 / predict_prob[index]) : 
				(predict_prob[index] - 1);
		}
		else {
			diff_nct = exp(diff_nct - log_pzx_n);
			diff_nct = already_softmax ? 
				(-diff_nct / predict_prob[index]) : 
				(predict_prob[index] - diff_nct);
		}
	}
}

template <typename Dtype>
void CTCLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
	if (!propagate_down[0]) {
		LOG(INFO) << "Skip CTC Loss BackwardCPU";
		return;
	}
	const int nt_count = N_*T_;
	const int nct_count = nt_count * C_;
	Blob<Dtype> *const data_blob = bottom[0];
	Dtype *const diff = data_blob->mutable_gpu_diff();
	// the data_diff is first used as log_prob_sum
	// before the final gradient calculation
	caffe_gpu_set<Dtype>(nct_count, kLogZero, diff);
	const Dtype *const indicator = (bottom.size() == 2) ? nullptr : bottom[1]->gpu_data();
	LogSumProbGPU<Dtype><<<CAFFE_GET_BLOCKS(nt_count), CAFFE_CUDA_NUM_THREADS>>>(
		nt_count, T_, 2 * L_ + 1, N_step_, C_step_, T_step_, prime_len_blob_.gpu_data(), l_primes_blob_.gpu_data(),
		log_alpha_beta_blob_.gpu_data(), log_alpha_beta_blob_.gpu_diff(), log_pzx_blob_.gpu_data(), indicator, diff);
	CalculateCTCGradientGPU<Dtype><<<CAFFE_GET_BLOCKS(nct_count), CAFFE_CUDA_NUM_THREADS>>>(
		nct_count, N_, N_step_, already_softmax_, log_pzx_blob_.gpu_data(), data_blob->gpu_data(), diff);
	// Scale gradient
	Dtype loss_coefficient = 0;
	if (indicator) // the recurrent data
	{
		caffe_gpu_asum<Dtype>(nt_count, indicator, &loss_coefficient);
		loss_coefficient += N_;
	}
	else {
		loss_coefficient = nt_count;
	}
	Dtype loss_weight = top[0]->cpu_diff()[0] / loss_coefficient;
	caffe_gpu_scal<Dtype>(nct_count, loss_weight, diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(CTCLossLayer);

}  // namespace caffe
