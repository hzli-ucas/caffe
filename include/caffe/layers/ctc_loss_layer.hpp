#ifndef CAFFE_CTC_LOSS_LAYER_HPP_
#define CAFFE_CTC_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Implementation of the CTC (Connectionist Temporal Classification) algorithm
 *        to train neural networks on unsegmented sequence data. The ctc_loss_layer
 *        can be adopted after either recurrent layers or convolutional layers.
 * 
 *
 * To split the computation into Forward and Backward passes the intermediate results
 * (alpha, beta, l_prime, log_pzx) are stored during the forward pass and are
 * reused during the backward pass.
 *
 * More details about the calculation process can be found in
 *   Alex Graves. Supervised Sequence Labelling with Recurrent Neural Networks
 *                Chapter 7  Connectionist Temporal Classification
 */
template <typename Dtype>
class CTCLossLayer : public Layer<Dtype> {
	// Didn't choose LossLayer as father class, because its
	// bottom blobs must be of an exact number.
public:
	explicit CTCLossLayer(const LayerParameter& param);
	virtual ~CTCLossLayer();

	virtual void LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "CTCLoss"; }

	// 2: outputs, label sequence
	// 3: outputs, indicator, label sequence
	virtual inline int MinBottomBlobs() const { return 2; }
	virtual inline int MaxBottomBlobs() const { return 3; }

	/**
	* @brief For convenience and backwards compatibility, instruct the Net to
	*        automatically allocate a single top Blob for LossLayers, into which
	*        they output their singleton loss, (even if the user didn't specify
	*        one in the prototxt, etc.).
	*/
	virtual inline bool AutoTopBlobs() const { return true; }
	virtual inline int ExactNumTopBlobs() const { return 1; }
	/**
	* We cannot backpropagate to the labels or the indicators of recurrent data,
	* ignore force_backward for these inputs.
	*/
	virtual inline bool AllowForceBackward(const int bottom_index) const {
		return bottom_index == 0;
	}

protected:
	/**
	* @brief Computes the loss and the error gradients for the input data
	*        in one step (due to optimization isses)
	*
	* @param bottom input Blob vector 
	*
	*   (length 3) the blob format of recurrent layers:
	*     bottom[0]: [T, N, C], data_blob
	*     bottom[1]: [T, N],    indi_blob, indicates the length of each sequence
	*                           (must be 0 at t = 0, and 1 from t = 1, and 0 if the sequence has ended)
	*     bottom[2]: [L, N],    label_blob, the target sequence
	*                           (must start at t = 0, and filled with -1 if the sequence has ended)
	*
	*   (length 2) the blob format  convolutional layers:
	*     bottom[0]: [N, C, T] or [N, C, T, 1] or [N, C, 1, T], data_blob
	*     bottom[1]: [N, L],    label_blob
	*
	*   (length 2) specify the blob format with nct_axis in ctc_param:
	*     e.g. 
	*     bottom[0]: [T, N, C], data_blob
	*     bottom[1]: [L, N],    label_blob
	*     ctc_param {
	*       nct_axis: 1 // the axis index for N
	*       nct_axis: 2 // the axis index for C
	*     }
	*   Where N is the mini-batch size,
	*       C is number of classes = |L|+1, the extra one channel for 'blank',
	*       T is the time steps of the sequences,
	*       L is max possible label length, mustn't be larger than T.
	*
	* @param top output Blob vector
	*
	*   (length 1)
	*      top[0]: the computed loss
	*/

	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

private:
	void GetTarget_gpu(const Dtype *predict_prob, const Dtype *label, Dtype *target_prob);
	void GetTarget_cpu(const Dtype *predict_prob, const Dtype *label, Dtype *target_prob);
	/**
	* @brief Normalize the input data with softmax
	*/
	void SequenceSoftmax(const Dtype *data, const int *seq_len, Dtype *probability);
	
	int N_axis_, C_axis_, T_axis_;
	// specify the channel index for blank label
	// the default blank index is (channels - 1)
	int blank_index_;
	int N_;	// batch size
	int C_;	// class number
	int T_;	// time steps
	int L_; // label size <= T
	int N_step_, C_step_, T_step_, label_N_step_, label_L_step_;
	// if there is a indicator blob, it must have shape [T, N]

	// Intermediate variables that are calculated during the forward pass
	// and reused during the backward pass

	/// prob stores the output probability predictions from the SoftmaxLayer.
	Blob<Dtype> prob_;
	// [N, 2L+1] blobs to store the l_primes of the sequences
	// (label_0, label_1, ...) -> (blank, label_0, blank, label_1, blank, ..., blank)
	Blob<int> l_primes_blob_;
	// [N] blob to store the prime lengths U = label_len * 2 + 1
	// the label_len is the actual label length, which is no larger than L
	Blob<int> prime_len_blob_;
	// [N, 2L+1, T] blobs to store the alpha and beta variables of each input sequence
	// blob.data() for alpha variables, blob.diff() for beta variables
	Blob<Dtype> log_alpha_beta_blob_;
	// [N] blob to store log(p(z|x)) for each sequence
	Blob<Dtype> log_pzx_blob_;
	// We use log_pzx_blob_.diff() to store the RNN data sequence lengths

	// parameters for modification on CTC
	vector<Dtype> gammas_, alphas_;
	int tmp_iter_;
	vector<int> stepvalues_;
	Blob<Dtype> label_count_; // [C]
};

}  // namespace caffe

#endif  // CAFFE_CTC_LOSS_LAYER_HPP_
