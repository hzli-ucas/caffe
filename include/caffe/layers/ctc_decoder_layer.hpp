#ifndef CAFFE_CTC_DECODER_LAYER_HPP_
#define CAFFE_CTC_DECODER_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief A layer that converts the probability distribution of label sequences into a label sequence.
 *
 * The decoding methods include 
 *     (1) in-place decoding, output predictions at each time-step;
 *     (2) best path decoding, merge the repeated labels and remove blanks;
 *     (3) beam search decoding, you know, beam search.
 * The default one is the best path decoding.
 */
template <typename Dtype>
class CTCDecoderLayer : public Layer<Dtype> {
public:
	explicit CTCDecoderLayer(const LayerParameter& param);
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "CTCDecoder"; }

	// probabilities (N x C x T),
	// RNN indicator (T x N) [optional]
	virtual inline int MinBottomBlobs() const { return 1; }
	virtual inline int MaxBottomBlobs() const { return 2; }

	// sequences (N x T, terminated with negative numbers)
	virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
	/**
	* @brief Decode the network output for each channel to labels
	*
	* @param bottom input Blob vector 
	*
	*   (length 2) the blob format of recurrent layers:
	*     bottom[0]: [T, N, C], data_blob
	*     bottom[1]: [T, N],    indi_blob, indicates the length of each sequence
	*                           (must be 0 at t = 0, and 1 from t = 1, and 0 if the sequence has ended)
	*
	*   (length 1) the blob format of convolutional layers:
	*     bottom[0]: [N, C, T] or [N, C, T, 1] or [N, C, 1, T], data_blob
	*
	*   (length 1) specify the blob format with nct_axis in ctc_param:
	*     Example 1: 
	*     bottom[0]: [T, N, C], data_blob
	*     ctc_param {
	*       nct_axis: 1 // the axis index for N
	*       nct_axis: 2 // the axis index for C
	*     }
	*     Example 2: 
	*     bottom[0]: [N, C, T], data_blob
	*     ctc_param {
	*       nct_axis: 0 // the axis index for N
	*       nct_axis: 1 // the axis index for C
	*       nct_axis: 3 // the axis index for T
	*     }
	*   Where N is the mini-batch size,
	*       C is number of classes = |L|+1, the extra one channel for 'blank',
	*       T is the time steps of the sequences.
	*
	* @param top output Blob vector
	*
	*   (length 1) the sequences must have lengths <= T, terminated with -1
	*      top[0]: [N, T], label sequence
	*/
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
							const vector<Blob<Dtype>*>& top);

	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
							const vector<bool>& propagate_down,
							const vector<Blob<Dtype>*>& bottom);
private:
	void SequenceSoftmax(Blob<Dtype> *data_blob, const Dtype *indicator);
	void Decode(Blob<Dtype> *data_blob, const Dtype *indicator, Blob<Dtype> *sequence_blob);
	// specify the channel index for blank label
	// the default blank-index is (channels - 1)
	int blank_index_;
	int N_;
	int C_;
	int T_;
	int N_step_, C_step_, T_step_;
	// if there is a indicator blob, it must have shape [T x N]
	CTCParameter ctc_param_;
	// parameter for beam search decoding
	int beam_width_;
};

}  // namespace caffe

#endif  // CAFFE_CTC_DECODER_LAYER_HPP_
