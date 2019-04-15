#ifndef CAFFE_SEQUENCE_ACCURACY_LAYER_HPP_
#define CAFFE_SEQUENCE_ACCURACY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the classification accuracy and edit distance for
 *        sequence recognition task.
 */
template <typename Dtype>
class SequenceAccuracyLayer : public Layer<Dtype> {
 public:
  explicit SequenceAccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SequenceAccuracy"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }

  // If there is one top blob, the blob contains sequence-level accuracy;
  // If there are two top blobs, the second blob contains (1) the total edit
  //   distance, (2) the total length of groundtruth sequences and (3) the
  //   total length of the predicted sequences. We can use AR = (1)/(2) to
  //   evaluate the character-level accuracy.
  // If there are three top blobs, the third blob contains the number of the
  //   correct characters. We can get correct rate by CR = top[3]/top[2](2).
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlos() const { return 3; }

 protected:
  /**
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times 1 \times 1) @f$
   *      the predicted label sequences, an integer-valued Blob with values
   *      @f$ l_n \in [-1, 0, 1, 2, ..., K - 1] @f$, where -1 indicates the
   *	  end of the sequcence.
   *   -# @f$ (N \times C \times 1 \times 1) @f$
   *      the groundtruth label sequences, an integer-valued Blob with values
   *      @f$ l_n \in [-1, 0, 1, 2, ..., K - 1] @f$, where -1 indicates the
   *	  end of the sequcence.
   * @param top output Blob vector (length 1/3)
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      the computed accuracy: @f$
   *        \frac{1}{N} \sum\limits_{n=1}^N \delta\{ \hat{l}_n = l_n \}
   *      @f$, where @f$
   *      \delta\{\mathrm{condition}\} = \left\{
   *         \begin{array}{lr}
   *            1 & \mbox{if condition} \\
   *            0 & \mbox{otherwise}
   *         \end{array} \right.
   *      @f$
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  /// @brief Not implemented -- SequenceAccuracyLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

  int N_;
  // Store the sequence capacity
  int prd_size_, gt_size_;
  // Use dynamic programming to calculate
  // the edit distance between sequences
  std::vector<std::vector<int> > edit_dis_;
  std::vector<std::vector<int> > edit_path_;
};

}  // namespace caffe

#endif  // CAFFE_ACCURACY_LAYER_HPP_
