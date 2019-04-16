#ifndef CAFFE_REVERSE_LAYER_HPP_
#define CAFFE_REVERSE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Reverses the data of the input Blob into the output blob.
 *
 * Note: This is a useful layer if you want to reverse
 *       a recurrent layer...or a convolutional layer:)
 *       You MUST either
 *       (1) specify the reverse scope with a indicator blob for a recurrent layer,
 *       or
 *       (2) specify the reverse axis, to reverse the whole blob along the axis by
 *          reverse_param { axis: 0 } // there is no default value of axis, in case of misuse
 */

template <typename Dtype>
class ReverseLayer : public Layer<Dtype> {
 public:
  explicit ReverseLayer(const LayerParameter& param);

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Reverse"; }

  virtual inline int MinNumBottomBlobs() const { return 1; }
  virtual inline int MaxNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int reverse_axis_, axis_outer_, axis_inner_, axis_length_, outer_step_;
  Blob<int> seq_len_blob_;
};

}  // namespace caffe

#endif  // CAFFE_REVERSE_LAYER_HPP_
