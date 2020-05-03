#ifndef CAFFE_LMDB_DATA_LAYER_HPP_
#define CAFFE_LMDB_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class LmdbDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit LmdbDataLayer(const LayerParameter& param);
  virtual ~LmdbDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "LmdbData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  void AlignCollate();
  void ShuffleIndices();
  void LoadSample(const int index, cv::Mat &cv_img, string &label);
  virtual void load_batch(Batch<Dtype>* batch);

  shared_ptr<db::DB> db_;
  shared_ptr<db::Cursor> cursor_;

  // RANDOM_SAMPLE
  vector<int> indices_;
  int index_, id_offset_;
  // KEEP_RATIO
  int samples_num_;

  map<char, int> label_index_;
};

}  // namespace caffe

#endif  // CAFFE_LMDB_DATA_LAYER_HPP_
