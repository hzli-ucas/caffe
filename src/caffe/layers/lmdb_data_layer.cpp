#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp> // imdecode
#include <opencv2/imgproc/imgproc.hpp> // resize
#include <opencv2/imgcodecs/legacy/constants_c.h>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>
#include <map>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/lmdb_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/format.hpp"

namespace caffe {

template <typename Dtype>
LmdbDataLayer<Dtype>::LmdbDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param) {
  db_.reset(db::GetDB("lmdb"));
  db_->Open(param.lmdb_data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());
}

template <typename Dtype>
LmdbDataLayer<Dtype>::~LmdbDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void LmdbDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.lmdb_data_param().batch_size();
  const int label_size = this->layer_param_.lmdb_data_param().label_size();
  const int new_height = this->layer_param_.lmdb_data_param().new_height();
  int new_width = this->layer_param_.lmdb_data_param().new_width();
  const int max_width = this->layer_param_.lmdb_data_param().max_width();
  const int min_width = this->layer_param_.lmdb_data_param().min_width();
  const int repeats = this->layer_param_.lmdb_data_param().repeats();
  id_offset_ = this->layer_param_.lmdb_data_param().id_offset();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  CHECK_GT(label_size, 0) << "Positive label size required";
  CHECK_GT(repeats, 0) << "Positive repeat times required";
  CHECK(new_height > 0 && new_width > 0) << "Invalid new_height = "
	  << new_height << " or new_width = " << new_width;
  CHECK_LE(new_height, new_width) << "The new_height = " << new_height 
	  << " should be no larger than new_width = " << new_width;
  CHECK(!min_width || !max_width || min_width < max_width);

  cursor_->Seek("num-samples");
  CHECK(cursor_->valid()) << "The lmdb data should contain a \"num-samples\" value.";
  samples_num_ = atoi(cursor_->value().c_str());
  LOG(INFO) << "A total of " << samples_num_ << " samples.";

  // Initialize the label-index map
  // [0~9, Aa~Zz]: [0~35]
  for (int i = 48; i < 58; ++i) {
	  label_index_[i] = i - 48;
  }
  for (int i = 65; i < 91; ++i) {
	  label_index_[i] = i - 55;
  }
  for (int i = 97; i < 123; ++i) {
	  label_index_[i] = i - 87;
  }

  // Read a data point, and use it to initialize the top blob.
  cv::Mat cv_img;
  LoadSample(caffe_rng_rand() % samples_num_, cv_img, string());
  switch (this->layer_param_.lmdb_data_param().sampler()) {
  case LmdbDataParameter_SampleOpt_SHUFFLE_RESIZE:
	  CHECK(!this->layer_param_.lmdb_data_param().has_min_width()
		  && !this->layer_param_.lmdb_data_param().has_max_width())
		  << "Try RANDOM_ALIGN sampler if you wanna keep aspect ratios for images.";
	  for (int i = 0; i < samples_num_; ++i) {
		  indices_.push_back(i);
	  }
	  // randomly shuffle data
	  LOG(INFO) << "Shuffling data";
	  prefetch_rng_.reset(new Caffe::RNG(caffe_rng_rand()));
	  ShuffleIndices();
	  index_ = 0;
	  break;
  case LmdbDataParameter_SampleOpt_RANDOM_ALIGN:
	  CHECK(!this->layer_param_.lmdb_data_param().has_new_width())
		  << "Try SHUFFLE_RESIZE sampler if you wanna resize images with a fixed width.";
	  LOG(INFO) << "Aligning data with aspect ratios";
	  AlignCollate();
	  new_width = cv_img.cols * new_height / cv_img.rows;
	  // Should be max_width >= new_width >= min_width
	  if (new_width < min_width)
		  new_width = min_width;
	  if (max_width && new_width > max_width)
		  new_width = max_width;
	  break;
  case LmdbDataParameter_SampleOpt_SEQUENTIAL:
	  if (this->layer_param_.lmdb_data_param().has_new_width()) {
		  CHECK(!this->layer_param_.lmdb_data_param().has_min_width()
			  && !this->layer_param_.lmdb_data_param().has_max_width())
			  << "Cannot resize images to a FIXED SIZE meanwhile KEEP RATIO."
			  << "Try choose one of them. Default to be KEEP RATIO.";
	  }
	  else {
		  if (this->layer_param_.lmdb_data_param().batch_size() != 1) {
			  LOG(WARNING) << "You are using KEEP RATIO mode on unsorted data,"
				  << "where images within a batch might be rescaled to an unreasonable width.";
		  }
		  new_width = cv_img.cols * new_height / cv_img.rows;
		  // Should be max_width >= new_width >= min_width
		  if (new_width < min_width)
			  new_width = min_width;
		  if (max_width && new_width > max_width)
			  new_width = max_width;
	  }
	  index_ = 0;
	  break;
  default:
	  LOG(ERROR) << "Unknown sampler: " 
		  << this->layer_param_.lmdb_data_param().sampler();
	  break;
  }
  cv::resize(cv_img, cv_img, cv::Size(new_width, new_height));
  cv_img = cv::repeat(cv_img, 1, repeats);

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  LOG_IF(INFO, Caffe::root_solver())
      << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(2, batch_size);
  label_shape[1] = label_size * repeats;
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
	  this->prefetch_[i]->label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void LmdbDataLayer<Dtype>::AlignCollate() {
	for (int i = 0; i < samples_num_; ++i) {
		indices_.push_back(i);
	}
	//std::multimap<Dtype, int> aspect_ratios;
	//for (int i = 0; i < samples_num_; ++i) {
	//	string index_str = format_int(i + id_offset_, 9);
	//	cursor_->Seek("image-" + index_str);
	//	CHECK(cursor_->valid()) << "Unable to find value by key: image-" << index_str;
	//	const string &data = cursor_->value();
	//	vector<char> vec_data(data.c_str(), data.c_str() + data.size());
	//	cv::Mat cv_img = cv::imdecode(vec_data, CV_LOAD_IMAGE_GRAYSCALE);
	//	if (!cv_img.data) {
	//		LOG(WARNING) << "Corrupted image " << index_str;
	//		continue;
	//	}
	//	aspect_ratios.insert(make_pair(Dtype(cv_img.cols) / cv_img.rows, i));
	//}
	//stringstream ss;
	//for (auto i : aspect_ratios) {
	//	indices_.push_back(i.second);
	//	ss << " " << i.second;
	//}
	//LOG(INFO) << "The image aspect ratio ranges from " << aspect_ratios.begin()->first
	//	<< " to " << aspect_ratios.rbegin()->first
	//	<< "\n"<<ss.str();
}

template <typename Dtype>
void LmdbDataLayer<Dtype>::ShuffleIndices() {
	caffe::rng_t* prefetch_rng =
		static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	shuffle(indices_.begin(), indices_.end(), prefetch_rng);
}

template <typename Dtype>
void LmdbDataLayer<Dtype>::LoadSample(const int index, cv::Mat &cv_img, string &label) {
	// We do not check whether the index is valid here,
	// since the index should be managed outside the function.
	// If this image is corrupted, we will load the next image
	// in the lmdb data, i.e. index+1.
	// We do not check if index+1 overflows, just assuming
	// the last image is not a corrupted one.
	// The image index starts from 1
	string index_str = format_int(index + id_offset_, 9);

	cursor_->Seek("image-" + index_str);
	CHECK(cursor_->valid()) << "Unable to find value by key: image-" << index_str;
	const string &data = cursor_->value();
	vector<char> vec_data(data.c_str(), data.c_str() + data.size());
	//cv_img.release();
	cv_img = cv::imdecode(vec_data, CV_LOAD_IMAGE_GRAYSCALE);
	if (!cv_img.data) {
		LOG(WARNING) << "Corrupted image " << index_str;
		return LoadSample(index + 1, cv_img, label);
	}

	cursor_->Seek("label-" + index_str);
	CHECK(cursor_->valid()) << "Unable to find value by key: label-" << index_str;
	//label.clear();
	label = cursor_->value();
}

// This function is called on prefetch thread
template<typename Dtype>
void LmdbDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  int batch_size = this->layer_param_.lmdb_data_param().batch_size();
  const int label_size = this->layer_param_.lmdb_data_param().label_size();
  const int repeats = this->layer_param_.lmdb_data_param().repeats();
  const int label_step = label_size * repeats;

  switch (this->layer_param_.lmdb_data_param().sampler()) {
  case LmdbDataParameter_SampleOpt_SHUFFLE_RESIZE:
  {
	  const int new_height = this->layer_param_.lmdb_data_param().new_height();
	  const int new_width = this->layer_param_.lmdb_data_param().new_width();
	  for (int item_id = 0; item_id < batch_size; ++item_id) {
		  timer.Start();
		  cv::Mat cv_img;
		  string label;
		  LoadSample(indices_[index_++], cv_img, label);
		  cv::resize(cv_img, cv_img, cv::Size(new_width, new_height));
		  if (label.size() > label_size) {
			  LOG(WARNING) << "The sample label is \"" << label
				  << "\", which exceeds defined label_size = " << label_size
				  << "Excess will be cut off. Should try a larger label_size.";
		  }
		  cv_img = cv::repeat(cv_img, 1, repeats);
		  std::ostringstream os;
		  for (int i = 0; i < repeats; ++i)
			  os << label;
		  label = os.str();
		  read_time += timer.MicroSeconds();
		  // Apply data transformations (mirror, scale, crop...)
		  timer.Start();
		  int offset = batch->data_.offset(item_id);
		  Dtype* top_data = batch->data_.mutable_cpu_data();
		  this->transformed_data_.set_cpu_data(top_data + offset);
		  this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
		  // Copy label.
		  Dtype* top_label = batch->label_.mutable_cpu_data() + item_id * label_step;
		  const int label_length = label.size();
		  for (int label_id = 0; label_id < label_length; ++label_id)
			  top_label[label_id] = label_index_[label[label_id]];
		  for (int label_id = label_length; label_id < label_step; ++label_id)
			  top_label[label_id] = -1;
		  trans_time += timer.MicroSeconds();
		  // go to the next iter
		  if (index_ >= indices_.size()) {
			  // We have reached the end. Restart from the first.
			  DLOG(INFO) << "Restarting data prefetching from start.";
			  index_ = 0;
			  ShuffleIndices();
		  }
	  }
	  break;
  }
  case LmdbDataParameter_SampleOpt_RANDOM_ALIGN:
  {
	  const int new_height = this->layer_param_.lmdb_data_param().new_height();
	  const int max_width = this->layer_param_.lmdb_data_param().max_width();
	  const int min_width = this->layer_param_.lmdb_data_param().min_width();
	  int new_width;
	  while (true) {
		  index_ = caffe_rng_rand() % (indices_.size() - batch_size + 1);
		  // Calculate the new_width accodrding to the medium ratio within the batch
		  cv::Mat example_img;
		  string example_label;
		  LoadSample(indices_[batch_size / 2 + index_], example_img, example_label);
		  if (example_label.length() < 3 || example_img.cols > example_img.rows * 20)
			  continue;
		  new_width = example_img.cols * new_height / example_img.rows;
		  break;
	  }
	  // Should be max_width >= new_width >= min_width
	  if (new_width < min_width)
		  new_width = min_width;
	  if (max_width && new_width > max_width)
		  new_width = max_width;
	  // Reshape according to the example_img, because
	  // the blob shape can be different for different batch
	  vector<int> top_shape = batch->data_.shape();
	  // Reshape batch according to the batch_size.
	  top_shape[0] = batch_size;
	  top_shape[3] = new_width * repeats;
	  batch->data_.Reshape(top_shape);
	  top_shape[0] = 1;
	  this->transformed_data_.Reshape(top_shape);
	  LOG(INFO) << "image width = " << new_width << ", height = " << new_height;
	  for (int item_id = 0; item_id < batch_size; ++item_id) {
		  timer.Start();
		  cv::Mat cv_img;
		  string label;
		  LoadSample(indices_[index_++], cv_img, label);
		  //LOG(INFO) << "image width = " << cv_img.cols << ", height = " << cv_img.rows << ", label: "<<label;
		  cv::resize(cv_img, cv_img, cv::Size(new_width, new_height));
		  if (label.size() > label_size) {
			  LOG(WARNING) << "The sample label is \"" << label
				  << "\", which exceeds defined label_size = " << label_size
				  << "Excess will be cut off. Should try a larger label_size.";
		  }
		  cv_img = cv::repeat(cv_img, 1, repeats);
		  std::ostringstream os;
		  for (int i = 0; i < repeats; ++i)
			  os << label;
		  label = os.str();
		  read_time += timer.MicroSeconds();
		  // Apply data transformations (mirror, scale, crop...)
		  timer.Start();
		  int offset = batch->data_.offset(item_id);
		  Dtype* top_data = batch->data_.mutable_cpu_data();
		  this->transformed_data_.set_cpu_data(top_data + offset);
		  this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
		  // Copy label.
		  Dtype* top_label = batch->label_.mutable_cpu_data() + item_id * label_step;
		  const int label_length = label.size();
		  for (int label_id = 0; label_id < label_length; ++label_id)
			  top_label[label_id] = label_index_[label[label_id]];
		  for (int label_id = label_length; label_id < label_step; ++label_id)
			  top_label[label_id] = -1;
		  trans_time += timer.MicroSeconds();
	  }
	  break;
  }
  case LmdbDataParameter_SampleOpt_SEQUENTIAL:
  {
	  const int new_height = this->layer_param_.lmdb_data_param().new_height();
	  int new_width;
	  if (this->layer_param_.lmdb_data_param().has_new_width()) {
		  new_width = this->layer_param_.lmdb_data_param().new_width();
	  } else {
		  const int max_width = this->layer_param_.lmdb_data_param().max_width();
		  const int min_width = this->layer_param_.lmdb_data_param().min_width();
		  if (batch_size + index_ > samples_num_)
			  batch_size = samples_num_ - index_;
		  // Calculate the new_width accodrding to the medium ratio within the batch
		  cv::Mat example_img;
		  LoadSample(batch_size / 2 + index_, example_img, string());
		  new_width = example_img.cols * new_height / example_img.rows;
		  // Should be max_width >= new_width >= min_width
		  if (new_width < min_width)
			  new_width = min_width;
		  if (max_width && new_width > max_width)
			  new_width = max_width;
		  // Reshape according to the example_img, because
		  // the blob shape can be different for different batch
		  vector<int> top_shape = batch->data_.shape();
		  // Reshape batch according to the batch_size.
		  top_shape[0] = batch_size;
		  top_shape[3] = new_width * repeats;
		  batch->data_.Reshape(top_shape);
		  top_shape[0] = 1;
		  this->transformed_data_.Reshape(top_shape);
	  }
	  for (int item_id = 0; item_id < batch_size; ++item_id) {
		  timer.Start();
		  cv::Mat cv_img;
		  string label;
		  LoadSample(index_++, cv_img, label);
		  cv::resize(cv_img, cv_img, cv::Size(new_width, new_height));
		  if (label.size() > label_size) {
			  LOG(WARNING) << "The sample label is \"" << label
				  << "\", which exceeds defined label_size = " << label_size
				  << "Excess will be cut off. Should try a larger label_size.";
		  }
		  cv_img = cv::repeat(cv_img, 1, repeats);
		  std::ostringstream os;
		  for (int i = 0; i < repeats; ++i)
			  os << label;
		  label = os.str();
		  read_time += timer.MicroSeconds();
		  // Apply data transformations (mirror, scale, crop...)
		  timer.Start();
		  int offset = batch->data_.offset(item_id);
		  Dtype* top_data = batch->data_.mutable_cpu_data();
		  this->transformed_data_.set_cpu_data(top_data + offset);
		  this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
		  // Copy label.
		  Dtype* top_label = batch->label_.mutable_cpu_data() + item_id * label_step;
		  const int label_length = label.size();
		  for (int label_id = 0; label_id < label_length; ++label_id)
			  top_label[label_id] = label_index_[label[label_id]];
		  for (int label_id = label_length; label_id < label_step; ++label_id)
			  top_label[label_id] = -1;
		  trans_time += timer.MicroSeconds();
		  // go to the next iter
		  if (index_ >= samples_num_) {
			  // We have reached the end. Restart from the first.
			  DLOG(INFO) << "Restarting data prefetching from start.";
			  index_ = 0;
		  }
	  }
	  break;
  }
  default:
	  LOG(ERROR) << "Unknown sampler: "
		  << this->layer_param_.lmdb_data_param().sampler();
	  break;
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(LmdbDataLayer);
REGISTER_LAYER_CLASS(LmdbData);

}  // namespace caffe
