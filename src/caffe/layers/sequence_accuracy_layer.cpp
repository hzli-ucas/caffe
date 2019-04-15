#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/sequence_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SequenceAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom.size(), 2);
	// used to reshape scalar; 0 axes.
	vector<int> top_shape(0);
	switch (top.size()){
	case 3:
		top[2]->Reshape(top_shape);
	case 2:
		// blob with three elements: edit-distance,
		// groundtruth- and predctied- sequence length
		top[1]->Reshape(vector<int>(1, 3));
	case 1:
		top[0]->Reshape(top_shape);
		break;
	default:
		LOG(FATAL) << "Need one or two or three output blobs: "
			<< "a) 1: accuracy, or "
			<< "b) 1: accuracy, 2: edit-distance {edit_dis, gt_len, pred_len}, or "
			<< "c) 1: accuracy, 2: edit-distance {edit_dis, gt_len, pred_len}, 3: correct-characters.";
		break;
	}
}

template <typename Dtype>
void SequenceAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const Blob<Dtype>* prd_blob = bottom[0];
	N_ = prd_blob->shape(0);
	prd_size_ = prd_blob->shape(1);
	CHECK_EQ(N_ * prd_size_, prd_blob->count());
	const Blob<Dtype>* gt_blob = bottom[1];
	CHECK_EQ(N_, gt_blob->shape(0));
	gt_size_ = gt_blob->shape(1);
	CHECK_EQ(N_ * gt_size_, gt_blob->count());
	switch (top.size()) {
	case 3:
		// matrix [prd_size, gt_size] is used to count correct
		// characters by recording the shortest edit path each
		// element is one of 0(diagonal), 1(left) or 2(above)
		edit_path_.resize(prd_size_);
		for (int i = 0; i < prd_size_; i++)
			edit_path_[i].resize(gt_size_);
	case 2:
		// matrix [prd_size+1, gt_size+1] is used to calculate
		// edit distance with dynamic programming
		edit_dis_.resize(prd_size_ + 1);
		for (int i = 0; i <= prd_size_; i++)
			edit_dis_[i].resize(gt_size_ + 1);
	}
}

template <typename Dtype>
void SequenceAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int accuracy = 0; // correctly recognized sequences
  int distance = 0, gt_length = 0, prd_length = 0, correct_char = 0;
  const Dtype* prd_data = bottom[0]->cpu_data();
  const Dtype* gt_data = bottom[1]->cpu_data();
  for (int n = 0; n < N_; ++n){
	  const Dtype* prd_data_n = prd_data + bottom[0]->offset(n);
	  const Dtype* gt_data_n = gt_data + bottom[1]->offset(n);
	  int prd_len, gt_len;
	  bool accurate = true;
	  for (prd_len = 0, gt_len = 0;
		  prd_len < prd_size_ && gt_len < gt_size_;
		  ++prd_len, ++gt_len)
	  {
		  // either sequence ends
		  if (prd_data_n[prd_len] < 0 || gt_data_n[gt_len] < 0)
			  break; // these two sequences are the same until then
		  if (prd_data_n[prd_len] != gt_data_n[gt_len]){
			  accurate = false;
			  break;
		  }
	  }
	  if (accurate){
		  // has reached the end of predicted sequence, but not groundtruth sequence
		  if ((prd_len == prd_size_ || prd_data_n[prd_len] < 0) && gt_len < gt_size_ && gt_data_n[gt_len] >= 0){
			  if (top.size() > 1){
				  // get the groundtruth sequence length
				  while (++gt_len < gt_size_ && gt_data_n[gt_len] >= 0);
				  distance += gt_len - prd_len;
				  gt_length += gt_len;
				  prd_length += prd_len;
				  if (top.size() > 2){
					  correct_char += prd_len;
				  }
			  }
			  continue; // did not add the accuracy
		  }
		  // has reached the end of groundtruth sequence, but not predicted sequence
		  if ((gt_len == gt_size_ || gt_data_n[gt_len] < 0) && prd_len < prd_size_ && prd_data_n[prd_len] >= 0){
			  if (top.size() > 1){
				  // get the predicted sequence length
				  while (++prd_len < prd_size_ && prd_data_n[prd_len] >= 0);
				  distance += prd_len - gt_len;
				  gt_length += gt_len;
				  prd_length += prd_len;
				  if (top.size() > 2){
					  correct_char += gt_len;
				  }
			  }
			  continue; // did not add the accuracy
		  }
		  // there must be gt_len == prd_len
		  ++accuracy;
		  if (top.size() > 1){
			  //distance += 0;
			  gt_length += gt_len;
			  prd_length += prd_len;
			  if (top.size() > 2){
				  // there is prd_len == gt_len
				  correct_char += gt_len;
			  }
		  }
		  continue;
	  }
	  // there must be prd_data_n[i] != gt_data_n[i]
	  if (top.size() == 1)
		  continue;
	  // get the length of predicted and groundtruth sequences
	  if (prd_len < prd_size_ && prd_data_n[prd_len] >= 0){
		  while (++prd_len < prd_size_ && prd_data_n[prd_len] >= 0);
	  }
	  if (gt_len < gt_size_ && gt_data_n[gt_len] >= 0){
		  while (++gt_len < gt_size_ && gt_data_n[gt_len] >= 0);
	  }
	  // Initialize the edit distance matrix
	  for (int i = 0; i <= prd_len; i++) edit_dis_[i][0] = i;
	  for (int i = 1; i <= gt_len; i++) edit_dis_[0][i] = i;
	  // each character in predicted sequence 
	  for (int i = 1; i <= prd_len; i++)
	  {
		  // each character in groundtruth sequences
		  for (int j = 1; j <= gt_len; j++)
		  {
			  // update the matrix
			  const int above = edit_dis_[i - 1][j] + 1;
			  const int left = edit_dis_[i][j - 1] + 1;
			  const int diag = (prd_data_n[i - 1] == gt_data_n[j - 1]) ? edit_dis_[i - 1][j - 1] :
				  edit_dis_[i - 1][j - 1] + 1;
			  // elements' values in edit path matrix
			  // 0 for diagonal, 1 for left, 2 for above
			  if (diag < above)
			  {
				  if (diag < left) // diag
				  {
					  edit_dis_[i][j] = diag;
					  if (top.size() > 2){
						  edit_path_[i - 1][j - 1] = 0;
					  }
				  }
				  else // left
				  {
					  edit_dis_[i][j] = left;
					  if (top.size() > 2){
						  edit_path_[i - 1][j - 1] = 1;
					  }
				  }
			  }
			  else {
				  if (above < left) // above
				  {
					  edit_dis_[i][j] = above;
					  if (top.size() > 2){
						  edit_path_[i - 1][j - 1] = 2;
					  }
				  }
				  else { // left
					  edit_dis_[i][j] = left;
					  if (top.size() > 2){
						  edit_path_[i - 1][j - 1] = 1;
					  }
				  }
			  } // get the minimum value among the three
		  }
	  } // dynamic programming for edit distance
	  // there is { if (top.size() == 1) continue; } above
	  // there must be top.size() >= 2
	  distance += edit_dis_[prd_len][gt_len];
	  gt_length += gt_len;
	  prd_length += prd_len;
	  if (top.size() > 2){
		  int i = prd_len - 1, j = gt_len - 1;
		  while (i >= 0 && j >= 0) {
			  switch (edit_path_[i][j]) {
			  case 0: // diag
				  if (prd_data_n[i] == gt_data_n[j])
					  ++correct_char;
				  --i;
				  --j;
				  break;
			  case 1: // left
				  --j;
				  break;
			  case 2: // above
				  --i;
				  break;
			  default:
				  LOG(ERROR) << "Invalid value in edit path matrix "
					  "(0: diagonal, 1: left, 2: above):\n"
					  "matrix[" << i << "][" << j << "] = "
					  << edit_path_[i][j];
				  break;
			  }
		  }
	  } // the number of correct characters
  } // for each sample in a mini-batch

  switch (top.size()){
  case 3:
	  top[2]->mutable_cpu_data()[0] = Dtype(correct_char) / N_;
  case 2:
	  top[1]->mutable_cpu_data()[0] = Dtype(distance) / N_;
	  top[1]->mutable_cpu_data()[1] = Dtype(gt_length) / N_;
	  top[1]->mutable_cpu_data()[2] = Dtype(prd_length) / N_;
  case 1:
	  top[0]->mutable_cpu_data()[0] = Dtype(accuracy) / N_;
  }
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(SequenceAccuracyLayer);
REGISTER_LAYER_CLASS(SequenceAccuracy);

}  // namespace caffe
