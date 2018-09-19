#include <vector>

#include "caffe/layers/ternary_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void TernaryConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
  
  ternary_weights_.ReshapeLike(*this->blobs_[0]);
  alphas_.Reshape(this->num_output_,1,1,1);
  weight_sum_multiplier_.Reshape(this->blobs_[0]->count(1),1,1,1);
  threshold_.Reshape(this->num_output_,1,1,1);
  
  skip_quantization_ = false;
}

template <typename Dtype>
void TernaryConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void TernaryConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

#ifdef CPU_ONLY
STUB_GPU(TernaryConvolutionLayer);
#endif

INSTANTIATE_CLASS(TernaryConvolutionLayer);
REGISTER_LAYER_CLASS(TernaryConvolution);
}  // namespace caffe
