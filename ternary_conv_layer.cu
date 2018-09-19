#include <vector>
#include <algorithm>

#include "caffe/layers/ternary_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void TernaryWeightQuant(const int n, const int weight_dim, const Dtype* weight, 
        const Dtype* threshold, Dtype* ternary_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    int i = index/weight_dim;
    Dtype ternary_code = weight[index] > Dtype(0) ? Dtype(1) : Dtype(-1);
    ternary_weight[index] = fabs(weight[index]) >= threshold[i] ? ternary_code : Dtype(0);
  }
}

template <typename Dtype>
__global__ void TernaryWeightForward(const int n, const int weight_dim, const Dtype* weight, 
        const Dtype* alpha, Dtype* ternary_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    int i = index/weight_dim;
    ternary_weight[index] = weight[index] * alpha[i];
  }
}

template <typename Dtype>
void TernaryConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // initialization for ternary parameters
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const int weight_dim = this->blobs_[0]->count(1);
  
  if (skip_quantization_ == false) {
    caffe_gpu_abs(this->blobs_[0]->count(), weight, ternary_weights_.mutable_gpu_data());
    caffe_gpu_set(weight_sum_multiplier_.count(),Dtype(1),weight_sum_multiplier_.mutable_gpu_data());
    const int nthreads = this->blobs_[0]->count();
    Dtype* threshold_ptr = threshold_.mutable_cpu_data();
  
    for (int i = 0; i < this->blobs_[0]->num(); i++) {
        Dtype* kernel_mutable_cpu_data = ternary_weights_.mutable_cpu_data()+i*this->blobs_[0]->count(1);
        std::sort(kernel_mutable_cpu_data, kernel_mutable_cpu_data+this->blobs_[0]->count(1));
        int r = 0;
        Dtype s = 0;
        Dtype loss_max = Dtype(1e-5);
        int idx = 1;
        for (int j = this->blobs_[0]->count(1)-1; j >=0; j--) {
            s += kernel_mutable_cpu_data[j];  r++;
            const Dtype loss = s*s/r;
            if (loss >= loss_max) {
                loss_max = loss;
                idx = j;
            }
        }
        threshold_ptr[i] = kernel_mutable_cpu_data[idx];
    }
  
    TernaryWeightQuant<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
            nthreads, weight_dim, weight, threshold_.gpu_data(), ternary_weights_.mutable_gpu_data());
  
    const int output_channel_num = this->num_output_;
    const int kernel_dim = this->kernel_dim_;
  
    caffe_gpu_mul(output_channel_num*kernel_dim, weight, ternary_weights_.gpu_data(), 
                    ternary_weights_.mutable_gpu_diff());
    caffe_gpu_gemv<Dtype>(CblasNoTrans, output_channel_num, kernel_dim, (Dtype)1., 									
							ternary_weights_.gpu_diff(), weight_sum_multiplier_.gpu_data(), 
                                (Dtype)0., alphas_.mutable_gpu_data());
    caffe_gpu_mul(output_channel_num*kernel_dim, ternary_weights_.gpu_data(), 
                    ternary_weights_.gpu_data(), ternary_weights_.mutable_gpu_diff());
    caffe_gpu_gemv<Dtype>(CblasNoTrans, output_channel_num, kernel_dim, 										
                            (Dtype)1., ternary_weights_.gpu_diff(),	weight_sum_multiplier_.gpu_data(), 
                                (Dtype)0., alphas_.mutable_gpu_diff());
    caffe_gpu_div(output_channel_num, alphas_.gpu_data(), alphas_.gpu_diff(), alphas_.mutable_gpu_data());
  
    TernaryWeightForward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
            nthreads, weight_dim, ternary_weights_.gpu_data(), alphas_.gpu_data(), ternary_weights_.mutable_gpu_data());
  }
  skip_quantization_ = this->phase_ == TEST;
  
  const Dtype* ternary_weights = ternary_weights_.gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, ternary_weights,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void TernaryConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* ternary_weights = ternary_weights_.gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, ternary_weights,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TernaryConvolutionLayer);

}  // namespace caffe
