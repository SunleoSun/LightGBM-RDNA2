/*!
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_RDNA2_RDNA2_TREE_LEARNER_HPP_
#define LIGHTGBM_SRC_TREELEARNER_RDNA2_RDNA2_TREE_LEARNER_HPP_

#include "../serial_tree_learner.h"
#include "rdna2_histogram_engine.hpp"

namespace LightGBM {

/*! \brief RDNA2 learner boundary. Host-side tree semantics intentionally remain SerialTreeLearner semantics. */
class RDNA2TreeLearner final : public SerialTreeLearner {
 public:
  explicit RDNA2TreeLearner(const Config* config) : SerialTreeLearner(config) {}

  void Init(const Dataset* train_data, bool is_constant_hessian) override {
    SerialTreeLearner::Init(train_data, is_constant_hessian);
    histogram_engine_.Init(train_data, config_->gpu_device_id);
  }

  void ResetTrainingDataInner(const Dataset* train_data, bool is_constant_hessian,
                              bool reset_multi_val_bin) override {
    SerialTreeLearner::ResetTrainingDataInner(train_data, is_constant_hessian, reset_multi_val_bin);
    histogram_engine_.Init(train_data, config_->gpu_device_id);
  }

 protected:
  void BeforeTrain() override {
    SerialTreeLearner::BeforeTrain();
    if (!config_->use_quantized_grad) {
      histogram_engine_.BeforeTrain(gradients_, hessians_);
    }
  }

  void ConstructHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract) override {
    const bool only_smaller_leaf_needed = larger_leaf_histogram_array_ == nullptr || use_subtract;
    if (!config_->use_quantized_grad && only_smaller_leaf_needed) {
      hist_t* ptr_smaller_leaf_hist_data = smaller_leaf_histogram_array_[0].RawData() - kHistOffset;
      if (histogram_engine_.ConstructHistogram(smaller_leaf_splits_->data_indices(),
                                               smaller_leaf_splits_->num_data_in_leaf(),
                                               ptr_smaller_leaf_hist_data)) {
        return;
      }
    }
    SerialTreeLearner::ConstructHistograms(is_feature_used, use_subtract);
  }

 private:
  RDNA2HistogramEngine histogram_engine_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_TREELEARNER_RDNA2_RDNA2_TREE_LEARNER_HPP_
