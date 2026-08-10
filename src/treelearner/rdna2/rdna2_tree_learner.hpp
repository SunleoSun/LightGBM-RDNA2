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

  ~RDNA2TreeLearner() override {
#ifdef TIMETAG
    if (profile_histogram_dispatch_calls_ > 0) {
      Log::Info("RDNA2 learner profile: dispatch_calls=%llu rdna2_calls=%llu serial_fallback_calls=%llu avg_used_features=%.1f/%d",
                static_cast<unsigned long long>(profile_histogram_dispatch_calls_),
                static_cast<unsigned long long>(profile_rdna2_histogram_calls_),
                static_cast<unsigned long long>(profile_serial_fallback_calls_),
                static_cast<double>(profile_used_features_total_) / profile_histogram_dispatch_calls_,
                num_features_);
    }
#endif
  }

  void Init(const Dataset* train_data, bool is_constant_hessian) override {
    SerialTreeLearner::Init(train_data, is_constant_hessian);
    histogram_engine_.Init(train_data, config_->gpu_device_id);
    histogram_engine_.RegisterDataIndices(data_partition_->indices(), static_cast<size_t>(num_data_));
  }

  void ResetTrainingDataInner(const Dataset* train_data, bool is_constant_hessian,
                              bool reset_multi_val_bin) override {
    histogram_engine_.UnregisterDataIndices();
    SerialTreeLearner::ResetTrainingDataInner(train_data, is_constant_hessian, reset_multi_val_bin);
    histogram_engine_.Init(train_data, config_->gpu_device_id);
    histogram_engine_.RegisterDataIndices(data_partition_->indices(), static_cast<size_t>(num_data_));
  }

 protected:
  void BeforeTrain() override {
    SerialTreeLearner::BeforeTrain();
    if (!config_->use_quantized_grad) {
      Common::FunctionTimer fun_timer("RDNA2TreeLearner::BeforeTrainH2D", global_timer);
      histogram_engine_.BeforeTrain(gradients_, hessians_);
    }
  }

  void ConstructHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract) override {
    Common::FunctionTimer fun_timer("RDNA2TreeLearner::ConstructHistograms", global_timer);
#ifdef TIMETAG
    ++profile_histogram_dispatch_calls_;
    profile_used_features_total_ += static_cast<uint64_t>(
        std::count(is_feature_used.begin(), is_feature_used.end(), static_cast<int8_t>(1)));
#endif
    const bool only_smaller_leaf_needed = larger_leaf_histogram_array_ == nullptr || use_subtract;
    if (!config_->use_quantized_grad && only_smaller_leaf_needed) {
      hist_t* ptr_smaller_leaf_hist_data = smaller_leaf_histogram_array_[0].RawData() - kHistOffset;
      if (histogram_engine_.ConstructHistogram(is_feature_used, smaller_leaf_splits_->data_indices(),
                                               smaller_leaf_splits_->num_data_in_leaf(),
                                               ptr_smaller_leaf_hist_data)) {
#ifdef TIMETAG
        ++profile_rdna2_histogram_calls_;
#endif
        return;
      }
    }
#ifdef TIMETAG
    ++profile_serial_fallback_calls_;
#endif
    SerialTreeLearner::ConstructHistograms(is_feature_used, use_subtract);
  }

 private:
  RDNA2HistogramEngine histogram_engine_;
#ifdef TIMETAG
  uint64_t profile_histogram_dispatch_calls_ = 0;
  uint64_t profile_rdna2_histogram_calls_ = 0;
  uint64_t profile_serial_fallback_calls_ = 0;
  uint64_t profile_used_features_total_ = 0;
#endif
};

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_TREELEARNER_RDNA2_RDNA2_TREE_LEARNER_HPP_
