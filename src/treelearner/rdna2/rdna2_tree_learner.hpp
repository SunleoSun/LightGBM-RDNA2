/*!
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_RDNA2_RDNA2_TREE_LEARNER_HPP_
#define LIGHTGBM_SRC_TREELEARNER_RDNA2_RDNA2_TREE_LEARNER_HPP_

#include "../serial_tree_learner.h"
#ifdef TIMETAG
#include "rdna2_best_split_engine.hpp"
#endif
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
    InitEngines();
  }

  void ResetTrainingDataInner(const Dataset* train_data, bool is_constant_hessian,
                              bool reset_multi_val_bin) override {
    histogram_engine_.ReleaseHostRegistrations();
    SerialTreeLearner::ResetTrainingDataInner(train_data, is_constant_hessian, reset_multi_val_bin);
    InitEngines();
  }

  void ResetConfig(const Config* config) override {
    CHECK_EQ(DeviceId(*config), DeviceId(*config_));
    const bool resize_pool = config->num_leaves != config_->num_leaves;
    if (resize_pool) {
      histogram_engine_.ReleaseHostRegistrations();
    }
    SerialTreeLearner::ResetConfig(config);
    if (resize_pool) {
      InitEngines();
    }
  }

  Tree* Train(const score_t* gradients, const score_t* hessians, bool is_first_tree) override {
    if (histogram_engine_.h128_eligible()) {
      SetCUDADevice(DeviceId(*config_), __FILE__, __LINE__);
    }
    std::unique_ptr<Tree> tree(SerialTreeLearner::Train(gradients, hessians, is_first_tree));
    histogram_engine_.WaitForPendingUploads();
    return tree.release();
  }

 protected:
  void BeforeTrain() override {
    SerialTreeLearner::BeforeTrain();
    if (!config_->use_quantized_grad) {
      Common::FunctionTimer fun_timer("RDNA2TreeLearner::BeforeTrainH2D", global_timer);
      histogram_engine_.BeforeTrain(gradients_, hessians_);
      if (histogram_engine_.h64_eligible() || histogram_engine_.h128_eligible()) {
        histogram_engine_.PreloadDataIndices(smaller_leaf_splits_->data_indices(),
                                             smaller_leaf_splits_->num_data_in_leaf());
      }
    }
  }

  void Split(Tree* tree, int best_leaf, int* left_leaf, int* right_leaf) override {
    histogram_engine_.WaitForPendingUploads();
    SerialTreeLearner::Split(tree, best_leaf, left_leaf, right_leaf);
    if (!config_->use_quantized_grad &&
        (histogram_engine_.h64_eligible() || histogram_engine_.h128_eligible())) {
      histogram_engine_.PreloadDataIndices(smaller_leaf_splits_->data_indices(),
                                           smaller_leaf_splits_->num_data_in_leaf());
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
    if (!config_->use_quantized_grad && histogram_engine_.h128_eligible() && only_smaller_leaf_needed) {
      hist_t* ptr_smaller_leaf_hist_data = smaller_leaf_histogram_array_[0].RawData() - kHistOffset;
      if (histogram_engine_.ConstructHistogram(is_feature_used, smaller_leaf_splits_->data_indices(),
                                                smaller_leaf_splits_->num_data_in_leaf(),
                                                ptr_smaller_leaf_hist_data, smaller_leaf_splits_->leaf_index(),
                                                larger_leaf_splits_ == nullptr ? -1 : larger_leaf_splits_->leaf_index(),
                                                use_subtract)) {
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

  // Serial owns split selection and the per-feature is_splittable flags used
  // to prune descendants. GPU top-K rescans do not preserve those flags.
#ifdef TIMETAG
  void FindBestSplitsFromHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract,
                                    const Tree* tree) override {
    SerialTreeLearner::FindBestSplitsFromHistograms(is_feature_used, use_subtract, tree);
    if (cegb_ == nullptr && best_split_engine_.eligible() &&
        config_->feature_fraction_bynode >= 1.0 && config_->interaction_constraints.empty()) {
      const std::vector<int8_t> node_used_features(static_cast<size_t>(num_features_), 1);
      const int smaller_leaf = smaller_leaf_splits_->leaf_index();
      if (histogram_engine_.best_split_pool_histogram_valid(smaller_leaf)) {
        best_split_engine_.ShadowFindDevice(config_, smaller_leaf_histogram_array_,
                                            histogram_engine_.best_split_pool_histogram(smaller_leaf),
                                            is_feature_used, node_used_features,
                                            smaller_leaf_splits_->sum_gradients(),
                                            smaller_leaf_splits_->sum_hessians(),
                                            smaller_leaf_splits_->num_data_in_leaf(),
                                            GetParentOutput(tree, smaller_leaf_splits_.get()),
                                            best_split_per_leaf_[smaller_leaf], "smaller-pool");
      }
      if (larger_leaf_splits_ != nullptr && larger_leaf_splits_->leaf_index() >= 0) {
        const int larger_leaf = larger_leaf_splits_->leaf_index();
        if (histogram_engine_.best_split_pool_histogram_valid(larger_leaf)) {
          best_split_engine_.ShadowFindDevice(config_, larger_leaf_histogram_array_,
                                              histogram_engine_.best_split_pool_histogram(larger_leaf),
                                              is_feature_used, node_used_features,
                                              larger_leaf_splits_->sum_gradients(),
                                              larger_leaf_splits_->sum_hessians(),
                                              larger_leaf_splits_->num_data_in_leaf(),
                                              GetParentOutput(tree, larger_leaf_splits_.get()),
                                              best_split_per_leaf_[larger_leaf], "larger-pool");
        }
      }
    }
  }
#endif

 private:
  static int DeviceId(const Config& config) {
    return config.gpu_device_id < 0 ? 0 : config.gpu_device_id;
  }

  void InitEngines() {
    const int device = DeviceId(*config_);
    int shadow_pool_leaves = 0;
#ifdef TIMETAG
    if (histogram_pool_.HasFullCapacity()) {
      shadow_pool_leaves = config_->num_leaves;
    }
#endif
    histogram_engine_.Init(train_data_, device, shadow_pool_leaves);
    histogram_engine_.RegisterDataIndices(data_partition_->indices(), static_cast<size_t>(num_data_));
#ifdef TIMETAG
    if (histogram_engine_.best_split_pool_enabled()) {
      best_split_engine_.Init(train_data_, device);
    }
#endif
  }

  RDNA2HistogramEngine histogram_engine_;
#ifdef TIMETAG
  RDNA2BestSplitEngine best_split_engine_;
  uint64_t profile_histogram_dispatch_calls_ = 0;
  uint64_t profile_rdna2_histogram_calls_ = 0;
  uint64_t profile_serial_fallback_calls_ = 0;
  uint64_t profile_used_features_total_ = 0;
#endif
};

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_TREELEARNER_RDNA2_RDNA2_TREE_LEARNER_HPP_
