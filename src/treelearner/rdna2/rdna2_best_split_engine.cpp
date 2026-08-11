/*!
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include "rdna2_best_split_engine.hpp"

#include <LightGBM/bin.h>

#include <chrono>
#include <cmath>

namespace LightGBM {

RDNA2BestSplitEngine::~RDNA2BestSplitEngine() {
#ifdef TIMETAG
  if (profile_calls_ > 0) {
    Log::Info("RDNA2 best-split shadow: calls=%llu raw_feature_match=%llu raw_choice_match=%llu exact_choice=%llu exact_payload=%llu h2d_ms=%.3f kernel_ms=%.3f d2h_ms=%.3f finalize_ms=%.3f",
              static_cast<unsigned long long>(profile_calls_),
              static_cast<unsigned long long>(profile_raw_feature_match_),
              static_cast<unsigned long long>(profile_raw_choice_match_),
              static_cast<unsigned long long>(profile_exact_choice_),
              static_cast<unsigned long long>(profile_exact_payload_),
              profile_h2d_ms_, profile_kernel_ms_, profile_d2h_ms_, profile_finalize_ms_);
  }
#endif
  if (stream_ != nullptr) {
    SynchronizeCUDAStream(stream_, __FILE__, __LINE__);
    CUDASUCCESS_OR_FATAL(cudaStreamDestroy(stream_));
    stream_ = nullptr;
  }
}

void RDNA2BestSplitEngine::Init(const Dataset* train_data, int gpu_device_id) {
  CHECK(train_data != nullptr);
  SetCUDADevice(gpu_device_id, __FILE__, __LINE__);
  if (stream_ == nullptr) {
    CUDASUCCESS_OR_FATAL(cudaStreamCreate(&stream_));
  }
  train_data_ = train_data;
  num_features_ = train_data_->num_features();
  num_total_bins_ = static_cast<size_t>(train_data_->NumTotalBin());
  eligible_ = num_features_ > 0;
  host_feature_meta_.resize(static_cast<size_t>(num_features_));
  for (int feature = 0; feature < num_features_; ++feature) {
    const BinMapper* mapper = train_data_->FeatureBinMapper(feature);
    if (mapper->bin_type() != BinType::NumericalBin || mapper->missing_type() != MissingType::None ||
        mapper->num_bin() > 128) {
      eligible_ = false;
    }
    FeatureMeta meta;
    meta.num_bin = static_cast<uint32_t>(mapper->num_bin());
    meta.offset = static_cast<uint8_t>(mapper->GetMostFreqBin() == 0);
    meta.real_feature = train_data_->RealFeatureIndex(feature);
    host_feature_meta_[static_cast<size_t>(feature)] = meta;
  }
  feature_meta_.InitFromHostVector(host_feature_meta_);
  host_hist_offsets_.resize(static_cast<size_t>(num_features_));
  host_used_features_.assign(static_cast<size_t>(num_features_), 0);
  hist_offsets_initialized_ = false;
  used_features_initialized_ = false;
  hist_offsets_.Resize(static_cast<size_t>(num_features_));
  used_features_.Resize(static_cast<size_t>(num_features_));
  results_.Resize(static_cast<size_t>(num_features_));
  histogram_.Resize(num_total_bins_ * 2);
  best_result_.Resize(1);
}
bool RDNA2BestSplitEngine::ShadowFind(const Config* config, FeatureHistogram* histogram_array,
                                       const std::vector<int8_t>& is_feature_used,
                                       const std::vector<int8_t>& node_used_features,
                                       double sum_gradients, double sum_hessians, data_size_t num_data,
                                       double parent_output, const SplitInfo& cpu_best, const char* leaf_name) {
  return ShadowFindImpl(config, histogram_array, nullptr, true, is_feature_used, node_used_features,
                        sum_gradients, sum_hessians, num_data, parent_output, &cpu_best, nullptr, leaf_name);
}

bool RDNA2BestSplitEngine::ShadowFindDevice(const Config* config, FeatureHistogram* histogram_array,
                                             const hist_t* device_histogram,
                                             const std::vector<int8_t>& is_feature_used,
                                             const std::vector<int8_t>& node_used_features,
                                             double sum_gradients, double sum_hessians, data_size_t num_data,
                                             double parent_output, const SplitInfo& cpu_best, const char* leaf_name) {
  if (device_histogram == nullptr) {
    return false;
  }
  return ShadowFindImpl(config, histogram_array, device_histogram, false, is_feature_used, node_used_features,
                        sum_gradients, sum_hessians, num_data, parent_output, &cpu_best, nullptr, leaf_name);
}

bool RDNA2BestSplitEngine::FindBestDeviceExact(const Config* config, FeatureHistogram* histogram_array,
                                                const hist_t* device_histogram,
                                                const std::vector<int8_t>& is_feature_used,
                                                const std::vector<int8_t>& node_used_features,
                                                double sum_gradients, double sum_hessians, data_size_t num_data,
                                                double parent_output, SplitInfo* exact_result) {
  if (device_histogram == nullptr || exact_result == nullptr) {
    return false;
  }
  return ShadowFindImpl(config, histogram_array, device_histogram, false, is_feature_used, node_used_features,
                        sum_gradients, sum_hessians, num_data, parent_output, nullptr, exact_result, "production");
}

bool RDNA2BestSplitEngine::ShadowFindImpl(const Config* config, FeatureHistogram* histogram_array,
                                           const hist_t* device_histogram, bool copy_host_histogram,
                                           const std::vector<int8_t>& is_feature_used,
                                           const std::vector<int8_t>& node_used_features,
                                           double sum_gradients, double sum_hessians, data_size_t num_data,
                                           double parent_output, const SplitInfo* cpu_best, SplitInfo* exact_result,
                                           const char* leaf_name) {
  if (!eligible_ || config == nullptr || histogram_array == nullptr || num_data <= 0 ||
      config->use_quantized_grad || config->extra_trees || !config->monotone_constraints.empty() ||
      !config->feature_contri.empty() || config->max_delta_step > 0.0 || config->path_smooth > kEpsilon ||
      is_feature_used.size() != static_cast<size_t>(num_features_) ||
      node_used_features.size() != static_cast<size_t>(num_features_)) {
    return false;
  }
  hist_t* host_base = histogram_array[0].RawData() - kHistOffset;
  bool upload_offsets = !hist_offsets_initialized_;
  bool upload_used_features = !used_features_initialized_;
  for (int feature = 0; feature < num_features_; ++feature) {
    if (upload_offsets) {
      const ptrdiff_t offset = histogram_array[feature].RawData() - host_base;
      if (offset < 0 || static_cast<size_t>(offset) >= num_total_bins_ * 2) {
        return false;
      }
      host_hist_offsets_[static_cast<size_t>(feature)] = static_cast<uint64_t>(offset);
    }
    const int8_t used = static_cast<int8_t>(
        is_feature_used[static_cast<size_t>(feature)] != 0 &&
        node_used_features[static_cast<size_t>(feature)] != 0);
    if (host_used_features_[static_cast<size_t>(feature)] != used) {
      host_used_features_[static_cast<size_t>(feature)] = used;
      upload_used_features = true;
    }
  }

#ifdef TIMETAG
  auto stage_start = std::chrono::steady_clock::now();
#endif
  if (copy_host_histogram) {
    CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(histogram_.RawData(), host_base,
                                         num_total_bins_ * 2 * sizeof(hist_t),
                                         cudaMemcpyHostToDevice, stream_));
    device_histogram = histogram_.RawDataReadOnly();
  }
  if (upload_offsets) {
    CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(hist_offsets_.RawData(), host_hist_offsets_.data(),
                                         host_hist_offsets_.size() * sizeof(uint64_t),
                                         cudaMemcpyHostToDevice, stream_));
    hist_offsets_initialized_ = true;
  }
  if (upload_used_features) {
    CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(used_features_.RawData(), host_used_features_.data(),
                                         host_used_features_.size() * sizeof(int8_t),
                                         cudaMemcpyHostToDevice, stream_));
    used_features_initialized_ = true;
  }
#ifdef TIMETAG
  auto stage_end = std::chrono::steady_clock::now();
  profile_h2d_ms_ += std::chrono::duration<double, std::milli>(stage_end - stage_start).count();
  stage_start = std::chrono::steady_clock::now();
#endif

  LaunchRDNA2BestSplitKernel(feature_meta_.RawDataReadOnly(), hist_offsets_.RawDataReadOnly(),
                             used_features_.RawDataReadOnly(), num_features_, device_histogram,
                             sum_gradients, sum_hessians, num_data, parent_output,
                             config->lambda_l1, config->lambda_l2, config->min_data_in_leaf,
                             config->min_sum_hessian_in_leaf, config->min_gain_to_split,
                             results_.RawData(), best_result_.RawData(), stream_);
  CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(&host_best_result_, best_result_.RawDataReadOnly(),
                                       sizeof(DeviceSplit), cudaMemcpyDeviceToHost, stream_));
  SynchronizeCUDAStream(stream_, __FILE__, __LINE__);
#ifdef TIMETAG
  stage_end = std::chrono::steady_clock::now();
  profile_kernel_ms_ += std::chrono::duration<double, std::milli>(stage_end - stage_start).count();
  ++profile_calls_;
#endif

  const DeviceSplit* gpu_best = host_best_result_.valid ? &host_best_result_ : nullptr;
  const bool gpu_valid = gpu_best != nullptr;
  SplitInfo finalized;
  bool finalized_valid = false;
  int finalized_inner_feature = -1;
#ifdef TIMETAG
  const auto finalize_start = std::chrono::steady_clock::now();
#endif
  if (gpu_valid) {
    finalized_inner_feature = train_data_->InnerFeatureIndex(gpu_best->feature);
    if (finalized_inner_feature >= 0) {
      histogram_array[finalized_inner_feature].FindBestThreshold(
          sum_gradients, sum_hessians, num_data, nullptr, parent_output, &finalized);
      finalized.feature = gpu_best->feature;
      finalized_valid = finalized.gain > kMinScore;
    }
  }
#ifdef TIMETAG
  const auto finalize_end = std::chrono::steady_clock::now();
  profile_finalize_ms_ += std::chrono::duration<double, std::milli>(finalize_end - finalize_start).count();
#endif
  if (exact_result != nullptr) {
    for (int feature = 0; feature < num_features_; ++feature) {
      if (host_used_features_[static_cast<size_t>(feature)] != 0 && feature != finalized_inner_feature) {
        histogram_array[feature].set_is_splittable(true);
      }
    }
    *exact_result = finalized;
  }
  if (cpu_best == nullptr) {
    return true;
  }

  const bool cpu_valid = cpu_best->gain > kMinScore;
  const bool raw_feature_match = cpu_valid == gpu_valid &&
      (!cpu_valid || cpu_best->feature == gpu_best->feature);
  const bool raw_choice_match = raw_feature_match && (!cpu_valid ||
      (cpu_best->threshold == gpu_best->threshold &&
       static_cast<uint8_t>(cpu_best->default_left) == gpu_best->default_left));
  const bool exact_choice = cpu_valid == finalized_valid &&
      (!cpu_valid || (cpu_best->feature == finalized.feature &&
                      cpu_best->threshold == finalized.threshold &&
                      cpu_best->default_left == finalized.default_left));
  const bool exact_payload = exact_choice && (!cpu_valid ||
      (cpu_best->left_count == finalized.left_count &&
       cpu_best->right_count == finalized.right_count &&
       cpu_best->left_output == finalized.left_output &&
       cpu_best->right_output == finalized.right_output &&
       cpu_best->gain == finalized.gain &&
       cpu_best->left_sum_gradient == finalized.left_sum_gradient &&
       cpu_best->left_sum_hessian == finalized.left_sum_hessian &&
       cpu_best->right_sum_gradient == finalized.right_sum_gradient &&
       cpu_best->right_sum_hessian == finalized.right_sum_hessian));
#ifdef TIMETAG
  if (raw_feature_match) {
    ++profile_raw_feature_match_;
  }
  if (raw_choice_match) {
    ++profile_raw_choice_match_;
  }
  if (exact_choice) {
    ++profile_exact_choice_;
  }
  if (exact_payload) {
    ++profile_exact_payload_;
  }
  if (!exact_choice) {
    Log::Warning("RDNA2 best-split finalized mismatch (%s): CPU feature=%d threshold=%u gain=%.17g, GPU feature=%d threshold=%u gain=%.17g, finalized feature=%d threshold=%u gain=%.17g",
                 leaf_name, cpu_best->feature, cpu_best->threshold, cpu_best->gain,
                 gpu_best == nullptr ? -1 : gpu_best->feature,
                 gpu_best == nullptr ? 0u : gpu_best->threshold,
                 gpu_best == nullptr ? kMinScore : gpu_best->gain,
                 finalized_valid ? finalized.feature : -1,
                 finalized_valid ? finalized.threshold : 0u,
                 finalized_valid ? finalized.gain : kMinScore);
  }
#endif
  return exact_choice;
}
}  // namespace LightGBM
