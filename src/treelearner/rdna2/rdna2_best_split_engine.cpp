/*!
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include "rdna2_best_split_engine.hpp"

#include <LightGBM/bin.h>

#include <chrono>
#include <cmath>
#include <cstring>

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
  if (host_candidate_histograms_ != nullptr) {
    CUDASUCCESS_OR_FATAL(cudaFreeHost(host_candidate_histograms_));
    host_candidate_histograms_ = nullptr;
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
  host_top_results_.resize(8);
  if (host_candidate_histograms_ == nullptr) {
    CUDASUCCESS_OR_FATAL(cudaHostAlloc(reinterpret_cast<void**>(&host_candidate_histograms_),
                                       8 * kCandidateHistogramValues * sizeof(hist_t),
                                       cudaHostAllocPortable));
  }
  hist_offsets_initialized_ = false;
  used_features_initialized_ = false;
  hist_offsets_.Resize(static_cast<size_t>(num_features_));
  used_features_.Resize(static_cast<size_t>(num_features_));
  results_.Resize(static_cast<size_t>(num_features_) * 2);
  histogram_.Resize(num_total_bins_ * 2);
  best_result_.Resize(1);
  top_results_.Resize(8);
  candidate_histograms_.Resize(8 * kCandidateHistogramValues);
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
                                                 double parent_output, cudaEvent_t histogram_ready_event,
                                                 SplitInfo* exact_result) {
  if (device_histogram == nullptr || exact_result == nullptr) {
    return false;
  }
  if (histogram_ready_event != nullptr) {
    CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(stream_, histogram_ready_event, 0));
  }
  return ShadowFindImpl(config, histogram_array, device_histogram, false, is_feature_used, node_used_features,
                         sum_gradients, sum_hessians, num_data, parent_output, nullptr, exact_result, "production");
}

bool RDNA2BestSplitEngine::FindBestDeviceExactPair(
    const Config* config,
    FeatureHistogram* first_histogram_array, const hist_t* first_device_histogram,
    double first_sum_gradients, double first_sum_hessians, data_size_t first_num_data,
    double first_parent_output, SplitInfo* first_exact_result,
    FeatureHistogram* second_histogram_array, const hist_t* second_device_histogram,
    double second_sum_gradients, double second_sum_hessians, data_size_t second_num_data,
    double second_parent_output, SplitInfo* second_exact_result,
    const std::vector<int8_t>& is_feature_used,
    const std::vector<int8_t>& node_used_features,
    cudaEvent_t histogram_ready_event) {
  if (!eligible_ || config == nullptr || first_histogram_array == nullptr || second_histogram_array == nullptr ||
      first_device_histogram == nullptr || second_device_histogram == nullptr ||
      first_exact_result == nullptr || second_exact_result == nullptr ||
      first_num_data <= 0 || second_num_data <= 0 || config->use_quantized_grad || config->extra_trees ||
      !config->monotone_constraints.empty() || !config->feature_contri.empty() ||
      config->max_delta_step > 0.0 || config->path_smooth > kEpsilon ||
      is_feature_used.size() != static_cast<size_t>(num_features_) ||
      node_used_features.size() != static_cast<size_t>(num_features_)) {
    return false;
  }
  if (histogram_ready_event != nullptr) {
    CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(stream_, histogram_ready_event, 0));
  }

  hist_t* host_base = first_histogram_array[0].RawData() - kHistOffset;
  bool upload_offsets = !hist_offsets_initialized_;
  bool upload_used_features = !used_features_initialized_;
  for (int feature = 0; feature < num_features_; ++feature) {
    if (upload_offsets) {
      const ptrdiff_t offset = first_histogram_array[feature].RawData() - host_base;
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

  constexpr int kExactTopK = 2;
#ifdef TIMETAG
  const auto kernel_start = std::chrono::steady_clock::now();
#endif
  LaunchRDNA2BestSplitPairKernelAndGather(
      feature_meta_.RawDataReadOnly(), hist_offsets_.RawDataReadOnly(), used_features_.RawDataReadOnly(),
      num_features_, first_device_histogram, first_sum_gradients, first_sum_hessians, first_num_data,
      second_device_histogram, second_sum_gradients, second_sum_hessians, second_num_data,
      config->lambda_l1, config->lambda_l2, config->min_data_in_leaf, config->min_sum_hessian_in_leaf,
      config->min_gain_to_split, kExactTopK, results_.RawData(), top_results_.RawData(),
      candidate_histograms_.RawData(), stream_);
  CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(host_top_results_.data(), top_results_.RawDataReadOnly(),
                                       2 * kExactTopK * sizeof(DeviceSplit),
                                       cudaMemcpyDeviceToHost, stream_));
  CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(host_candidate_histograms_, candidate_histograms_.RawDataReadOnly(),
                                       2 * kExactTopK * kCandidateHistogramValues * sizeof(hist_t),
                                       cudaMemcpyDeviceToHost, stream_));
  SynchronizeCUDAStream(stream_, __FILE__, __LINE__);
#ifdef TIMETAG
  const auto kernel_end = std::chrono::steady_clock::now();
  profile_kernel_ms_ += std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count();
  profile_calls_ += 2;
  const auto finalize_start = std::chrono::steady_clock::now();
#endif

  std::vector<int> first_features;
  std::vector<int> second_features;
  first_features.reserve(kExactTopK);
  second_features.reserve(kExactTopK);
  auto materialize_histograms = [&](FeatureHistogram* histogram_array, int result_offset,
                                    std::vector<int>* inner_features) {
    for (int candidate_index = 0; candidate_index < kExactTopK; ++candidate_index) {
      const int slot = result_offset + candidate_index;
      const DeviceSplit& candidate = host_top_results_[static_cast<size_t>(slot)];
      if (!candidate.valid || candidate.inner_feature < 0) {
        continue;
      }
      const int inner_feature = candidate.inner_feature;
      inner_features->push_back(inner_feature);
      const auto& meta = host_feature_meta_[static_cast<size_t>(inner_feature)];
      const size_t hist_values = static_cast<size_t>(meta.num_bin - meta.offset) * 2;
      std::memcpy(histogram_array[inner_feature].RawData(),
                  host_candidate_histograms_ + static_cast<size_t>(slot) * kCandidateHistogramValues,
                  hist_values * sizeof(hist_t));
    }
  };
  materialize_histograms(first_histogram_array, 0, &first_features);
  materialize_histograms(second_histogram_array, kExactTopK, &second_features);

  auto finalize_leaf = [&](FeatureHistogram* histogram_array, const std::vector<int>& inner_features,
                           double sum_gradients, double sum_hessians, data_size_t num_data,
                           double parent_output, SplitInfo* exact_result) {
    SplitInfo finalized;
    std::vector<uint8_t> was_finalized(static_cast<size_t>(num_features_), 0);
    for (int inner_feature : inner_features) {
      was_finalized[static_cast<size_t>(inner_feature)] = 1;
      train_data_->FixHistogram(inner_feature, sum_gradients, sum_hessians,
                                histogram_array[inner_feature].RawData());
      SplitInfo candidate_split;
      histogram_array[inner_feature].FindBestThreshold(
          sum_gradients, sum_hessians, num_data, nullptr, parent_output, &candidate_split);
      candidate_split.feature = train_data_->RealFeatureIndex(inner_feature);
      if (candidate_split > finalized) {
        finalized = candidate_split;
      }
    }
    for (int feature = 0; feature < num_features_; ++feature) {
      if (host_used_features_[static_cast<size_t>(feature)] != 0 &&
          was_finalized[static_cast<size_t>(feature)] == 0) {
        histogram_array[feature].set_is_splittable(true);
      }
    }
    *exact_result = finalized;
  };
  finalize_leaf(first_histogram_array, first_features, first_sum_gradients, first_sum_hessians,
                first_num_data, first_parent_output, first_exact_result);
  finalize_leaf(second_histogram_array, second_features, second_sum_gradients, second_sum_hessians,
                second_num_data, second_parent_output, second_exact_result);
#ifdef TIMETAG
  const auto finalize_end = std::chrono::steady_clock::now();
  profile_finalize_ms_ += std::chrono::duration<double, std::milli>(finalize_end - finalize_start).count();
#endif
  return true;
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
  // GPU arithmetic is only a nominator; always canonically rescan the two best GPU features
  // in production so close CPU/GPU FP-order differences cannot change the selected feature.
  constexpr int kExactTopK = 2;
  const int top_k = exact_result != nullptr ? kExactTopK : 1;

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
                             config->min_sum_hessian_in_leaf, config->min_gain_to_split, top_k,
                             results_.RawData(), best_result_.RawData(), top_results_.RawData(), stream_);
  if (top_k == 1) {
    CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(&host_best_result_, best_result_.RawDataReadOnly(),
                                         sizeof(DeviceSplit), cudaMemcpyDeviceToHost, stream_));
  } else {
    CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(host_top_results_.data(), top_results_.RawDataReadOnly(),
                                         static_cast<size_t>(top_k) * sizeof(DeviceSplit),
                                         cudaMemcpyDeviceToHost, stream_));
  }
  SynchronizeCUDAStream(stream_, __FILE__, __LINE__);
#ifdef TIMETAG
  stage_end = std::chrono::steady_clock::now();
  profile_kernel_ms_ += std::chrono::duration<double, std::milli>(stage_end - stage_start).count();
  ++profile_calls_;
#endif

  const DeviceSplit* gpu_best = nullptr;
  if (top_k == 1) {
    gpu_best = host_best_result_.valid ? &host_best_result_ : nullptr;
  } else if (!host_top_results_.empty() && host_top_results_[0].valid) {
    gpu_best = &host_top_results_[0];
  }
  const bool gpu_valid = gpu_best != nullptr;
  SplitInfo finalized;
  bool finalized_valid = false;
  std::vector<int> finalized_inner_features;
#ifdef TIMETAG
  const auto finalize_start = std::chrono::steady_clock::now();
#endif
  if (gpu_valid) {
    const int candidate_count = top_k == 1 ? 1 : top_k;
    finalized_inner_features.reserve(static_cast<size_t>(candidate_count));
    for (int candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
      const DeviceSplit& candidate = top_k == 1 ? *gpu_best : host_top_results_[static_cast<size_t>(candidate_index)];
      if (!candidate.valid) {
        continue;
      }
      const int inner_feature = train_data_->InnerFeatureIndex(candidate.feature);
      if (inner_feature < 0) {
        continue;
      }
      finalized_inner_features.push_back(inner_feature);
      if (exact_result != nullptr && !copy_host_histogram) {
        const auto& meta = host_feature_meta_[static_cast<size_t>(inner_feature)];
        const size_t hist_values = static_cast<size_t>(meta.num_bin - meta.offset) * 2;
        CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(
            histogram_array[inner_feature].RawData(),
            device_histogram + host_hist_offsets_[static_cast<size_t>(inner_feature)],
            hist_values * sizeof(hist_t), cudaMemcpyDeviceToHost, stream_));
      }
    }
    if (exact_result != nullptr && !copy_host_histogram && !finalized_inner_features.empty()) {
      SynchronizeCUDAStream(stream_, __FILE__, __LINE__);
    }
    for (int inner_feature : finalized_inner_features) {
      if (exact_result != nullptr && !copy_host_histogram) {
        train_data_->FixHistogram(inner_feature, sum_gradients, sum_hessians,
                                  histogram_array[inner_feature].RawData());
      }
      SplitInfo candidate_split;
      histogram_array[inner_feature].FindBestThreshold(
          sum_gradients, sum_hessians, num_data, nullptr, parent_output, &candidate_split);
      candidate_split.feature = train_data_->RealFeatureIndex(inner_feature);
      if (candidate_split > finalized) {
        finalized = candidate_split;
      }
    }
    finalized_valid = finalized.gain > kMinScore;
  }
#ifdef TIMETAG
  const auto finalize_end = std::chrono::steady_clock::now();
  profile_finalize_ms_ += std::chrono::duration<double, std::milli>(finalize_end - finalize_start).count();
#endif
  if (exact_result != nullptr) {
    std::vector<uint8_t> was_finalized(static_cast<size_t>(num_features_), 0);
    for (int inner_feature : finalized_inner_features) {
      was_finalized[static_cast<size_t>(inner_feature)] = 1;
    }
    for (int feature = 0; feature < num_features_; ++feature) {
      if (host_used_features_[static_cast<size_t>(feature)] != 0 &&
          was_finalized[static_cast<size_t>(feature)] == 0) {
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
