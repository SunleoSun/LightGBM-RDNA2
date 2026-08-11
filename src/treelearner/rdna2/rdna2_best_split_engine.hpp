/*!
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_RDNA2_RDNA2_BEST_SPLIT_ENGINE_HPP_
#define LIGHTGBM_SRC_TREELEARNER_RDNA2_RDNA2_BEST_SPLIT_ENGINE_HPP_

#include <LightGBM/cuda/cuda_utils.hu>
#include <LightGBM/dataset.h>
#include <LightGBM/utils/log.h>

#include <cstdint>
#include <vector>

#include "../feature_histogram.hpp"
#include "../split_info.hpp"

namespace LightGBM {

class RDNA2BestSplitEngine {
 public:
  struct FeatureMeta {
    uint32_t num_bin;
    uint8_t offset;
    int real_feature;
  };

  struct DeviceSplit {
    int feature;
    int inner_feature;
    uint32_t threshold;
    data_size_t left_count;
    data_size_t right_count;
    double left_output;
    double right_output;
    double gain;
    double left_sum_gradient;
    double left_sum_hessian;
    double right_sum_gradient;
    double right_sum_hessian;
    uint8_t default_left;
    uint8_t valid;
  };

  static constexpr size_t kCandidateHistogramValues = 256;

  RDNA2BestSplitEngine() = default;
  ~RDNA2BestSplitEngine();

  void Init(const Dataset* train_data, int gpu_device_id);

  bool eligible() const { return eligible_; }

  bool ShadowFind(const Config* config, FeatureHistogram* histogram_array,
                  const std::vector<int8_t>& is_feature_used,
                  const std::vector<int8_t>& node_used_features,
                  double sum_gradients, double sum_hessians, data_size_t num_data,
                  double parent_output, const SplitInfo& cpu_best, const char* leaf_name);

  bool ShadowFindDevice(const Config* config, FeatureHistogram* histogram_array,
                        const hist_t* device_histogram,
                        const std::vector<int8_t>& is_feature_used,
                        const std::vector<int8_t>& node_used_features,
                        double sum_gradients, double sum_hessians, data_size_t num_data,
                        double parent_output, const SplitInfo& cpu_best, const char* leaf_name);
  bool FindBestDeviceExact(const Config* config, FeatureHistogram* histogram_array,
                            const hist_t* device_histogram,
                            const std::vector<int8_t>& is_feature_used,
                            const std::vector<int8_t>& node_used_features,
                            double sum_gradients, double sum_hessians, data_size_t num_data,
                            double parent_output, SplitInfo* exact_result);
  bool FindBestDeviceExactPair(const Config* config,
                               FeatureHistogram* first_histogram_array, const hist_t* first_device_histogram,
                               double first_sum_gradients, double first_sum_hessians, data_size_t first_num_data,
                               double first_parent_output, SplitInfo* first_exact_result,
                               FeatureHistogram* second_histogram_array, const hist_t* second_device_histogram,
                               double second_sum_gradients, double second_sum_hessians, data_size_t second_num_data,
                               double second_parent_output, SplitInfo* second_exact_result,
                               const std::vector<int8_t>& is_feature_used,
                               const std::vector<int8_t>& node_used_features);

 private:
  bool ShadowFindImpl(const Config* config, FeatureHistogram* histogram_array,
                      const hist_t* device_histogram, bool copy_host_histogram,
                      const std::vector<int8_t>& is_feature_used,
                      const std::vector<int8_t>& node_used_features,
                      double sum_gradients, double sum_hessians, data_size_t num_data,
                      double parent_output, const SplitInfo* cpu_best, SplitInfo* exact_result,
                      const char* leaf_name);
  const Dataset* train_data_ = nullptr;
  int num_features_ = 0;
  size_t num_total_bins_ = 0;
  bool eligible_ = false;
  cudaStream_t stream_ = nullptr;
  std::vector<FeatureMeta> host_feature_meta_;
  std::vector<uint64_t> host_hist_offsets_;
  std::vector<int8_t> host_used_features_;
  std::vector<DeviceSplit> host_top_results_;
  hist_t* host_candidate_histograms_ = nullptr;
  bool hist_offsets_initialized_ = false;
  bool used_features_initialized_ = false;
  DeviceSplit host_best_result_{};
  CUDAVector<FeatureMeta> feature_meta_;
  CUDAVector<uint64_t> hist_offsets_;
  CUDAVector<int8_t> used_features_;
  CUDAVector<hist_t> histogram_;
  CUDAVector<DeviceSplit> results_;
  CUDAVector<DeviceSplit> best_result_;
  CUDAVector<DeviceSplit> top_results_;
  CUDAVector<hist_t> candidate_histograms_;
#ifdef TIMETAG
  uint64_t profile_calls_ = 0;
  uint64_t profile_raw_feature_match_ = 0;
  uint64_t profile_raw_choice_match_ = 0;
  uint64_t profile_exact_choice_ = 0;
  uint64_t profile_exact_payload_ = 0;
  double profile_h2d_ms_ = 0.0;
  double profile_kernel_ms_ = 0.0;
  double profile_d2h_ms_ = 0.0;
  double profile_finalize_ms_ = 0.0;
#endif
};

void LaunchRDNA2BestSplitKernel(const RDNA2BestSplitEngine::FeatureMeta* feature_meta,
                                 const uint64_t* hist_offsets, const int8_t* used_features,
                                 int num_features, const hist_t* histogram,
                                 double sum_gradients, double sum_hessians, data_size_t num_data,
                                 double parent_output, double lambda_l1, double lambda_l2,
                                 data_size_t min_data_in_leaf, double min_sum_hessian_in_leaf,
                                 double min_gain_to_split, int top_k,
                                 RDNA2BestSplitEngine::DeviceSplit* results,
                                 RDNA2BestSplitEngine::DeviceSplit* best_result,
                                 RDNA2BestSplitEngine::DeviceSplit* top_results, cudaStream_t stream);
void LaunchRDNA2BestSplitPairKernelAndGather(
    const RDNA2BestSplitEngine::FeatureMeta* feature_meta, const uint64_t* hist_offsets,
    const int8_t* used_features, int num_features,
    const hist_t* first_histogram, double first_sum_gradients, double first_sum_hessians,
    data_size_t first_num_data, const hist_t* second_histogram, double second_sum_gradients,
    double second_sum_hessians, data_size_t second_num_data, double lambda_l1, double lambda_l2,
    data_size_t min_data_in_leaf, double min_sum_hessian_in_leaf, double min_gain_to_split, int top_k,
    RDNA2BestSplitEngine::DeviceSplit* results, RDNA2BestSplitEngine::DeviceSplit* top_results,
    hist_t* candidate_histograms, cudaStream_t stream);

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_TREELEARNER_RDNA2_RDNA2_BEST_SPLIT_ENGINE_HPP_
