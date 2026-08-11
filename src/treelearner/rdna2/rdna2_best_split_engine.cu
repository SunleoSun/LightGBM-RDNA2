/*!
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include "rdna2_best_split_engine.hpp"

namespace LightGBM {
namespace {

constexpr int kBestSplitThreads = 256;  // eight native RDNA2 wave32s; one feature per wave
constexpr int kWaveSize = 32;
constexpr int kBinsPerLane = 4;

__device__ inline double ThresholdL1Device(double value, double l1) {
  const double magnitude = fmax(0.0, fabs(value) - l1);
  return value > 0.0 ? magnitude : (value < 0.0 ? -magnitude : 0.0);
}

__device__ inline double LeafGain(double grad, double hess, double l1, double l2) {
  const double reg_grad = l1 > 0.0 ? ThresholdL1Device(grad, l1) : grad;
  return (reg_grad * reg_grad) / (hess + l2);
}

__device__ inline double LeafOutput(double grad, double hess, double l1, double l2) {
  const double reg_grad = l1 > 0.0 ? ThresholdL1Device(grad, l1) : grad;
  return -reg_grad / (hess + l2);
}

__global__ void RDNA2BestSplitKernel(
    const RDNA2BestSplitEngine::FeatureMeta* feature_meta, const uint64_t* hist_offsets,
    const int8_t* used_features, int num_features,
    const hist_t* first_histogram, const hist_t* second_histogram,
    double first_sum_gradients, double first_sum_hessians, data_size_t first_num_data,
    double second_sum_gradients, double second_sum_hessians, data_size_t second_num_data,
    double lambda_l1, double lambda_l2, data_size_t min_data_in_leaf,
    double min_sum_hessian_in_leaf, double min_gain_to_split, int leaf_count,
    RDNA2BestSplitEngine::DeviceSplit* results) {
  constexpr int kWavesPerBlock = kBestSplitThreads / kWaveSize;
  const int wave = static_cast<int>(threadIdx.x) / kWaveSize;
  const int lane = static_cast<int>(threadIdx.x) & (kWaveSize - 1);
  const int feature = static_cast<int>(blockIdx.x) * kWavesPerBlock + wave;
  const int leaf = static_cast<int>(blockIdx.y);
  if (feature >= num_features || leaf >= leaf_count) {
    return;
  }
  const hist_t* histogram = leaf == 0 ? first_histogram : second_histogram;
  const double raw_sum_gradients = leaf == 0 ? first_sum_gradients : second_sum_gradients;
  const double raw_sum_hessians = leaf == 0 ? first_sum_hessians : second_sum_hessians;
  const data_size_t num_data = leaf == 0 ? first_num_data : second_num_data;
  results += static_cast<size_t>(leaf) * static_cast<size_t>(num_features);

  RDNA2BestSplitEngine::DeviceSplit out{};
  out.feature = feature_meta[feature].real_feature;
  out.inner_feature = feature;
  out.gain = kMinScore;
  out.default_left = 1;
  out.valid = 0;
  if (used_features[feature] == 0) {
    if (lane == 0) {
      results[feature] = out;
    }
    return;
  }

  const auto meta = feature_meta[feature];
  const hist_t* data = histogram + hist_offsets[feature];
  const int offset = static_cast<int>(meta.offset);
  const int num_thresholds = static_cast<int>(meta.num_bin) - 1;
  const double sum_gradient = raw_sum_gradients;
  const double sum_hessian = raw_sum_hessians;
  const double cnt_factor = static_cast<double>(num_data) / sum_hessian;
  const double min_gain_shift = LeafGain(sum_gradient, sum_hessian, lambda_l1, lambda_l2) + min_gain_to_split;

  double local_grad[kBinsPerLane] = {0.0, 0.0, 0.0, 0.0};
  double local_hess[kBinsPerLane] = {0.0, 0.0, 0.0, 0.0};
  data_size_t local_count[kBinsPerLane] = {0, 0, 0, 0};
  double lane_grad_total = 0.0;
  double lane_hess_total = 0.0;
  data_size_t lane_count_total = 0;
#pragma unroll
  for (int j = 0; j < kBinsPerLane; ++j) {
    const int pos = lane * kBinsPerLane + j;
    if (pos < num_thresholds) {
      const int t = static_cast<int>(meta.num_bin) - 1 - offset - pos;
      const double grad = static_cast<double>(data[t * 2]);
      const double hess = static_cast<double>(data[t * 2 + 1]);
      lane_grad_total += grad;
      lane_hess_total += hess;
      lane_count_total += static_cast<data_size_t>(hess * cnt_factor + 0.5f);
    }
    local_grad[j] = lane_grad_total;
    local_hess[j] = lane_hess_total;
    local_count[j] = lane_count_total;
  }
  const double own_grad_total = lane_grad_total;
  const double own_hess_total = lane_hess_total;
  const data_size_t own_count_total = lane_count_total;
#pragma unroll
  for (int delta = 1; delta < kWaveSize; delta <<= 1) {
    const double prev_grad = __shfl_up(lane_grad_total, delta, kWaveSize);
    const double prev_hess = __shfl_up(lane_hess_total, delta, kWaveSize);
    const data_size_t prev_count = __shfl_up(lane_count_total, delta, kWaveSize);
    if (lane >= delta) {
      lane_grad_total += prev_grad;
      lane_hess_total += prev_hess;
      lane_count_total += prev_count;
    }
  }
  const double prefix_grad = lane_grad_total - own_grad_total;
  const double prefix_hess = lane_hess_total - own_hess_total;
  const data_size_t prefix_count = lane_count_total - own_count_total;

  double best_gain = kMinScore;
  int best_pos = 0x7fffffff;
  uint32_t best_threshold = meta.num_bin;
  data_size_t best_left_count = 0;
  double best_sum_left_gradient = 0.0;
  double best_sum_left_hessian = 0.0;
#pragma unroll
  for (int j = 0; j < kBinsPerLane; ++j) {
    const int pos = lane * kBinsPerLane + j;
    if (pos >= num_thresholds) {
      continue;
    }
    const double sum_right_gradient = prefix_grad + local_grad[j];
    const double sum_right_hessian = static_cast<double>(kEpsilon) + prefix_hess + local_hess[j];
    const data_size_t right_count = prefix_count + local_count[j];
    if (right_count < min_data_in_leaf || sum_right_hessian < min_sum_hessian_in_leaf) {
      continue;
    }
    const data_size_t left_count = num_data - right_count;
    if (left_count < min_data_in_leaf) {
      continue;
    }
    const double sum_left_hessian = sum_hessian - sum_right_hessian;
    if (sum_left_hessian < min_sum_hessian_in_leaf) {
      continue;
    }
    const double sum_left_gradient = sum_gradient - sum_right_gradient;
    const double current_gain = LeafGain(sum_left_gradient, sum_left_hessian, lambda_l1, lambda_l2) +
                                LeafGain(sum_right_gradient, sum_right_hessian, lambda_l1, lambda_l2);
    if (current_gain > min_gain_shift && current_gain > best_gain) {
      best_gain = current_gain;
      best_pos = pos;
      const int t = static_cast<int>(meta.num_bin) - 1 - offset - pos;
      best_threshold = static_cast<uint32_t>(t - 1 + offset);
      best_left_count = left_count;
      best_sum_left_gradient = sum_left_gradient;
      best_sum_left_hessian = sum_left_hessian;
    }
  }

  int winner_lane = lane;
#pragma unroll
  for (int delta = kWaveSize / 2; delta > 0; delta >>= 1) {
    const double other_gain = __shfl_down(best_gain, delta, kWaveSize);
    const int other_pos = __shfl_down(best_pos, delta, kWaveSize);
    const int other_lane = __shfl_down(winner_lane, delta, kWaveSize);
    if (lane + delta < kWaveSize &&
        (other_gain > best_gain || (other_gain == best_gain && other_pos < best_pos))) {
      best_gain = other_gain;
      best_pos = other_pos;
      winner_lane = other_lane;
    }
  }
  const uint32_t selected_threshold = __shfl(best_threshold, winner_lane, kWaveSize);
  const data_size_t selected_left_count = __shfl(best_left_count, winner_lane, kWaveSize);
  const double selected_left_gradient = __shfl(best_sum_left_gradient, winner_lane, kWaveSize);
  const double selected_left_hessian = __shfl(best_sum_left_hessian, winner_lane, kWaveSize);
  if (lane == 0) {
    if (best_gain > min_gain_shift) {
      out.threshold = selected_threshold;
      out.left_count = selected_left_count;
      out.right_count = num_data - out.left_count;
      out.left_sum_gradient = selected_left_gradient;
      out.left_sum_hessian = selected_left_hessian - static_cast<double>(kEpsilon);
      out.right_sum_gradient = sum_gradient - out.left_sum_gradient;
      out.right_sum_hessian = sum_hessian - selected_left_hessian - static_cast<double>(kEpsilon);
      out.left_output = LeafOutput(out.left_sum_gradient, selected_left_hessian, lambda_l1, lambda_l2);
      out.right_output = LeafOutput(sum_gradient - out.left_sum_gradient,
                                   sum_hessian - selected_left_hessian, lambda_l1, lambda_l2);
      out.gain = best_gain - min_gain_shift;
      out.valid = 1;
    }
    results[feature] = out;
  }
}

__device__ inline bool BetterDeviceSplit(const RDNA2BestSplitEngine::DeviceSplit& candidate,
                                         const RDNA2BestSplitEngine::DeviceSplit& current) {
  if (!candidate.valid) {
    return false;
  }
  if (!current.valid) {
    return true;
  }
  return candidate.gain > current.gain ||
         (candidate.gain == current.gain && candidate.feature < current.feature);
}

__global__ void RDNA2BestSplitReduceKernel(const RDNA2BestSplitEngine::DeviceSplit* results,
                                             int num_features,
                                             RDNA2BestSplitEngine::DeviceSplit* best_result) {
  __shared__ RDNA2BestSplitEngine::DeviceSplit shared[kBestSplitThreads];
  const int tid = static_cast<int>(threadIdx.x);
  RDNA2BestSplitEngine::DeviceSplit local{};
  local.feature = -1;
  local.gain = kMinScore;
  local.valid = 0;
  for (int feature = tid; feature < num_features; feature += kBestSplitThreads) {
    const auto candidate = results[feature];
    if (BetterDeviceSplit(candidate, local)) {
      local = candidate;
    }
  }
  shared[tid] = local;
  __syncthreads();
  for (int stride = kBestSplitThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride && BetterDeviceSplit(shared[tid + stride], shared[tid])) {
      shared[tid] = shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    best_result[0] = shared[0];
  }
}

constexpr int kMaxTopK = 8;

__global__ void RDNA2BestSplitTopKKernel(
    const RDNA2BestSplitEngine::DeviceSplit* all_results, int num_features, int top_k,
    RDNA2BestSplitEngine::DeviceSplit* all_top_results, const hist_t* first_histogram,
    const hist_t* second_histogram, const uint64_t* hist_offsets,
    const RDNA2BestSplitEngine::FeatureMeta* feature_meta, hist_t* all_candidate_histograms,
    int leaf_count) {
  const int leaf = static_cast<int>(blockIdx.x);
  if (leaf >= leaf_count) {
    return;
  }
  const RDNA2BestSplitEngine::DeviceSplit* results =
      all_results + static_cast<size_t>(leaf) * static_cast<size_t>(num_features);
  RDNA2BestSplitEngine::DeviceSplit* top_results =
      all_top_results + static_cast<size_t>(leaf) * static_cast<size_t>(top_k);
  const hist_t* histogram = leaf == 0 ? first_histogram : second_histogram;
  hist_t* candidate_histograms = all_candidate_histograms == nullptr ? nullptr :
      all_candidate_histograms + static_cast<size_t>(leaf) * static_cast<size_t>(top_k) *
          RDNA2BestSplitEngine::kCandidateHistogramValues;
  __shared__ RDNA2BestSplitEngine::DeviceSplit shared[kBestSplitThreads];
  __shared__ int selected_features[kMaxTopK];
  const int tid = static_cast<int>(threadIdx.x);
  for (int rank = 0; rank < top_k; ++rank) {
    RDNA2BestSplitEngine::DeviceSplit local{};
    local.feature = -1;
    local.gain = kMinScore;
    local.valid = 0;
    for (int feature = tid; feature < num_features; feature += kBestSplitThreads) {
      const auto candidate = results[feature];
      bool already_selected = false;
#pragma unroll
      for (int selected = 0; selected < kMaxTopK; ++selected) {
        if (selected < rank && candidate.inner_feature == selected_features[selected]) {
          already_selected = true;
        }
      }
      if (!already_selected && BetterDeviceSplit(candidate, local)) {
        local = candidate;
      }
    }
    shared[tid] = local;
    __syncthreads();
    for (int stride = kBestSplitThreads / 2; stride > 0; stride >>= 1) {
      if (tid < stride && BetterDeviceSplit(shared[tid + stride], shared[tid])) {
        shared[tid] = shared[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      top_results[rank] = shared[0];
      selected_features[rank] = shared[0].valid ? shared[0].inner_feature : -1;
    }
    __syncthreads();
  }
  if (candidate_histograms == nullptr || histogram == nullptr) {
    return;
  }
  for (int rank = 0; rank < top_k; ++rank) {
    const int inner_feature = selected_features[rank];
    if (inner_feature < 0) {
      continue;
    }
    const auto meta = feature_meta[inner_feature];
    const size_t histogram_values = static_cast<size_t>(meta.num_bin - meta.offset) * 2;
    if (static_cast<size_t>(tid) < histogram_values) {
      candidate_histograms[static_cast<size_t>(rank) * RDNA2BestSplitEngine::kCandidateHistogramValues +
                           static_cast<size_t>(tid)] =
          histogram[hist_offsets[inner_feature] + static_cast<size_t>(tid)];
    }
  }
}

}  // namespace

void LaunchRDNA2BestSplitKernel(const RDNA2BestSplitEngine::FeatureMeta* feature_meta,
                                 const uint64_t* hist_offsets, const int8_t* used_features,
                                 int num_features, const hist_t* histogram,
                                 double sum_gradients, double sum_hessians, data_size_t num_data,
                                 double parent_output, double lambda_l1, double lambda_l2,
                                 data_size_t min_data_in_leaf, double min_sum_hessian_in_leaf,
                                 double min_gain_to_split, int top_k,
                                 RDNA2BestSplitEngine::DeviceSplit* results,
                                 RDNA2BestSplitEngine::DeviceSplit* best_result,
                                 RDNA2BestSplitEngine::DeviceSplit* top_results, cudaStream_t stream) {
  (void)parent_output;
  const dim3 block(kBestSplitThreads);
  constexpr int kWavesPerBlock = kBestSplitThreads / kWaveSize;
  const dim3 grid(static_cast<unsigned int>((num_features + kWavesPerBlock - 1) / kWavesPerBlock), 1);
  RDNA2BestSplitKernel<<<grid, block, 0, stream>>>(
      feature_meta, hist_offsets, used_features, num_features, histogram, nullptr,
      sum_gradients, sum_hessians, num_data, 0.0, 0.0, 0,
      lambda_l1, lambda_l2, min_data_in_leaf, min_sum_hessian_in_leaf, min_gain_to_split, 1, results);
  CUDASUCCESS_OR_FATAL(cudaGetLastError());
  if (top_k <= 1) {
    RDNA2BestSplitReduceKernel<<<1, block, 0, stream>>>(results, num_features, best_result);
  } else {
    RDNA2BestSplitTopKKernel<<<1, block, 0, stream>>>(
        results, num_features, top_k, top_results, histogram, nullptr, hist_offsets, feature_meta, nullptr, 1);
  }
  CUDASUCCESS_OR_FATAL(cudaGetLastError());
}

void LaunchRDNA2BestSplitPairKernelAndGather(
    const RDNA2BestSplitEngine::FeatureMeta* feature_meta, const uint64_t* hist_offsets,
    const int8_t* used_features, int num_features,
    const hist_t* first_histogram, double first_sum_gradients, double first_sum_hessians,
    data_size_t first_num_data, const hist_t* second_histogram, double second_sum_gradients,
    double second_sum_hessians, data_size_t second_num_data, double lambda_l1, double lambda_l2,
    data_size_t min_data_in_leaf, double min_sum_hessian_in_leaf, double min_gain_to_split, int top_k,
    RDNA2BestSplitEngine::DeviceSplit* results, RDNA2BestSplitEngine::DeviceSplit* top_results,
    hist_t* candidate_histograms, cudaStream_t stream) {
  const dim3 block(kBestSplitThreads);
  constexpr int kWavesPerBlock = kBestSplitThreads / kWaveSize;
  const dim3 grid(static_cast<unsigned int>((num_features + kWavesPerBlock - 1) / kWavesPerBlock), 2);
  RDNA2BestSplitKernel<<<grid, block, 0, stream>>>(
      feature_meta, hist_offsets, used_features, num_features, first_histogram, second_histogram,
      first_sum_gradients, first_sum_hessians, first_num_data,
      second_sum_gradients, second_sum_hessians, second_num_data,
      lambda_l1, lambda_l2, min_data_in_leaf, min_sum_hessian_in_leaf, min_gain_to_split, 2, results);
  CUDASUCCESS_OR_FATAL(cudaGetLastError());
  RDNA2BestSplitTopKKernel<<<2, block, 0, stream>>>(
      results, num_features, top_k, top_results, first_histogram, second_histogram, hist_offsets,
      feature_meta, candidate_histograms, 2);
  CUDASUCCESS_OR_FATAL(cudaGetLastError());
}

}  // namespace LightGBM
