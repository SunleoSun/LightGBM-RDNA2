/*!
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include "rdna2_histogram_engine.hpp"

namespace LightGBM {
namespace {

constexpr int kH64Threads = 256;
constexpr int kH64FeaturesPerTuple = 4;
constexpr int kH64Bins = 64;
constexpr int kH64Banks = 4;
constexpr int kH64EntriesPerBank = kH64FeaturesPerTuple * kH64Bins * 2;
constexpr int kH64SharedEntries = kH64Banks * kH64EntriesPerBank;

__global__ void RDNA2RootHistogramH64Kernel(
    const uint32_t* packed_features,
    const score_t* gradients,
    const score_t* hessians,
    const data_size_t* data_indices,
    const uint32_t* group_bin_offsets,
    const data_size_t dataset_num_data,
    const data_size_t leaf_num_data,
    const int num_groups,
    hist_t* histogram) {
  __shared__ hist_t shared_hist[kH64SharedEntries];

  const int tid = static_cast<int>(threadIdx.x);
  for (int i = tid; i < kH64SharedEntries; i += kH64Threads) {
    shared_hist[i] = 0.0;
  }
  __syncthreads();

  const int tuple = static_cast<int>(blockIdx.x);
  const int group_base = tuple * kH64FeaturesPerTuple;
  const int bank = (tid >> 3) & (kH64Banks - 1);
  const int feature_rotation = tid & (kH64FeaturesPerTuple - 1);
  const uint32_t* tuple_data = packed_features + static_cast<size_t>(tuple) * dataset_num_data;

  for (data_size_t leaf_pos = static_cast<data_size_t>(tid); leaf_pos < leaf_num_data; leaf_pos += kH64Threads) {
    const data_size_t row = data_indices == nullptr ? leaf_pos : data_indices[leaf_pos];
    const uint32_t packed = tuple_data[row];
    const hist_t grad = static_cast<hist_t>(gradients[row]);
    const hist_t hess = static_cast<hist_t>(hessians[row]);
#pragma unroll
    for (int slot = 0; slot < kH64FeaturesPerTuple; ++slot) {
      const int feature = (slot + feature_rotation) & (kH64FeaturesPerTuple - 1);
      const int group = group_base + feature;
      if (group < num_groups) {
        const uint32_t bin = (packed >> (feature * 8)) & 0xffu;
        const int base = bank * kH64EntriesPerBank +
                         ((feature * kH64Bins + static_cast<int>(bin)) * 2);
        atomicAdd(shared_hist + base, grad);
        atomicAdd(shared_hist + base + 1, hess);
      }
    }
  }
  __syncthreads();

  const int feature = tid & (kH64FeaturesPerTuple - 1);
  const int bin = tid >> 2;
  const int group = group_base + feature;
  if (group < num_groups) {
    const uint32_t begin = group_bin_offsets[group];
    const uint32_t end = group_bin_offsets[group + 1];
    const uint32_t num_bins = end - begin;
    if (static_cast<uint32_t>(bin) < num_bins) {
      hist_t grad_sum = 0.0;
      hist_t hess_sum = 0.0;
#pragma unroll
      for (int reduce_bank = 0; reduce_bank < kH64Banks; ++reduce_bank) {
        const int base = reduce_bank * kH64EntriesPerBank +
                         ((feature * kH64Bins + bin) * 2);
        grad_sum += shared_hist[base];
        hess_sum += shared_hist[base + 1];
      }
      const size_t output = static_cast<size_t>(begin + static_cast<uint32_t>(bin)) * 2;
      histogram[output] = grad_sum;
      histogram[output + 1] = hess_sum;
    }
  }
}

}  // namespace

void RDNA2HistogramEngine::BeforeTrain(const score_t* gradients, const score_t* hessians) {
  if (!h64_eligible_ || gradients == nullptr || hessians == nullptr) {
    return;
  }
  CopyFromHostToCUDADevice(device_gradients(), gradients, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  CopyFromHostToCUDADevice(device_hessians(), hessians, static_cast<size_t>(num_data_), __FILE__, __LINE__);
}

bool RDNA2HistogramEngine::ConstructH64(
    const data_size_t* data_indices, data_size_t num_data, hist_t* host_histogram) {
  if (!h64_eligible_ || host_histogram == nullptr || num_data <= 0) {
    return false;
  }

  const data_size_t* device_indices = nullptr;
  if (data_indices != nullptr && num_data < num_data_) {
    CopyFromHostToCUDADevice(device_data_indices(), data_indices, static_cast<size_t>(num_data), __FILE__, __LINE__);
    device_indices = device_data_indices();
  }

  CUDASUCCESS_OR_FATAL(cudaMemset(device_histogram(), 0, num_total_bins_ * 2 * sizeof(hist_t)));
  const int num_groups = static_cast<int>(dense_feature_groups_.size());
  const dim3 grid(static_cast<unsigned int>(num_feature4_));
  const dim3 block(kH64Threads);
  RDNA2RootHistogramH64Kernel<<<grid, block>>>(
      reinterpret_cast<const uint32_t*>(packed_features()),
      device_gradients(), device_hessians(), device_indices, group_bin_offsets(), num_data_, num_data,
      num_groups, device_histogram());
  SynchronizeCUDADevice(__FILE__, __LINE__);
  CopyFromCUDADeviceToHost(host_histogram, device_histogram(), num_total_bins_ * 2, __FILE__, __LINE__);
  SynchronizeCUDADevice(__FILE__, __LINE__);
  return true;
}

}  // namespace LightGBM
