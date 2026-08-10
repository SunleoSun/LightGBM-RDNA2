/*!
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include "rdna2_histogram_engine.hpp"

namespace LightGBM {
namespace {

constexpr int kHistogramThreads = 256;
constexpr int kFeaturesPerTuple = 4;

template <int NUM_BINS, int NUM_BANKS, bool USE_INDICES>
__global__ void RDNA2HistogramKernel(
    const uint32_t* packed_features,
    const score_t* gradients,
    const score_t* hessians,
    const data_size_t* data_indices,
    const uint32_t* group_bin_offsets,
    const uint8_t* feature4_masks,
    const data_size_t dataset_num_data,
    const data_size_t leaf_num_data,
    const int num_groups,
    hist_t* histogram) {
  static_assert(NUM_BANKS == 2 || NUM_BANKS == 4 || NUM_BANKS == 8, "RDNA2 histogram bank count must be a power-of-two tuning point");
  constexpr int kEntriesPerBank = kFeaturesPerTuple * NUM_BINS * 2;
  constexpr int kSharedEntries = NUM_BANKS * kEntriesPerBank;
  __shared__ hist_t shared_hist[kSharedEntries];

  const int tid = static_cast<int>(threadIdx.x);
  const int tuple = static_cast<int>(blockIdx.x);
  const uint8_t active_mask = feature4_masks[tuple];
  if (active_mask == 0) {
    return;
  }
  for (int i = tid; i < kSharedEntries; i += kHistogramThreads) {
    shared_hist[i] = 0.0;
  }
  __syncthreads();

  const int group_base = tuple * kFeaturesPerTuple;
  const int bank = (tid >> 3) & (NUM_BANKS - 1);
  const int feature_rotation = tid & (kFeaturesPerTuple - 1);
  const uint32_t* tuple_data = packed_features + static_cast<size_t>(tuple) * dataset_num_data;

  if constexpr (NUM_BINS == 128) {
    data_size_t leaf_pos = static_cast<data_size_t>(tid);
    if (leaf_pos < leaf_num_data) {
      data_size_t row = USE_INDICES ? data_indices[leaf_pos] : leaf_pos;
      uint32_t packed = tuple_data[row];
      hist_t grad = static_cast<hist_t>(gradients[row]);
      hist_t hess = static_cast<hist_t>(hessians[row]);
      while (true) {
        const data_size_t next_leaf_pos = leaf_pos + kHistogramThreads;
        const bool has_next = next_leaf_pos < leaf_num_data;
        data_size_t next_row = row;
        uint32_t next_packed = packed;
        hist_t next_grad = grad;
        hist_t next_hess = hess;
        if (has_next) {
          next_row = USE_INDICES ? data_indices[next_leaf_pos] : next_leaf_pos;
          next_packed = tuple_data[next_row];
          next_grad = static_cast<hist_t>(gradients[next_row]);
          next_hess = static_cast<hist_t>(hessians[next_row]);
        }
#pragma unroll
        for (int slot = 0; slot < kFeaturesPerTuple; ++slot) {
          const int feature = (slot + feature_rotation) & (kFeaturesPerTuple - 1);
          const int group = group_base + feature;
          if (group < num_groups && (active_mask & (1u << feature)) != 0) {
            const uint32_t bin = (packed >> (feature * 8)) & 0xffu;
            const int base = bank * kEntriesPerBank + ((feature * NUM_BINS + static_cast<int>(bin)) * 2);
            atomicAdd(shared_hist + base, grad);
            atomicAdd(shared_hist + base + 1, hess);
          }
        }
        if (!has_next) {
          break;
        }
        leaf_pos = next_leaf_pos;
        row = next_row;
        packed = next_packed;
        grad = next_grad;
        hess = next_hess;
      }
    }
  } else {
    for (data_size_t leaf_pos = static_cast<data_size_t>(tid); leaf_pos < leaf_num_data; leaf_pos += kHistogramThreads) {
      const data_size_t row = USE_INDICES ? data_indices[leaf_pos] : leaf_pos;
      const uint32_t packed = tuple_data[row];
      const hist_t grad = static_cast<hist_t>(gradients[row]);
      const hist_t hess = static_cast<hist_t>(hessians[row]);
#pragma unroll
      for (int slot = 0; slot < kFeaturesPerTuple; ++slot) {
        const int feature = (slot + feature_rotation) & (kFeaturesPerTuple - 1);
        const int group = group_base + feature;
        if (group < num_groups && (active_mask & (1u << feature)) != 0) {
          const uint32_t bin = (packed >> (feature * 8)) & 0xffu;
          const int base = bank * kEntriesPerBank + ((feature * NUM_BINS + static_cast<int>(bin)) * 2);
          atomicAdd(shared_hist + base, grad);
          atomicAdd(shared_hist + base + 1, hess);
        }
      }
    }
  }
  __syncthreads();

  constexpr int kOutputsPerTuple = kFeaturesPerTuple * NUM_BINS;
  for (int output_index = tid; output_index < kOutputsPerTuple; output_index += kHistogramThreads) {
    const int feature = output_index & (kFeaturesPerTuple - 1);
    const int bin = output_index >> 2;
    const int group = group_base + feature;
    if (group < num_groups && (active_mask & (1u << feature)) != 0) {
      const uint32_t begin = group_bin_offsets[group];
      const uint32_t end = group_bin_offsets[group + 1];
      const uint32_t num_bins = end - begin;
      if (static_cast<uint32_t>(bin) < num_bins) {
        hist_t grad_sum = 0.0;
        hist_t hess_sum = 0.0;
#pragma unroll
        for (int reduce_bank = 0; reduce_bank < NUM_BANKS; ++reduce_bank) {
          const int base = reduce_bank * kEntriesPerBank + ((feature * NUM_BINS + bin) * 2);
          grad_sum += shared_hist[base];
          hess_sum += shared_hist[base + 1];
        }
        const size_t output = static_cast<size_t>(begin + static_cast<uint32_t>(bin)) * 2;
        histogram[output] = grad_sum;
        histogram[output + 1] = hess_sum;
      }
    }
  }
}

template <int NUM_BINS, int NUM_TUPLES, int NUM_BANKS, bool USE_INDICES>
__global__ void RDNA2HistogramSuperTileKernel(
    const uint32_t* packed_features,
    const score_t* gradients,
    const score_t* hessians,
    const data_size_t* data_indices,
    const uint32_t* group_bin_offsets,
    const uint8_t* feature4_masks,
    const data_size_t dataset_num_data,
    const data_size_t leaf_num_data,
    const int num_feature4,
    const int num_groups,
    hist_t* histogram) {
  constexpr int kTileFeatures = kFeaturesPerTuple * NUM_TUPLES;
  constexpr int kEntriesPerBank = kTileFeatures * NUM_BINS * 2;
  constexpr int kSharedEntries = NUM_BANKS * kEntriesPerBank;
  __shared__ hist_t shared_hist[kSharedEntries];

  const int tid = static_cast<int>(threadIdx.x);
  const int tile = static_cast<int>(blockIdx.x);
  const int tuple_base = tile * NUM_TUPLES;
  uint32_t tile_active_mask = 0;
#pragma unroll
  for (int tuple_offset = 0; tuple_offset < NUM_TUPLES; ++tuple_offset) {
    const int tuple = tuple_base + tuple_offset;
    if (tuple < num_feature4) {
      tile_active_mask |= static_cast<uint32_t>(feature4_masks[tuple]) << (tuple_offset * kFeaturesPerTuple);
    }
  }
  if (tile_active_mask == 0) {
    return;
  }
  for (int i = tid; i < kSharedEntries; i += kHistogramThreads) {
    shared_hist[i] = 0.0;
  }
  __syncthreads();

  const int group_base = tuple_base * kFeaturesPerTuple;
  const int bank = (tid >> 3) & (NUM_BANKS - 1);
  const int feature_rotation = tid & (kFeaturesPerTuple - 1);

  constexpr uint32_t kAllTileFeaturesMask = (1u << kTileFeatures) - 1u;
  if (tile_active_mask == kAllTileFeaturesMask) {
    for (data_size_t leaf_pos = static_cast<data_size_t>(tid); leaf_pos < leaf_num_data; leaf_pos += kHistogramThreads) {
      const data_size_t row = USE_INDICES ? data_indices[leaf_pos] : leaf_pos;
      const hist_t grad = static_cast<hist_t>(gradients[row]);
      const hist_t hess = static_cast<hist_t>(hessians[row]);
#pragma unroll
      for (int tuple_offset = 0; tuple_offset < NUM_TUPLES; ++tuple_offset) {
        const int tuple = tuple_base + tuple_offset;
        const uint32_t packed = packed_features[static_cast<size_t>(tuple) * dataset_num_data + row];
#pragma unroll
        for (int slot = 0; slot < kFeaturesPerTuple; ++slot) {
          const int local_feature = (slot + feature_rotation) & (kFeaturesPerTuple - 1);
          const int tile_feature = tuple_offset * kFeaturesPerTuple + local_feature;
          const uint32_t bin = (packed >> (local_feature * 8)) & 0xffu;
          const int base = bank * kEntriesPerBank + ((tile_feature * NUM_BINS + static_cast<int>(bin)) * 2);
          atomicAdd(shared_hist + base, grad);
          atomicAdd(shared_hist + base + 1, hess);
        }
      }
    }
  } else {
    for (data_size_t leaf_pos = static_cast<data_size_t>(tid); leaf_pos < leaf_num_data; leaf_pos += kHistogramThreads) {
      const data_size_t row = USE_INDICES ? data_indices[leaf_pos] : leaf_pos;
      const hist_t grad = static_cast<hist_t>(gradients[row]);
      const hist_t hess = static_cast<hist_t>(hessians[row]);
#pragma unroll
      for (int tuple_offset = 0; tuple_offset < NUM_TUPLES; ++tuple_offset) {
        const int tuple = tuple_base + tuple_offset;
        if (tuple < num_feature4) {
          const uint32_t packed = packed_features[static_cast<size_t>(tuple) * dataset_num_data + row];
#pragma unroll
          for (int slot = 0; slot < kFeaturesPerTuple; ++slot) {
            const int local_feature = (slot + feature_rotation) & (kFeaturesPerTuple - 1);
            const int tile_feature = tuple_offset * kFeaturesPerTuple + local_feature;
            const int group = group_base + tile_feature;
            if (group < num_groups && (tile_active_mask & (1u << tile_feature)) != 0) {
              const uint32_t bin = (packed >> (local_feature * 8)) & 0xffu;
              const int base = bank * kEntriesPerBank + ((tile_feature * NUM_BINS + static_cast<int>(bin)) * 2);
              atomicAdd(shared_hist + base, grad);
              atomicAdd(shared_hist + base + 1, hess);
            }
          }
        }
      }
    }
  }
  __syncthreads();

  constexpr int kOutputsPerTile = kTileFeatures * NUM_BINS;
  for (int output_index = tid; output_index < kOutputsPerTile; output_index += kHistogramThreads) {
    const int tile_feature = output_index % kTileFeatures;
    const int bin = output_index / kTileFeatures;
    const int group = group_base + tile_feature;
    if (group < num_groups && (tile_active_mask & (1u << tile_feature)) != 0) {
      const uint32_t begin = group_bin_offsets[group];
      const uint32_t end = group_bin_offsets[group + 1];
      const uint32_t num_bins = end - begin;
      if (static_cast<uint32_t>(bin) < num_bins) {
        hist_t grad_sum = 0.0;
        hist_t hess_sum = 0.0;
#pragma unroll
        for (int reduce_bank = 0; reduce_bank < NUM_BANKS; ++reduce_bank) {
          const int base = reduce_bank * kEntriesPerBank + ((tile_feature * NUM_BINS + bin) * 2);
          grad_sum += shared_hist[base];
          hess_sum += shared_hist[base + 1];
        }
        const size_t output = static_cast<size_t>(begin + static_cast<uint32_t>(bin)) * 2;
        histogram[output] = grad_sum;
        histogram[output + 1] = hess_sum;
      }
    }
  }
}

}  // namespace

void RDNA2HistogramEngine::BeforeTrain(const score_t* gradients, const score_t* hessians) {
  if ((!h64_eligible_ && !h128_eligible_) || gradients == nullptr || hessians == nullptr) {
    return;
  }
#ifdef TIMETAG
  const auto profile_start = std::chrono::steady_clock::now();
#endif
  CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(device_gradients(), gradients,
                                       static_cast<size_t>(num_data_) * sizeof(score_t),
                                       cudaMemcpyHostToDevice, stream()));
  CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(device_hessians(), hessians,
                                       static_cast<size_t>(num_data_) * sizeof(score_t),
                                       cudaMemcpyHostToDevice, stream()));
#ifdef TIMETAG
  SynchronizeCUDAStream(stream(), __FILE__, __LINE__);
  const auto profile_end = std::chrono::steady_clock::now();
  ProfileAddGradientH2D(
      std::chrono::duration<double, std::milli>(profile_end - profile_start).count());
#endif
}

bool RDNA2HistogramEngine::ConstructHistogram(
    const std::vector<int8_t>& is_feature_used, const data_size_t* data_indices,
    data_size_t num_data, hist_t* host_histogram) {
  if ((!h64_eligible_ && !h128_eligible_) || host_histogram == nullptr || num_data <= 0 ||
      is_feature_used.size() != static_cast<size_t>(num_features_)) {
    return false;
  }

#ifdef TIMETAG
  double index_h2d_ms = 0.0;
  double memset_ms = 0.0;
  double kernel_ms = 0.0;
  double d2h_ms = 0.0;
#endif
  for (size_t tuple = 0; tuple < num_feature4_; ++tuple) {
    uint8_t mask = 0;
    const int group_base = static_cast<int>(tuple) * kFeaturesPerTuple;
    for (int lane = 0; lane < kFeaturesPerTuple; ++lane) {
      const int group = group_base + lane;
      if (group >= static_cast<int>(host_group_feature_indices_.size())) {
        break;
      }
      const int feature = host_group_feature_indices_[static_cast<size_t>(group)];
      if (feature >= 0 && is_feature_used[static_cast<size_t>(feature)] != 0) {
        mask |= static_cast<uint8_t>(1u << lane);
      }
    }
    host_feature4_masks_[tuple] = mask;
  }
  CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(feature4_masks_.RawData(), host_feature4_masks_.data(),
                                       host_feature4_masks_.size() * sizeof(uint8_t),
                                       cudaMemcpyHostToDevice, stream()));

  const data_size_t* device_indices = nullptr;
  const bool needs_indices = data_indices != nullptr && num_data < num_data_;
  if (needs_indices) {
    const bool use_preloaded_indices = ConsumePreloadedDataIndices(data_indices, num_data);
    if (use_preloaded_indices) {
      device_indices = device_data_indices();
#ifdef TIMETAG
      ++profile_index_preload_hits_;
#endif
    } else {
#ifdef TIMETAG
      ++profile_index_fallback_copies_;
      const auto index_start = std::chrono::steady_clock::now();
#endif
      CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(device_data_indices(), data_indices,
                                           static_cast<size_t>(num_data) * sizeof(data_size_t),
                                           cudaMemcpyHostToDevice, stream()));
      device_indices = device_data_indices();
#ifdef TIMETAG
      SynchronizeCUDAStream(stream(), __FILE__, __LINE__);
      const auto index_end = std::chrono::steady_clock::now();
      index_h2d_ms = std::chrono::duration<double, std::milli>(index_end - index_start).count();
#endif
    }
  }

#ifdef TIMETAG
  auto stage_start = std::chrono::steady_clock::now();
#endif

  hist_t* mapped_histogram = EnsureCanonicalHistogramMapped(host_histogram);
  hist_t* kernel_histogram = mapped_histogram != nullptr ? mapped_histogram : device_histogram();

  const int num_groups = static_cast<int>(dense_feature_groups_.size());
  const dim3 block(kHistogramThreads);
  constexpr int kH64SuperTileTuples = 2;
  constexpr data_size_t kH64SuperTileMinRows = 12288;
  const bool use_indices = device_indices != nullptr;
  if (h64_eligible_ && num_data >= kH64SuperTileMinRows) {
    const int num_tiles = (static_cast<int>(num_feature4_) + kH64SuperTileTuples - 1) / kH64SuperTileTuples;
    const dim3 grid(static_cast<unsigned int>(num_tiles));
    if (use_indices) {
      RDNA2HistogramSuperTileKernel<64, kH64SuperTileTuples, 4, true><<<grid, block, 0, stream()>>>(
          reinterpret_cast<const uint32_t*>(packed_features()),
          device_gradients(), device_hessians(), device_indices, group_bin_offsets(),
          feature4_masks(), num_data_, num_data, static_cast<int>(num_feature4_),
          num_groups, kernel_histogram);
    } else {
      RDNA2HistogramSuperTileKernel<64, kH64SuperTileTuples, 4, false><<<grid, block, 0, stream()>>>(
          reinterpret_cast<const uint32_t*>(packed_features()),
          device_gradients(), device_hessians(), device_indices, group_bin_offsets(),
          feature4_masks(), num_data_, num_data, static_cast<int>(num_feature4_),
          num_groups, kernel_histogram);
    }
  } else {
    const dim3 grid(static_cast<unsigned int>(num_feature4_));
    if (h64_eligible_) {
      if (use_indices) {
        RDNA2HistogramKernel<64, 4, true><<<grid, block, 0, stream()>>>(
            reinterpret_cast<const uint32_t*>(packed_features()),
            device_gradients(), device_hessians(), device_indices, group_bin_offsets(),
            feature4_masks(), num_data_, num_data, num_groups, kernel_histogram);
      } else {
        RDNA2HistogramKernel<64, 4, false><<<grid, block, 0, stream()>>>(
            reinterpret_cast<const uint32_t*>(packed_features()),
            device_gradients(), device_hessians(), device_indices, group_bin_offsets(),
            feature4_masks(), num_data_, num_data, num_groups, kernel_histogram);
      }
    } else if (use_indices) {
      RDNA2HistogramKernel<128, 4, true><<<grid, block, 0, stream()>>>(
          reinterpret_cast<const uint32_t*>(packed_features()),
          device_gradients(), device_hessians(), device_indices, group_bin_offsets(),
          feature4_masks(), num_data_, num_data, num_groups, kernel_histogram);
    } else {
      RDNA2HistogramKernel<128, 4, false><<<grid, block, 0, stream()>>>(
          reinterpret_cast<const uint32_t*>(packed_features()),
          device_gradients(), device_hessians(), device_indices, group_bin_offsets(),
          feature4_masks(), num_data_, num_data, num_groups, kernel_histogram);
    }
  }
  SynchronizeCUDAStream(stream(), __FILE__, __LINE__);
#ifdef TIMETAG
  auto stage_end = std::chrono::steady_clock::now();
  kernel_ms = std::chrono::duration<double, std::milli>(stage_end - stage_start).count();
  stage_start = std::chrono::steady_clock::now();
#endif
  double host_copy_ms = 0.0;
  if (mapped_histogram == nullptr) {
    CopyFromCUDADeviceToHostAsync(host_histogram_staging(), device_histogram(), num_total_bins_ * 2,
                                  stream(), __FILE__, __LINE__);
    SynchronizeCUDAStream(stream(), __FILE__, __LINE__);
#ifdef TIMETAG
    stage_end = std::chrono::steady_clock::now();
    d2h_ms = std::chrono::duration<double, std::milli>(stage_end - stage_start).count();
#endif
    const auto host_copy_start = std::chrono::steady_clock::now();
    std::memcpy(host_histogram, host_histogram_staging(), num_total_bins_ * 2 * sizeof(hist_t));
#ifdef TIMETAG
    const auto host_copy_end = std::chrono::steady_clock::now();
    host_copy_ms = std::chrono::duration<double, std::milli>(host_copy_end - host_copy_start).count();
#endif
  }
#ifdef TIMETAG
  ProfileAddHistogramCall(index_h2d_ms, memset_ms, kernel_ms, d2h_ms, host_copy_ms);
#endif
  return true;
}

}  // namespace LightGBM
