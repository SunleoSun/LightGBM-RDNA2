/*!
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include "rdna2_dataset_population.hpp"

#include <LightGBM/bin.h>
#include <LightGBM/cuda/cuda_utils.hu>
#include <LightGBM/utils/log.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

namespace LightGBM {
namespace {

constexpr int kFeatureTile = 8;
constexpr int kRowTile = 32;
constexpr int kThreads = kFeatureTile * kRowTile;
constexpr int kMaxBins = 128;
constexpr int kPopulationPipelineSlots = 2;
constexpr size_t kMaxInputChunkBytes = 64ull * 1024ull * 1024ull;
constexpr int kMaxChunkRows = 4096;

__global__ void RDNA2PopulateDenseBinsKernel(
    const float* input, uint8_t* output, const double* upper_bounds,
    const uint16_t* num_bins, const uint8_t* missing_nan,
    int chunk_rows, int num_features) {
  __shared__ uint8_t tile[kRowTile][kFeatureTile];
  const int tid = static_cast<int>(threadIdx.x);
  const int load_row = tid / kFeatureTile;
  const int load_feature = tid % kFeatureTile;
  const int row_base = static_cast<int>(blockIdx.y) * kRowTile;
  const int feature_base = static_cast<int>(blockIdx.x) * kFeatureTile;
  const int row = row_base + load_row;
  const int feature = feature_base + load_feature;

  uint8_t bin = 0;
  if (row < chunk_rows && feature < num_features) {
    const int bins = static_cast<int>(num_bins[feature]);
    if (bins > 0) {
      double value = static_cast<double>(input[static_cast<size_t>(row) * num_features + feature]);
      if (isnan(value) && missing_nan[feature] != 0) {
        bin = static_cast<uint8_t>(bins - 1);
      } else {
        if (isnan(value)) {
          value = 0.0;
        }
        int left = 0;
        int right = bins - 1 - static_cast<int>(missing_nan[feature] != 0);
        while (left < right) {
          const int mid = (right + left - 1) / 2;
          if (value <= upper_bounds[static_cast<size_t>(feature) * kMaxBins + mid]) {
            right = mid;
          } else {
            left = mid + 1;
          }
        }
        bin = static_cast<uint8_t>(left);
      }
    }
  }
  tile[load_row][load_feature] = bin;
  __syncthreads();

  const int store_feature = tid / kRowTile;
  const int store_row = tid % kRowTile;
  const int out_feature = feature_base + store_feature;
  const int out_row = row_base + store_row;
  if (out_feature < num_features && out_row < chunk_rows) {
    output[static_cast<size_t>(out_feature) * chunk_rows + out_row] = tile[store_row][store_feature];
  }
}

int ChooseChunkRows(int num_rows, int num_cols) {
  if (num_rows <= 0 || num_cols <= 0) {
    return 0;
  }
  const size_t bytes_per_row = static_cast<size_t>(num_cols) * sizeof(float);
  int rows_by_bytes = static_cast<int>(std::max<size_t>(1, kMaxInputChunkBytes / bytes_per_row));
  int chunk_rows = std::min({num_rows, kMaxChunkRows, rows_by_bytes});
  if (chunk_rows >= kRowTile) {
    chunk_rows = (chunk_rows / kRowTile) * kRowTile;
  }
  return std::max(1, chunk_rows);
}

class RDNA2DatasetPopulationContext {
 public:
  ~RDNA2DatasetPopulationContext() { Release(); }

  void Ensure(size_t input_elems, size_t output_elems, size_t bounds_elems, size_t feature_count) {
    for (int slot = 0; slot < kPopulationPipelineSlots; ++slot) {
      if (streams_[slot] == nullptr) {
        CUDASUCCESS_OR_FATAL(cudaStreamCreate(&streams_[slot]));
      }
    }
    if (input_elems > input_capacity_) {
      for (int slot = 0; slot < kPopulationPipelineSlots; ++slot) {
        if (host_inputs_[slot] != nullptr) CUDASUCCESS_OR_FATAL(cudaFreeHost(host_inputs_[slot]));
        if (device_inputs_[slot] != nullptr) CUDASUCCESS_OR_FATAL(cudaFree(device_inputs_[slot]));
        CUDASUCCESS_OR_FATAL(cudaHostAlloc(reinterpret_cast<void**>(&host_inputs_[slot]),
                                           input_elems * sizeof(float), cudaHostAllocPortable));
        CUDASUCCESS_OR_FATAL(cudaMalloc(reinterpret_cast<void**>(&device_inputs_[slot]),
                                        input_elems * sizeof(float)));
      }
      input_capacity_ = input_elems;
    }
    if (output_elems > output_capacity_) {
      for (int slot = 0; slot < kPopulationPipelineSlots; ++slot) {
        if (host_outputs_[slot] != nullptr) CUDASUCCESS_OR_FATAL(cudaFreeHost(host_outputs_[slot]));
        if (device_outputs_[slot] != nullptr) CUDASUCCESS_OR_FATAL(cudaFree(device_outputs_[slot]));
        CUDASUCCESS_OR_FATAL(cudaHostAlloc(reinterpret_cast<void**>(&host_outputs_[slot]),
                                           output_elems * sizeof(uint8_t), cudaHostAllocPortable));
        CUDASUCCESS_OR_FATAL(cudaMalloc(reinterpret_cast<void**>(&device_outputs_[slot]),
                                        output_elems * sizeof(uint8_t)));
      }
      output_capacity_ = output_elems;
    }
    if (bounds_elems > bounds_capacity_) {
      if (device_upper_bounds_ != nullptr) CUDASUCCESS_OR_FATAL(cudaFree(device_upper_bounds_));
      CUDASUCCESS_OR_FATAL(cudaMalloc(reinterpret_cast<void**>(&device_upper_bounds_),
                                      bounds_elems * sizeof(double)));
      bounds_capacity_ = bounds_elems;
    }
    if (feature_count > feature_capacity_) {
      if (device_num_bins_ != nullptr) CUDASUCCESS_OR_FATAL(cudaFree(device_num_bins_));
      if (device_missing_nan_ != nullptr) CUDASUCCESS_OR_FATAL(cudaFree(device_missing_nan_));
      CUDASUCCESS_OR_FATAL(cudaMalloc(reinterpret_cast<void**>(&device_num_bins_),
                                      feature_count * sizeof(uint16_t)));
      CUDASUCCESS_OR_FATAL(cudaMalloc(reinterpret_cast<void**>(&device_missing_nan_),
                                      feature_count * sizeof(uint8_t)));
      feature_capacity_ = feature_count;
    }
  }

  void Release() {
    for (int slot = 0; slot < kPopulationPipelineSlots; ++slot) {
      if (streams_[slot] != nullptr) { cudaStreamSynchronize(streams_[slot]); }
    }
    if (device_missing_nan_ != nullptr) cudaFree(device_missing_nan_);
    if (device_num_bins_ != nullptr) cudaFree(device_num_bins_);
    if (device_upper_bounds_ != nullptr) cudaFree(device_upper_bounds_);
    for (int slot = 0; slot < kPopulationPipelineSlots; ++slot) {
      if (device_outputs_[slot] != nullptr) cudaFree(device_outputs_[slot]);
      if (device_inputs_[slot] != nullptr) cudaFree(device_inputs_[slot]);
      if (host_outputs_[slot] != nullptr) cudaFreeHost(host_outputs_[slot]);
      if (host_inputs_[slot] != nullptr) cudaFreeHost(host_inputs_[slot]);
      if (streams_[slot] != nullptr) cudaStreamDestroy(streams_[slot]);
      streams_[slot] = nullptr;
      host_inputs_[slot] = nullptr;
      host_outputs_[slot] = nullptr;
      device_inputs_[slot] = nullptr;
      device_outputs_[slot] = nullptr;
    }
    device_upper_bounds_ = nullptr;
    device_num_bins_ = nullptr;
    device_missing_nan_ = nullptr;
    input_capacity_ = 0;
    output_capacity_ = 0;
    bounds_capacity_ = 0;
    feature_capacity_ = 0;
  }

  std::mutex mutex_;
  std::array<cudaStream_t, kPopulationPipelineSlots> streams_{};
  std::array<float*, kPopulationPipelineSlots> host_inputs_{};
  std::array<uint8_t*, kPopulationPipelineSlots> host_outputs_{};
  std::array<float*, kPopulationPipelineSlots> device_inputs_{};
  std::array<uint8_t*, kPopulationPipelineSlots> device_outputs_{};
  double* device_upper_bounds_ = nullptr;
  uint16_t* device_num_bins_ = nullptr;
  uint8_t* device_missing_nan_ = nullptr;
  size_t input_capacity_ = 0;
  size_t output_capacity_ = 0;
  size_t bounds_capacity_ = 0;
  size_t feature_capacity_ = 0;
};

RDNA2DatasetPopulationContext& PopulationContext() {
  static RDNA2DatasetPopulationContext context;
  return context;
}

}  // namespace

bool RDNA2DenseFloat32DatasetPopulationNeedsPrepare(int num_rows, int num_cols) {
  const int chunk_rows = ChooseChunkRows(num_rows, num_cols);
  if (chunk_rows <= 0) {
    return false;
  }
  const size_t chunk_elems = static_cast<size_t>(chunk_rows) * num_cols;
  auto& context = PopulationContext();
  std::lock_guard<std::mutex> lock(context.mutex_);
  return context.streams_[0] == nullptr || context.streams_[1] == nullptr ||
         context.input_capacity_ < chunk_elems ||
         context.output_capacity_ < chunk_elems ||
         context.bounds_capacity_ < static_cast<size_t>(num_cols) * kMaxBins ||
         context.feature_capacity_ < static_cast<size_t>(num_cols);
}

bool RDNA2PrepareDenseFloat32DatasetPopulation(int num_rows, int num_cols, int gpu_device_id) {
  const int chunk_rows = ChooseChunkRows(num_rows, num_cols);
  if (chunk_rows <= 0) {
    return false;
  }
  const int device = gpu_device_id < 0 ? 0 : gpu_device_id;
  SetCUDADevice(device, __FILE__, __LINE__);
  cudaDeviceProp prop{};
  CUDASUCCESS_OR_FATAL(cudaGetDeviceProperties(&prop, device));
  if (std::strncmp(prop.gcnArchName, "gfx1030", 7) != 0) {
    return false;
  }

  const size_t chunk_elems = static_cast<size_t>(chunk_rows) * num_cols;
  auto& context = PopulationContext();
  std::lock_guard<std::mutex> lock(context.mutex_);
  context.Ensure(chunk_elems, chunk_elems,
                 static_cast<size_t>(num_cols) * kMaxBins,
                 static_cast<size_t>(num_cols));
  return true;
}

bool RDNA2PopulateDenseFloat32Dataset(Dataset* dataset, const float* data,
                                      int num_rows, int num_cols,
                                      int gpu_device_id) {
  const auto function_start = std::chrono::steady_clock::now();
  if (dataset == nullptr || data == nullptr || dataset->has_raw() ||
      dataset->num_data() != num_rows || dataset->num_total_features() != num_cols ||
      !dataset->CanLoadDenseFeatureMajorCanonicalBins(num_cols)) {
    return false;
  }

  std::vector<double> upper_bounds(static_cast<size_t>(num_cols) * kMaxBins, 0.0);
  std::vector<uint16_t> num_bins(static_cast<size_t>(num_cols), 0);
  std::vector<uint8_t> missing_nan(static_cast<size_t>(num_cols), 0);
  for (int col = 0; col < num_cols; ++col) {
    const int feature = dataset->InnerFeatureIndex(col);
    if (feature < 0) {
      continue;
    }
    const BinMapper* mapper = dataset->FeatureBinMapper(feature);
    if (mapper == nullptr || mapper->bin_type() != BinType::NumericalBin ||
        mapper->num_bin() <= 0 || mapper->num_bin() > kMaxBins) {
      return false;
    }
    num_bins[static_cast<size_t>(col)] = static_cast<uint16_t>(mapper->num_bin());
    missing_nan[static_cast<size_t>(col)] =
        mapper->missing_type() == MissingType::NaN ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
    const int searchable_bins = mapper->num_bin() - static_cast<int>(missing_nan[static_cast<size_t>(col)] != 0);
    for (int bin = 0; bin < searchable_bins; ++bin) {
      upper_bounds[static_cast<size_t>(col) * kMaxBins + bin] = mapper->BinToValue(bin);
    }
  }

  const auto metadata_done = std::chrono::steady_clock::now();
  const int device = gpu_device_id < 0 ? 0 : gpu_device_id;
  SetCUDADevice(device, __FILE__, __LINE__);
  cudaDeviceProp prop{};
  CUDASUCCESS_OR_FATAL(cudaGetDeviceProperties(&prop, device));
  if (std::strncmp(prop.gcnArchName, "gfx1030", 7) != 0) {
    return false;
  }

  const int chunk_rows = ChooseChunkRows(num_rows, num_cols);
  if (chunk_rows <= 0) {
    return false;
  }
  const size_t input_chunk_elems = static_cast<size_t>(chunk_rows) * num_cols;
  const size_t output_chunk_elems = static_cast<size_t>(chunk_rows) * num_cols;

  auto& context = PopulationContext();
  std::lock_guard<std::mutex> lock(context.mutex_);
  const auto start = std::chrono::steady_clock::now();
  const double metadata_ms = std::chrono::duration<double, std::milli>(metadata_done - function_start).count();
  const double device_check_ms = std::chrono::duration<double, std::milli>(start - metadata_done).count();
  context.Ensure(input_chunk_elems, output_chunk_elems, upper_bounds.size(), num_bins.size());
  CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(context.device_upper_bounds_, upper_bounds.data(),
                                       upper_bounds.size() * sizeof(double), cudaMemcpyHostToDevice, context.streams_[0]));
  CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(context.device_num_bins_, num_bins.data(),
                                       num_bins.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, context.streams_[0]));
  CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(context.device_missing_nan_, missing_nan.data(),
                                       missing_nan.size() * sizeof(uint8_t), cudaMemcpyHostToDevice, context.streams_[0]));
  CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(context.streams_[0]));

  const auto setup_done = std::chrono::steady_clock::now();
  double host_stage_ms = 0.0;
  double gpu_wait_ms = 0.0;
  double host_load_ms = 0.0;
  std::array<int, kPopulationPipelineSlots> pending_row_offsets{};
  std::array<int, kPopulationPipelineSlots> pending_row_counts{};
  pending_row_offsets.fill(-1);
  const dim3 block(kThreads);

  auto finish_slot = [&](int slot) {
    if (pending_row_offsets[slot] < 0) {
      return true;
    }
    const auto wait_begin = std::chrono::steady_clock::now();
    CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(context.streams_[slot]));
    const auto wait_end = std::chrono::steady_clock::now();
    gpu_wait_ms += std::chrono::duration<double, std::milli>(wait_end - wait_begin).count();

    const auto load_begin = std::chrono::steady_clock::now();
    const bool loaded = dataset->LoadDenseFeatureMajorCanonicalBinRange(
        context.host_outputs_[slot], num_cols, pending_row_offsets[slot], pending_row_counts[slot]);
    const auto load_end = std::chrono::steady_clock::now();
    host_load_ms += std::chrono::duration<double, std::milli>(load_end - load_begin).count();
    pending_row_offsets[slot] = -1;
    pending_row_counts[slot] = 0;
    return loaded;
  };

  int chunk_index = 0;
  for (int row_offset = 0; row_offset < num_rows; row_offset += chunk_rows, ++chunk_index) {
    const int slot = chunk_index % kPopulationPipelineSlots;
    if (!finish_slot(slot)) {
      return false;
    }

    const int this_rows = std::min(chunk_rows, num_rows - row_offset);
    const size_t input_bytes = static_cast<size_t>(this_rows) * num_cols * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(this_rows) * num_cols * sizeof(uint8_t);

    const auto stage_begin = std::chrono::steady_clock::now();
    std::memcpy(context.host_inputs_[slot],
                data + static_cast<size_t>(row_offset) * num_cols, input_bytes);
    const auto stage_end = std::chrono::steady_clock::now();
    host_stage_ms += std::chrono::duration<double, std::milli>(stage_end - stage_begin).count();

    CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(context.device_inputs_[slot], context.host_inputs_[slot],
                                         input_bytes, cudaMemcpyHostToDevice, context.streams_[slot]));
    const dim3 grid((num_cols + kFeatureTile - 1) / kFeatureTile,
                    (this_rows + kRowTile - 1) / kRowTile);
    RDNA2PopulateDenseBinsKernel<<<grid, block, 0, context.streams_[slot]>>>(
        context.device_inputs_[slot], context.device_outputs_[slot], context.device_upper_bounds_,
        context.device_num_bins_, context.device_missing_nan_, this_rows, num_cols);
    CUDASUCCESS_OR_FATAL(cudaGetLastError());
    CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(context.host_outputs_[slot], context.device_outputs_[slot],
                                         output_bytes, cudaMemcpyDeviceToHost, context.streams_[slot]));
    pending_row_offsets[slot] = row_offset;
    pending_row_counts[slot] = this_rows;
  }

  for (int slot = 0; slot < kPopulationPipelineSlots; ++slot) {
    if (!finish_slot(slot)) {
      return false;
    }
  }

  const auto end = std::chrono::steady_clock::now();
  const double setup_ms = std::chrono::duration<double, std::milli>(setup_done - start).count();
  const double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
  Log::Info("RDNA2 dataset population: rows=%d features=%d chunk_rows=%d chunks=%d slots=%d metadata=%.3f ms device_check=%.3f ms setup=%.3f ms host_stage=%.3f ms gpu_wait=%.3f ms host_load=%.3f ms total=%.3f ms",
            num_rows, num_cols, chunk_rows, (num_rows + chunk_rows - 1) / chunk_rows,
            kPopulationPipelineSlots, metadata_ms, device_check_ms, setup_ms, host_stage_ms,
            gpu_wait_ms, host_load_ms, total_ms);
  return true;
}

}  // namespace LightGBM
