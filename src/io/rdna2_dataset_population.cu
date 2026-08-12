/*!
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include "rdna2_dataset_population.hpp"

#include <LightGBM/bin.h>
#include <LightGBM/cuda/cuda_utils.hu>
#include <LightGBM/utils/log.h>

#include <algorithm>
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
constexpr size_t kMaxInputChunkBytes = 128ull * 1024ull * 1024ull;
constexpr int kMaxChunkRows = 8192;

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
    if (stream_ == nullptr) {
      CUDASUCCESS_OR_FATAL(cudaStreamCreate(&stream_));
    }
    if (input_elems > input_capacity_) {
      if (host_input_ != nullptr) CUDASUCCESS_OR_FATAL(cudaFreeHost(host_input_));
      if (device_input_ != nullptr) CUDASUCCESS_OR_FATAL(cudaFree(device_input_));
      CUDASUCCESS_OR_FATAL(cudaHostAlloc(reinterpret_cast<void**>(&host_input_),
                                         input_elems * sizeof(float), cudaHostAllocPortable));
      CUDASUCCESS_OR_FATAL(cudaMalloc(reinterpret_cast<void**>(&device_input_),
                                      input_elems * sizeof(float)));
      input_capacity_ = input_elems;
    }
    if (output_elems > output_capacity_) {
      if (host_output_ != nullptr) CUDASUCCESS_OR_FATAL(cudaFreeHost(host_output_));
      if (device_output_ != nullptr) CUDASUCCESS_OR_FATAL(cudaFree(device_output_));
      CUDASUCCESS_OR_FATAL(cudaHostAlloc(reinterpret_cast<void**>(&host_output_),
                                         output_elems * sizeof(uint8_t), cudaHostAllocPortable));
      CUDASUCCESS_OR_FATAL(cudaMalloc(reinterpret_cast<void**>(&device_output_),
                                      output_elems * sizeof(uint8_t)));
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
    if (stream_ != nullptr) { cudaStreamSynchronize(stream_); }
    if (device_missing_nan_ != nullptr) cudaFree(device_missing_nan_);
    if (device_num_bins_ != nullptr) cudaFree(device_num_bins_);
    if (device_upper_bounds_ != nullptr) cudaFree(device_upper_bounds_);
    if (device_output_ != nullptr) cudaFree(device_output_);
    if (device_input_ != nullptr) cudaFree(device_input_);
    if (host_output_ != nullptr) cudaFreeHost(host_output_);
    if (host_input_ != nullptr) cudaFreeHost(host_input_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
    stream_ = nullptr;
    host_input_ = nullptr;
    host_output_ = nullptr;
    device_input_ = nullptr;
    device_output_ = nullptr;
    device_upper_bounds_ = nullptr;
    device_num_bins_ = nullptr;
    device_missing_nan_ = nullptr;
    input_capacity_ = 0;
    output_capacity_ = 0;
    bounds_capacity_ = 0;
    feature_capacity_ = 0;
  }

  std::mutex mutex_;
  cudaStream_t stream_ = nullptr;
  float* host_input_ = nullptr;
  uint8_t* host_output_ = nullptr;
  float* device_input_ = nullptr;
  uint8_t* device_output_ = nullptr;
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
                                       upper_bounds.size() * sizeof(double), cudaMemcpyHostToDevice, context.stream_));
  CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(context.device_num_bins_, num_bins.data(),
                                       num_bins.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, context.stream_));
  CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(context.device_missing_nan_, missing_nan.data(),
                                       missing_nan.size() * sizeof(uint8_t), cudaMemcpyHostToDevice, context.stream_));
  CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(context.stream_));

  const auto setup_done = std::chrono::steady_clock::now();
  double host_stage_ms = 0.0;
  double gpu_ms = 0.0;
  double host_load_ms = 0.0;
  const dim3 block(kThreads);
  for (int row_offset = 0; row_offset < num_rows; row_offset += chunk_rows) {
    const int this_rows = std::min(chunk_rows, num_rows - row_offset);
    const size_t input_bytes = static_cast<size_t>(this_rows) * num_cols * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(this_rows) * num_cols * sizeof(uint8_t);

    const auto stage_begin = std::chrono::steady_clock::now();
    std::memcpy(context.host_input_, data + static_cast<size_t>(row_offset) * num_cols, input_bytes);
    const auto stage_end = std::chrono::steady_clock::now();
    host_stage_ms += std::chrono::duration<double, std::milli>(stage_end - stage_begin).count();

    const auto gpu_begin = std::chrono::steady_clock::now();
    CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(context.device_input_, context.host_input_, input_bytes,
                                         cudaMemcpyHostToDevice, context.stream_));
    const dim3 grid((num_cols + kFeatureTile - 1) / kFeatureTile,
                    (this_rows + kRowTile - 1) / kRowTile);
    RDNA2PopulateDenseBinsKernel<<<grid, block, 0, context.stream_>>>(
        context.device_input_, context.device_output_, context.device_upper_bounds_,
        context.device_num_bins_, context.device_missing_nan_, this_rows, num_cols);
    CUDASUCCESS_OR_FATAL(cudaGetLastError());
    CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(context.host_output_, context.device_output_, output_bytes,
                                         cudaMemcpyDeviceToHost, context.stream_));
    CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(context.stream_));
    const auto gpu_end = std::chrono::steady_clock::now();
    gpu_ms += std::chrono::duration<double, std::milli>(gpu_end - gpu_begin).count();

    const auto load_begin = std::chrono::steady_clock::now();
    if (!dataset->LoadDenseFeatureMajorCanonicalBinRange(
            context.host_output_, num_cols, row_offset, this_rows)) {
      return false;
    }
    const auto load_end = std::chrono::steady_clock::now();
    host_load_ms += std::chrono::duration<double, std::milli>(load_end - load_begin).count();
  }

  const auto end = std::chrono::steady_clock::now();
  const double setup_ms = std::chrono::duration<double, std::milli>(setup_done - start).count();
  const double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
  Log::Info("RDNA2 dataset population: rows=%d features=%d chunk_rows=%d chunks=%d metadata=%.3f ms device_check=%.3f ms setup=%.3f ms host_stage=%.3f ms gpu=%.3f ms host_load=%.3f ms total=%.3f ms",
            num_rows, num_cols, chunk_rows, (num_rows + chunk_rows - 1) / chunk_rows,
            metadata_ms, device_check_ms, setup_ms, host_stage_ms, gpu_ms, host_load_ms, total_ms);
  return true;
}

}  // namespace LightGBM
