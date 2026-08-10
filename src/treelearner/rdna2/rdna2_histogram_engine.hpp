/*!
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_RDNA2_RDNA2_HISTOGRAM_ENGINE_HPP_
#define LIGHTGBM_SRC_TREELEARNER_RDNA2_RDNA2_HISTOGRAM_ENGINE_HPP_

#include <LightGBM/cuda/cuda_utils.hu>
#include <LightGBM/dataset.h>
#include <LightGBM/utils/log.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <vector>
#include <unordered_set>

#include "../../io/dense_bin.hpp"

namespace LightGBM {

class RDNA2HistogramEngine {
 public:
  struct PackedFeature4 {
    uint32_t value;
  };

  RDNA2HistogramEngine() = default;

  ~RDNA2HistogramEngine() {
#ifdef TIMETAG
    if (profile_histogram_calls_ > 0) {
      Log::Info("RDNA2 profile: hist_calls=%llu grad_h2d_ms=%.3f index_h2d_ms=%.3f memset_ms=%.3f kernel_ms=%.3f d2h_ms=%.3f host_copy_ms=%.3f index_preload_hits=%llu index_fallback_copies=%llu",
                static_cast<unsigned long long>(profile_histogram_calls_), profile_grad_h2d_ms_,
                profile_index_h2d_ms_, profile_memset_ms_, profile_kernel_ms_, profile_d2h_ms_,
                profile_host_copy_ms_, static_cast<unsigned long long>(profile_index_preload_hits_),
                static_cast<unsigned long long>(profile_index_fallback_copies_));
    }
#endif
    if (stream_ != nullptr) {
      SynchronizeCUDAStream(stream_, __FILE__, __LINE__);
    }
    UnregisterDataIndices();
    for (auto* ptr : registered_histogram_buffers_) {
      CUDASUCCESS_OR_FATAL(cudaHostUnregister(ptr));
    }
    registered_histogram_buffers_.clear();
    if (stream_ != nullptr) {
      CUDASUCCESS_OR_FATAL(cudaStreamDestroy(stream_));
      stream_ = nullptr;
    }
    if (host_histogram_staging_ != nullptr) {
      CUDASUCCESS_OR_FATAL(cudaFreeHost(host_histogram_staging_));
      host_histogram_staging_ = nullptr;
      host_histogram_staging_size_ = 0;
    }
  }

  void Init(const Dataset* train_data, int gpu_device_id) {
    CHECK(train_data != nullptr);
    SetCUDADevice(gpu_device_id, __FILE__, __LINE__);

    cudaDeviceProp device_prop;
    CUDASUCCESS_OR_FATAL(cudaGetDeviceProperties(&device_prop, gpu_device_id));
    if (std::strncmp(device_prop.gcnArchName, "gfx1030", 7) != 0) {
      Log::Fatal("device_type=rdna2 currently requires gfx1030; detected %s", device_prop.gcnArchName);
    }
    if (stream_ == nullptr) {
      CUDASUCCESS_OR_FATAL(cudaStreamCreate(&stream_));
    }

    train_data_ = train_data;
    num_data_ = train_data_->num_data();
    InvalidatePreloadedDataIndices();
    dense_feature_groups_.clear();
    dense_feature_num_bins_.clear();

    const int num_feature_groups = train_data_->num_feature_groups();
    dense_feature_groups_.reserve(num_feature_groups);
    dense_feature_num_bins_.reserve(num_feature_groups);
    for (int group = 0; group < num_feature_groups; ++group) {
      if (train_data_->IsMultiGroup(group)) {
        continue;
      }
      const int num_bins = train_data_->FeatureGroupNumBin(group);
      if (num_bins > 255) {
        continue;
      }
      dense_feature_groups_.push_back(group);
      dense_feature_num_bins_.push_back(num_bins);
    }

    num_feature4_ = (dense_feature_groups_.size() + 3) / 4;
    std::vector<int> host_group_feature_indices(static_cast<size_t>(num_feature_groups), -1);
    for (int feature = 0; feature < train_data_->num_features(); ++feature) {
      const int group = train_data_->Feature2Group(feature);
      if (group >= 0 && group < num_feature_groups) {
        if (host_group_feature_indices[static_cast<size_t>(group)] >= 0) {
          host_group_feature_indices[static_cast<size_t>(group)] = -2;
        } else {
          host_group_feature_indices[static_cast<size_t>(group)] = feature;
        }
      }
    }
    const bool all_dense_single =
        dense_feature_groups_.size() == static_cast<size_t>(num_feature_groups) &&
        !dense_feature_groups_.empty() &&
        std::all_of(host_group_feature_indices.begin(), host_group_feature_indices.end(),
                    [](int feature) { return feature >= 0; });
    h64_eligible_ = all_dense_single &&
                    std::all_of(dense_feature_num_bins_.begin(), dense_feature_num_bins_.end(),
                                [](int bins) { return bins <= 64; });
    h128_eligible_ = all_dense_single &&
                     std::all_of(dense_feature_num_bins_.begin(), dense_feature_num_bins_.end(),
                                 [](int bins) { return bins <= 128; });

    num_total_bins_ = static_cast<size_t>(train_data_->NumTotalBin());
    const size_t histogram_values = num_total_bins_ * 2;
    if (host_histogram_staging_size_ != histogram_values) {
      if (host_histogram_staging_ != nullptr) {
        CUDASUCCESS_OR_FATAL(cudaFreeHost(host_histogram_staging_));
        host_histogram_staging_ = nullptr;
      }
      CUDASUCCESS_OR_FATAL(cudaHostAlloc(reinterpret_cast<void**>(&host_histogram_staging_),
                                         histogram_values * sizeof(hist_t), cudaHostAllocPortable));
      host_histogram_staging_size_ = histogram_values;
    }
    std::vector<uint32_t> host_group_bin_offsets(static_cast<size_t>(num_feature_groups) + 1);
    for (int group = 0; group <= num_feature_groups; ++group) {
      host_group_bin_offsets[static_cast<size_t>(group)] =
          static_cast<uint32_t>(train_data_->GroupBinBoundary(group));
    }

    const auto start = std::chrono::steady_clock::now();
    std::vector<PackedFeature4> host_packed(static_cast<size_t>(num_feature4_) * static_cast<size_t>(num_data_));

    #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
    for (int tuple = 0; tuple < static_cast<int>(num_feature4_); ++tuple) {
      const int group_base = tuple * 4;
      const int lanes = std::min(4, static_cast<int>(dense_feature_groups_.size()) - group_base);
      BinIterator* bin_iters[4] = {nullptr, nullptr, nullptr, nullptr};
      DenseBinIterator<uint8_t, false>* dense8[4] = {nullptr, nullptr, nullptr, nullptr};
      bool all_dense8 = lanes == 4;
      for (int lane = 0; lane < lanes; ++lane) {
        bin_iters[lane] = train_data_->FeatureGroupIterator(dense_feature_groups_[group_base + lane]);
        dense8[lane] = dynamic_cast<DenseBinIterator<uint8_t, false>*>(bin_iters[lane]);
        all_dense8 = all_dense8 && dense8[lane] != nullptr;
      }

      if (all_dense8) {
        DenseBinIterator<uint8_t, false> iter0 = *dense8[0];
        DenseBinIterator<uint8_t, false> iter1 = *dense8[1];
        DenseBinIterator<uint8_t, false> iter2 = *dense8[2];
        DenseBinIterator<uint8_t, false> iter3 = *dense8[3];
        for (data_size_t row = 0; row < num_data_; ++row) {
          const uint32_t packed = static_cast<uint32_t>(iter0.RawGet(row)) |
                                  (static_cast<uint32_t>(iter1.RawGet(row)) << 8) |
                                  (static_cast<uint32_t>(iter2.RawGet(row)) << 16) |
                                  (static_cast<uint32_t>(iter3.RawGet(row)) << 24);
          host_packed[static_cast<size_t>(tuple) * static_cast<size_t>(num_data_) + row].value = packed;
        }
        continue;
      }

      for (int lane = 0; lane < lanes; ++lane) {
        if (auto* lane_dense8 = dynamic_cast<DenseBinIterator<uint8_t, false>*>(bin_iters[lane])) {
          DenseBinIterator<uint8_t, false> iter = *lane_dense8;
          for (data_size_t row = 0; row < num_data_; ++row) {
            host_packed[static_cast<size_t>(tuple) * static_cast<size_t>(num_data_) + row].value |=
                static_cast<uint32_t>(iter.RawGet(row)) << (lane * 8);
          }
        } else if (auto* dense4 = dynamic_cast<DenseBinIterator<uint8_t, true>*>(bin_iters[lane])) {
          DenseBinIterator<uint8_t, true> iter = *dense4;
          for (data_size_t row = 0; row < num_data_; ++row) {
            host_packed[static_cast<size_t>(tuple) * static_cast<size_t>(num_data_) + row].value |=
                static_cast<uint32_t>(iter.RawGet(row)) << (lane * 8);
          }
        } else {
          Log::Fatal("RDNA2 dense packing requires DenseBin/Dense4bitsBin for feature group %d",
                     dense_feature_groups_[group_base + lane]);
        }
      }
    }

    const auto packed_ready = std::chrono::steady_clock::now();
    packed_features_.Resize(host_packed.size());
    group_bin_offsets_.InitFromHostVector(host_group_bin_offsets);
    group_feature_indices_.InitFromHostVector(host_group_feature_indices);
    feature_used_.Resize(static_cast<size_t>(train_data_->num_features()));
    gradients_.Resize(static_cast<size_t>(num_data_));
    hessians_.Resize(static_cast<size_t>(num_data_));
    data_indices_.Resize(static_cast<size_t>(num_data_));
    histogram_.Resize(num_total_bins_ * 2);
    const auto allocation_done = std::chrono::steady_clock::now();
    CopyFromHostToCUDADevice(packed_features_.RawData(), host_packed.data(), host_packed.size(), __FILE__, __LINE__);
    const auto upload_done = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> pack_elapsed = packed_ready - start;
    const std::chrono::duration<double, std::milli> allocation_elapsed = allocation_done - packed_ready;
    const std::chrono::duration<double, std::milli> upload_elapsed = upload_done - allocation_done;
    const double packed_mib = static_cast<double>(host_packed.size() * sizeof(PackedFeature4)) / (1024.0 * 1024.0);
    Log::Info("RDNA2 packed dataset: arch=%s rows=%d dense_groups=%d feature4=%d size=%.2f MiB h64=%s pack=%.3f ms alloc=%.3f ms H2D=%.3f ms",
              device_prop.gcnArchName, static_cast<int>(num_data_), static_cast<int>(dense_feature_groups_.size()),
              static_cast<int>(num_feature4_), packed_mib,
              h64_eligible_ ? "yes" : (h128_eligible_ ? "h128" : "no"),
              pack_elapsed.count(), allocation_elapsed.count(), upload_elapsed.count());
  }

  void BeforeTrain(const score_t* gradients, const score_t* hessians);

  bool ConstructHistogram(const std::vector<int8_t>& is_feature_used, const data_size_t* data_indices,
                          data_size_t num_data, hist_t* host_histogram);

  void PreloadDataIndices(const data_size_t* data_indices, data_size_t num_data) {
    InvalidatePreloadedDataIndices();
    if (data_indices == nullptr || num_data <= 0 || num_data >= num_data_) {
      return;
    }
    CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(device_data_indices(), data_indices,
                                         static_cast<size_t>(num_data) * sizeof(data_size_t),
                                         cudaMemcpyHostToDevice, stream()));
    preloaded_data_indices_ = data_indices;
    preloaded_data_count_ = num_data;
    preloaded_data_valid_ = true;
  }

  void RegisterDataIndices(const data_size_t* data_indices, size_t count) {
    UnregisterDataIndices();
    if (data_indices == nullptr || count == 0) {
      return;
    }
    const cudaError_t err = cudaHostRegister(const_cast<data_size_t*>(data_indices),
                                             count * sizeof(data_size_t), cudaHostRegisterPortable);
    if (err != cudaSuccess) {
      Log::Warning("RDNA2 could not pin DataPartition indices; leaf-index H2D will use pageable memory: %s",
                   cudaGetErrorString(err));
      return;
    }
    registered_data_indices_ = data_indices;
    registered_data_indices_count_ = count;
  }

  void UnregisterDataIndices() {
    InvalidatePreloadedDataIndices();
    if (registered_data_indices_ != nullptr) {
      if (stream_ != nullptr) {
        SynchronizeCUDAStream(stream_, __FILE__, __LINE__);
      }
      CUDASUCCESS_OR_FATAL(cudaHostUnregister(const_cast<data_size_t*>(registered_data_indices_)));
      registered_data_indices_ = nullptr;
      registered_data_indices_count_ = 0;
    }
  }

  bool ConsumePreloadedDataIndices(const data_size_t* data_indices, data_size_t num_data) {
    const bool matches = preloaded_data_valid_ && preloaded_data_indices_ == data_indices &&
                         preloaded_data_count_ == num_data;
    InvalidatePreloadedDataIndices();
    return matches;
  }

  bool EnsureCanonicalHistogramPinned(hist_t* host_histogram) {
    if (registered_histogram_buffers_.find(host_histogram) != registered_histogram_buffers_.end()) {
      return true;
    }
    const size_t bytes = num_total_bins_ * 2 * sizeof(hist_t);
    const cudaError_t err = cudaHostRegister(host_histogram, bytes, cudaHostRegisterPortable);
    if (err != cudaSuccess) {
      Log::Warning("RDNA2 could not pin canonical histogram buffer; falling back to staging copy: %s",
                   cudaGetErrorString(err));
      return false;
    }
    registered_histogram_buffers_.insert(host_histogram);
    return true;
  }

  bool h64_eligible() const { return h64_eligible_; }
  bool h128_eligible() const { return h128_eligible_; }
  data_size_t num_data() const { return num_data_; }
  size_t num_feature4() const { return num_feature4_; }
  size_t num_total_bins() const { return num_total_bins_; }
  const PackedFeature4* packed_features() const { return packed_features_.RawDataReadOnly(); }
  const uint32_t* group_bin_offsets() const { return group_bin_offsets_.RawDataReadOnly(); }
  const int* group_feature_indices() const { return group_feature_indices_.RawDataReadOnly(); }
  int8_t* device_feature_used() { return feature_used_.RawData(); }
  score_t* device_gradients() { return gradients_.RawData(); }
  score_t* device_hessians() { return hessians_.RawData(); }
  data_size_t* device_data_indices() { return data_indices_.RawData(); }
  hist_t* device_histogram() { return histogram_.RawData(); }
  hist_t* host_histogram_staging() { return host_histogram_staging_; }
  cudaStream_t stream() const { return stream_; }
  const std::vector<int>& dense_feature_groups() const { return dense_feature_groups_; }
  const std::vector<int>& dense_feature_num_bins() const { return dense_feature_num_bins_; }

#ifdef TIMETAG
  void ProfileAddGradientH2D(double elapsed_ms) { profile_grad_h2d_ms_ += elapsed_ms; }
  void ProfileAddHistogramCall(double index_h2d_ms, double memset_ms, double kernel_ms,
                               double d2h_ms, double host_copy_ms) {
    profile_index_h2d_ms_ += index_h2d_ms;
    profile_memset_ms_ += memset_ms;
    profile_kernel_ms_ += kernel_ms;
    profile_d2h_ms_ += d2h_ms;
    profile_host_copy_ms_ += host_copy_ms;
    ++profile_histogram_calls_;
  }
#endif

 private:
  void InvalidatePreloadedDataIndices() {
    preloaded_data_indices_ = nullptr;
    preloaded_data_count_ = 0;
    preloaded_data_valid_ = false;
  }

  const Dataset* train_data_ = nullptr;
  data_size_t num_data_ = 0;
  size_t num_feature4_ = 0;
  size_t num_total_bins_ = 0;
  bool h64_eligible_ = false;
  bool h128_eligible_ = false;
  std::vector<int> dense_feature_groups_;
  std::vector<int> dense_feature_num_bins_;
  CUDAVector<PackedFeature4> packed_features_;
  CUDAVector<uint32_t> group_bin_offsets_;
  CUDAVector<int> group_feature_indices_;
  CUDAVector<int8_t> feature_used_;
  CUDAVector<score_t> gradients_;
  CUDAVector<score_t> hessians_;
  CUDAVector<data_size_t> data_indices_;
  CUDAVector<hist_t> histogram_;
  hist_t* host_histogram_staging_ = nullptr;
  size_t host_histogram_staging_size_ = 0;
  cudaStream_t stream_ = nullptr;
  std::unordered_set<hist_t*> registered_histogram_buffers_;
  const data_size_t* registered_data_indices_ = nullptr;
  size_t registered_data_indices_count_ = 0;
  const data_size_t* preloaded_data_indices_ = nullptr;
  data_size_t preloaded_data_count_ = 0;
  bool preloaded_data_valid_ = false;
#ifdef TIMETAG
  uint64_t profile_histogram_calls_ = 0;
  double profile_grad_h2d_ms_ = 0.0;
  double profile_index_h2d_ms_ = 0.0;
  double profile_memset_ms_ = 0.0;
  double profile_kernel_ms_ = 0.0;
  double profile_d2h_ms_ = 0.0;
  double profile_host_copy_ms_ = 0.0;
  uint64_t profile_index_preload_hits_ = 0;
  uint64_t profile_index_fallback_copies_ = 0;
#endif
};

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_TREELEARNER_RDNA2_RDNA2_HISTOGRAM_ENGINE_HPP_
