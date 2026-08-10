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

#include "../../io/dense_bin.hpp"

namespace LightGBM {

class RDNA2HistogramEngine {
 public:
  struct PackedFeature4 {
    uint32_t value;
  };

  RDNA2HistogramEngine() = default;

  void Init(const Dataset* train_data, int gpu_device_id) {
    CHECK(train_data != nullptr);
    SetCUDADevice(gpu_device_id, __FILE__, __LINE__);

    cudaDeviceProp device_prop;
    CUDASUCCESS_OR_FATAL(cudaGetDeviceProperties(&device_prop, gpu_device_id));
    if (std::strncmp(device_prop.gcnArchName, "gfx1030", 7) != 0) {
      Log::Fatal("device_type=rdna2 currently requires gfx1030; detected %s", device_prop.gcnArchName);
    }

    train_data_ = train_data;
    num_data_ = train_data_->num_data();
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
    h64_eligible_ = !dense_feature_groups_.empty() &&
                    std::all_of(dense_feature_num_bins_.begin(), dense_feature_num_bins_.end(),
                                [](int bins) { return bins <= 64; });

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
    const auto allocation_done = std::chrono::steady_clock::now();
    CopyFromHostToCUDADevice(packed_features_.RawData(), host_packed.data(), host_packed.size(), __FILE__, __LINE__);
    const auto upload_done = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> pack_elapsed = packed_ready - start;
    const std::chrono::duration<double, std::milli> allocation_elapsed = allocation_done - packed_ready;
    const std::chrono::duration<double, std::milli> upload_elapsed = upload_done - allocation_done;
    const double packed_mib = static_cast<double>(host_packed.size() * sizeof(PackedFeature4)) / (1024.0 * 1024.0);
    Log::Info("RDNA2 packed dataset: arch=%s rows=%d dense_groups=%d feature4=%d size=%.2f MiB h64=%s pack=%.3f ms alloc=%.3f ms H2D=%.3f ms",
              device_prop.gcnArchName, static_cast<int>(num_data_), static_cast<int>(dense_feature_groups_.size()),
              static_cast<int>(num_feature4_), packed_mib, h64_eligible_ ? "yes" : "no",
              pack_elapsed.count(), allocation_elapsed.count(), upload_elapsed.count());
  }
  bool h64_eligible() const { return h64_eligible_; }
  data_size_t num_data() const { return num_data_; }
  size_t num_feature4() const { return num_feature4_; }
  const PackedFeature4* packed_features() const { return packed_features_.RawDataReadOnly(); }
  const std::vector<int>& dense_feature_groups() const { return dense_feature_groups_; }
  const std::vector<int>& dense_feature_num_bins() const { return dense_feature_num_bins_; }

 private:
  const Dataset* train_data_ = nullptr;
  data_size_t num_data_ = 0;
  size_t num_feature4_ = 0;
  bool h64_eligible_ = false;
  std::vector<int> dense_feature_groups_;
  std::vector<int> dense_feature_num_bins_;
  CUDAVector<PackedFeature4> packed_features_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_TREELEARNER_RDNA2_RDNA2_HISTOGRAM_ENGINE_HPP_
