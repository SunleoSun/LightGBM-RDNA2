/*!
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_RDNA2_RDNA2_TREE_LEARNER_HPP_
#define LIGHTGBM_SRC_TREELEARNER_RDNA2_RDNA2_TREE_LEARNER_HPP_

#include "../serial_tree_learner.h"

namespace LightGBM {

/*! \brief RDNA2 learner boundary. Host-side tree semantics intentionally remain SerialTreeLearner semantics. */
class RDNA2TreeLearner final : public SerialTreeLearner {
 public:
  explicit RDNA2TreeLearner(const Config* config) : SerialTreeLearner(config) {}

 protected:
  void ConstructHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract) override {
    // Phase 0 correctness baseline: keep canonical SerialTreeLearner histogram construction.
    // RDNA2HistogramEngine will replace only this boundary once the backend routing is proven.
    SerialTreeLearner::ConstructHistograms(is_feature_used, use_subtract);
  }
};

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_TREELEARNER_RDNA2_RDNA2_TREE_LEARNER_HPP_
