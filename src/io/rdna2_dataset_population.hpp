/*!
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_IO_RDNA2_DATASET_POPULATION_HPP_
#define LIGHTGBM_SRC_IO_RDNA2_DATASET_POPULATION_HPP_

#include <LightGBM/dataset.h>

namespace LightGBM {

#ifdef USE_CUDA
bool RDNA2DenseFloat32DatasetPopulationNeedsPrepare(int num_rows, int num_cols);
bool RDNA2PrepareDenseFloat32DatasetPopulation(int num_rows, int num_cols, int gpu_device_id);

bool RDNA2PopulateDenseFloat32Dataset(Dataset* dataset, const float* data,
                                      int num_rows, int num_cols,
                                      int gpu_device_id);
#endif

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_IO_RDNA2_DATASET_POPULATION_HPP_
