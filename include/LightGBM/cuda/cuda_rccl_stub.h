/*
 * Minimal RCCL API surface for single-GPU ROCm builds on Windows.
 *
 * AMD's Windows HIP SDK does not ship RCCL. LightGBM's single-GPU CUDA/ROCm
 * path still references RCCL types in shared headers, even though it never
 * initializes an RCCL communicator when num_gpu == 1. These definitions keep
 * that path buildable while making every attempted RCCL operation fail
 * explicitly instead of silently pretending multi-GPU support exists.
 */
#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_RCCL_STUB_H_
#define LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_RCCL_STUB_H_

#include <cstddef>

using ncclComm_t = void*;

struct ncclUniqueId {
  char internal[128];
};

enum ncclResult_t {
  ncclSuccess = 0,
  ncclInvalidUsage = 5
};

enum ncclDataType_t {
  ncclInt32 = 0,
  ncclInt64 = 1,
  ncclFloat32 = 2,
  ncclFloat64 = 3
};

enum ncclRedOp_t {
  ncclSum = 0,
  ncclProd = 1,
  ncclMax = 2,
  ncclMin = 3,
  ncclAvg = 4
};

inline const char* ncclGetErrorString(ncclResult_t result) {
  return result == ncclSuccess ? "no error" : "RCCL is unavailable in the Windows HIP SDK";
}

inline ncclResult_t ncclGetUniqueId(ncclUniqueId*) { return ncclInvalidUsage; }
inline ncclResult_t ncclGroupStart() { return ncclInvalidUsage; }
inline ncclResult_t ncclGroupEnd() { return ncclInvalidUsage; }
inline ncclResult_t ncclCommInitRank(ncclComm_t*, int, ncclUniqueId, int) { return ncclInvalidUsage; }
inline ncclResult_t ncclAllReduce(const void*, void*, std::size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t) {
  return ncclInvalidUsage;
}

#endif  // LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_RCCL_STUB_H_
