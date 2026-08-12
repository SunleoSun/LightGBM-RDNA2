#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

constexpr int kFeatureTile = 8;
constexpr int kRowTile = 32;
constexpr int kThreads = kFeatureTile * kRowTile;

#define HIP_CHECK(expr) do { \
  const hipError_t err = (expr); \
  if (err != hipSuccess) { \
    std::fprintf(stderr, "HIP error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, hipGetErrorString(err)); \
    std::exit(2); \
  } \
} while (0)

__global__ void PopulateBinsKernel(const float* input, uint8_t* output,
                                   const double* upper_bounds,
                                   const uint16_t* num_bins,
                                   int chunk_rows, int total_rows,
                                   int num_features, int row_offset,
                                   int max_bins) {
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
    double value = static_cast<double>(input[static_cast<size_t>(row) * num_features + feature]);
    int bins = static_cast<int>(num_bins[feature]);
    int left = 0;
    int right = bins - 1;
    while (left < right) {
      const int mid = (left + right - 1) / 2;
      if (value <= upper_bounds[static_cast<size_t>(feature) * max_bins + mid]) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }
    bin = static_cast<uint8_t>(left);
  }
  tile[load_row][load_feature] = bin;
  __syncthreads();

  const int store_feature = tid / kRowTile;
  const int store_row = tid % kRowTile;
  const int out_feature = feature_base + store_feature;
  const int out_row = row_base + store_row;
  if (out_feature < num_features && out_row < chunk_rows) {
    output[static_cast<size_t>(out_feature) * total_rows + row_offset + out_row] =
        tile[store_row][store_feature];
  }
}

float ElapsedMs(hipEvent_t a, hipEvent_t b) {
  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, a, b));
  return ms;
}

}  // namespace

int main(int argc, char** argv) {
  const int rows = argc > 1 ? std::atoi(argv[1]) : 40000;
  const int features = argc > 2 ? std::atoi(argv[2]) : 3000;
  const int bins = argc > 3 ? std::atoi(argv[3]) : 64;
  const int chunk_rows = argc > 4 ? std::atoi(argv[4]) : std::min(rows, 8192);
  const int repeats = argc > 5 ? std::atoi(argv[5]) : 5;
  const bool stage_copy = argc > 6 ? std::atoi(argv[6]) != 0 : false;
  if (rows <= 0 || features <= 0 || bins < 2 || bins > 128 || chunk_rows <= 0) {
    std::fprintf(stderr, "invalid arguments\n");
    return 1;
  }

  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  HIP_CHECK(hipSetDevice(0));
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));

  const int actual_chunk_rows = std::min(rows, chunk_rows);
  const size_t input_elems = static_cast<size_t>(actual_chunk_rows) * features;
  const size_t input_bytes = input_elems * sizeof(float);
  const size_t output_bytes = static_cast<size_t>(rows) * features * sizeof(uint8_t);
  const size_t bounds_elems = static_cast<size_t>(features) * bins;

  float* host_input = nullptr;
  uint8_t* host_output = nullptr;
  std::vector<float> pageable_source(stage_copy ? input_elems : 0);
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&host_input), input_bytes, hipHostMallocPortable));
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&host_output), output_bytes, hipHostMallocPortable));
  std::vector<double> host_bounds(bounds_elems);
  std::vector<uint16_t> host_num_bins(features, static_cast<uint16_t>(bins));
  std::mt19937 rng(20260812);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (size_t i = 0; i < input_elems; ++i) {
    const float value = dist(rng);
    host_input[i] = value;
    if (stage_copy) pageable_source[i] = value;
  }
  for (int f = 0; f < features; ++f) {
    for (int b = 0; b < bins; ++b) {
      host_bounds[static_cast<size_t>(f) * bins + b] = -3.0 + 6.0 * (b + 1) / bins;
    }
  }

  float* dev_input = nullptr;
  uint8_t* dev_output = nullptr;
  double* dev_bounds = nullptr;
  uint16_t* dev_num_bins = nullptr;
  HIP_CHECK(hipMalloc(&dev_input, input_bytes));
  HIP_CHECK(hipMalloc(&dev_output, output_bytes));
  HIP_CHECK(hipMalloc(&dev_bounds, bounds_elems * sizeof(double)));
  HIP_CHECK(hipMalloc(&dev_num_bins, static_cast<size_t>(features) * sizeof(uint16_t)));
  HIP_CHECK(hipMemcpy(dev_bounds, host_bounds.data(), bounds_elems * sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dev_num_bins, host_num_bins.data(), static_cast<size_t>(features) * sizeof(uint16_t), hipMemcpyHostToDevice));

  hipEvent_t start{}, after_h2d{}, after_kernel{}, after_d2h{};
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&after_h2d));
  HIP_CHECK(hipEventCreate(&after_kernel));
  HIP_CHECK(hipEventCreate(&after_d2h));

  double total_h2d = 0.0;
  double total_kernel = 0.0;
  double total_d2h = 0.0;
  double total_wall = 0.0;
  const dim3 block(kThreads);

  for (int rep = -1; rep < repeats; ++rep) {
    const auto wall_begin = std::chrono::steady_clock::now();
    HIP_CHECK(hipEventRecord(start, stream));
    for (int row_offset = 0; row_offset < rows; row_offset += actual_chunk_rows) {
      const int this_rows = std::min(actual_chunk_rows, rows - row_offset);
      const size_t this_input_bytes = static_cast<size_t>(this_rows) * features * sizeof(float);
      if (stage_copy) {
        std::memcpy(host_input, pageable_source.data(), this_input_bytes);
      }
      HIP_CHECK(hipMemcpyAsync(dev_input, host_input, this_input_bytes, hipMemcpyHostToDevice, stream));
      const dim3 grid((features + kFeatureTile - 1) / kFeatureTile,
                      (this_rows + kRowTile - 1) / kRowTile);
      hipLaunchKernelGGL(PopulateBinsKernel, grid, block, 0, stream,
                         dev_input, dev_output, dev_bounds, dev_num_bins, this_rows, rows,
                         features, row_offset, bins);
      HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipEventRecord(after_h2d, stream));
    HIP_CHECK(hipEventRecord(after_kernel, stream));
    HIP_CHECK(hipMemcpyAsync(host_output, dev_output, output_bytes, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipEventRecord(after_d2h, stream));
    HIP_CHECK(hipEventSynchronize(after_d2h));
    const auto wall_end = std::chrono::steady_clock::now();
    if (rep >= 0) {
      // H2D and kernels are interleaved per chunk; use wall-minus-final-D2H for their combined time.
      const double d2h_ms = ElapsedMs(after_kernel, after_d2h);
      const double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();
      total_d2h += d2h_ms;
      total_wall += wall_ms;
      total_h2d += 0.0;
      total_kernel += wall_ms - d2h_ms;
    }
  }

  const double avg_wall = total_wall / repeats;
  const double avg_stream_compute = total_kernel / repeats;
  const double avg_d2h = total_d2h / repeats;
  const double total_input_gib = static_cast<double>(rows) * features * sizeof(float) / (1024.0 * 1024.0 * 1024.0);
  const double total_output_gib = static_cast<double>(rows) * features / (1024.0 * 1024.0 * 1024.0);
  std::printf("device=%s rows=%d features=%d bins=%d chunk_rows=%d chunks=%d stage_copy=%s input=%.3fGiB output=%.3fGiB\n",
              prop.name, rows, features, bins, actual_chunk_rows,
              (rows + actual_chunk_rows - 1) / actual_chunk_rows, stage_copy ? "yes" : "no",
              total_input_gib, total_output_gib);
  std::printf("avg_wall_ms=%.3f interleaved_h2d_plus_kernel_ms=%.3f final_d2h_ms=%.3f effective_input_GBps=%.2f effective_output_GBps=%.2f checksum=%u\n",
              avg_wall, avg_stream_compute, avg_d2h,
              (static_cast<double>(rows) * features * sizeof(float) / 1.0e9) / (avg_stream_compute / 1000.0),
              (static_cast<double>(rows) * features / 1.0e9) / (avg_d2h / 1000.0),
              static_cast<unsigned>(host_output[0]));

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(after_h2d));
  HIP_CHECK(hipEventDestroy(after_kernel));
  HIP_CHECK(hipEventDestroy(after_d2h));
  HIP_CHECK(hipFree(dev_input));
  HIP_CHECK(hipFree(dev_output));
  HIP_CHECK(hipFree(dev_bounds));
  HIP_CHECK(hipFree(dev_num_bins));
  HIP_CHECK(hipHostFree(host_input));
  HIP_CHECK(hipHostFree(host_output));
  HIP_CHECK(hipStreamDestroy(stream));
  return 0;
}
