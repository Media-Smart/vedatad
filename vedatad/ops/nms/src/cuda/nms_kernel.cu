// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/DeviceGuard.h>

#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>

#include <vector>
#include <iostream>

int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__ inline float devIoU(float const * const a, float const * const b) {
  float left = max(a[0], b[0]), right = min(a[1], b[1]);
  float interS = max(right - left, 0.f);
  float Sa = a[1] - a[0];
  float Sb = b[1] - b[0];
  return interS / (Sa + Sb - interS);
}

__global__ void nms_kernel(const int n_segments, const float nms_overlap_thresh,
                           const float *dev_segments, unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_segments - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_segments - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_segments[threadsPerBlock * 3];
  if (threadIdx.x < col_size) {
    block_segments[threadIdx.x * 3 + 0] =
        dev_segments[(threadsPerBlock * col_start + threadIdx.x) * 3 + 0];
    block_segments[threadIdx.x * 3 + 1] =
        dev_segments[(threadsPerBlock * col_start + threadIdx.x) * 3 + 1];
    block_segments[threadIdx.x * 3 + 2] =
        dev_segments[(threadsPerBlock * col_start + threadIdx.x) * 3 + 2];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_segment_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_segment = dev_segments + cur_segment_idx * 3;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_segment, block_segments + i * 3) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = THCCeilDiv(n_segments, threadsPerBlock);
    dev_mask[cur_segment_idx * col_blocks + col_start] = t;
  }
}

// segments is a N x 3 tensor
at::Tensor nms_cuda_forward(const at::Tensor segments, float nms_overlap_thresh) {

  // Ensure CUDA uses the input tensor device.
  at::DeviceGuard guard(segments.device());

  using scalar_t = float;
  AT_ASSERTM(segments.device().is_cuda(), "segments must be a CUDA tensor");
  auto scores = segments.select(1, 2);
  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));
  auto segments_sorted = segments.index_select(0, order_t);

  int segments_num = segments.size(0);

  const int col_blocks = THCCeilDiv(segments_num, threadsPerBlock);

  scalar_t* segments_dev = segments_sorted.data_ptr<scalar_t>();

  THCState *state = at::globalContext().lazyInitCUDA(); // TODO replace with getTHCState

  unsigned long long* mask_dev = NULL;
  //THCudaCheck(THCudaMalloc(state, (void**) &mask_dev,
  //                      segments_num * col_blocks * sizeof(unsigned long long)));

  mask_dev = (unsigned long long*) THCudaMalloc(state, segments_num * col_blocks * sizeof(unsigned long long));

  dim3 blocks(THCCeilDiv(segments_num, threadsPerBlock),
              THCCeilDiv(segments_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  nms_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(segments_num,
                                  nms_overlap_thresh,
                                  segments_dev,
                                  mask_dev);

  std::vector<unsigned long long> mask_host(segments_num * col_blocks);
  THCudaCheck(cudaMemcpyAsync(
			  &mask_host[0],
			  mask_dev,
			  sizeof(unsigned long long) * segments_num * col_blocks,
			  cudaMemcpyDeviceToHost,
			  at::cuda::getCurrentCUDAStream()
			  ));

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  at::Tensor keep = at::empty({segments_num}, segments.options().dtype(at::kLong).device(at::kCPU));
  int64_t* keep_out = keep.data_ptr<int64_t>();

  int num_to_keep = 0;
  for (int i = 0; i < segments_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  THCudaFree(state, mask_dev);
  // TODO improve this part
  return order_t.index({
      keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep).to(
          order_t.device(), keep.scalar_type())});
}
