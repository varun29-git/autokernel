#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

// ABI belongs to the trusted adapter. Each CTA quantizes one 128-value group.
extern "C" __global__ void fp8_rows(
    const __nv_bfloat16* input, uint8_t* codes, float* scales,
    int64_t rows, int64_t width, int64_t row_stride, int64_t column_stride,
    int64_t scale_stride, float epsilon) {
  const int64_t row = blockIdx.x;
  const int64_t column = (int64_t)blockIdx.y * 128 + threadIdx.x;
  const int lane = threadIdx.x;
  __shared__ float reduction[BLOCK_SIZE];
  float value = 0.0f;
  if (row < rows && lane < 128 && column < width)
    value = __bfloat162float(input[row * row_stride + column * column_stride]);
  reduction[lane] = fabsf(value);
  __syncthreads();
  for (int offset = BLOCK_SIZE / 2; offset; offset >>= 1) {
    if (lane < offset) {
      const float a = reduction[lane], b = reduction[lane + offset];
      reduction[lane] = (isnan(a) || isnan(b)) ? nanf("") : fmaxf(a, b);
    }
    __syncthreads();
  }
  float amplitude = reduction[0];
  if (!isnan(amplitude)) amplitude = fmaxf(amplitude, epsilon);
  float scale = amplitude > 0 ? fmaxf(amplitude * (1.0f / 448.0f), 1.1754943508222875e-38f)
                            : (amplitude == 0 ? 1.0f : amplitude);
  if (row < rows && lane == 0) scales[row + (int64_t)blockIdx.y * scale_stride] = scale;
  if (row < rows && lane < 128 && column < width) {
    float quotient = __fdiv_rn(value, scale);
    if (!isnan(quotient)) quotient = fminf(fmaxf(quotient, -448.0f), 448.0f);
    __nv_fp8_e4m3 encoded(quotient);
    codes[row * width + column] = encoded.__x;
  }
}
