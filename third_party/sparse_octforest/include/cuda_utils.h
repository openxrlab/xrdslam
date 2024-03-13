#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#include <torch/extension.h>
#include <cuda_runtime.h>

#define TOTAL_THREADS 512
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be an float tensor")
#define CUDA_CHECK_ERRORS()                                           \
  {                                                                   \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err)                                           \
    {                                                                 \
      fprintf(stderr, "CUDA kernel failed : %s\n%s at L:%d in %s\n",  \
              cudaGetErrorString(err), __PRETTY_FUNCTION__, __LINE__, \
              __FILE__);                                              \
      exit(-1);                                                       \
    }                                                                 \
  }

inline int imax(int a, int b)
{
    return a > b ? a : b;
}

inline int imin(int a, int b)
{
    return a < b ? a : b;
}

inline int opt_n_threads(int work_size)
{
  const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);

  return imax(imin(1 << pow_2, TOTAL_THREADS), 1);
}

#endif
