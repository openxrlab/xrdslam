#include "cuda_utils.h"
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void get_features_kernel(
    const int *__restrict__ points, //(b, n, 3)
    const int *__restrict__ children,
    int *__restrict__ features,
    const int num_batch,
    const int num_sample)
{
    int n = num_sample;
    int b = blockIdx.x;
    int p = threadIdx.x;

    int cnt = 0;

    points += b * n * 3 + p * 3;
    features += b * n + p;
    int x = points[0];
    int y = points[1];
    int z = points[2];

    int size = children[8];

    unsigned edge = size / 2;
    int d = 0;
    while(children[d * 9 + 8] != 1) {
        int childid = ((x & edge) > 0) + 2 * ((y & edge) > 0) + 4 * ((z & edge) > 0);
        if (children[d * 9 + childid] == -1) {
            return;
        }
        d = children[d * 9 + childid];
        edge /= 2;
    }
    features[0] = d;
}

void get_features_kernel_wrapper(const int *points,
    const int *children,
    int *features,
    const int num_batch,
    const int num_sample)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    get_features_kernel<<<num_batch, num_sample, 0, stream>>>(points, children, features, num_batch, num_sample);
    CUDA_CHECK_ERRORS();
}
