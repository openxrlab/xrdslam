import os.path as osp

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))

setup(name='dpvo',
      packages=find_packages(),
      ext_modules=[
          CUDAExtension('cuda_corr',
                        sources=[
                            'altcorr/correlation.cpp',
                            'altcorr/correlation_kernel.cu'
                        ],
                        extra_compile_args={
                            'cxx': ['-O3'],
                            'nvcc': ['-O3'],
                        }),
          CUDAExtension('cuda_ba',
                        sources=['fastba/ba_dpvo.cpp', 'fastba/ba_cuda.cu'],
                        extra_compile_args={
                            'cxx': ['-O3'],
                            'nvcc': ['-O3'],
                        }),
          CUDAExtension('lietorch_backends',
                        include_dirs=[
                            osp.join(ROOT, 'lietorch/include'),
                            osp.join(ROOT, 'eigen-3.4.0')
                        ],
                        sources=[
                            'lietorch/src/lietorch.cpp',
                            'lietorch/src/lietorch_gpu.cu',
                            'lietorch/src/lietorch_cpu.cpp'
                        ],
                        extra_compile_args={
                            'cxx': ['-O3'],
                            'nvcc': ['-O3'],
                        }),
      ],
      cmdclass={'build_ext': BuildExtension})
