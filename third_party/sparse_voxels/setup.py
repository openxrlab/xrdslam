# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

current_dir = os.getcwd()
_ext_sources = glob.glob('src/*.cpp') + glob.glob('src/*.cu')
include_path = os.path.join(current_dir, 'include')

setup(name='grid',
      ext_modules=[
          CUDAExtension(
              name='grid',
              sources=_ext_sources,
              include_dirs=[include_path],
              extra_compile_args={
                  'cxx': ['-O2', '-I./include'],
                  'nvcc': ['-O2', '-I./include'],
              },
          )
      ],
      cmdclass={'build_ext': BuildExtension})
