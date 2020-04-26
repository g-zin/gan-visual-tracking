#!/bin/bash

mkdir -p build/temp.linux-x86_64-3.7/src/

g++ -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -fPIC -I/project/6019510/zin1991/anas/ENV/lib/python3.7/site-packages/torch/lib/include -I/project/6019510/zin1991/anas/ENV/lib/python3.7/site-packages/torch/lib/include/torch/csrc/api/include -I/project/6019510/zin1991/anas/ENV/lib/python3.7/site-packages/torch/lib/include/TH -I/project/6019510/zin1991/anas/ENV/lib/python3.7/site-packages/torch/lib/include/THC -I/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/intel2016.4/cuda/8.0.44/include -I/cvmfs/soft.computecanada.ca/easybuild/software/2017/Core/python/3.7.0/include/python3.7m -c src/roi_align_cuda.cpp -o build/temp.linux-x86_64-3.7/src/roi_align_cuda.o -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=roi_align_cuda -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++11

/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/intel2016.4/cuda/8.0.44/bin/nvcc -I/project/6019510/zin1991/anas/ENV/lib/python3.7/site-packages/torch/lib/include -I/project/6019510/zin1991/anas/ENV/lib/python3.7/site-packages/torch/lib/include/torch/csrc/api/include -I/project/6019510/zin1991/anas/ENV/lib/python3.7/site-packages/torch/lib/include/TH -I/project/6019510/zin1991/anas/ENV/lib/python3.7/site-packages/torch/lib/include/THC -I/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/intel2016.4/cuda/8.0.44/include -I/cvmfs/soft.computecanada.ca/easybuild/software/2017/Core/python/3.7.0/include/python3.7m -c src/roi_align_kernel_c.cu -o build/temp.linux-x86_64-3.7/src/roi_align_kernel_c.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --compiler-options '-fPIC' -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=roi_align_cuda -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++11

mkdir -p build/lib.linux-x86_64-3.7

g++ -pthread -shared build/temp.linux-x86_64-3.7/src/roi_align_cuda.o build/temp.linux-x86_64-3.7/src/roi_align_kernel_c.o -L/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/intel2016.4/cuda/8.0.44/lib64 -L/cvmfs/soft.computecanada.ca/easybuild/software/2017/Core/python/3.7.0/lib -lcudart -lpython3.7m -o build/lib.linux-x86_64-3.7/roi_align_cuda.cpython-37m-x86_64-linux-gnu.so

cp build/lib.linux-x86_64-3.7/roi_align_cuda.cpython-37m-x86_64-linux-gnu.so .
