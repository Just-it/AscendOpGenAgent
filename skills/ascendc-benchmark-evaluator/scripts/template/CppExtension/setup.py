import os
import sys
import glob
import sysconfig
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension

import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
USE_NINJA = os.getenv('USE_NINJA') == '1'
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

source_files = glob.glob(os.path.join(BASE_DIR, "csrc", "*.cpp"), recursive=True)

python_include_dir = sysconfig.get_path('include')
if not python_include_dir:
    python_include_dir = os.path.join(sys.prefix, 'include', f'python{sys.version_info.major}.{sys.version_info.minor}')

exts = []
ext = NpuExtension(
    name="custom_ops_lib",
    sources=source_files,
    extra_compile_args = [
        '-I' + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/acl/inc"),
        '-I' + python_include_dir,
    ],
)
exts.append(ext)

setup(
    name="custom_ops",
    version='1.0',
    keywords='custom_ops',
    ext_modules=exts,
    packages=find_packages(),
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=USE_NINJA)},
)
