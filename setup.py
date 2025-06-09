#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import re
import sys
from pathlib import Path
from typing import List

import pccm
from ccimport import compat
from pccm.extension import PCCMBuild, PCCMExtension
from setuptools import find_packages, setup
from setuptools.extension import Extension

# 使用するCUDAのバージョン（例: "11.8"）。
CUMM_CUDA_VERSION = os.environ.get("CUMM_CUDA_VERSION", "")

# --- パッケージのメタデータ定義 ---
NAME = 'spconv'
RELEASE_NAME = NAME
deps = ["cumm"]

# CUDAバージョンに基づいてリリース名と依存関係を調整
if CUMM_CUDA_VERSION:
    cuda_ver_str = CUMM_CUDA_VERSION.replace(".", "")  # 11.8 -> 118
    RELEASE_NAME += f"-cu{cuda_ver_str}"
    deps = [f"cumm-cu{cuda_ver_str}>=0.7.11, <1.0.0"]
else:
    # CUMM_CUDA_VERSIONが設定されていない場合はビルドを中止
    raise RuntimeError("CUMM_CUDA_VERSION must be set. e.g. export CUMM_CUDA_VERSION=11.8")

DESCRIPTION = 'spatial sparse convolution'
URL = 'https://github.com/traveller59/spconv'
EMAIL = 'yanyan.sub@outlook.com'
AUTHOR = 'Yan Yan'
REQUIRES_PYTHON = '>=3.9'

# --- 依存パッケージの定義 ---
REQUIRED = [
    "pccm>=0.4.16",
    "ccimport>=0.4.4",
    "pybind11>=2.6.0",
    "fire",
    "numpy<2.0.0",
    *deps
]

EXTRAS = {}

here = os.path.abspath(os.path.dirname(__file__))
sys.path.append(str(Path(__file__).parent))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# バージョン情報の読み込みと設定
about = {}
with open('version.txt', 'r') as f:
    version = f.read().strip()

version_path = Path(here) / NAME / '__version__.py'
with open(version_path, 'w') as f:
    f.write(f"__version__ = '{version}'\n")
about['__version__'] = version

# --- AOT (事前コンパイル) の設定 ---

# setup.py build_ext 実行時に pccm のビルドプロセスを呼び出すように設定
cmdclass = {
    'build_ext': PCCMBuild,
}

# C++拡張のビルドに必要なモジュールをインポート
from cumm.common import CompileInfo
from cumm.conv.main import ConvMainUnitTest
from cumm.gemm.main import GemmMainUnitTest
from spconv.core import (IMPLGEMM_AMPERE_PARAMS, IMPLGEMM_SIMT_PARAMS,
                         IMPLGEMM_TURING_PARAMS, IMPLGEMM_VOLTA_PARAMS,
                         SHUFFLE_AMPERE_PARAMS, SHUFFLE_SIMT_PARAMS,
                         SHUFFLE_TURING_PARAMS, SHUFFLE_VOLTA_PARAMS)
from spconv.csrc.hash.core import HashTable
from spconv.csrc.sparse.all import SpconvOps
from spconv.csrc.sparse.alloc import ExternalAllocator
from spconv.csrc.sparse.convops import (ConvGemmOps, ConvTunerSimple,
                                        ExternalSpconvMatmul, GemmTunerSimple)
from spconv.csrc.sparse.inference import InferenceOps
from spconv.csrc.utils import BoxOps, PointCloudCompress

# ビルド対象のカーネルパラメータをすべて結合
all_shuffle = SHUFFLE_SIMT_PARAMS + SHUFFLE_VOLTA_PARAMS + SHUFFLE_TURING_PARAMS + SHUFFLE_AMPERE_PARAMS
all_imp = (IMPLGEMM_SIMT_PARAMS + IMPLGEMM_VOLTA_PARAMS +
           IMPLGEMM_TURING_PARAMS + IMPLGEMM_AMPERE_PARAMS)

cu = GemmMainUnitTest(all_shuffle)
cu.namespace = "cumm.gemm.main"
convcu = ConvMainUnitTest(all_imp)
convcu.namespace = "cumm.conv.main"

# CUDAバージョンに基づいて使用するC++標準を決定
std = "c++17"
cuda_ver_tuple = tuple(map(int, CUMM_CUDA_VERSION.split(".")))
if cuda_ver_tuple[0] < 11:
    std = "c++14"

# pccmに渡すC++コード定義体のリストを作成
gemmtuner = GemmTunerSimple(cu)
gemmtuner.namespace = "csrc.sparse.convops.gemmops"
convtuner = ConvTunerSimple(convcu)
convtuner.namespace = "csrc.sparse.convops.convops"
convops = ConvGemmOps(gemmtuner, convtuner)
convops.namespace = "csrc.sparse.convops.spops"

cus = [
    gemmtuner, convtuner, convops, SpconvOps(), BoxOps(), HashTable(),
    CompileInfo(), ExternalAllocator(), PointCloudCompress(),
    ExternalSpconvMatmul(), InferenceOps()
]
# GPU関連のコアコンポーネントを追加
cus.extend([cu, convcu])

# setuptoolsに渡す拡張モジュールを定義
ext_modules: List[Extension] = [
    PCCMExtension(
        cus,
        "spconv/core_cc",  # 生成されるモジュール名
        Path(__file__).resolve().parent / "spconv",  # ソース生成先
        std=std,
        disable_pch=True,
        verbose=True
    )
]

# --- パッケージング実行 ---
setup(
    name=RELEASE_NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='Apache License 2.0',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)
