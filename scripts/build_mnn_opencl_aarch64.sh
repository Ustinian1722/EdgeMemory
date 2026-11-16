#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/d/A-IQ-9075
cross_libs="$repo/.local/cross/sysroot/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$cross_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cmake -S "$repo/.local/mnn-opencl-source" \
  -B "$repo/.local/mnn-opencl-build" \
  -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$repo/.local/cross/toolchain-geniex.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DMNN_OPENCL=ON \
  -DMNN_USE_SYSTEM_LIB=OFF \
  -DMNN_BUILD_SHARED_LIBS=ON \
  -DMNN_SEP_BUILD=OFF \
  -DMNN_BUILD_TOOLS=OFF \
  -DMNN_BUILD_CONVERTER=OFF \
  -DMNN_BUILD_TRAIN=OFF \
  -DMNN_BUILD_TEST=OFF

cmake --build "$repo/.local/mnn-opencl-build" --target MNN --parallel 6

cmake -S "$repo/edgememory/native/mnn_clip_bridge" \
  -B "$repo/.local/mnn-clip-bridge-build" \
  -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$repo/.local/cross/toolchain-geniex.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DMNN_SOURCE_DIR="$repo/.local/mnn-opencl-source" \
  -DMNN_BUILD_DIR="$repo/.local/mnn-opencl-build"

cmake --build "$repo/.local/mnn-clip-bridge-build" --parallel 6
