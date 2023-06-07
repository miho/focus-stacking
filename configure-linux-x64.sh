#!/bin/bash

# configure VCPKG location
export VCPKG_ROOT="$HOME/software/vcpkg/"

# configure triplet
export VCPKG_TRIPLET="x64-linux"

#export VCPKG_FORCE_SYSTEM_BINARIES=1

export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:build/vcpkg_installed/${VCPKG_TRIPLET}/

# If vcpkg opencv on linux is used, make sure to set -DWITH_GTK=ON in portfile
#-DOpenCV_DIR=./build/vcpkg_installed/arm64-linux/share/opencv/ 

# If jetson version is used set
#-DOpenCV_DIR=/usr/lib/aarch64-linux-gnu/cmake/opencv4/ 

cmake -B build -S . -DCMAKE_CXX_COMPILER=g++-10 -DCMAKE_C_COMPILER=gcc-10 -DCMAKE_TOOLCHAIN_FILE="${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET="${VCPKG_TRIPLET}"
