#!/bin/bash

# build configured project
export VCPKG_FORCE_SYSTEM_BINARIES=1

cmake --build build --config Release -j4
