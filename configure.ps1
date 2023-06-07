
# configure VCPKG location
$env:VCPKG_ROOT="\dev\vcpkg\"

# configure triplet
$env:VCPKG_TRIPLET="x64-windows"

cmake -B build -S . -G"Visual Studio 16 2019" -DCMAKE_TOOLCHAIN_FILE="${env:VCPKG_ROOT}\scripts\buildsystems\vcpkg.cmake" -DVCPKG_TARGET_TRIPLET="${env:VCPKG_TRIPLET}" -DVCPKG_BUILD_TYPE=release -DUEYE_DIR_SEARCH_INC="deps\win-x64\IDS\uEye\develop\include" -DUEYE_DIR_SEARCH_LIB="deps\win-x64\IDS\uEye\develop\lib" -DDAHENG_DIR_SEARCH_INC="deps\win-x64\GalaxySDK\Samples\C++ SDK\inc" -DDAHENG_DIR_SEARCH_LIB="deps\win-x64\GalaxySDK\Samples\C++ SDK\lib\x64"
