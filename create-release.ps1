# build configured project

#cmake --build build --config Debug -j8
cmake --build build --config Release -j8 # -DVCPKG_BUILD_TYPE=release

Remove-Item -Path build\dist -Recurse -Force 
Remove-Item -Path build\bfx-strobocore.zip -Recurse -Force

mkdir -p build\dist\

cp -r build\bin\ build\dist\bfx-strobocore

cp README.md build\dist\bfx-strobocore
7z a build\bfx-strobocore.zip .\build\dist\bfx-strobocore
