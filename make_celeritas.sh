#!/bin/bash

BUILD_DIR="/workspace/Celeritas/build"
# 检查 build 文件夹是否存在

# 检查 build 文件夹是否存在
if [ ! -d "$BUILD_DIR" ]; then
    # 如果不存在，则创建目录并执行 CMake 配置
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # 运行 CMake 配置
    cmake .. /tmp/pip-req-build-ba9g0dvc \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=/tmp/pip-req-build-ba9g0dvc/build/lib.linux-x86_64-3.7/celeritas \
    -DPYTHON_EXECUTABLE=/opt/conda/bin/python \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE \
    -DUSE_CUDA=TRUE \
    -DUSE_OMP=TRUE \
    -DCMAKE_INSTALL_RPATH=$ORIGIN
else
    # 如果存在，则直接进入 build 目录
    cd "$BUILD_DIR"
fi

# 使用 make 构建项目
make -j64

# 将生成的可执行文件复制到 /opt/conda/bin/ 目录
cp celeritas_train /opt/conda/bin/
cp celeritas_eval /opt/conda/bin/
