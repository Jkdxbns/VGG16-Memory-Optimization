#!/bin/bash
set -e

echo "Rebuilding..."
cd "$(dirname "$0")/../build"
rm -rf *

cmake -DTorch_DIR=/home/jkdxbns/general/lib/python3.10/site-packages/torch/share/cmake/Torch ..
make -j$(nproc)

