#!/bin/bash

set -eux

# 71: epoch=0, avg_prob=0.0998688
# 71: epoch=10, avg_prob=0.679469
# 71: epoch=20, avg_prob=0.763363
# 71: epoch=30, avg_prob=0.771592
g++ pytorch_test.cpp -o pytorch_test -O3 -D LOCAL # -D_GLIBCXX_DEBUG -fsanitize=address
time ./pytorch_test

trap "rm -rf pytorch_test" EXIT SIGINT

python3 -m venv py_local_env
source py_local_env/bin/activate
trap "deactivate" EXIT
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu


time python pytorch_test.py
