#!/bin/bash

trap "rm -rf pytorch_test" EXIT SIGINT

python3 -m venv py_local_env
source py_local_env/bin/activate
trap "deactivate" EXIT
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

g++ pytorch_test.cpp -o pytorch_test -pthread -O3 -D LOCAL -std=c++0x # -D_GLIBCXX_DEBUG -fsanitize=address
time ./pytorch_test

time python pytorch_test.py
