#!/bin/bash

rm -rf main test nn_test
g++ nn_test.cpp -o nn_test -pthread -O2 -D LOCAL -std=c++0x -D_GLIBCXX_DEBUG -fsanitize=address
