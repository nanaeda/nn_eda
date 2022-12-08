#!/bin/bash

set -eux

trap "rm -rf codingame_test" EXIT SIGINT
rm -rf codingame_nn.txt

g++ codingame_test.cpp -o codingame_test -O2 -D LOCAL -D_GLIBCXX_DEBUG -fsanitize=address
time ./codingame_test

g++ codingame_test.cpp -o codingame_test -O2 -D_GLIBCXX_DEBUG -fsanitize=address
time ./codingame_test
