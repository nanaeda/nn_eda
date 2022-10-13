#!/bin/bash

trap "rm -rf codingame_test" EXIT SIGINT
rm -rf codingame_nn.txt

g++ codingame_test.cpp -o codingame_test -pthread -O2 -D LOCAL -std=c++0x -D_GLIBCXX_DEBUG -fsanitize=address
time ./codingame_test

g++ codingame_test.cpp -o codingame_test -pthread -O2 -std=c++0x -D_GLIBCXX_DEBUG -fsanitize=address
time ./codingame_test
