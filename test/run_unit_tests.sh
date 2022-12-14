#!/bin/bash

set -eux

trap "rm -rf unit_test" EXIT SIGINT
trap "rm -rf nn_io_test.txt" EXIT SIGINT
rm -rf nn_test_export.txt  # nn_test writes this.

g++ unit_test.cpp -o unit_test -O2 -D LOCAL -D_GLIBCXX_DEBUG -fsanitize=address
time ./unit_test
