There are 3 different types of tests in this directory.
All of them can be executed from their shell script.


## run_unit_tests.sh

This runs typical unit tests in unit_test.cpp.
This writes nn_test_export.txt to be later embedded into unit_test.cpp manually.


## run_pytorch_tests.sh

This runs pytorch_test.cpp and pytorch_test.py to check if my library trains models in a similar way with PyTorch.
Both of the code build the same NN architecture and train it using similar artificial data.
They output a validation loss on each epoch, which need to be compared manually.


## run_codingame_tests.sh

This helps check if the code works well on codingame with some manual work.
This script runs codingame_test.cpp with and without -D_LOCAL. 

1. Run codingame_test.cpp with -D_LOCAL, which exports a model to codingame_nn.txt.
2. Copy and paste the epxorted model from codingame_nn.txt to codingame_test.cpp.
3. Run codingame_test.cpp without -D_LOCAL.
4. Submit codingame_test.cpp to codingame along with nn.cpp.
5. Confirm the outputs from 3 and 4 exactly match.

