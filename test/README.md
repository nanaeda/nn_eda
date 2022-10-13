There are 3 different types of tests in this directory.
All of them can be executed from shell scripts.


## run_unit_tests.sh

This runs typical unit tests in unit_test.cpp.
This writes nn_test_export.txt to be later manually embedded into unit_test.cpp.


## run_pytorch_tests.sh

This runs pytorch_test.cpp and pytorch_test.py to check if my library trains models in a similar way with PyTorch.
Both of the code build the same NN architecture and train it using similar artificial data.
They output a validation loss on each epoch, and they need to be compared manually.


## run_codingame_tests.sh

1. run local training and export the trained to codingame_nn.txt.
2. import the model from codingame_nn.txt to codingame_test.cpp.
3. run local validation.
4. submit the code to codingame and run the validation.
5. compare the outputs from 3 and 4.
