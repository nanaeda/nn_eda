NnEda is my personal C++ neural network library for the dedicated usage in online programming competitions. Any backward-compatibility isn't guaranteed because code length is a critical factor in programming competitions and legacy functions are removed immediately. So, it's highly recommended to use this library as a reference for your own implementation.

Tested only on CodinGame.


## Usage of nn.cpp

This library only supports MLP models with softmax multi-head outputs.
The trained model can be exported as a string.

### Example Code

The following is an example code for a boardgame, reversi.

```

// Create a model.
vector<int> fc_widths = {64, 256, 128};
vector<int> head_widths = {2, 64}; // (value output, policy output)
Trainer trainer(fc_widths, head_widths);

// Train one sample.
float learning_rate = 1e-3;

vector<float> reversi_board(64);

float win_rate = ...;
int selected_move = ...;
vector<float> value_label = {1 - win_rate, win_rate}; // Sigmoid is achieved by Softmax with 2 outputs.
vector<float> policy_label = vector<float>(64, 0);
policy_label[selected_move] = 1.0;

trainer.train({reversi_board}, {{value_label, policy_label}}, learning_rate);

// Export model
int k_bits = 15; // discretize weight values into 2^{k_bits} integers.
string model_str = NnIo::serialize(trainer, k_bits); // Print or save this string as needed.

// Import model
Inferrer loaded_inferrer = NnIo::deserialize(model_str); // Embed or load model string as needed.
Trainer loaded_trainer(loaded_inferrer); // Trainer is a wrapper of Inferer. Use Trainer if code size isn't a concern.

// Make inference
loaded_inferrer.forward(reversi_board); 
float predicted_win_rate = loaded_inferrer.get_prediction(0, 1); 
float predicted_policy_prob_at_47 = loaded_inferrer.get_prediction(1, 47);

```

It's a bit confusing to have Trainer and Inferrer. 
I've split out Trainer out Inferrer so that Trainer code can removed on submission to save the code length. 

