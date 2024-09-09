#include "../nn.cpp"

#include <bits/stdc++.h>
#include <random>
#include <sys/time.h>
#include <utility>
#include <x86intrin.h>

#define rep(i, n) for (int i = 0; (i) < ((int) (n)); (i)++)
#define sz(v) ((int) ((v).size()))
#define all(v) (v).begin(), (v).end()
#define debug(v) { cerr << __LINE__ << ": " << (#v) << "=" << (v) << endl; }
#define debug2(v0, v1) { cerr << __LINE__ << ": " << (#v0) << "=" << (v0) << ", " << (#v1) << "=" << (v1) << endl; }
#define debug3(v0, v1, v2) { cerr << __LINE__ << ": " << (#v0) << "=" << (v0) << ", " << (#v1) << "=" << (v1) <<  ", " << (#v2) << "=" << (v2) << endl; }

using namespace std;
using namespace nn_eda;


float my_rand(float mini, float maxi)
{
  int mask = (1 << 25) - 1;
  float f = ((float) (rand() & mask)) / mask;
  return mini + (maxi - mini) * f;
}

vector<float> gen_input(int dim, int target)
{
  vector<float> res;
  rep(i, dim) res.push_back(-my_rand(0, 2));
  res[target] = -1;
  return res;
}

vector<float> gen_label(int dim, int target)
{
  vector<float> res(dim, 0);
  res[target] = 1.0;
  return res;
}

int get_target(int dim)
{
  return rand() % dim;
}

double validate_model(Nn &nn, int dim, int n)
{
  srand(3456);

  double total_prob = 0.0;
  rep(loop, n){
    int target = get_target(dim);
    vector<float> input = gen_input(dim, target);
    total_prob += nn.forward(input)[target];
  }
  double avg_prob = total_prob / n;
  return avg_prob;
}

Nn train_model(vector<int> widths, int num_epochs, int num_samples, int batch_size, double learning_rate)
{
  srand(2345);

  Nn nn(widths);

  rep(epoch, num_epochs){
    // validation
    {
      double avg_prob = validate_model(nn, widths[0], num_samples);
      debug2(epoch, avg_prob);
    }

    // train
    {
      rep(loop, num_samples / batch_size){
        vector<vector<float>> inputs;
        vector<vector<float>> labels;
        rep(i, batch_size){
          int target = get_target(widths[0]);
          inputs.push_back(gen_input(widths[0], target));
          labels.push_back(gen_label(widths[0], target));
        }
        nn.train(inputs, labels, learning_rate);
      }
    }
  }

  return nn;
}

int main()
{
  vector<int> widths({10, 50, 50, 10});
  int num_epochs = 100;
  int num_samples = (int) 1e5;
  int batch_size = 32;
  double learning_rate = 3e-3;
  train_model(widths, num_epochs, num_samples, batch_size, learning_rate);
}
