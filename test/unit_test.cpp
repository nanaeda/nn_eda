#include "../nn.cpp"

#include <bits/stdc++.h>
#include <random>
#include <sys/time.h>
#include <utility>
#include <x86intrin.h>
#include "dqn_sampler.cpp"

#define rep(i, n) for (int i = 0; (i) < ((int) (n)); (i)++)
#define sz(v) ((int) ((v).size()))
#define all(v) (v).begin(), (v).end()
#define debug(v) { cerr << __LINE__ << ": " << (#v) << "=" << (v) << endl; }
#define debug2(v0, v1) { cerr << __LINE__ << ": " << (#v0) << "=" << (v0) << ", " << (#v1) << "=" << (v1) << endl; }
#define debug3(v0, v1, v2) { cerr << __LINE__ << ": " << (#v0) << "=" << (v0) << ", " << (#v1) << "=" << (v1) <<  ", " << (#v2) << "=" << (v2) << endl; }

using namespace std;
using namespace nn_eda;


class Assert 
{
public:
  static void equals(float f0, float f1, float delta = 1e-6) 
  {
    assert(abs(f0 - f1) <= delta);
  }

  static void equals(int i0, int i1) 
  {
    assert(i0 == i1);
  }
};

void declare_test_name(string test_name)
{
  cout << endl << "[" << test_name << "]" << endl;
}

#define run_test(test_func) { \
  cout << "  running: " << (#test_func) << endl; \
  test_func(); \
} 

float my_rand(float mini, float maxi)
{
  int mask = (1 << 25) - 1;
  float f = ((float) (rand() & mask)) / mask;
  return mini + (maxi - mini) * f;
}

float random_float(float mini, float maxi)
{
  int mask = (1 << 25) - 1;
  float f = ((float) (rand() & mask)) / mask;
  return mini + (maxi - mini) * f;
}

class DqnSamplerTester
{
public:
  static void test()
  {
    declare_test_name("DqnSamplerTester");
    run_test(test_0);
  }

  static void test_0()
  {
    srand(1234);

    int n = 20;

    vector<double> weights;
    rep(i, n) weights.push_back(random_float(0, 2));
    double total_weight = 0.0;
    rep(i, n) total_weight += weights[i];

    DqnSampler<int> sampler(n);
    rep(i, n / 2) sampler.add_or_overwrite(i, 1.0);
    rep(i, n) sampler.add_or_overwrite(i, weights[(n / 2 + i) % sz(weights)]);

    int num_attempts = 1000000;
    vector<int> counts(n, 0);
    rep(i, num_attempts) ++counts[sampler.get_index()];

    rep(i, n){
      double expected = weights[i] / total_weight * num_attempts;
      assert(abs(expected - counts[i]) / counts[i] < 0.05);
    } 
  }
};

class ScalerTester
{
public:
  static void test()
  {
    declare_test_name("ScalerTester");
    run_test(test_scale_0);
    run_test(test_scale_1);
    run_test(test_scale_2);
    run_test(test_unscale_0);
    run_test(test_unscale_1);
    run_test(test_scale_unscale);
  }

  static void test_scale_unscale()
  {
    float mini = 234;
    float maxi = 2333;
    Scaler scaler(mini, maxi);
    rep(i, 100){
      float f = my_rand(mini, maxi);
      Assert::equals(f, scaler.unscale(scaler.scale(f)));
    }
  }

  static void test_scale_0()
  {
    vector<float> v({10.0, 20.0, 15.0});
    Scaler scaler = Scaler::create(v);
    Assert::equals(-1.0, scaler.scale(0.0));
    Assert::equals(0.0, scaler.scale(10.0));
    Assert::equals(0.5, scaler.scale(15.0));
    Assert::equals(1.0, scaler.scale(20.0));
    Assert::equals(2.0, scaler.scale(30.0));
  }

  static void test_scale_1()
  {
    vector<float> v({10.0, 10.0, 10.0});
    Scaler scaler = Scaler::create(v);
    Assert::equals(0.0, scaler.scale(10.0));
  }

  static void test_scale_2()
  {
    Scaler scaler(-2.0, 3.0);
    Assert::equals(0.0, scaler.scale(-2.0));
    Assert::equals(0.2, scaler.scale(-1.0));
    Assert::equals(0.4, scaler.scale(0.0));
    Assert::equals(0.5, scaler.scale(0.5));
    Assert::equals(0.6, scaler.scale(1.0));
    Assert::equals(0.8, scaler.scale(2.0));
    Assert::equals(1.0, scaler.scale(3.0));
  }

  static void test_unscale_0()
  {
    Scaler scaler(20, 30);
    Assert::equals(20.0, scaler.unscale(0));
    Assert::equals(25.0, scaler.unscale(0.5));
    Assert::equals(30.0, scaler.unscale(1.0));
  }

  static void test_unscale_1()
  {
    Scaler scaler(-1.5, 1.5);
    Assert::equals(-1.5, scaler.unscale(0));
    Assert::equals(-0.5, scaler.unscale(0.333333333333333333));
    Assert::equals(1.5, scaler.unscale(1.0));
  }
};

class NnDataGenerator
{
public:
  virtual vector<float> gen_input(int in_dim) = 0;
  virtual vector<float> gen_label(vector<float> input, int out_dim) = 0;
};

class NnMaxDataGenerator : NnDataGenerator
{
public:

  vector<float> gen_input(int in_dim)
  {
    vector<float> v;
    rep(i, in_dim) v.push_back(my_rand(-1, 1));
    return v;
  }

  vector<float> gen_label(vector<float> v, int out_dim)
  {
    vector<float> total(out_dim, 0);
    rep(i, sz(v)) total[i % sz(total)] += v[i];

    int best = 0;
    rep(i, sz(total))if(total[best] < total[i]) best = i;
    
    vector<float> res(out_dim, 0);
    res[best] = 1.0;
    return res;
  }
};

class NnOneToOneDataGenerator : NnDataGenerator
{
public:

  vector<float> gen_input(int in_dim)
  {
    vector<float> v(in_dim, 0);
    v[rand() % in_dim] = 1;
    return v;
  }

  vector<float> gen_label(vector<float> v, int out_dim)
  {
    vector<float> res(out_dim, 0);
    rep(i, sz(v))if(0 < v[i]) res[i % sz(res)] = 1;
    return res;
  }
};

class NnQcutDataGenerator : NnDataGenerator
{
public:

  vector<float> gen_input(int in_dim)
  {
    vector<float> v;
    rep(i, in_dim) v.push_back(my_rand(0, 1));
    return v;
  }

  vector<float> gen_label(vector<float> v, int out_dim)
  {
    assert(sz(v) == out_dim);

    vector<float> res(out_dim, 0);
    double total = 0;
    rep(i, sz(v)) total += v[i];
    res[(int) total] = 1;
    return res;
  }
};

class NnAlwaysLabelTwoDataGenerator : NnDataGenerator
{
public:

  vector<float> gen_input(int in_dim)
  {
    vector<float> v;
    rep(i, in_dim) v.push_back(my_rand(0, 1));
    return v;
  }

  vector<float> gen_label(vector<float> v, int out_dim)
  {
    assert(sz(v) == out_dim);

    vector<float> res(out_dim, 0);
    res[2] = 1;
    return res;
  }
};

class NnTester
{
public:
  static void test()
  {
    declare_test_name("NnTester");
    run_test(test_simple_train);
    run_test(test_serde);
    run_test(test_dqn);
    run_test(test_max_data);
    run_test(test_one_to_one_data);
    run_test(test_qcut_data);
  }

  class DqnState
  {
  public:
    int cur;
    int action;
    int next;
    bool is_last;
    bool completed;
    DqnState(int cur, int action, int next) : cur(cur), action(action), next(next), is_last(false) {}
    DqnState(int cur, int action, int next, bool completed) : cur(cur), action(action), next(next), is_last(true), completed(completed) {}
  };

  static void test_simple_train()
  {
    srand(1234);

    const int dim_inputs = 10;
    const int dim_outputs = 3;
    Nn nn({dim_inputs, 50, 50}, {dim_outputs});
    int num_epochs = 30;
    int num_iters = 512;
    int batch_size = 32;
    int signs[dim_inputs][dim_outputs];
    rep(i, dim_inputs)rep(j, dim_outputs) signs[i][j] = (rand() % 2) ? 1 : -1;

    rep(epoch, num_epochs){
      double loss = 0.0;
      int success = 0;
      rep(iter, num_iters){
        vector<vector<float>> all_inputs;
        vector<vector<vector<float>>> labels;
        rep(batch, batch_size){
          vector<float> inputs;
          rep(i, dim_inputs) inputs.push_back(random_float(-1.0, 1.0));
          all_inputs.push_back(inputs);

          vector<pair<int, int>> v;
          rep(i, dim_outputs){
            double total = 0;
            rep(j, dim_inputs) total += signs[j][i] * inputs[j];
            v.push_back(make_pair(total, i));
          }
          sort(v.begin(), v.end());

          vector<float> l(dim_outputs, 0);
          l[v[0].second] = 1;

          labels.push_back({l});

          float *f = nn.forward(inputs);
          if(0.5 < f[v[0].second]) success += 1;
        }
        loss += nn.train(all_inputs, labels, 3e-2);
      }
      double accuracy = (double) success / num_iters / batch_size;
      debug3(epoch, accuracy, loss / num_iters);
      if(epoch + 1 == num_epochs){
        assert(accuracy > 0.93);
        assert(loss / num_iters < 0.12);
      }
    }
  }

  static void test_dqn()
  {
    srand(1234);
    int ranks = 4;
    int num_epochs = 300;
    int num_iters = 64;
    int batch_size = 64;
    int buffer_size = 1 << 20;
    int max_length = 500;
    int target_update = 4;
    double learning_rate = 1e-3;

    Nn policy({ranks, 32}, vector<int>(ranks + 1, 2));
    Nn target = policy;
    vector<vector<float>> nn_inputs;
    rep(i, ranks){
      vector<float> v(ranks, 0);
      v[i] = 1.0;
      nn_inputs.push_back(v);
    }

    auto compute_reward = [&](DqnState &s){
      if(s.is_last) return s.completed ? 1.0 : 0.0;
      int best_action = 0;

      policy.forward(nn_inputs[s.next]);
      rep(i, ranks + 1)if(policy.get_prediction(best_action, 1) < policy.get_prediction(i, 1)) best_action = i;

      target.forward(nn_inputs[s.next]);
      return 0.98 * target.get_prediction(best_action, 1);
    };

    int dat_index = 0;
    DqnSampler<DqnState> sampler(buffer_size);
    rep(epoch, num_epochs){
      int total_steps = 0;
      rep(data_i, num_iters){
        {
          vector<DqnState> hist;
          for(int loop = 0, cur = 0; loop < max_length && cur < ranks; ++loop){
            policy.forward(nn_inputs[cur]);

            int t = 0;
            rep(i, ranks + 1)if(policy.get_prediction(t, 1) < policy.get_prediction(i, 1)) t = i;
            if(random_float(0, 1) < max(0.2, 1.0 - epoch / (num_epochs / 2.0))) t = rand() % (ranks + 1);

            int next;
            if(t == cur + 1 || t == cur) next = t;
            else next = max(0, cur - 1);

            hist.push_back(DqnState(cur, t, next));
            cur = next;
          }
          total_steps += sz(hist);
          hist.back().is_last = true;
          hist.back().completed = sz(hist) != max_length;

          for(DqnState &s: hist){
            policy.forward(nn_inputs[s.cur]);
            sampler.add_or_overwrite(s, abs(policy.get_prediction(s.action, 1) - compute_reward(s)));
          }
        }
        {
          vector<vector<float>> inputs;
          vector<vector<vector<float>>> rewards;
          rep(batch_i, batch_size){
            int dqn_i = sampler.get_index();
            DqnState &s = sampler.get(dqn_i);

            policy.forward(nn_inputs[s.cur]);
            float reward = compute_reward(s);
            sampler.update_weight(dqn_i, abs(policy.get_prediction(s.action, 1) - reward));

            inputs.push_back(nn_inputs[s.cur]);
            vector<vector<float>> v(ranks + 1);
            v[s.action] = {1 - reward, reward};

            rewards.push_back(v);
          }
          policy.train(inputs, rewards, learning_rate);
        }
      }
      if(epoch % 5 == 0) debug2(epoch, total_steps / num_iters);
      if(epoch + 1 == num_epochs) assert(total_steps / num_iters < 10.0);
      if(epoch % target_update == 0) target = policy;

      if(epoch % 50 == 0){
        rep(i, ranks){
          target.forward(nn_inputs[i]);
          rep(j, ranks + 1) printf("%0.6f ", target.get_prediction(j, 1));
          cout << endl;
        }
      }
    }
  }

  static void test_serde()
  {
    vector<int> fc_widths({23, 45, 67, 89});
    vector<int> head_widths({5, 6, 7});
    srand(345);
    Nn nn0(fc_widths, head_widths);
    srand(678);
    Nn nn1(fc_widths, head_widths);
    compare_nn_outputs(nn0, nn0, fc_widths, head_widths, true);
    compare_nn_outputs(nn1, nn1, fc_widths, head_widths, true);

    compare_nn_outputs(nn0, nn1, fc_widths, head_widths, false);
    nn1.import_weights(nn0.export_weights());
    compare_nn_outputs(nn0, nn1, fc_widths, head_widths, true);
  }

  static void compare_nn_outputs(Nn &nn0, Nn &nn1, vector<int> &fc_widths, vector<int> &head_widths, bool should_match)
  {
    int n = 1;
    rep(loop, n){
      vector<float> v;
      rep(i, fc_widths[0]) v.push_back((rand() % 10000 - 5000) / 1000.0);
      nn0.forward(v);
      nn1.forward(v);
      rep(i, sz(head_widths))rep(j, head_widths[i]){
        cerr << nn0.get_prediction(i, j) << ", " << nn1.get_prediction(i, j) << endl;
        assert((nn0.get_prediction(i, j) == nn1.get_prediction(i, j)) == should_match);
      }
    }
  }

  static void test_max_data()
  {
    NnMaxDataGenerator generator;
    test_training((NnDataGenerator*) &generator, {25, 50, 50}, {10}, -log(0.8));
  }

  static void test_one_to_one_data()
  {
    NnOneToOneDataGenerator generator;
    test_training((NnDataGenerator*) &generator, {25, 50, 50}, {10}, -log(0.95));
  }

  static void test_always_label_two_data()
  {
    NnAlwaysLabelTwoDataGenerator generator;
    test_training((NnDataGenerator*) &generator, {25, 50, 50}, {10}, -log(0.99));
  }

  static void test_qcut_data()
  {
    NnQcutDataGenerator generator;
    test_training((NnDataGenerator*) &generator, {10, 50, 50}, {10}, -log(0.8));
  }

  static void test_training(NnDataGenerator *generator, vector<int> fc_widths, vector<int> head_widths, double target, int num_epochs = 10)
  {
    srand(1234);
    Nn nn(fc_widths, head_widths);

    vector<double> losses;
    rep(epoch, num_epochs){
      debug(epoch);

      double total_loss = 0;
      int n = 2000;
      rep(loop, n){
        vector<float> input = generator->gen_input(fc_widths[0]);
        vector<float> label = generator->gen_label(input, head_widths[0]);
        float *pred = nn.forward(input);
        rep(i, sz(label)) total_loss += (-label[i] * log(max(pred[i], 1e-6f)));
      }
      losses.push_back(total_loss / n);
      debug(losses.back());

      // train
      rep(loop, 2000){
        int batch = 16;
        vector<vector<float>> inputs;
        vector<vector<vector<float>>> labels;
        rep(i, batch){
          inputs.push_back(generator->gen_input(fc_widths[0]));
          labels.push_back({generator->gen_label(inputs.back(), head_widths[0])});
        }
        nn.train(inputs, labels, 0.01);
      }
    }
    debug2(losses.back(), target);
    assert(losses.back() < target);
  }

};

class Base32768Tester
{
public:
  static void test()
  {
    declare_test_name("Base32768Tester");
    run_test(test_16_bit_integers);
    run_test(test_mulitple);
  }

  static void test_16_bit_integers()
  {
    int k_bits = 16;
    vector<int> original;
    rep(i, (1 << k_bits)) original.push_back(i);
    string encoded = Base32768::encode_k_bit_integer(original, k_bits);
    vector<int> decoded = Base32768::decode(encoded);

    Assert::equals((int) original.size(), (int) decoded.size());
    for (int i = 0; i < original.size(); ++i) {
      Assert::equals(original[i], decoded[i]);
    }
  }

  static void test_mulitple()
  {
    for (int seed = 1234; seed < 1234 + 100; ++seed) {
      for (int k_bits = 1; k_bits <= 16; ++k_bits) {
        srand(seed);
        int mask = (1 << k_bits) - 1;
        int length = rand() % 10000;
        vector<int> original;
        rep(i, length) original.push_back(rand() % mask);
        string encoded = Base32768::encode_k_bit_integer(original, k_bits);
        vector<int> decoded = Base32768::decode(encoded);

        Assert::equals((int) original.size(), (int) decoded.size());
        for (int i = 0; i < original.size(); ++i) {
          Assert::equals(original[i], decoded[i]);
        }
      }
    }
  }

  static void test_file_io()
  {
    int k_bits = 16;
    vector<int> original;
    rep(i, (1 << k_bits)) original.push_back(i);
    string encoded = Base32768::encode_k_bit_integer(original, k_bits);

    string filename = "test_encode_k_bit_integer.txt";
    ofstream ofs(filename);
    ofs << encoded;
    ifstream ifs(filename);
    string loaded;
    ifs >> loaded;

    vector<int> decoded = Base32768::decode(loaded);

    Assert::equals((int) original.size(), (int) decoded.size());
    for (int i = 0; i < original.size(); ++i) {
      Assert::equals(original[i], decoded[i]);
    }
  }
};

class NnIoTester
{
public:
  static void test()
  {
    declare_test_name("NnIoTester");
    run_test(test_obj_creation);
    run_test(test_io);
    run_test(test_raw_io);
  }

  static void test_raw_io()
  { 
    string path = "nn_io_test.txt";
    vector<int> fc_widths({20, 50, 50, 10});
    vector<int> head_widths({5, 9, 4});
    Nn nn0(fc_widths, head_widths);
    NnIo::write_raw(nn0, path);
    Nn nn1 = NnIo::read_raw(path);
    NnTester::compare_nn_outputs(nn0, nn1, fc_widths, head_widths, true);
  }

  static void test_obj_creation()
  {
    for(int k_bits = 16; 10 < k_bits; --k_bits){
      vector<int> fc_widths({20, 50, 50, 10});
      vector<int> head_widths({9, 4, 5});
      Nn nn0(fc_widths, head_widths);
      NnIo::Obj obj = NnIo::to_obj(nn0, k_bits);
      Nn nn1 = NnIo::from_obj(obj);
      compare_models(nn0, nn1, k_bits, fc_widths, head_widths);
    }
  }

  static void compare_models(Nn &nn0, Nn &nn1, int k_bits, vector<int> fc_widths, vector<int> head_widths)
  {
      int n = 1000;
      double total_diff = 0.0;
      rep(loop, n){
        vector<float> input;
        rep(i, fc_widths[0]) input.push_back(my_rand(-5, 5));
        float *f0 = nn0.forward(input);
        float *f1 = nn1.forward(input);
        double diff = 0;
        rep(i, head_widths[0]) diff += abs(f0[i] - f1[i]);
        total_diff += diff;
      }
      double avg_diff = total_diff / n;

      debug2(k_bits, avg_diff);
      assert(avg_diff < 0.01);
  }

  class Batch 
  {
  public:
    vector<vector<float>> inputs;
    vector<vector<vector<float>>> labels;
  };

  /**
   * Used for README.
   */
  static void test_io()
  {
    vector<Batch> data;
    double learning_rate = 0.01;
    int encode_bits = 10;
    string out_path = "nn_test_export.txt";

    vector<int> fc_widths({4, 8, 8});
    vector<int> head_widths({3});
    Nn nn(fc_widths, head_widths);

    for (Batch batch: data) {
      nn.train(batch.inputs, batch.labels, learning_rate);
    }

    NnIo::Obj io_obj = NnIo::to_obj(nn, encode_bits);
    io_obj.write(out_path);

    // Copy-and-pasted from ${out_path}.
    auto loaded = nn_eda::NnIo::from_obj(nn_eda::NnIo::Obj(
      std::vector<int>({4, 8, 8, }),
      std::vector<int>({3, }),
      "㈪㊯㈠꿻旼狀髷劋蚩뚘诒㲚礘䷧귱㛬綗㑡墲䧶匀俽㯰㹌꾼㸻㷘匱鷉忏脥貑聽沗忹战䕁俘楩铽届뜕皛氥렋鉆玍轐培襙堞溱梄㼑綡漇䍋奐䂊䜚廱脦㶈蕖䃸䔃袅擦䳑麛蹆囫㚬놽봓頙粀㈶检莥頡閘齸珼氺珼氺珼氺珼氺珼氺珼氺珼氺珼樬",
      -0.71865761280059814453,
      0.67901515960693359375,
      10
    ));


    compare_models(nn, loaded, encode_bits, fc_widths, head_widths);
  }
};

int main()
{
  DqnSamplerTester::test();
  ScalerTester::test();
  Base32768Tester::test();
  NnIoTester::test();
  NnTester::test();

  // Assert::equals(2.0, 1.0, 0.5);
}


