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
    run_test(test_serde);
    run_test(test_policy_gradient);
    run_test(test_max_data);
    run_test(test_one_to_one_data);
    run_test(test_qcut_data);
  }

  static void test_policy_gradient()
  {
    srand(1234);
    int num_epochs = 100;
    int num_data = 128;
    int num_trains = 64;
    int batch_size = 32;
    int max_length = 1000;
    double learning_rate = 1e-2;

    Nn nn(vector<int>({5, 30, 30, 6}));
    rep(epoch, num_epochs){
      vector<vector<pair<int, int>>> dat;
      int total_steps = 0;
      rep(data_i, num_data){
        vector<pair<int, int>> hist;
        for(int loop = 0, cur = 0; loop < max_length && cur < 5; ++loop){
          vector<float> v(5, 0);
          v[cur] = 1.0;
          float *scores = nn.forward(v);
          float rem = random_float(0, 0.99999);
          int t = 0;
          while(t < 6){
            rem -= scores[t];
            if(rem <= 0) break;
            ++t;
          }
          if(0 <= rem) debug(rem);
          if(!(rem <= 0)) debug(rem);
          assert(rem <= 0);
          if(random_float(0, 1) < 0.05) t = rand() % 6;

          hist.push_back(make_pair(cur, t));
          if(t <= cur + 1) cur = t;
          else cur = max(0, cur - 1);
        }
        total_steps += sz(hist);
        dat.push_back(hist);
      }
      debug(total_steps / num_data);
      if(epoch + 1 == num_epochs) assert(total_steps / num_data < 10.0);

      rep(train_i, num_trains){
        vector<vector<float>> inputs;
        vector<float> rewards;
        vector<int> action_indexes;
        rep(batch_i, batch_size){
          vector<pair<int, int>> &v = dat[rand() % sz(dat)];
          int t = rand() % sz(v);
          double reward = (sz(v) == max_length) ? -1 : 1;

          vector<float> input(5, 0);
          input[v[t].first] = 1.0;
          inputs.push_back(input);
          rewards.push_back(reward * pow(0.99, sz(v) - t));
          action_indexes.push_back(v[t].second);
        }
        nn.train_policy_gradient(inputs, rewards, action_indexes, learning_rate);
      }
    }
  }

  static void test_serde()
  {
    vector<int> widths({23, 45, 67, 89});
    srand(345);
    Nn nn0(widths);
    srand(678);
    Nn nn1(widths);
    compare_nn_outputs(nn0, nn0, widths, true);
    compare_nn_outputs(nn1, nn1, widths, true);

    compare_nn_outputs(nn0, nn1, widths, false);
    nn1.import_weights(nn0.export_weights());
    compare_nn_outputs(nn0, nn1, widths, true);
  }

  static void compare_nn_outputs(Nn &nn0, Nn &nn1, vector<int> &widths, bool should_match)
  {
    int n = 100;
    rep(loop, n){
      vector<float> v;
      rep(i, widths[0]) v.push_back((rand() % 10000 - 5000) / 1000.0);
      float *res0 = nn0.forward(v);
      float *res1 = nn1.forward(v);
      rep(i, widths.back()) assert((res0[i] == res1[i]) == should_match);
    }
  }

  static void test_max_data()
  {
    NnMaxDataGenerator generator;
    test_training((NnDataGenerator*) &generator, vector<int>({25, 50, 50, 10}), -log(0.8));
  }

  static void test_one_to_one_data()
  {
    NnOneToOneDataGenerator generator;
    test_training((NnDataGenerator*) &generator, vector<int>({25, 50, 50, 10}), -log(0.95));
  }

  static void test_always_label_two_data()
  {
    NnAlwaysLabelTwoDataGenerator generator;
    test_training((NnDataGenerator*) &generator, vector<int>({25, 50, 50, 10}), -log(0.99));
  }

  static void test_qcut_data()
  {
    NnQcutDataGenerator generator;
    test_training((NnDataGenerator*) &generator, vector<int>({10, 50, 50, 10}), -log(0.8));
  }

  static void test_training(NnDataGenerator *generator, vector<int> widths, double target, int num_epochs = 10)
  {
    srand(1234);
    Nn nn(widths);

    vector<double> losses;
    rep(epoch, num_epochs){
      debug(epoch);

      double total_loss = 0;
      int n = 2000;
      rep(loop, n){
        vector<float> input = generator->gen_input(widths[0]);
        vector<float> label = generator->gen_label(input, widths.back());
        float *pred = nn.forward(input);
        rep(i, sz(label)) total_loss += (-label[i] * log(max(pred[i], 1e-6f)));
      }
      losses.push_back(total_loss / n);
      debug(losses.back());

      // train
      rep(loop, 2000){
        int batch = 16;
        vector<vector<float>> inputs;
        vector<vector<float>> labels;
        rep(i, batch){
          inputs.push_back(generator->gen_input(widths[0]));
          labels.push_back(generator->gen_label(inputs.back(), widths.back()));
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
    vector<int> widths({20, 50, 50, 10});
    Nn nn0(widths);
    NnIo::write_raw(nn0, path);
    Nn nn1 = NnIo::read_raw(path);
    NnTester::compare_nn_outputs(nn0, nn1, widths, true);
  }

  static void test_obj_creation()
  {
    for(int k_bits = 16; 10 < k_bits; --k_bits){
      vector<int> widths({20, 50, 50, 10});
      Nn nn0(widths);
      NnIo::Obj obj = NnIo::to_obj(nn0, k_bits);
      Nn nn1 = NnIo::from_obj(obj);
      compare_models(nn0, nn1, k_bits, widths);
    }
  }

  static void compare_models(Nn &nn0, Nn &nn1, int k_bits, vector<int> widths)
  {
      int n = 1000;
      double total_diff = 0.0;
      rep(loop, n){
        vector<float> input;
        rep(i, widths[0]) input.push_back(my_rand(-5, 5));
        float *f0 = nn0.forward(input);
        float *f1 = nn1.forward(input);
        double diff = 0;
        rep(i, widths.back()) diff += abs(f0[i] - f1[i]);
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
    vector<vector<float>> labels;
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

    vector<int> widths({4, 8, 8, 3});
    nn_eda::Nn nn(widths);

    for (Batch batch: data) {
      nn.train(batch.inputs, batch.labels, learning_rate);
    }

    nn_eda::NnIo::Obj io_obj = nn_eda::NnIo::to_obj(nn, encode_bits);
    io_obj.write(out_path);

    // Copy-and-pasted from ${out_path}.
    nn_eda::Nn loaded = nn_eda::NnIo::from_obj(nn_eda::NnIo::Obj(
      std::vector<int>({4, 8, 8, 3, }),
      "㈪㊯㈠傷話啹년升뫆볮䄝麜맀亴蘫됧郄珩漲逫㥈籋非鸳䢠鶛檯膢㖗踁珤焝㦚抢缗憠們댤濄叆侣듬垖婕鑉鼲淞끵瓚髜寅奉册俗䲐戠卞鍿忓两띨燄烆鎰憤廨隮埉楻붶罦巌㌷仑粆齜茉㰉薇湺蒾跪柣琜瀻琜瀻琜瀻琜瀻琜瀻琜瀻琜瀻琜測",
      -0.73765462636947631836,
      0.69456773996353149414,
      10
    ));

    compare_models(nn, loaded, encode_bits, widths);
  }
};

int main()
{
  ScalerTester::test();
  Base32768Tester::test();
  NnIoTester::test();
  NnTester::test();

  // Assert::equals(2.0, 1.0, 0.5);
}


