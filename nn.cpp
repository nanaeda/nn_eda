#pragma GCC target("avx2")

#include <bits/stdc++.h>
#include <x86intrin.h>


namespace nn_eda
{
  using namespace std;
  typedef vector<int> vi;
  typedef vector<float> vf;
  typedef vector<vf> vvf;

  class Nn
  {
  public:
    Nn(vi widths)
    {
      assert(2 <= widths.size());

      this->widths = widths;
      this->ws = create_zero_f3(widths);
      this->bs = create_zero_f2(widths);
      this->outs = create_zero_f2(widths);
      this->softmax_out = new float[widths.back()];

      grad_ws = create_zero_f3(widths);
      grad_bs = create_zero_f2(widths);
      grad_os = create_zero_f2(widths);
      grad_momentum_ws = create_zero_f3(widths);
      grad_momentum_bs = create_zero_f2(widths);

      for (int d = 0; d + 1 < widths.size(); ++d) {
        for (int i = 0; i < widths[d]; ++i) {
          for (int j = 0; j < widths[d + 1]; ++j) {
            ws[d][i][j] = random_float(-1, 1.0) * sqrtf(6.0f / (float) (widths[d] + widths[d + 1]));
          }
        }
      }
    }

    /**
     * Note that this is used by the "train" method when you change this.
     */
    float* forward(vf &v)
    {
      assert(widths[0] == v.size());

      for (int i = 0; i < widths[0]; ++i) outs[0][i] = v[i];

      for (int d = 0; (d + 1) < widths.size(); ++d) {
        memcpy(outs[d + 1], bs[d + 1], sizeof(float) * widths[d + 1]);

        for (int i = 0; i < widths[d]; ++i) {
          float &a = outs[d][i];
          if (0 < d) a = max(a, 0.0f);
          if (a == 0) continue;

          float *out_ptr = outs[d + 1];
          float *out_ptr_end = outs[d + 1] + widths[d + 1];
          float *w_ptr = ws[d][i];

          __m256 m256_a = _mm256_set1_ps(a);
          while(out_ptr < out_ptr_end){
            __m256 out = _mm256_loadu_ps(out_ptr);
            __m256 w = _mm256_loadu_ps(w_ptr);
            out = _mm256_add_ps(_mm256_mul_ps(m256_a, w), out);
            _mm256_storeu_ps(out_ptr, out);

            out_ptr += 8;
            w_ptr += 8;
          }
        }
      }

      // softmax
      {
        float softmax_max = outs[widths.size() - 1][0];
        for (int i = 0; i < widths.back(); ++i) {
          softmax_max = max(softmax_max, outs[widths.size() - 1][i]);
        }
        float total = 0.0;
        for (int i = 0; i < widths.back(); ++i) {
          softmax_out[i] = expf(outs[widths.size() - 1][i] - softmax_max);
          total += softmax_out[i];
        }
        for (int i = 0; i < widths.back(); ++i) {
          softmax_out[i] /= total;
        }
      }
      return softmax_out;
    }

    vf export_weights()
    {
      vf res;
      for (int d = 0; (d + 1) < widths.size(); ++d) {
        for (int i = 0; i < widths[d]; ++i) {
          for (int j = 0; j < widths[d + 1]; ++j) {
            res.push_back(ws[d][i][j]);
          }
        }
      }
      for (int d = 0; d < widths.size(); ++d) {
        for (int i = 0; i < widths[d]; ++i) {
          res.push_back(bs[d][i]);
        }
      }
      return res;
    }

    void import_weights(vf v)
    {
      int index = 0;
      for (int d = 0; (d + 1) < widths.size(); ++d) {
        for (int i = 0; i < widths[d]; ++i) {
          for (int j = 0; j < widths[d + 1]; ++j) {
            ws[d][i][j] = v[index++];
          }
        }
      }
      for (int d = 0; d < widths.size(); ++d) {
        for (int i = 0; i < widths[d]; ++i) {
          bs[d][i] += v[index++];
        }
      }
      assert(v.size() == index);
    }


    double train(vvf inputs, vvf labels, double learning_rate)
    {
      return train(inputs, labels, vf(), vi(), learning_rate, false);
    }

    void train_policy_gradient(vvf inputs, vf rewards, vi action_indexes, double learning_rate)
    {
      train(inputs, vvf(), rewards, action_indexes, learning_rate, true);
    }

    double train(vvf inputs, 
                 vvf labels, 
                 vf rewards,
                 vi action_indexes,
                 double learning_rate, 
                 bool is_policy_gradient)
    {
      const double norm = 1.0 / inputs.size();
      const double momentum = 0.9;
      const double decay = 1e-4;
      const double logloss_eps = 1e-6;

      for (int d = 0; d < widths.size(); ++d) {
        memset(grad_bs[d], 0, sizeof(float) * widths[d]);
        memset(grad_os[d], 0, sizeof(float) * widths[d]);
      }
      for (int d = 0; d + 1 < widths.size(); ++d) {
        for (int i = 0; i < widths[d]; ++i) {
          memset(grad_ws[d][i], 0, sizeof(float) * widths[d + 1]);
        }
      }

      double total_loss = 0.0;
      for (int input_index = 0; input_index < inputs.size(); ++input_index) {
        vf input = inputs[input_index];

        // forward
        float *forward_out = forward(input);

        // grad of outputs.
        if(is_policy_gradient){
          for (int i = 0; i < widths.back(); ++i) {
            grad_os[widths.size() - 1][i] = rewards[input_index] * (softmax_out[i] - (i == action_indexes[input_index] ? 1 : 0));
          }
        }else{
          vf label = labels[input_index];
          for (int i = 0; i < widths.back(); ++i) {
            total_loss += label[i] * -log(max((float) logloss_eps, forward_out[i]));
          }
          for (int i = 0; i < widths.back(); ++i) {
            grad_os[widths.size() - 1][i] = softmax_out[i] - label[i];
          }
        }
        for (int d = widths.size() - 2; 0 <= d; --d) {
          for (int i = 0; i < widths[d]; ++i) {
            grad_os[d][i] = 0;
            bool relu_applied = (d + 1) < (widths.size() - 1);
            for (int j = 0; j < widths[d + 1]; ++j) {
              if (relu_applied && (outs[d + 1][j] == 0)) continue;
              grad_os[d][i] += grad_os[d + 1][j] * ws[d][i][j];
            }
          }
        }

        // grad of parameters
        for (int d = 1; d < widths.size(); ++d){
          for (int i = 0; i < widths[d]; ++i){
            if ((d + 1 < widths.size()) && outs[d][i] <= 0){
              assert(outs[d][i] == 0);
              continue;
            }
            grad_bs[d][i] += grad_os[d][i];
            for (int j = 0; j < widths[d - 1]; ++j) {
              grad_ws[d - 1][j][i] += grad_os[d][i] * outs[d - 1][j];
            }
          }
        }
      }

      for (int d = 0; d < widths.size(); ++d) {
        for (int i = 0; i < widths[d]; ++i) {
          float &g = grad_momentum_bs[d][i];
          g = (momentum * g + (grad_bs[d][i] * norm) + decay * bs[d][i]);
          bs[d][i] -= learning_rate * g;
        }
      }
      for (int d = 0; d + 1 < widths.size(); ++d) {
        for (int i = 0; i < widths[d]; ++i) {
          for (int j = 0; j < widths[d + 1]; ++j) {
            float &g = grad_momentum_ws[d][i][j];
            g = (momentum * g + (grad_ws[d][i][j] * norm) + decay * ws[d][i][j]);
            ws[d][i][j] -= learning_rate * g;
          }
        }
      }

      return total_loss / inputs.size();
    }

    ~Nn()
    {
      free_memory();
    }

    vi widths;

  private:

    float ***ws;
    float **bs;
    float **outs;
    float *softmax_out;

    float ***grad_ws;
    float **grad_bs;
    float **grad_os;
    float ***grad_momentum_ws;
    float **grad_momentum_bs;

    static float** create_zero_f2(vi &widths)
    {
      float **a = new float*[widths.size()];
      for (int i = 0; i < widths.size(); ++i) {
        int width8 = increment8(widths[i]);
        a[i] = new float[width8];
        memset(a[i], 0, sizeof(float) * width8);
      }
      return a;
    }

    static float*** create_zero_f3(vi &widths)
    {
      float ***ws = new float**[widths.size() - 1];
      for (int i = 0; (i + 1) < widths.size(); ++i) {
        ws[i] = new float*[widths[i]];
        for (int j = 0; j < widths[i]; ++j) {
          int width8 = increment8(widths[i + 1]);
          ws[i][j] = new float[width8];
          memset(ws[i][j], 0, sizeof(float) * width8);
        }
      }
      return ws;
    }

    void free_f3(float ***a)
    {
      for (int i = 0; i + 1 < widths.size(); ++i) {
        for (int j = 0; j < widths[i]; ++j) {
          delete [] a[i][j];
        }
        delete [] a[i];
      }
      delete [] a;
    }

    void free_f2(float **a)
    {
      for (int i = 0; i < widths.size(); ++i) {
        delete [] a[i];
      }
      delete [] a;
    }

    static int increment8(int i)
    {
      while(i % 8) ++i;
      return i;
    }

    static float random_float(float mini, float maxi)
    {
      int mask = (1 << 25) - 1;
      float f = ((float) (rand() & mask)) / mask;
      return mini + (maxi - mini) * f;
    }

    void free_memory()
    {
      this->widths = widths;

      free_f3(this->ws);
      free_f2(this->bs);
      free_f2(this->outs);
      
      free_f3(grad_ws);
      free_f2(grad_bs);
      free_f2(grad_os);
      free_f3(grad_momentum_ws);
      free_f2(grad_momentum_bs);

      delete [] softmax_out;
    }

  };

  class Scaler
  {
  public:
    Scaler(float mini, float maxi)
    {
      assert(mini <= maxi);
      this->mini = mini;
      this->maxi = (mini == maxi) ? (mini + 1e-6) : maxi;
    }

    float scale(float f) 
    {
      return (f - mini) / (maxi - mini);
    }

    float unscale(float f)
    {
      assert(0 <= f && f <= 1.0);
      return (maxi - mini) * f + mini;
    }

    static Scaler create(vf &v) 
    {
      float mini = v[0];
      float maxi = v[0];
      for (float f: v) {
        mini = min(mini, f);
        maxi = max(maxi, f);
      }
      return Scaler(mini, maxi);
    }

    float mini;
    float maxi;
  };

  // Couldn't find how to put this in Base32768...
  // https://bowwowforeach.hatenablog.com/entry/2022/07/05/195417
  constexpr int CHAR_RANGES[3][2] = {
    {0x3220, 0x4DB4},
    {0x4DC0, 0x9FEE},
    {0xAC00, 0xD7A2},
  };

  constexpr int CHAR_RANGE_LENGTHS[3] = {
    CHAR_RANGES[0][1] - CHAR_RANGES[0][0],
    CHAR_RANGES[1][1] - CHAR_RANGES[1][0],
    CHAR_RANGES[2][1] - CHAR_RANGES[2][0],
  };

  class Base32768 
  {
  public:
    /**
     * All elements in "v" must be in the range of [0, 1 << k_bits).
     */
    static string encode_k_bit_integer(vi &v, int k_bits)
    {
      assert(k_bits <= 16); // to avoid overflow.

      int mask_k = get_bit_mask(k_bits);
      vi char16s;

      // k_bits
      char16s.push_back(k_bits);

      // length.
      char16s.push_back(v.size() & MASK_15);
      char16s.push_back(v.size() >> 15);
      
      // contents
      for (int i: convert_base(v, k_bits, 15)) char16s.push_back(i);

      u16string u16s = to_u16string(char16s);

      wstring_convert<codecvt_utf8_utf16<char16_t>, char16_t> converter;
      return converter.to_bytes(u16s);
    }

    static vi decode(string &u8s)
    {
      wstring_convert<codecvt_utf8_utf16<char16_t>, char16_t> converter;
      u16string u16s = converter.from_bytes(u8s);

      // k_bits
      int k_bits = c2i(u16s[0]);
      assert(k_bits <= 16);

      // length
      int length = c2i(u16s[1]) | (c2i(u16s[2]) << 15);
      
      // contents
      vi v;
      for (int i = 3; i < u16s.size(); ++i) v.push_back(c2i(u16s[i]));

      vi res = convert_base(v, 15, k_bits);
      while (length < res.size()) res.pop_back();

      return res;
    }

  private:

    static constexpr int MASK_15 = (1 << 15) - 1;

    static int get_bit_mask(int i)
    {
      return (1 << i) - 1;
    }

    static vi convert_base(vi &v, int curr_base_bits, int next_base_bits) 
    {
      vi res;

      int curr_mask = get_bit_mask(curr_base_bits);
      int cur = 0;
      int num_bits = 0;
      for (int i: v) {
        cur = (cur << curr_base_bits) | fit_int(i, 0, curr_mask);
        num_bits += curr_base_bits;
        while (next_base_bits <= num_bits) {
          int shift = num_bits - next_base_bits;
          int t = cur >> shift;
          res.push_back(t);
          num_bits -= next_base_bits;
          cur ^= t << shift;
        }
      }
      {
        // 0-filling.
        int shift = next_base_bits - num_bits;
        int t = cur << shift;
        res.push_back(t);
      }
      return res;
    }

    static int fit_int(int i, int mini, int maxi) 
    {
      if (i < mini) {
        cout << "Found a value out of the range. " << i << endl;
        return mini;
      }
      if (maxi < i) {
        cout << "Found a value out of the range. " << i << endl;
        return maxi;
      }
      return i;
    }

    static u16string to_u16string(vi &v)
    {
      char16_t *a = new char16_t[v.size() + 1];

      for (int i = 0; i < v.size(); ++i) a[i] = i2c(v[i]);

      a[v.size()] = 0; // terminator.

      u16string res(a);

      delete [] a;

      return res;
    }

    static char16_t i2c(int t)
    {
      assert(0 <= t && t <= MASK_15);

      for (int i = 0; i < 3; ++i) {
        if (t < CHAR_RANGE_LENGTHS[i]) return (char16_t) (t + CHAR_RANGES[i][0]);
        t -= CHAR_RANGE_LENGTHS[i];
      }

      cout << "something went wrong... : " << t << endl;
      assert(false);
      exit(0);
    }

    static int c2i(char16_t t)
    {
      int res = 0;
      for (int i = 0; i < 3; ++i) {
        if (CHAR_RANGES[i][0] <= t && t <= CHAR_RANGES[i][1]) return res + (t - CHAR_RANGES[i][0]);
        res += CHAR_RANGE_LENGTHS[i];
      }

      cout << "something went wrong... : " << t << endl;
      assert(false);
      exit(0);
    }
  };

  class NnIo
  {
  public:
    class Obj
    {
    public:

      Obj(
        vi widths,
        string serialized_weights,
        float weight_mini,
        float weight_maxi,
        int k_bits
      ) {
        this->widths = widths;
        this->serialized_weights = serialized_weights;
        this->weight_mini = weight_mini;
        this->weight_maxi = weight_maxi;
        this->k_bits = k_bits;
      }

      vi widths;
      string serialized_weights;
      float weight_mini;
      float weight_maxi;
      int k_bits;

      void write(string path)
      {
        ofstream ofs(path);
        ofs << setprecision(20);
        ofs << "nn_eda::NnIo::from_obj(nn_eda::NnIo::Obj(" << endl;
        ofs << "vi({";
        for (int width: widths) ofs << width << ", ";
        ofs << "})," << endl;
        ofs << "\"" << serialized_weights << "\"," << endl;
        ofs << weight_mini << "," << endl;
        ofs << weight_maxi << "," << endl;
        ofs << k_bits << "));" << endl;
      }
    };

    static Nn from_obj(Obj obj) 
    {
      vi int_weights = Base32768::decode(obj.serialized_weights);

      Scaler scaler(obj.weight_mini, obj.weight_maxi);
      vf weights;
      for(int w: int_weights) weights.push_back(scaler.unscale((float) w / get_bit_mask(obj.k_bits)));

      Nn nn(obj.widths);
      nn.import_weights(weights);
      return nn;
    }

    static Obj to_obj(Nn &nn, int k_bits) 
    {
      vf weights = nn.export_weights();
      Scaler scaler = Scaler::create(weights);
      vi int_weights;
      for(float w: weights) int_weights.push_back((int) (scaler.scale(w) * get_bit_mask(k_bits) + 0.5));

      return Obj(
        nn.widths,
        Base32768::encode_k_bit_integer(int_weights, k_bits),
        scaler.mini,
        scaler.maxi,
        k_bits
      );
    }


    static void write_raw(Nn &nn, string path)
    {
      ofstream ofs(path, ios::binary);
      int num_widths = nn.widths.size();
      ofs.write(reinterpret_cast<const char*>(&num_widths), sizeof(int));
      for (int width: nn.widths) ofs.write(reinterpret_cast<const char*>(&width), sizeof(int));

      vf weights = nn.export_weights();
      int num_weights = weights.size();
      ofs.write(reinterpret_cast<const char*>(&num_weights), sizeof(int));
      for (float w: weights) ofs.write(reinterpret_cast<const char*>(&w), sizeof(float));
    }

    static Nn read_raw(string path)
    {
      ifstream ifs(path, ios::binary);
      int num_widths;
      ifs.read(reinterpret_cast<char*>(&num_widths), sizeof(int));
      vi widths(num_widths);
      for(int i = 0; i < num_widths; ++i) ifs.read(reinterpret_cast<char*>(&widths[i]), sizeof(int));

      Nn nn(widths);

      int num_weights;
      ifs.read(reinterpret_cast<char*>(&num_weights), sizeof(int));
      vf weights(num_weights);
      for(int i = 0; i < num_weights; ++i) ifs.read(reinterpret_cast<char*>(&weights[i]), sizeof(float));
      nn.import_weights(weights);
      return nn;
    }

  private:
    static int get_bit_mask(int i)
    {
      return (1 << i) - 1;
    }

  };
}
