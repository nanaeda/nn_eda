#pragma GCC target("avx2")

#include <bits/stdc++.h>
#include <x86intrin.h>


namespace nn_eda
{
  class Util
  {
  public:
    static int get_bit_mask(int i)
    {
      return (1 << i) - 1;
    }
  };

  class Nn
  {
  public:

    Nn(std::vector<int> widths, int seed = 1234)
    {
      assert(2 <= widths.size());

      srand(seed);

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
    float* forward(std::vector<float> &v)
    {
      assert(widths[0] == v.size());

      for (int i = 0; i < widths[0]; ++i) outs[0][i] = v[i];

      for (int d = 0; (d + 1) < widths.size(); ++d) {
        memcpy(outs[d + 1], bs[d + 1], sizeof(float) * widths[d + 1]);

        for (int i = 0; i < widths[d]; ++i) {
          float &a = outs[d][i];
          if (0 < d) a = std::max(a, 0.0f);
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
          softmax_max = std::max(softmax_max, outs[widths.size() - 1][i]);
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

    std::vector<float> export_weights()
    {
      std::vector<float> res;
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

    void import_weights(std::vector<float> v)
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

    void train(std::vector<float> input, std::vector<float> label, double learning_rate)
    {
      train(std::vector<std::vector<float>>({input}), std::vector<std::vector<float>>({label}), learning_rate);
    }

    void train(std::vector<std::vector<float>> inputs, std::vector<std::vector<float>> labels, double learning_rate)
    {
      for (int d = 0; d < widths.size(); ++d) {
        memset(grad_bs[d], 0, sizeof(float) * widths[d]);
        memset(grad_os[d], 0, sizeof(float) * widths[d]);
      }
      for (int d = 0; d + 1 < widths.size(); ++d) {
        for (int i = 0; i < widths[d]; ++i) {
          memset(grad_ws[d][i], 0, sizeof(float) * widths[d + 1]);
        }
      }

      for (int input_index = 0; input_index < inputs.size(); ++input_index) {
        std::vector<float> input = inputs[input_index];
        std::vector<float> label = labels[input_index];

        // forward
        forward(input);

        // grad of outputs.
        for (int i = 0; i < widths.back(); ++i) {
          grad_os[widths.size() - 1][i] = softmax_out[i] - label[i];
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

      double norm = 1.0 / inputs.size();
      double momentum = 0.9;
      double decay = 1e-4;
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
    }

    ~Nn()
    {
      free_memory();
    }

    std::vector<int> widths;

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

    static float** create_zero_f2(std::vector<int> &widths)
    {
      float **a = new float*[widths.size()];
      for (int i = 0; i < widths.size(); ++i) {
        int width8 = increment8(widths[i]);
        a[i] = new float[width8];
        memset(a[i], 0, sizeof(float) * width8);
      }
      return a;
    }

    static float*** create_zero_f3(std::vector<int> &widths)
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

    static Scaler create(std::vector<float> &v) 
    {
      float mini = v[0];
      float maxi = v[0];
      for (float f: v) {
        mini = std::min(mini, f);
        maxi = std::max(maxi, f);
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
    static std::string encode_k_bit_integer(std::vector<int> &v, int k_bits)
    {
      assert(k_bits <= 16); // to avoid overflow.

      int mask_k = Util::get_bit_mask(k_bits);
      std::vector<int> char16s;

      // k_bits
      char16s.push_back(k_bits);

      // length.
      char16s.push_back(v.size() & MASK_15);
      char16s.push_back(v.size() >> 15);
      
      // contents
      for (int i: convert_base(v, k_bits, 15)) char16s.push_back(i);

      std::u16string u16s = to_u16string(char16s);

      std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> converter;
      return converter.to_bytes(u16s);
    }

    static std::vector<int> decode(std::string &u8s)
    {
      std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> converter;
      std::u16string u16s = converter.from_bytes(u8s);

      // k_bits
      int k_bits = c2i(u16s[0]);
      assert(k_bits <= 16);

      // length
      int length = c2i(u16s[1]) | (c2i(u16s[2]) << 15);
      
      // contents
      std::vector<int> v;
      for (int i = 3; i < u16s.size(); ++i) v.push_back(c2i(u16s[i]));

      std::vector<int> res = convert_base(v, 15, k_bits);
      while (length < res.size()) res.pop_back();

      return res;
    }

  private:

    static constexpr int MASK_15 = (1 << 15) - 1;

    static std::vector<int> convert_base(std::vector<int> &v, int curr_base_bits, int next_base_bits) 
    {
      std::vector<int> res;

      int curr_mask = Util::get_bit_mask(curr_base_bits);
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
        std::cout << "Found a value out of the range. " << i << std::endl;
        return mini;
      }
      if (maxi < i) {
        std::cout << "Found a value out of the range. " << i << std::endl;
        return maxi;
      }
      return i;
    }

    static std::u16string to_u16string(std::vector<int> &v)
    {
      char16_t *a = new char16_t[v.size() + 1];

      for (int i = 0; i < v.size(); ++i) a[i] = i2c(v[i]);

      a[v.size()] = 0; // terminator.

      std::u16string res(a);

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

      std::cout << "something went wrong... : " << t << std::endl;
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

      std::cout << "something went wrong... : " << t << std::endl;
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
        std::vector<int> widths,
        std::string serialized_weights,
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

      std::vector<int> widths;
      std::string serialized_weights;
      float weight_mini;
      float weight_maxi;
      int k_bits;

      void write(std::string path)
      {
        std::ofstream ofs(path);
        ofs << std::setprecision(20);
        ofs << "nn_eda::NnIo::from_obj(nn_eda::NnIo::Obj(" << std::endl;
        ofs << "std::vector<int>({";
        for (int width: widths) ofs << width << ", ";
        ofs << "})," << std::endl;
        ofs << "\"" << serialized_weights << "\"," << std::endl;
        ofs << weight_mini << "," << std::endl;
        ofs << weight_maxi << "," << std::endl;
        ofs << k_bits << "));" << std::endl;
      }
    };

    static Nn from_obj(Obj obj) 
    {
      std::vector<int> int_weights = Base32768::decode(obj.serialized_weights);

      Scaler scaler(obj.weight_mini, obj.weight_maxi);
      std::vector<float> weights;
      for(int w: int_weights) weights.push_back(scaler.unscale((float) w / Util::get_bit_mask(obj.k_bits)));

      Nn nn(obj.widths);
      nn.import_weights(weights);
      return nn;
    }

    static Obj to_obj(Nn &nn, int k_bits) 
    {
      std::vector<float> weights = nn.export_weights();
      Scaler scaler = Scaler::create(weights);
      std::vector<int> int_weights;
      for(float w: weights) int_weights.push_back((int) (scaler.scale(w) * Util::get_bit_mask(k_bits) + 0.5));

      return Obj(
        nn.widths,
        Base32768::encode_k_bit_integer(int_weights, k_bits),
        scaler.mini,
        scaler.maxi,
        k_bits
      );
    }


    static void write_raw(Nn &nn, std::string path)
    {
      std::ofstream ofs(path, std::ios::binary);
      int num_widths = nn.widths.size();
      ofs.write(reinterpret_cast<const char*>(&num_widths), sizeof(int));
      for (int width: nn.widths) ofs.write(reinterpret_cast<const char*>(&width), sizeof(int));

      std::vector<float> weights = nn.export_weights();
      int num_weights = weights.size();
      ofs.write(reinterpret_cast<const char*>(&num_weights), sizeof(int));
      for (float w: weights) ofs.write(reinterpret_cast<const char*>(&w), sizeof(float));
    }

    static Nn read_raw(std::string path)
    {
      std::ifstream ifs(path, std::ios::binary);
      int num_widths;
      ifs.read(reinterpret_cast<char*>(&num_widths), sizeof(int));
      std::vector<int> widths(num_widths);
      for(int i = 0; i < num_widths; ++i) ifs.read(reinterpret_cast<char*>(&widths[i]), sizeof(int));

      Nn nn(widths);

      int num_weights;
      ifs.read(reinterpret_cast<char*>(&num_weights), sizeof(int));
      std::vector<float> weights(num_weights);
      for(int i = 0; i < num_weights; ++i) ifs.read(reinterpret_cast<char*>(&weights[i]), sizeof(float));
      nn.import_weights(weights);
      return nn;
    }
  };
}
