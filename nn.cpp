#pragma GCC target("avx2")

#include <bits/stdc++.h>
#include <x86intrin.h>


namespace nn_eda
{
  using namespace std;
  typedef vector<int> vi;
  typedef vector<float> vf;
  typedef vector<vf> vvf;
  #pragma push_macro("FOR")
  #define FOR(i, n) for(int i = 0; i < ((int) (n)); ++i)

  class Nn
  {
  public:
    Nn(vi ls)
    {
      assert(2 <= ls.size());

      this->ls = ls;
      prepare_arrays(ls);

      FOR(d, ls.size() - 1)FOR(i, ls[d])FOR(j, ls[d + 1]){
        int mask = (1 << 25) - 1;
        float f = ((float) (rand() & mask)) / mask;
        ws[offsets_3[d][i] + j] = (2 * f - 1) * sqrtf(6.0f / (ls[d] + ls[d + 1]));
      }
    }

    void prepare_arrays(vector<int> &ls)
    {
      {
        offsets_2.push_back(0);
        for(int l: ls) offsets_2.push_back(offsets_2.back() + l);
      }
      {
        int index = 0;
        FOR(i, ls.size() - 1){
          offsets_3.push_back(vector<int>({index}));
          FOR(j, ls[i]){
            index += ls[i + 1];
            offsets_3.back().push_back(index);
          }
        }
      }

      ws = new float[offsets_3.back().back()];
      bs = new float[offsets_2.back()];
      outs = new float[offsets_2.back()];
      last_out = new float[ls.back()];
      memset(ws, 0, sizeof(float) * offsets_3.back().back());
      memset(bs, 0, sizeof(float) * offsets_2.back());
      memset(outs, 0, sizeof(float) * offsets_2.back());
      memset(last_out, 0, sizeof(float) * ls.back());

      grad_ws = new float[offsets_3.back().back()];
      grad_bs = new float[offsets_2.back()];
      grad_os = new float[offsets_2.back()];
      momentum_ws = new float[offsets_3.back().back()];
      momentum_bs = new float[offsets_2.back()];
      memset(grad_ws, 0, sizeof(float) * offsets_3.back().back());
      memset(grad_bs, 0, sizeof(float) * offsets_2.back());
      memset(grad_os, 0, sizeof(float) * offsets_2.back());
      memset(momentum_ws, 0, sizeof(float) * offsets_3.back().back());
      memset(momentum_bs, 0, sizeof(float) * offsets_2.back());
    }

    Nn(const Nn &nn)
    {
      ls = nn.ls;
      prepare_arrays(ls);
      import_weights(nn.export_weights());
    }

    Nn& operator=(const Nn &nn)
    {
      free_memory();

      ls = nn.ls;
      prepare_arrays(ls);
      import_weights(nn.export_weights());
      return *this;
    }

    float* forward_sigmoid(vf &v)
    { 
      return forward(v, true);
    }

    float* forward_softmax(vf &v)
    {
      return forward(v, false);
    }

    vf export_weights() const
    {
      vf res;
      FOR(i, offsets_3.back().back()) res.push_back(ws[i]);
      FOR(i, offsets_2.back()) res.push_back(bs[i]);
      return res;
    }

    void import_weights(vf v)
    {
      int index = 0;
      FOR(i, offsets_3.back().back()) ws[i] = v[index++];
      FOR(i, offsets_2.back()) bs[i] = v[index++];
      assert(v.size() == index);
    }

    double train_softmax(vvf inputs, vvf labels, double lr)
    {
      return train(inputs, labels, vf(), vi(), lr, false);
    }

    double train_sigmoid(vvf inputs, vf labels, vi actions, double lr)
    {
      return train(inputs, vvf(), labels, actions, lr, true);
    }

    ~Nn()
    {
      free_memory();
    }

    vi ls;

  private:
    vector<int> offsets_2;
    vector<vector<int>> offsets_3;

    float *ws;
    float *bs;
    float *outs;
    float *last_out;

    float *grad_ws;
    float *grad_bs;
    float *grad_os;
    float *momentum_ws;
    float *momentum_bs;

    double train(vvf inputs, 
                 vvf labels, 
                 vf sig_labels,
                 vi sig_actions,
                 double lr, 
                 bool is_sigmoid)
    {
      double norm = 1.0 / inputs.size();
      double momentum = 0.9;
      double decay = 1e-4;

      memset(grad_bs, 0, sizeof(float) * offsets_2.back());
      memset(grad_os, 0, sizeof(float) * offsets_2.back());
      memset(grad_ws, 0, sizeof(float) * offsets_3.back().back());

      double total_loss = 0.0;
      FOR(input_index, inputs.size()){
        vf input = inputs[input_index];

        float *forward_out = forward(input, is_sigmoid);

        if(is_sigmoid){
          FOR(i, ls.back()){
            float *g = grad_os + offsets_2[ls.size() - 1];
            if(i == sig_actions[input_index]){
              double label = sig_labels[input_index];
              total_loss -= label * log(max(last_out[i], 1e-6f)) + (1 - label) * log(max(1 - last_out[i], 1e-6f));
              g[i] = last_out[i] - label;
            }else{
              g[i] = 0;
            }
          }
        }else{
          float *g = grad_os + offsets_2[ls.size() - 1];
          vf &label = labels[input_index];
          FOR(i, ls.back()){
            total_loss += label[i] * -log(max(1e-6f, forward_out[i]));
            g[i] = last_out[i] - label[i];
          }
        }
        for (int d = ls.size() - 2; 0 <= d; --d) {
          float *offset_grad_os_0 = grad_os + offsets_2[d + 0];
          float *offset_grad_os_1 = grad_os + offsets_2[d + 1];
          FOR(i, ls[d]){
            float *offset_ws = ws + offsets_3[d][i];
            offset_grad_os_0[i] = 0;
            bool relu_applied = (d + 1) < (ls.size() - 1);
            FOR(j, ls[d + 1]){
              if (relu_applied && (outs[offsets_2[d + 1] + j] == 0)) continue;
              offset_grad_os_0[i] += offset_grad_os_1[j] * offset_ws[j];
            }
          }
        }

        for (int d = 1; d < ls.size(); ++d){
          FOR(i, ls[d]){
            if ((d + 1 < ls.size()) && outs[offsets_2[d] + i] <= 0){
              assert(outs[offsets_2[d] + i] == 0);
              continue;
            }

            float *offset_outs = outs + offsets_2[d - 1];
            float grad_o = grad_os[offsets_2[d] + i];
            grad_bs[offsets_2[d] + i] += grad_o;
            FOR(j, ls[d - 1]){
              grad_ws[offsets_3[d - 1][j] + i] += grad_o * offset_outs[j];
            }
          }
        }
      }

      FOR(d, ls.size()){
        FOR(i, ls[d]){
          float &g = momentum_bs[offsets_2[d] + i];
          g = (momentum * g + (grad_bs[offsets_2[d] + i] * norm) + decay * bs[offsets_2[d] + i]);
          bs[offsets_2[d] + i] -= lr * g;
        }
      }
      FOR(d, ls.size() - 1){
        FOR(i, ls[d]){
          float *offset_grad_ws = grad_ws + offsets_3[d][i];
          float *offset_ws = ws + offsets_3[d][i];
          FOR(j, ls[d + 1]){
            float &g = momentum_ws[offsets_3[d][i] + j];
            g = (momentum * g + (offset_grad_ws[j] * norm) + decay * offset_ws[j]);
            offset_ws[j] -= lr * g;
          }
        }
      }

      return total_loss / inputs.size();
    }

    float* forward(vf &v, bool is_sigmoid)
    {
      assert(ls[0] == v.size());

      FOR(i, ls[0]) outs[offsets_2[0] + i] = v[i];

      FOR(d, ls.size() - 1){
        memcpy(outs + offsets_2[d + 1], bs + offsets_2[d + 1], sizeof(float) * ls[d + 1]);

        FOR(i, ls[d]){
          float &a = outs[offsets_2[d] + i];
          if (0 < d) a = max(a, 0.0f);
          if (a == 0) continue;

          float *out_ptr = outs + offsets_2[d + 1];
          float *out_ptr_end = outs + offsets_2[d + 1] + ls[d + 1];
          float *w_ptr = ws + offsets_3[d][i];

          __m256 m256_a = _mm256_set1_ps(a);
          while(out_ptr + 8 <= out_ptr_end){
            __m256 out = _mm256_loadu_ps(out_ptr);
            __m256 w = _mm256_loadu_ps(w_ptr);
            out = _mm256_add_ps(_mm256_mul_ps(m256_a, w), out);
            _mm256_storeu_ps(out_ptr, out);

            out_ptr += 8;
            w_ptr += 8;
          }
          while(out_ptr < out_ptr_end){
            *out_ptr += a * (*w_ptr);
            ++out_ptr;
            ++w_ptr;
          }
        }
      }

      if(is_sigmoid){
        FOR(i, ls.back()) last_out[i] = 1.0 / (1.0 + expf(-outs[offsets_2[ls.size() - 1] + i]));
      }else{
        float maxi = outs[offsets_2[ls.size() - 1] + 0];
        FOR(i, ls.back()) maxi = max(maxi, outs[offsets_2[ls.size() - 1] + i]);

        float total = 0.0;
        FOR(i, ls.back()){
          last_out[i] = expf(outs[offsets_2[ls.size() - 1] + i] - maxi);
          total += last_out[i];
        }
        FOR(i, ls.back()) last_out[i] /= total;
      }
      return last_out;
    }

    void free_memory()
    {
      delete [] ws;
      delete [] bs;
      delete [] outs;

      delete [] grad_ws;
      delete [] grad_bs;
      delete [] grad_os;
      delete [] momentum_ws;
      delete [] momentum_bs;

      delete [] last_out;
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

  // https://bowwowforeach.hatenablog.com/entry/2022/07/05/195417
  int CHAR_RANGES[3][2] = {
    {0x3220, 0x4DB4},
    {0x4DC0, 0x9FEE},
    {0xAC00, 0xD7A2},
  };

  int CHAR_RANGE_LENGTHS[3] = {
    CHAR_RANGES[0][1] - CHAR_RANGES[0][0],
    CHAR_RANGES[1][1] - CHAR_RANGES[1][0],
    CHAR_RANGES[2][1] - CHAR_RANGES[2][0],
  };

  class Base32768 
  {
  public:
    static string encode_k_bit_integer(vi &v, int k_bits)
    {
      assert(k_bits <= 16); // to avoid overflow.

      int mask_k = (1 << k_bits) - 1;
      vi c16s;

      c16s.push_back(k_bits);

      c16s.push_back(v.size() & MASK_15);
      c16s.push_back(v.size() >> 15);
      
      for (int i: convert_base(v, k_bits, 15)) c16s.push_back(i);

      u16string u16s = to_u16string(c16s);

      wstring_convert<codecvt_utf8_utf16<char16_t>, char16_t> converter;
      return converter.to_bytes(u16s);
    }

    static vi decode(string &u8s)
    {
      wstring_convert<codecvt_utf8_utf16<char16_t>, char16_t> converter;
      u16string u16s = converter.from_bytes(u8s);

      int k_bits = c2i(u16s[0]);
      assert(k_bits <= 16);

      int length = c2i(u16s[1]) | (c2i(u16s[2]) << 15);
      
      vi v;
      for (int i = 3; i < u16s.size(); ++i) v.push_back(c2i(u16s[i]));

      vi res = convert_base(v, 15, k_bits);
      while (length < res.size()) res.pop_back();

      return res;
    }

  private:

    static constexpr int MASK_15 = (1 << 15) - 1;

    static vi convert_base(vi &v, int curr_base_bits, int next_base_bits) 
    {
      vi res;

      int curr_mask = (1 << curr_base_bits) - 1;
      int cur = 0;
      int num_bits = 0;
      for (int i: v) {
        cur = (cur << curr_base_bits) | i;
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

    static u16string to_u16string(vi &v)
    {
      char16_t *a = new char16_t[v.size() + 1];

      FOR(i, v.size()) a[i] = i2c(v[i]);

      a[v.size()] = 0; // terminator.

      u16string res(a);

      delete [] a;

      return res;
    }

    static char16_t i2c(int t)
    {
      assert(0 <= t && t <= MASK_15);

      FOR(i, 3){
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
      FOR(i, 3){
        if (CHAR_RANGES[i][0] <= t && t <= CHAR_RANGES[i][1]) return res + (t - CHAR_RANGES[i][0]);
        res += CHAR_RANGE_LENGTHS[i];
      }

      assert(false);
    }
  };

  class NnIo
  {
  public:
    class Obj
    {
    public:

      Obj(
        vi ls,
        string w_str,
        float w_mini,
        float w_maxi,
        int k_bits
      ) : ls(ls), w_str(w_str), w_mini(w_mini), w_maxi(w_maxi), k_bits(k_bits) {}

      vi ls;
      string w_str;
      float w_mini;
      float w_maxi;
      int k_bits;

      void write(string path)
      {
        ofstream ofs(path);
        ofs << setprecision(20);
        ofs << "nn_eda::NnIo::from_obj(nn_eda::NnIo::Obj(" << endl;
        ofs << "vi({";
        for (int width: ls) ofs << width << ", ";
        ofs << "})," << endl;
        ofs << "\"" << w_str << "\"," << endl;
        ofs << w_mini << "," << endl;
        ofs << w_maxi << "," << endl;
        ofs << k_bits << "));" << endl;
      }
    };

    static Nn from_obj(Obj obj) 
    {
      vi int_weights = Base32768::decode(obj.w_str);

      Scaler scaler(obj.w_mini, obj.w_maxi);
      vf weights;
      for(int w: int_weights) weights.push_back(scaler.unscale((float) w / ((1 << obj.k_bits) - 1)));

      Nn nn(obj.ls);
      nn.import_weights(weights);
      return nn;
    }

    static Obj to_obj(Nn &nn, int k_bits) 
    {
      vf weights = nn.export_weights();
      Scaler scaler = Scaler::create(weights);
      vi int_weights;
      for(float w: weights) int_weights.push_back((int) (scaler.scale(w) * ((1 << k_bits) - 1) + 0.5));

      return Obj(
        nn.ls,
        Base32768::encode_k_bit_integer(int_weights, k_bits),
        scaler.mini,
        scaler.maxi,
        k_bits
      );
    }

    static void write_raw(Nn &nn, string path)
    {
      ofstream ofs(path, ios::binary);
      int num_ls = nn.ls.size();
      ofs.write(reinterpret_cast<const char*>(&num_ls), sizeof(int));
      for (int width: nn.ls) ofs.write(reinterpret_cast<const char*>(&width), sizeof(int));

      vf weights = nn.export_weights();
      int num_weights = weights.size();
      ofs.write(reinterpret_cast<const char*>(&num_weights), sizeof(int));
      for (float w: weights) ofs.write(reinterpret_cast<const char*>(&w), sizeof(float));
    }

    static Nn read_raw(string path)
    {
      ifstream ifs(path, ios::binary);
      int num_ls;
      ifs.read(reinterpret_cast<char*>(&num_ls), sizeof(int));
      vi ls(num_ls);
      FOR(i, num_ls) ifs.read(reinterpret_cast<char*>(&ls[i]), sizeof(int));

      Nn nn(ls);

      int num_weights;
      ifs.read(reinterpret_cast<char*>(&num_weights), sizeof(int));
      vf weights(num_weights);
      FOR(i, num_weights) ifs.read(reinterpret_cast<char*>(&weights[i]), sizeof(float));
      nn.import_weights(weights);
      return nn;
    }
  
  };

  #undef FOR
  #pragma pop_macro("FOR")
}
