#pragma GCC target("avx2")

#include <bits/stdc++.h>
#include <x86intrin.h>



namespace nn_eda
{
  using namespace std;
  typedef vector<int> vi;
  typedef vector<float> vf;
  #pragma push_macro("FOR")
  #define FOR(i, n) for(int i = 0; i < ((int) (n)); ++i)

  class Inferrer
  {
  public:
    Inferrer(const vi &fc_widths, const vi &head_widths)
    {
      set_widths(fc_widths, head_widths);
      
      allocate_memory();
      initialize_weights(); 
    }

    Inferrer(const Inferrer &obj)
    {
      set_widths(obj.fc_widths, obj.head_widths);

      allocate_memory();
      import_weights(obj.export_weights());
    }

    Inferrer& operator=(const Inferrer &obj)
    {
      free_memory();

      set_widths(obj.fc_widths, obj.head_widths);

      allocate_memory();
      import_weights(obj.export_weights());

      return *this;
    }

    ~Inferrer()
    {
      free_memory();
    }

    void import_weights(const vf &v)
    {
      int index = 0;
      FOR(i, w_offsets.back()) ws[i] = v[index++];
      FOR(i, b_offsets.back()) bs[i] = v[index++];
      assert(v.size() == index);
    }

    vf export_weights() const
    {
      vf res;
      FOR(i, w_offsets.back()) res.push_back(ws[i]);
      FOR(i, b_offsets.back()) res.push_back(bs[i]);
      return res;
    }

    float get_prediction(int head_i, int index)
    {
      return last_out[head_offsets[head_i] + index];
    }

    void forward(const vf &v)
    {
      assert(all_widths[0] == v.size());

      FOR(i, all_widths[0]) outs[b_offsets[0] + i] = v[i];

      FOR(d, all_widths.size() - 1){
        memcpy(outs + b_offsets[d + 1], bs + b_offsets[d + 1], sizeof(float) * all_widths[d + 1]);

        FOR(i, all_widths[d]){
          float &a = outs[b_offsets[d] + i];
          if (0 < d) a = max(a, 0.0f);
          if (a == 0) continue;

          float *out_ptr = outs + b_offsets[d + 1];
          float *out_ptr_end = out_ptr + all_widths[d + 1];
          float *w_ptr = ws + get_w_offset(d, i);

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

      FOR(head_i, head_widths.size()){
        float *out_ptr = outs + b_offsets[all_widths.size() - 1] + head_offsets[head_i];
        float *last_out_ptr = last_out + head_offsets[head_i];

        float maxi = out_ptr[0];
        FOR(i, head_widths[head_i]) maxi = max(maxi, out_ptr[i]);

        float total = 0.0;
        FOR(i, head_widths[head_i]){
          last_out_ptr[i] = expf(out_ptr[i] - maxi);
          total += last_out_ptr[i];
        }
        FOR(i, head_widths[head_i]) last_out_ptr[i] /= total;
      }
    }

    vi fc_widths;
    vi head_widths;

  protected:

    vi all_widths;
    vi head_offsets;

    vi b_offsets;
    vi w_offsets;

    float *ws;
    float *bs;
    float *outs;
    float *last_out;

    int get_w_offset(int i0, int i1)
    {
      return w_offsets[i0] + i1 * all_widths[i0 + 1];
    }

    void set_widths(const vi &fc_widths, const vi &head_widths)
    {
      this->fc_widths = fc_widths;
      this->head_widths = head_widths;

      head_offsets = vi({0});
      for(int w: head_widths) head_offsets.push_back(head_offsets.back() + w);

      all_widths = fc_widths;
      all_widths.push_back(head_offsets.back());
    }

    void allocate_memory()
    {
      b_offsets.push_back(0);
      FOR(i, all_widths.size()) b_offsets.push_back(b_offsets.back() + all_widths[i]);

      w_offsets.push_back(0);
      FOR(i, all_widths.size() - 1) w_offsets.push_back(w_offsets.back() + all_widths[i] * all_widths[i + 1]);

      ws = new float[w_offsets.back()];
      bs = new float[b_offsets.back()];
      outs = new float[b_offsets.back()];
      last_out = new float[all_widths.back()];
    }

    void free_memory()
    {
      delete [] ws;
      delete [] bs;
      delete [] outs;
      delete [] last_out;
    }

    void initialize_weights()
    {
      memset(bs, 0, sizeof(float) * b_offsets.back());

      FOR(d, all_widths.size() - 1)FOR(i, all_widths[d]){
        float *offset_ws = ws + get_w_offset(d, i);
        FOR(j, all_widths[d + 1]){
          int mask = (1 << 25) - 1;
          float f = ((float) (rand() & mask)) / mask;
          offset_ws[j] = (2 * f - 1) * sqrtf(6.0f / (all_widths[d] + all_widths[d + 1]));
        }
      }
    }

  };

  class Scaler
  {
  public:
    Scaler(float mini, float maxi) : 
      mini(mini + ((mini == maxi) ? 1e-6 : 0)),
      maxi(maxi - ((mini == maxi) ? 1e-6 : 0))
    {
      assert(mini <= maxi);
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

    static Scaler create(const vf &v) 
    {
      return Scaler(
        *min_element(v.begin(), v.end()),
        *max_element(v.begin(), v.end())
      );
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
    static string encode_k_bit_integer(const vi &v, int k_bits)
    {
      assert(k_bits <= 16); // to avoid overflow.

      int mask_k = (1 << k_bits) - 1;
      vi c16s;

      c16s.push_back(k_bits);

      c16s.push_back(v.size() & MASK_15);
      c16s.push_back(v.size() >> 15);
      
      for (int i: convert_base(v, k_bits, 15)) c16s.push_back(i);

      u16string u16s = to_u16string(c16s);

      return wstring_convert<codecvt_utf8_utf16<char16_t>, char16_t>().to_bytes(u16s);
    }

    static vi decode(const string &u8s)
    {
      u16string u16s = wstring_convert<codecvt_utf8_utf16<char16_t>, char16_t>().from_bytes(u8s);

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

    static vi convert_base(const vi &v, int curr_base_bits, int next_base_bits) 
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

    static u16string to_u16string(const vi &v)
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

      assert(false);
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
  
    static Inferrer deserialize(const string &str)
    {
      stringstream ss(str);

      int len;

      ss >> len;
      vi fc_widths(len);
      FOR(i, len) ss >> fc_widths[i];

      ss >> len;
      vi head_widths(len);
      FOR(i, len) ss >> head_widths[i];

      float w_mini, w_maxi;
      int k_bits;
      string w_str;
      ss >> w_mini >> w_maxi >> k_bits >> w_str;

      vi int_weights = Base32768::decode(w_str);

      Scaler scaler(w_mini, w_maxi);
      vf weights;
      for(int w: int_weights) weights.push_back(scaler.unscale((float) w / ((1 << k_bits) - 1)));

      Inferrer res(fc_widths, head_widths);
      res.import_weights(weights);
      return res;
    }

    static string serialize(Inferrer &inferrer, int k_bits) 
    {
      vf weights = inferrer.export_weights();
      Scaler scaler = Scaler::create(weights);
      vi int_weights;
      for(float w: weights) int_weights.push_back((int) (scaler.scale(w) * ((1 << k_bits) - 1) + 0.5));
      string w_str = Base32768::encode_k_bit_integer(int_weights, k_bits);

      stringstream ss;
      ss << setprecision(20);
      ss << inferrer.fc_widths.size() << " ";
      for(int w: inferrer.fc_widths) ss << w << " ";
      ss << inferrer.head_widths.size() << " ";
      for(int w: inferrer.head_widths) ss << w << " ";
      ss << scaler.mini << " " << scaler.maxi << " " << k_bits << " " << w_str;
      return ss.str();
    }
  };


// CUT_BEGIN

  class Trainer : public Inferrer
  {
  public:
    Trainer(const vi &fc_widths, const vi &head_widths) : Inferrer(fc_widths, head_widths)
    {
      allocate_memory();
    }

    Trainer(const Trainer &trainer) : Inferrer(trainer)
    {
      allocate_memory();
    }

    Trainer(const Inferrer &inferrer) : Inferrer(inferrer)
    {
      allocate_memory();
    }

    Trainer& operator=(const Trainer &trainer)
    {
      Inferrer::operator=(trainer);

      free_memory();
      allocate_memory();

      return *this;
    }

    ~Trainer()
    {
      free_memory();
    }

    double train(const vector<vf> &inputs, const vector<vector<vf>> &labels, double lr)
    {
      double norm = 1.0 / inputs.size();
      double momentum = 0.9;
      double decay = 1e-4;

      memset(grad_bs, 0, sizeof(float) * b_offsets.back());
      memset(grad_os, 0, sizeof(float) * b_offsets.back());
      memset(grad_ws, 0, sizeof(float) * w_offsets.back());

      double total_loss = 0.0;
      FOR(input_index, inputs.size()){
        vf input = inputs[input_index];

        forward(input);

        FOR(head_i, head_widths.size()){
          float *g = grad_os + b_offsets[all_widths.size() - 1] + head_offsets[head_i];
          const vf &label = labels[input_index][head_i];
          if(label.empty()) continue;
          FOR(i, head_widths[head_i]){
            float out = last_out[head_offsets[head_i] + i];
            total_loss += label[i] * -log(max(1e-6f, out));
            g[i] = out - label[i];
          }
        }

        float local_store[8];
        for (int d = all_widths.size() - 2; 1 <= d; --d) {
          FOR(i, all_widths[d]){
            float total = 0;
            if(0 < outs[b_offsets[d] + i]){
              float *os_ptr = grad_os + b_offsets[d + 1];
              float *os_ptr_end = os_ptr + all_widths[d + 1];
              float *ws_ptr = ws + get_w_offset(d, i);

              __m256 sum = _mm256_setzero_ps();
              while(os_ptr + 8 <= os_ptr_end){
                __m256 o = _mm256_loadu_ps(os_ptr);
                __m256 w = _mm256_loadu_ps(ws_ptr);
                sum = _mm256_add_ps(_mm256_mul_ps(o, w), sum);
                os_ptr += 8;
                ws_ptr += 8;
              }
              _mm256_storeu_ps(local_store, sum);

              FOR(k, 8) total += local_store[k];
              
              while(os_ptr < os_ptr_end){
                total += (*os_ptr) * (*ws_ptr);
                ++os_ptr;
                ++ws_ptr;
              }
            }
            grad_os[b_offsets[d] + i] = total;
          }
        }

        for (int d = 1; d < all_widths.size(); ++d){
          FOR(i, all_widths[d]) grad_bs[b_offsets[d] + i] += grad_os[b_offsets[d] + i];;
          FOR(j, all_widths[d - 1]){

            float out = outs[b_offsets[d - 1] + j];
            float *os_ptr = grad_os + b_offsets[d];
            float *os_ptr_end = os_ptr + all_widths[d];
            float *grad_ws_ptr = grad_ws + get_w_offset(d - 1, j);

            __m256 m256_out = _mm256_set1_ps(out);
            while(os_ptr + 8 <= os_ptr_end){
              __m256 o = _mm256_loadu_ps(os_ptr);
              __m256 grad_w = _mm256_loadu_ps(grad_ws_ptr);
              grad_w = _mm256_add_ps(_mm256_mul_ps(m256_out, o), grad_w);
              _mm256_storeu_ps(grad_ws_ptr, grad_w);

              os_ptr += 8;
              grad_ws_ptr += 8;
            }
            while(os_ptr < os_ptr_end){
              *grad_ws_ptr += out * (*os_ptr);
              ++os_ptr;
              ++grad_ws_ptr;
            }
          }
        }
      }

      FOR(d, all_widths.size()){
        FOR(i, all_widths[d]){
          float &g = momentum_bs[b_offsets[d] + i];
          g = (momentum * g + (grad_bs[b_offsets[d] + i] * norm) + decay * bs[b_offsets[d] + i]);
          bs[b_offsets[d] + i] -= lr * g;
        }
      }
      FOR(d, all_widths.size() - 1){
        FOR(i, all_widths[d]){
          float *offset_grad_ws = grad_ws + get_w_offset(d, i);
          float *offset_ws = ws + get_w_offset(d, i);
          float *offset_momentum_ws = momentum_ws + get_w_offset(d, i);
          FOR(j, all_widths[d + 1]){
            float &g = offset_momentum_ws[j];
            g = (momentum * g + (offset_grad_ws[j] * norm) + decay * offset_ws[j]);
            offset_ws[j] -= lr * g;
          }
        }
      }

      return total_loss / inputs.size();
    }

  private:

    float *grad_ws;
    float *grad_bs;
    float *grad_os;
    float *momentum_ws;
    float *momentum_bs;

    void allocate_memory()
    {
      grad_ws = new float[w_offsets.back()];
      grad_bs = new float[b_offsets.back()];
      grad_os = new float[b_offsets.back()];
      momentum_ws = new float[w_offsets.back()];
      momentum_bs = new float[b_offsets.back()];
      memset(grad_ws, 0, sizeof(float) * w_offsets.back());
      memset(grad_bs, 0, sizeof(float) * b_offsets.back());
      memset(grad_os, 0, sizeof(float) * b_offsets.back());
      memset(momentum_ws, 0, sizeof(float) * w_offsets.back());
      memset(momentum_bs, 0, sizeof(float) * b_offsets.back());
    }

    void free_memory()
    {
      delete [] grad_ws;
      delete [] grad_bs;
      delete [] grad_os;
      delete [] momentum_ws;
      delete [] momentum_bs;
    }

  };

// CUT_END

  #undef FOR
  #pragma pop_macro("FOR")
}

