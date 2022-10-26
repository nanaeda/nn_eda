#include <bits/stdc++.h>


template<typename T> class DqnSampler
{
public:
  DqnSampler(int max_capacity) : max_cap(max_capacity)
  {
    n2 = 1;  
    while(n2 <= max_cap) n2 *= 2;
    total_weights = std::vector<double>(n2 * 2, 0);
    tail = 0;
  }

  void add_or_overwrite(T &a, double weight)
  {
    update_weight(tail, weight);
    if(as.size() < max_cap){
      as.push_back(a);
    }else{
      as[tail] = a;
    }
    tail = (tail + 1) % max_cap;
  }

  void update_weight(int i, double weight)
  {
    i += n2;
    total_weights[i] = weight;
    while(1 < i){
      i /= 2;
      total_weights[i] = total_weights[i * 2] + total_weights[i * 2 + 1];
    }
  }

  int get_index()
  {
    assert(0 < as.size());
    const int mask = (1 << 25) - 1;
    double rem = (((double) (rand() & mask)) / mask) * total_weights[1];
    int i = 1;
    while(i < n2){
      if(rem <= total_weights[i * 2]){
        i = i * 2;
      }else{
        rem -= total_weights[i * 2];
        i = i * 2 + 1;
      }
    }
    return std::min(i - n2, (int) (as.size() - 1));
  }

  T& get(int index)
  {
    return as[index];
  }


private:
  std::vector<double> total_weights;
  std::vector<T> as;
  int max_cap;
  int n2;  
  int tail;
};

