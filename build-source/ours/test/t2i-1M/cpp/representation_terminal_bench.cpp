// Standalone, single-thread terminal scorer. No raw vectors or graph are loaded.
// Input primary scores and primary-transformed queries come from a frozen pool.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

extern "C" int terminal_bench(
    int queries, int dim, int groups, int width, int count, int additive, int topk,
    uint64_t node_count, const float* query, const float* projection,
    const float* books, const uint8_t* codes, const uint64_t* offsets,
    const uint32_t* ids, const float* primary, int profile,
    float* output_scores, uint32_t* output_ids, double* times) {
  if (queries <= 0 || dim <= 0 || groups <= 0 || width <= 0 || count != 256 ||
      topk <= 0 || (additive ? width != dim : groups*width != dim) || offsets[0]) return 1;
  uint64_t max_pool = 0;
  for (int q=0; q<queries; ++q) {
    if (offsets[q+1] < offsets[q] || offsets[q+1]-offsets[q] < uint64_t(topk)) return 2;
    max_pool = std::max(max_pool, offsets[q+1]-offsets[q]);
  }
  for (uint64_t i=0; i<offsets[queries]; ++i) if (ids[i] >= node_count) return 3;
  using Clock = std::chrono::steady_clock;
  auto ms = [](Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b-a).count();
  };
  std::vector<float> phi(dim), lut(groups*count), scores(max_pool);
  std::vector<uint32_t> order(max_pool);
  for (int j=0; j<5; ++j) times[j]=0;
  auto all_start=Clock::now();
  for (int q=0; q<queries; ++q) {
    Clock::time_point start, t1, t2, t3;
    if (profile) start=Clock::now();
    // projection is stored output-major; no per-query allocations.
    for (int j=0; j<dim; ++j) {
      float x=0;
      for (int k=0; k<dim; ++k) x += query[q*dim+k]*projection[j*dim+k];
      phi[j]=x;
    }
    if (profile) t1=Clock::now();
    for (int g=0; g<groups; ++g) {
      const float* f=phi.data()+(additive ? 0 : g*width);
      for (int c=0; c<count; ++c) {
        const float* book=books+(g*count+c)*width;
        float x=0;
        for (int j=0; j<width; ++j) x+=f[j]*book[j];
        lut[g*count+c]=x;
      }
    }
    if (profile) t2=Clock::now();
    uint64_t lo=offsets[q], n=offsets[q+1]-lo;
    for (uint64_t i=0; i<n; ++i) {
      const uint8_t* code=codes+uint64_t(ids[lo+i])*groups;
      float x=primary[lo+i];
      for (int g=0; g<groups; ++g) x+=lut[g*count+code[g]];
      scores[i]=x;
      if (output_scores) output_scores[lo+i]=x;
      order[i]=uint32_t(i);
    }
    if (profile) t3=Clock::now();
    std::partial_sort(order.begin(), order.begin()+topk, order.begin()+n,
        [&](uint32_t a, uint32_t b) {
          return scores[a]>scores[b] || (scores[a]==scores[b] && ids[lo+a]<ids[lo+b]);
        });
    for (int k=0; k<topk; ++k) output_ids[q*topk+k]=ids[lo+order[k]];
    if (profile) {
      times[1]+=ms(start,t1); times[2]+=ms(t1,t2);
      times[3]+=ms(t2,t3); times[4]+=ms(t3,Clock::now());
    }
  }
  times[0]=ms(all_start,Clock::now());
  for (int j=0; j<5; ++j) times[j]/=queries;
  return 0;
}
