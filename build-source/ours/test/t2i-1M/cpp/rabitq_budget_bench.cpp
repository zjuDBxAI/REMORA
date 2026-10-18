// Isolated experiment: reuse the production primary scorer, beam search and
// tie breaking verbatim, without editing/rebuilding the running ANN binaries.
#define main included_primary_search_main
#include "../../../src/primary_ji_search_main.cpp"
#undef main
#include <omp.h>
#include <numeric>

namespace {
thread_local std::string budget_error;
using BudgetClock = std::chrono::steady_clock;
double elapsed(BudgetClock::time_point start) {
  return std::chrono::duration<double, std::milli>(BudgetClock::now()-start).count();
}
struct BudgetContext {
  RabitqPrimary primary;
  int groups, dim, width;
  std::vector<float> projection, books;
  std::vector<uint8_t> codes;
  BudgetContext(const char* path, uint64_t n, int g, const float* p,
                const float* b, const uint8_t* c)
      : primary(path, "exact"), groups(g), dim(primary.scorer_dim()), width(g ? dim/g : 0) {
    if (n != primary.nodes() || (g != 0 && g != 32) || (g && dim%g))
      throw std::runtime_error("Node count / 256-bit Product shape mismatch");
    if (g) {
      projection.assign(p,p+dim*dim); books.assign(b,b+256*dim);
      codes.assign(c,c+n*g);
    }
  }
  void Refine(const float* q, const uint32_t* ids, size_t n, float* scores) const {
    if (!groups) return;
    std::vector<float> phi(dim), lut(groups*256);
    for (int j=0;j<dim;++j) {
      float value=0;
      for (int k=0;k<dim;++k) value+=q[k]*projection[j*dim+k];
      phi[j]=value;
    }
    for (int g=0;g<groups;++g) for (int c=0;c<256;++c) {
      float value=0;
      for (int j=0;j<width;++j) value+=phi[g*width+j]*books[(g*256+c)*width+j];
      lut[g*256+c]=value;
    }
    for (size_t i=0;i<n;++i) {
      const uint8_t* code=codes.data()+uint64_t(ids[i])*groups;
      for (int g=0;g<groups;++g) scores[i]+=lut[g*256+code[g]];
    }
  }
};
// Prepare exactly once per query, and reuse that context in the stock search.
struct PreparedPrimary {
  const RabitqPrimary& primary;
  const RabitqPrimary::Context& query;
  auto Prepare(const float*, uint32_t) const { return &query; }
  uint32_t scorer_dim() const { return primary.scorer_dim(); }
  const float* Transformed(const RabitqPrimary::Context* q) const { return primary.Transformed(*q); }
  void ScoreBatch(const RabitqPrimary::Context* q, const uint32_t* ids, size_t n, float* s) const {
    primary.ScoreBatch(*q,ids,n,s);
  }
};
void select_ids(const uint32_t* ids, const float* scores, size_t n, int k, uint32_t* out) {
  std::vector<uint32_t> order(n); std::iota(order.begin(),order.end(),0);
  std::partial_sort(order.begin(),order.begin()+k,order.end(),[&](uint32_t a,uint32_t b){
    return scores[a]>scores[b] || (scores[a]==scores[b] && ids[a]<ids[b]);
  });
  for(int j=0;j<k;++j) out[j]=ids[order[j]];
}
}
extern "C" const char* rb_budget_error() { return budget_error.c_str(); }
extern "C" void* rb_budget_create(const char* path,uint64_t n,int g,const float* p,
                                   const float* b,const uint8_t* c) {
  try { return new BudgetContext(path,n,g,p,b,c); }
  catch(const std::exception& e) { budget_error=e.what(); return nullptr; }
}
extern "C" void rb_budget_destroy(void* p) { delete static_cast<BudgetContext*>(p); }
extern "C" void* rb_budget_graph(const char* path) {
  try { return new ours::RoarGraph(ours::RoarGraph::Load(path)); }
  catch(const std::exception& e) { budget_error=e.what(); return nullptr; }
}
extern "C" void rb_budget_graph_destroy(void* p) { delete static_cast<ours::RoarGraph*>(p); }

// Fixed-pool timings recompute BOTH primary scorers from compressed codes.
// Raw base vectors and cached primary scores are not available in this kernel.
extern "C" int rb_budget_pools(void* ptr,int nq,int d,const float* queries,
    const uint64_t* offsets,const uint32_t* ids,int k,int profile,
    float* out_scores,uint32_t* out_ids,double* times) {
  try {
    const auto& c=*static_cast<BudgetContext*>(ptr);
    if(nq<1 || d!=int(c.primary.original_dim()) || offsets[0] || k<1)
      throw std::runtime_error("Invalid pool dimensions");
    for(int q=0;q<nq;++q) if(offsets[q+1]<offsets[q]+k) throw std::runtime_error("Short pool");
    for(uint64_t i=0;i<offsets[nq];++i) if(ids[i]>=c.primary.nodes()) throw std::runtime_error("Invalid node ID");
    for(int q=0;q<nq;++q) {
      size_t n=offsets[q+1]-offsets[q]; const uint32_t* ix=ids+offsets[q];
      std::vector<float> scores(n);
      const auto begin=BudgetClock::now();
      auto context=c.primary.Prepare(queries+uint64_t(q)*d,d);
      c.primary.ScoreBatch(context,ix,n,scores.data());
      double primary_ms=profile?elapsed(begin):0;
      auto stage=profile?BudgetClock::now():begin;
      c.Refine(c.primary.Transformed(context),ix,n,scores.data());
      double refine_ms=profile?elapsed(stage):0;
      stage=profile?BudgetClock::now():begin;
      select_ids(ix,scores.data(),n,k,out_ids+q*k);
      times[4*q]=elapsed(begin); times[4*q+1]=primary_ms;
      times[4*q+2]=refine_ms; times[4*q+3]=profile?elapsed(stage):0;
      if(out_scores) std::copy(scores.begin(),scores.end(),out_scores+offsets[q]);
    }
    return 0;
  } catch(const std::exception& e) {budget_error=e.what();return 1;}
}

// All arms use the SAME original 3-bit graph. 4-bit changes traversal scores,
// not topology. Primary+JI keeps primary-only traversal, terminal refinement.
extern "C" int rb_budget_search(void* ptr,void* gp,int nq,int d,const float* queries,
    int beam,int k,int profile,uint32_t* out_ids,uint32_t* retained,double* stats) {
  try {
    const auto& c=*static_cast<BudgetContext*>(ptr);
    const auto& graph=*static_cast<ours::RoarGraph*>(gp);
    if(nq<1 || d!=int(c.primary.original_dim()) || k<1 || beam<k || graph.size()!=c.primary.nodes())
      throw std::runtime_error("Invalid graph/query dimensions");
    for(int q=0;q<nq;++q) {
      const float* query=queries+uint64_t(q)*d;
      const auto begin=BudgetClock::now();
      auto context=c.primary.Prepare(query,d);
      PreparedPrimary prepared{c.primary,context};
      auto result=SearchOne(prepared,nullptr,graph,query,d,beam,beam,0,nullptr,
                            false,false,false,false,false,false,0);
      double primary_ms=profile?elapsed(begin):0;
      auto stage=profile?BudgetClock::now():begin;
      c.Refine(c.primary.Transformed(context),result.ids.data(),result.ids.size(),result.scores.data());
      double refine_ms=profile?elapsed(stage):0;
      stage=profile?BudgetClock::now():begin;
      select_ids(result.ids.data(),result.scores.data(),result.ids.size(),k,out_ids+q*k);
      stats[8*q]=elapsed(begin); stats[8*q+1]=primary_ms;stats[8*q+2]=refine_ms;
      stats[8*q+3]=profile?elapsed(stage):0;stats[8*q+4]=result.visited;
      stats[8*q+5]=result.comparisons;stats[8*q+6]=result.expansions;stats[8*q+7]=result.pool;
      if(retained) std::copy(result.ids.begin(),result.ids.end(),retained+uint64_t(q)*beam);
    }
    return 0;
  } catch(const std::exception& e) {budget_error=e.what();return 1;}
}

// Offline only. Reuse the exact saved rotation, dataset order and encoder.
// Re-encode probes at 3 bits first: reject a different base or encoder config.
extern "C" int rb_budget_encode4(const char* old_path,const char* base_path,
                                  const char* output_path,int threads) {
  try {
    if(threads<1 || std::filesystem::exists(output_path)) throw std::runtime_error("Invalid thread count or output exists");
    Mapping old(old_path);RabitqHeader h{};
    if(old.size()<sizeof(h)) throw std::runtime_error("Truncated source");
    std::memcpy(&h,old.data(),sizeof(h));
    if(std::memcmp(h.magic,"RBQROAR1",8) || h.version!=1 || h.total_bits!=3 || h.metric_type!=rabitqlib::METRIC_IP ||
       old.size()!=sizeof(h)+h.rotator_bytes+h.num_points*(h.bin_bytes_per_vec+h.ex_bytes_per_vec))
      throw std::runtime_error("Expected complete 3-bit IP source");
    ours::MappedFbin base(base_path);
    if(base.rows()!=h.num_points || base.dim()!=h.dim) throw std::runtime_error("Wrong base shape");
    auto make_rotator=[&](){
      auto r=std::unique_ptr<rabitqlib::Rotator<float>>(rabitqlib::choose_rotator<float>(h.dim,
          static_cast<rabitqlib::RotatorType>(h.rotator_type),h.padded_dim));
      r->load(static_cast<const char*>(old.data())+sizeof(h));return r;
    };
    auto rot=make_rotator();
    const char* old_bin=static_cast<const char*>(old.data())+sizeof(h)+h.rotator_bytes;
    const char* old_ex=old_bin+h.num_points*h.bin_bytes_per_vec;
    std::vector<float> v(h.padded_dim),zero(h.padded_dim,0);
    std::vector<char> pb(h.bin_bytes_per_vec),pe(h.ex_bytes_per_vec);
    for(uint64_t j=0;j<128;++j) {
      uint64_t i=j*(h.num_points-1)/127;rot->rotate(base.Row(i),v.data());
      std::fill(pb.begin(),pb.end(),0);std::fill(pe.begin(),pe.end(),0);
      rabitqlib::quant::quantize_split_single(v.data(),zero.data(),h.padded_dim,2,pb.data(),pe.data(),
          rabitqlib::METRIC_IP,rabitqlib::quant::RabitqConfig());
      // Integer payload must match exactly. Float calibration may differ by
      // a few ULPs across compiler versions/FMA contraction, not by scale.
      const size_t nb=h.padded_dim/8,ne=h.padded_dim*2/8;
      auto calibration_close=[](const char* a,const char* b,size_t count) {
        for(size_t t=0;t<count;++t) {
          float x,y;std::memcpy(&x,a+4*t,4);std::memcpy(&y,b+4*t,4);
          if(!std::isfinite(x) || !std::isfinite(y) || std::abs(x-y)>1e-6f+1e-5f*std::abs(y)) return false;
        }
        return true;
      };
      if(std::memcmp(pb.data(),old_bin+i*h.bin_bytes_per_vec,nb) ||
         std::memcmp(pe.data(),old_ex+i*h.ex_bytes_per_vec,ne) ||
         !calibration_close(pb.data()+nb,old_bin+i*h.bin_bytes_per_vec+nb,3) ||
         !calibration_close(pe.data()+ne,old_ex+i*h.ex_bytes_per_vec+ne,2))
        throw std::runtime_error("3-bit re-encoding probe failed: wrong base/config/library");
    }
    h.total_bits=4;h.bin_bytes_per_vec=rabitqlib::BinDataMap<float>::data_bytes(h.padded_dim);
    h.ex_bytes_per_vec=rabitqlib::ExDataMap<float>::data_bytes(h.padded_dim,3);
    std::vector<char> bin(h.num_points*h.bin_bytes_per_vec,0),ex(h.num_points*h.ex_bytes_per_vec,0);
    #pragma omp parallel num_threads(threads)
    {
      std::vector<float> rotated(h.padded_dim),centroid(h.padded_dim,0);
      #pragma omp for schedule(static)
      for(uint64_t i=0;i<h.num_points;++i) {
        rot->rotate(base.Row(i),rotated.data());
        rabitqlib::quant::quantize_split_single(rotated.data(),centroid.data(),h.padded_dim,3,
            bin.data()+i*h.bin_bytes_per_vec,ex.data()+i*h.ex_bytes_per_vec,rabitqlib::METRIC_IP,
            rabitqlib::quant::RabitqConfig());
      }
    }
    std::ofstream out(output_path,std::ios::binary);
    out.write(reinterpret_cast<const char*>(&h),sizeof(h));
    out.write(static_cast<const char*>(old.data())+sizeof(h),h.rotator_bytes);
    out.write(bin.data(),bin.size());out.write(ex.data(),ex.size());out.close();
    if(!out) throw std::runtime_error("Failed writing 4-bit codes");
    return 0;
  } catch(const std::exception& e) {budget_error=e.what();return 1;}
}
