// Score/selection consistency and a warm-code microbenchmark, NOT graph QPS.
#include "ours/qvrr_primary.h"
#include "ours/mapped_fbin.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

int main(int argc,char** argv) {
  try {
    if(argc!=5) throw std::runtime_error("usage: audit model codes queries.fbin ids.ibin");
    ours::QvrrPrimaryIndex primary(argv[1],argv[2]);
    ours::MappedFbin queries(argv[3]);
    std::ifstream stream(argv[4],std::ios::binary); uint32_t rows=0,cols=0;
    stream.read(reinterpret_cast<char*>(&rows),4); stream.read(reinterpret_cast<char*>(&cols),4);
    if(!rows || !cols || rows>queries.rows()) throw std::runtime_error("Invalid audit shape");
    std::vector<uint32_t> ids(uint64_t(rows)*cols);
    stream.read(reinterpret_cast<char*>(ids.data()),ids.size()*4);
    if(!stream) throw std::runtime_error("Truncated IDs");
    double max_error=0; uint64_t different=0,top10_different=0;
    std::vector<float> reference(cols),batched(cols);
    for(uint32_t q=0;q<rows;++q) {
      auto ctx=primary.Prepare(queries.Row(q),queries.dim());
      const auto* nodes=ids.data()+uint64_t(q)*cols;
      for(uint32_t j=0;j<cols;++j) reference[j]=primary.Score(ctx,nodes[j]);
      primary.ScoreBatch(ctx,nodes,cols,batched.data());
      for(uint32_t j=0;j<cols;++j) {
        double error=std::abs(double(reference[j])-batched[j]);
        max_error=std::max(max_error,error);
        different+=std::memcmp(&reference[j],&batched[j],4)!=0;
        if(error>2e-5 || !std::isfinite(batched[j])) throw std::runtime_error("Scalar/SIMD mismatch");
      }
      // Deterministic score/ID tie breaking. Record rather than hide near ties.
      std::vector<uint32_t> a(cols),b(cols); std::iota(a.begin(),a.end(),0); b=a;
      const auto rank=[&](const std::vector<float>& s) {
        return [&s,nodes](uint32_t u,uint32_t v) { return s[u]!=s[v] ? s[u]>s[v] : nodes[u]<nodes[v]; };
      };
      std::sort(a.begin(),a.end(),rank(reference)); std::sort(b.begin(),b.end(),rank(batched));
      a.resize(std::min<uint32_t>(10,cols)); b.resize(a.size());
      std::sort(a.begin(),a.end()); std::sort(b.begin(),b.end()); top10_different+=a!=b;
      // Every short/masked batch must agree, including final mapped record.
      for(size_t count: {size_t(0),size_t(1),size_t(3),size_t(4),size_t(7),size_t(15),size_t(16),size_t(17),size_t(31)}) {
        count=std::min<size_t>(count,cols);
        primary.ScoreBatch(ctx,nodes,count,batched.data());
        for(size_t j=0;j<count;++j) if(std::abs(batched[j]-reference[j])>2e-5) throw std::runtime_error("Tail mismatch");
      }
    }
    auto ctx=primary.Prepare(queries.Row(0),queries.dim());
    volatile double checksum=0;
    const auto measure=[&](bool batch) {
      std::vector<double> times;
      for(int trial=0;trial<5;++trial) {
        auto start=std::chrono::steady_clock::now();
        for(int repeat=0;repeat<128;++repeat) {
          if(batch) primary.ScoreBatch(ctx,ids.data(),cols,batched.data());
          else for(uint32_t j=0;j<cols;++j) batched[j]=primary.Score(ctx,ids[j]);
          checksum=checksum+batched[repeat%cols];
        }
        times.push_back(std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count()/(128.*cols));
      }
      std::sort(times.begin(),times.end()); return times[2];
    };
    const auto scalar_ns=measure(false),simd_ns=measure(true);
    std::cout<<"{\"kernel\":\""<<primary.scoring_kernel()<<"\",\"queries\":"<<rows
      <<",\"pairs\":"<<uint64_t(rows)*cols<<",\"max_scalar_simd_error\":"<<max_error
      <<",\"bitwise_different_scores\":"<<different<<",\"top10_set_different_queries\":"<<top10_different
      <<",\"scalar_ns_per_score\":"<<scalar_ns<<",\"simd_ns_per_score\":"<<simd_ns
      <<",\"speedup\":"<<scalar_ns/simd_ns<<",\"timing_scope\":\"warm-code kernel microbenchmark, not graph QPS\"}\n";
    return 0;
  } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
