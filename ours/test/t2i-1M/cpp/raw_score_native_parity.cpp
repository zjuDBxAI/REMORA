// Exercise the production reader, padded preparation and scalar/batch kernels.
#include "ours/ji_online.h"
#include "ours/mapped_fbin.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

int main(int argc, char **argv) {
  try {
    if (argc != 9) throw std::invalid_argument("model codes queries start rows nodes output padded");
    auto ji = ours::JiOnlineIndex::Load(argv[1],argv[2]);
    ours::MappedFbin q(argv[3]);
    const uint32_t start=std::stoul(argv[4]), rows=std::stoul(argv[5]), n=std::stoul(argv[6]);
    if (!rows || !n || start+uint64_t(rows)>q.rows() || n>ji.node_count())
      throw std::invalid_argument("bad parity subset");
    const bool padded=std::stoi(argv[8])!=0;
    std::vector<uint32_t> ids(n); std::iota(ids.begin(),ids.end(),0);
    std::vector<float> scores(n);
    std::ofstream out(argv[7],std::ios::binary);
    out.write(reinterpret_cast<const char*>(&rows),4);
    out.write(reinterpret_cast<const char*>(&n),4);
    ours::JiQueryWorkspace workspace;
    const float *lut_buffer=nullptr, *phi_buffer=nullptr, *pad_buffer=nullptr;
    for (uint32_t i=0;i<rows;++i) {
      auto query = padded ? ji.PrepareQueryZeroPadded(q.Row(start+i),q.dim())
                          : ji.PrepareQuery(q.Row(start+i),q.dim());
      auto reference = padded ? ji.PrepareQueryZeroPadded(q.Row(start+i),q.dim(),false)
                              : ji.PrepareQuery(q.Row(start+i),q.dim(),false);
      ours::JiPrepareTimings timings;
      const auto &terminal=ji.PrepareTerminalQuery(q.Row(start+i),q.dim(),workspace,&timings);
      if(terminal.lookup.size()!=query.lookup.size() ||
         std::memcmp(terminal.lookup.data(),query.lookup.data(),query.lookup.size()*sizeof(float)) ||
         std::memcmp(&terminal.mean_bias,&query.mean_bias,sizeof(float)) ||
         !std::isinf(terminal.max_correction_upper) || timings.upper_bound_ms!=0)
        throw std::runtime_error("terminal/legacy query context mismatch");
      if(i && (lut_buffer!=workspace.context.lookup.data() || phi_buffer!=workspace.phi.data() ||
               pad_buffer!=workspace.padded.data()))
        throw std::runtime_error("terminal workspace reallocated between queries");
      lut_buffer=workspace.context.lookup.data(); phi_buffer=workspace.phi.data();
      pad_buffer=workspace.padded.data();
      for (size_t k=0;k<query.lookup.size();++k)
        if (std::abs(query.lookup[k]-reference.lookup[k]) > 3e-6f+3e-5f*std::abs(reference.lookup[k]))
          throw std::runtime_error("SIMD/reference query LUT mismatch");
      for (size_t count : {size_t(0),size_t(1),size_t(15),size_t(16),size_t(17),size_t(31),size_t(33)}) {
        if (count>n) continue;
        std::vector<uint32_t> random_ids(count);
        for(size_t j=0;j<count;++j) random_ids[j]=(j*7919+1237)%ji.node_count();
        ji.ComputeCorrectionsBatch(query,random_ids.data(),count,scores.data());
        for(size_t j=0;j<count;++j)
          if(scores[j]!=ji.Correction(query,random_ids[j])) throw std::runtime_error("SIMD tail/random ID mismatch");
      }
      ji.ComputeCorrectionsBatch(query,ids.data(),n,scores.data());
      for (uint32_t j=0;j<n;++j)
        if (scores[j]!=ji.Correction(query,j)) throw std::runtime_error("scalar/batch mismatch");
      out.write(reinterpret_cast<const char*>(scores.data()),n*sizeof(float));
    }
    if (!out) throw std::runtime_error("cannot write parity output");
    std::cout << ji.scoring_kernel() << " bits=" << ji.budget_bits()
              << " runtime_cache_bytes=" << ji.runtime_cache_bytes() << '\n';
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
