// Deliberately compile the exact production adapters into the offline auditor.
// Never approximate a calibrated primary via an unverified reconstruction.
#define main production_search_main
#include "../../../src/primary_ji_search_main.cpp"
#undef main

template<class Primary>
int Audit(const Arguments &args, const Primary &primary, const std::string &ids_path) {
  ours::MappedFbin queries(args.queries);
  std::ifstream in(ids_path,std::ios::binary);
  uint32_t rows=0, cols=0;
  in.read(reinterpret_cast<char*>(&rows),4); in.read(reinterpret_cast<char*>(&cols),4);
  if(!in || !rows || !cols || rows>queries.rows()) throw std::runtime_error("bad audit ID header");
  std::vector<uint32_t> ids(static_cast<size_t>(rows)*cols);
  in.read(reinterpret_cast<char*>(ids.data()),ids.size()*4);
  if(!in) throw std::runtime_error("truncated audit IDs");
  std::unique_ptr<ours::JiOnlineIndex> ji;
  if(!args.disable_ji) ji=std::make_unique<ours::JiOnlineIndex>(
      ours::JiOnlineIndex::Load(args.ji_model,args.ji_side_codes));
  std::vector<float> scores(ids.size()), corrections(cols);
  for(uint32_t r=0;r<rows;++r) {
    auto context=primary.Prepare(queries.Row(r),queries.dim());
    const uint32_t *row=ids.data()+static_cast<size_t>(r)*cols;
    for(uint32_t k=0;k<cols;++k) if(row[k]>=primary.nodes()) throw std::runtime_error("audit node out of bounds");
    float *out=scores.data()+static_cast<size_t>(r)*cols;
    primary.ScoreBatch(context,row,cols,out);
    if(ji) {
      auto q=ji->PrepareQueryZeroPadded(queries.Row(r),queries.dim());
      ji->ComputeCorrectionsBatch(q,row,cols,corrections.data());
      for(uint32_t k=0;k<cols;++k) out[k]+=corrections[k];
    }
  }
  WriteMatrix(args.out_prefix+".scores.fbin",rows,cols,scores);
  return 0;
}

int main(int argc,char **argv) {
  try {
    std::vector<char*> filtered{argv[0]}; std::string ids;
    for(int i=1;i<argc;++i) {
      if(std::string(argv[i])=="--audit-ids") {
        if(++i==argc) throw std::runtime_error("missing audit IDs"); ids=argv[i];
      } else filtered.push_back(argv[i]);
    }
    auto args=ParseArguments(static_cast<int>(filtered.size()),filtered.data());
    if(ids.empty()) throw std::runtime_error("--audit-ids is required");
    if(args.primary_kind=="fp32") return Audit(args,Fp32Primary(args.primary_codes),ids);
    if(args.primary_kind=="extended-rabitq") return Audit(args,RabitqPrimary(args.primary_codes,args.rabitq_config),ids);
    if(args.primary_kind=="saq") return Audit(args,SaqPrimary(args.primary_model,args.primary_codes),ids);
    return Audit(args,GenericPrimary(args.primary_model,args.primary_codes),ids);
  } catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
