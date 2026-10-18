// Offline fixed-pair audit, deliberately using the fast search adapters.
#define main production_search_main
#include "../../../experimental/traversal-fast-v1/src/primary_ji_search_main.cpp"
#undef main

template<class Primary>
int AuditScores(const Arguments &args, const Primary &primary, const std::string &path) {
  ours::MappedFbin queries(args.queries);
  std::ifstream in(path, std::ios::binary);
  uint32_t rows = 0, cols = 0;
  in.read(reinterpret_cast<char*>(&rows), 4);
  in.read(reinterpret_cast<char*>(&cols), 4);
  if (!in || !rows || !cols || rows > queries.rows())
    throw std::runtime_error("invalid audit ID header");
  std::vector<uint32_t> ids(static_cast<size_t>(rows) * cols);
  in.read(reinterpret_cast<char*>(ids.data()), ids.size() * sizeof(uint32_t));
  if (!in) throw std::runtime_error("truncated audit IDs");
  std::unique_ptr<ours::JiOnlineIndex> ji;
  if (!args.disable_ji) ji = std::make_unique<ours::JiOnlineIndex>(
      ours::JiOnlineIndex::Load(args.ji_model, args.ji_side_codes));
  std::vector<float> scores(ids.size());
  ours::JiQueryWorkspace ji_query;
  for (uint32_t q = 0; q < rows; ++q) {
    auto context = primary.Prepare(queries.Row(q), queries.dim());
    const uint32_t *row = ids.data() + static_cast<size_t>(q) * cols;
    float *out = scores.data() + static_cast<size_t>(q) * cols;
    for (uint32_t k = 0; k < cols; ++k)
      if (row[k] >= primary.nodes()) throw std::runtime_error("audit node out of range");
    primary.ScoreBatch(context, row, cols, out);
    if (ji) {
      const auto &prepared = ji->PrepareQueryReused(queries.Row(q), queries.dim(), ji_query, false);
      ji->AddCorrectionsBatch(prepared, row, cols, out);
    }
  }
  WriteMatrix(args.out_prefix + ".scores.fbin", rows, cols, scores);
  return 0;
}

int main(int argc, char **argv) {
  try {
    std::vector<char*> filtered{argv[0]};
    std::string ids;
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "--audit-ids") {
        if (++i == argc) throw std::runtime_error("missing audit ID path");
        ids = argv[i];
      } else filtered.push_back(argv[i]);
    }
    auto args = ParseArguments(static_cast<int>(filtered.size()), filtered.data());
    if (ids.empty()) throw std::runtime_error("--audit-ids required");
    if (args.primary_kind == "fp32") return AuditScores(args, Fp32Primary(args.primary_codes), ids);
    if (args.primary_kind == "extended-rabitq")
      return AuditScores(args, RabitqPrimary(args.primary_codes, args.rabitq_config), ids);
    if (args.primary_kind == "saq")
      return AuditScores(args, SaqPrimary(args.primary_model, args.primary_codes), ids);
    return AuditScores(args, GenericPrimary(args.primary_model, args.primary_codes), ids);
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
