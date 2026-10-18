#include "ours/mapped_fbin.h"
#include "index_bipartite.h"
#include <omp.h>
#include <iostream>
#include <map>
#include <string>

int main(int argc, char** argv) {
  try {
    std::map<std::string, std::string> options;
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 == argc) throw std::invalid_argument("missing option value");
      options[argv[i]] = argv[i + 1];
    }
    const ours::MappedFbin base(options.at("--base"));
    const ours::MappedFbin queries(options.at("--queries"));
    const auto threads = std::stoul(options.at("--threads"));
    if (!threads || base.dim() != queries.dim() || base.rows() + queries.rows() > UINT32_MAX)
      throw std::invalid_argument("invalid FP build dimensions, threads or node count");
    omp_set_num_threads(threads);
    efanna2e::Parameters params;
    params.Set<uint32_t>("M_sq", std::stoul(options.at("--topk")));
    params.Set<uint32_t>("M_pjbp", std::stoul(options.at("--M_pjbp")));
    params.Set<uint32_t>("L_pjpq", std::stoul(options.at("--L_pjpq")));
    params.Set<uint32_t>("num_threads", threads);
    efanna2e::IndexBipartite index(base.dim(), base.rows() + queries.rows(),
                                  efanna2e::INNER_PRODUCT, nullptr);
    index.LoadLearnBaseKNN(options.at("--knn").c_str());
    index.BuildRoarGraph(queries.rows(), queries.Row(0), base.rows(), base.Row(0), params);
    index.SaveProjectionGraph(options.at("--graph").c_str());
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
