#include <omp.h>
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include "efanna2e/parameters.h"
#include "index_bipartite.h"
#include "ours/drb_model.h"

namespace po = boost::program_options;

namespace {

struct MatrixHeader {
    uint32_t rows = 0;
    uint32_t dim = 0;
};

struct PairLookupCheck {
    uint32_t pairs = 0;
    double mean_absolute_error = 0.0;
    double max_absolute_error = 0.0;
    double prepared_mean_absolute_error = 0.0;
    double prepared_max_absolute_error = 0.0;
};

MatrixHeader ReadMatrixHeader(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open matrix: " + path);
    MatrixHeader header;
    input.read(reinterpret_cast<char*>(&header.rows), sizeof(header.rows));
    input.read(reinterpret_cast<char*>(&header.dim), sizeof(header.dim));
    if (!input || header.rows == 0 || header.dim == 0) {
        throw std::runtime_error("invalid matrix header: " + path);
    }
    return header;
}

std::string JsonEscape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (char ch : value) {
        switch (ch) {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped += ch; break;
        }
    }
    return escaped;
}

uint64_t PeakRssBytes() {
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
#if defined(__APPLE__)
    return static_cast<uint64_t>(usage.ru_maxrss);
#else
    return static_cast<uint64_t>(usage.ru_maxrss) * 1024ULL;
#endif
}

class ResidualPqBuildOracle final : public efanna2e::BaseBuildDistanceOracle {
   public:
    ResidualPqBuildOracle(const std::string& model_path,
                          const std::string& codes_path,
                          const std::string& side_codes_path)
        : index_(ours::DrbPqIndex::Load(model_path, codes_path,
                                        side_codes_path)) {
        if (index_.refinement_kind() != ours::RefinementKind::kResidualPq) {
            throw std::invalid_argument(
                "quantized RoarGraph builder requires a residual-PQ online model");
        }
    }

    void PreparePairLookup(uint32_t threads) {
        index_.PrepareResidualPqBasePairLookup();
        workspaces_.resize(std::max<uint32_t>(threads, 1U));
    }

    uint64_t NodeCount() const override { return index_.node_count(); }
    uint32_t Dimension() const override { return index_.dim(); }

    float Compare(uint32_t lhs, uint32_t rhs) const override {
        ++CurrentWorkspace().direct_comparisons;
        return index_.ResidualPqBasePairDistance(lhs, rhs);
    }

    void PrepareAnchor(uint32_t anchor) const override {
        ThreadWorkspace& workspace = CurrentWorkspace();
        index_.PrepareResidualPqBaseAnchor(anchor, &workspace.context);
        ++workspace.prepared_anchors;
    }

    float ComparePrepared(uint32_t candidate,
                          uint32_t anchor) const override {
        ThreadWorkspace& workspace = CurrentWorkspace();
        if (!workspace.context.ready ||
            workspace.context.anchor_id != anchor) {
            throw std::runtime_error(
                "prepared base-pair anchor does not match the active search");
        }
        ++workspace.prepared_comparisons;
        return index_.ResidualPqPreparedBaseAnchorDistance(workspace.context,
                                                           candidate);
    }

    void ComparePreparedBatch(const uint32_t* candidates,
                              size_t count,
                              uint32_t anchor,
                              float* distances) const override {
        ThreadWorkspace& workspace = CurrentWorkspace();
        if (!workspace.context.ready ||
            workspace.context.anchor_id != anchor) {
            throw std::runtime_error(
                "prepared base-pair anchor does not match the active batch");
        }
        workspace.prepared_comparisons += count;
        index_.ResidualPqPreparedBaseAnchorDistanceBatch(
            workspace.context, candidates, count, distances);
    }

    uint32_t FindEntryPoint() const override {
        const std::vector<float> center = index_.MeanResidualPqReconstruction();
        uint32_t global_best = 0;
        double global_distance = std::numeric_limits<double>::infinity();

#pragma omp parallel
        {
            std::vector<float> reconstruction(index_.dim());
            uint32_t local_best = 0;
            double local_distance = std::numeric_limits<double>::infinity();
#pragma omp for schedule(static)
            for (uint64_t node = 0; node < index_.node_count(); ++node) {
                index_.ReconstructResidualPqNode(
                    static_cast<uint32_t>(node), reconstruction.data(),
                    index_.dim());
                double distance = 0.0;
                for (uint32_t coordinate = 0; coordinate < index_.dim();
                     ++coordinate) {
                    const double delta =
                        static_cast<double>(center[coordinate]) -
                        reconstruction[coordinate];
                    distance += delta * delta;
                }
                if (distance < local_distance ||
                    (distance == local_distance && node < local_best)) {
                    local_distance = distance;
                    local_best = static_cast<uint32_t>(node);
                }
            }
#pragma omp critical
            {
                if (local_distance < global_distance ||
                    (local_distance == global_distance &&
                     local_best < global_best)) {
                    global_distance = local_distance;
                    global_best = local_best;
                }
            }
        }
        entry_point_ = global_best;
        entry_point_squared_distance_ = global_distance;
        return global_best;
    }

    uint64_t PairLookupBytes() const { return index_.BasePairLookupBytes(); }
    uint64_t PreparedAnchorBytesPerThread() const {
        return static_cast<uint64_t>(index_.dim()) * sizeof(float) +
               (static_cast<uint64_t>(index_.subspaces()) *
                    index_.codewords() +
                static_cast<uint64_t>(index_.residual_groups()) *
                    index_.residual_codewords()) * sizeof(float);
    }
    uint64_t PreparedAnchorCount() const {
        uint64_t count = 0;
        for (const ThreadWorkspace& workspace : workspaces_) {
            count += workspace.prepared_anchors;
        }
        return count;
    }
    uint64_t PreparedComparisonCount() const {
        uint64_t count = 0;
        for (const ThreadWorkspace& workspace : workspaces_) {
            count += workspace.prepared_comparisons;
        }
        return count;
    }
    uint64_t DirectComparisonCount() const {
        uint64_t count = 0;
        for (const ThreadWorkspace& workspace : workspaces_) {
            count += workspace.direct_comparisons;
        }
        return count;
    }
    PairLookupCheck CheckPairLookup(uint32_t pairs, uint64_t seed) const {
        PairLookupCheck result;
        result.pairs = pairs;
        if (pairs == 0) return result;
        std::mt19937_64 generator(seed);
        std::uniform_int_distribution<uint32_t> nodes(
            0, static_cast<uint32_t>(index_.node_count() - 1));
        std::vector<float> lhs(index_.dim());
        std::vector<float> rhs(index_.dim());
        double total_error = 0.0;
        double prepared_total_error = 0.0;
        ours::ResidualPqBaseAnchorContext prepared;
        for (uint32_t sample = 0; sample < pairs; ++sample) {
            const uint32_t lhs_id = nodes(generator);
            const uint32_t rhs_id = nodes(generator);
            index_.ReconstructResidualPqNode(lhs_id, lhs.data(), index_.dim());
            index_.ReconstructResidualPqNode(rhs_id, rhs.data(), index_.dim());
            double reconstructed_score = 0.0;
            for (uint32_t coordinate = 0; coordinate < index_.dim();
                 ++coordinate) {
                reconstructed_score += static_cast<double>(lhs[coordinate]) *
                                       rhs[coordinate];
            }
            const double error = std::abs(
                static_cast<double>(index_.ResidualPqBasePairScore(lhs_id, rhs_id)) -
                reconstructed_score);
            total_error += error;
            result.max_absolute_error =
                std::max(result.max_absolute_error, error);
            index_.PrepareResidualPqBaseAnchor(rhs_id, &prepared);
            const double prepared_error = std::abs(
                static_cast<double>(index_.ResidualPqPreparedBaseAnchorScore(
                    prepared, lhs_id)) - reconstructed_score);
            prepared_total_error += prepared_error;
            result.prepared_max_absolute_error = std::max(
                result.prepared_max_absolute_error, prepared_error);
        }
        result.mean_absolute_error = total_error / pairs;
        result.prepared_mean_absolute_error = prepared_total_error / pairs;
        return result;
    }
    uint32_t EntryPoint() const { return entry_point_; }
    double EntryPointSquaredDistance() const {
        return entry_point_squared_distance_;
    }
    uint32_t PrimarySubspaces() const { return index_.subspaces(); }
    uint32_t ResidualGroups() const { return index_.residual_groups(); }

   private:
    struct alignas(64) ThreadWorkspace {
        ours::ResidualPqBaseAnchorContext context;
        uint64_t prepared_anchors = 0;
        uint64_t prepared_comparisons = 0;
        uint64_t direct_comparisons = 0;
    };

    ThreadWorkspace& CurrentWorkspace() const {
        const uint32_t thread = omp_in_parallel()
                                    ? static_cast<uint32_t>(omp_get_thread_num())
                                    : 0U;
        if (thread >= workspaces_.size()) {
            throw std::runtime_error(
                "base-pair workspace count is smaller than OpenMP threads");
        }
        return workspaces_[thread];
    }

    ours::DrbPqIndex index_;
    mutable std::vector<ThreadWorkspace> workspaces_;
    mutable uint32_t entry_point_ = 0;
    mutable double entry_point_squared_distance_ = 0.0;
};

void WriteSummary(const std::string& path,
                  const std::string& model_path,
                  const std::string& codes_path,
                  const std::string& side_codes_path,
                  const std::string& query_path,
                  const std::string& knn_path,
                  const std::string& graph_path,
                  const ResidualPqBuildOracle& oracle,
                  const PairLookupCheck& pair_check,
                  uint32_t query_count,
                  uint32_t topk,
                  uint32_t threads,
                  double model_load_seconds,
                  double pair_lookup_seconds,
                  double graph_build_seconds) {
    const std::filesystem::path output(path);
    if (!output.parent_path().empty()) {
        std::filesystem::create_directories(output.parent_path());
    }
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("cannot write summary: " + path);
    const auto absolute = [](const std::string& value) {
        return std::filesystem::absolute(value).string();
    };
    stream << std::setprecision(17)
           << "{\n"
           << "  \"version\": 1,\n"
           << "  \"method\": \"code_native_pq_rpq_roargraph_build\",\n"
           << "  \"distance\": \"negative_inner_product_of_pq_plus_rpq_reconstructions\",\n"
           << "  \"base_vectors_loaded\": false,\n"
           << "  \"materialized_base_vectors\": false,\n"
           << "  \"query_base_relation\": \"exact_all_database_pq_rpq_knn\",\n"
           << "  \"base_base_comparison\": "
              "\"prepared_anchor_lut_for_projection_search_plus_direct_pair_lookup_for_pruning\",\n"
           << "  \"nodes\": " << oracle.NodeCount() << ",\n"
           << "  \"dim\": " << oracle.Dimension() << ",\n"
           << "  \"queries\": " << query_count << ",\n"
           << "  \"query_base_topk\": " << topk << ",\n"
           << "  \"threads\": " << threads << ",\n"
           << "  \"primary_subspaces\": " << oracle.PrimarySubspaces()
           << ",\n"
           << "  \"residual_groups\": " << oracle.ResidualGroups() << ",\n"
           << "  \"entry_point\": " << oracle.EntryPoint() << ",\n"
           << "  \"entry_point_squared_distance\": "
           << oracle.EntryPointSquaredDistance() << ",\n"
           << "  \"pair_lookup_bytes\": " << oracle.PairLookupBytes() << ",\n"
           << "  \"prepared_anchor_bytes_per_thread\": "
           << oracle.PreparedAnchorBytesPerThread() << ",\n"
           << "  \"base_pair_comparisons\": {\n"
           << "    \"prepared_anchors\": " << oracle.PreparedAnchorCount()
           << ",\n"
           << "    \"prepared_anchor_comparisons\": "
           << oracle.PreparedComparisonCount() << ",\n"
           << "    \"direct_pair_comparisons\": "
           << oracle.DirectComparisonCount() << "\n"
           << "  },\n"
           << "  \"pair_lookup_self_check\": {\n"
           << "    \"pairs\": " << pair_check.pairs << ",\n"
           << "    \"mean_absolute_error\": "
           << pair_check.mean_absolute_error << ",\n"
           << "    \"max_absolute_error\": "
           << pair_check.max_absolute_error << ",\n"
           << "    \"prepared_mean_absolute_error\": "
           << pair_check.prepared_mean_absolute_error << ",\n"
           << "    \"prepared_max_absolute_error\": "
           << pair_check.prepared_max_absolute_error << "\n"
           << "  },\n"
           << "  \"peak_rss_bytes\": " << PeakRssBytes() << ",\n"
           << "  \"graph_bytes\": " << std::filesystem::file_size(graph_path)
           << ",\n"
           << "  \"seconds\": {\n"
           << "    \"model_and_codes_load\": " << model_load_seconds << ",\n"
           << "    \"pair_lookup_build\": " << pair_lookup_seconds << ",\n"
           << "    \"roargraph_build\": " << graph_build_seconds << "\n"
           << "  },\n"
           << "  \"throughput\": {\n"
           << "    \"graph_nodes_per_second\": "
           << oracle.NodeCount() / std::max(graph_build_seconds, 1e-12)
           << "\n"
           << "  },\n"
           << "  \"files\": {\n"
           << "    \"online_model\": \"" << JsonEscape(absolute(model_path))
           << "\",\n"
           << "    \"primary_codes\": \"" << JsonEscape(absolute(codes_path))
           << "\",\n"
           << "    \"residual_side_codes\": \""
           << JsonEscape(absolute(side_codes_path)) << "\",\n"
           << "    \"sampled_queries\": \"" << JsonEscape(absolute(query_path))
           << "\",\n"
           << "    \"query_base_knn\": \"" << JsonEscape(absolute(knn_path))
           << "\",\n"
           << "    \"graph\": \"" << JsonEscape(absolute(graph_path)) << "\"\n"
           << "  }\n"
           << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path;
    std::string codes_path;
    std::string side_codes_path;
    std::string sampled_query_path;
    std::string knn_path;
    std::string graph_path;
    std::string summary_path;
    uint32_t m_sq = 100;
    uint32_t m_pjbp = 35;
    uint32_t l_pjpq = 500;
    uint32_t threads = static_cast<uint32_t>(omp_get_num_procs());
    uint32_t self_check_pairs = 1024;

    po::options_description options("Arguments");
    options.add_options()
        ("help,h", "Print command-line help")
        ("model", po::value<std::string>(&model_path)->required(),
         "Exported primary-PQ plus residual-PQ online model")
        ("codes", po::value<std::string>(&codes_path)->required(),
         "Primary PQ codes")
        ("side_codes", po::value<std::string>(&side_codes_path)->required(),
         "Residual-PQ side codes")
        ("sampled_query_data_path",
         po::value<std::string>(&sampled_query_path)->required(),
         "Sampled build-query fbin (metadata only)")
        ("learn_base_nn_path", po::value<std::string>(&knn_path)->required(),
         "Query-to-base KNN produced directly from PQ/RPQ codes")
        ("projection_index_save_path",
         po::value<std::string>(&graph_path)->required(),
         "Output RoarGraph projection index")
        ("summary_path", po::value<std::string>(&summary_path)->default_value(""),
         "Optional JSON build summary")
        ("M_sq", po::value<uint32_t>(&m_sq)->default_value(100),
         "Query-to-base neighbors consumed per build query")
        ("M_pjbp", po::value<uint32_t>(&m_pjbp)->default_value(35),
         "Projection graph degree")
        ("L_pjpq", po::value<uint32_t>(&l_pjpq)->default_value(500),
         "Projection construction search width")
        ("num_threads,T", po::value<uint32_t>(&threads)->default_value(threads),
         "OpenMP construction threads")
        ("self_check_pairs",
         po::value<uint32_t>(&self_check_pairs)->default_value(1024),
         "Random direct-lookup versus reconstruction parity checks");

    try {
        po::variables_map variables;
        po::store(po::parse_command_line(argc, argv, options), variables);
        if (variables.count("help")) {
            std::cout << options;
            return 0;
        }
        po::notify(variables);
        if (threads == 0 || m_sq == 0 || m_pjbp == 0 || l_pjpq == 0) {
            throw std::invalid_argument("build parameters must be positive");
        }

        const MatrixHeader query_header = ReadMatrixHeader(sampled_query_path);
        const MatrixHeader knn_header = ReadMatrixHeader(knn_path);
        if (query_header.rows != knn_header.rows || knn_header.dim < m_sq) {
            throw std::invalid_argument(
                "sampled-query and query-base KNN headers do not match");
        }

        const auto load_started = std::chrono::steady_clock::now();
        ResidualPqBuildOracle oracle(model_path, codes_path, side_codes_path);
        const double model_load_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          load_started).count();
        if (query_header.dim != oracle.Dimension()) {
            throw std::invalid_argument(
                "sampled-query dimension does not match the quantized model");
        }

        const auto lookup_started = std::chrono::steady_clock::now();
        oracle.PreparePairLookup(threads);
        const double pair_lookup_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          lookup_started).count();
        const PairLookupCheck pair_check =
            oracle.CheckPairLookup(self_check_pairs, 20260903);
        if (pair_check.max_absolute_error > 1e-6 ||
            pair_check.prepared_max_absolute_error > 1e-6) {
            throw std::runtime_error(
                "PQ/RPQ pair lookup failed reconstruction parity check");
        }

        const std::filesystem::path graph_output(graph_path);
        if (!graph_output.parent_path().empty()) {
            std::filesystem::create_directories(graph_output.parent_path());
        }
        efanna2e::Parameters parameters;
        parameters.Set<uint32_t>("M_sq", m_sq);
        parameters.Set<uint32_t>("M_pjbp", m_pjbp);
        parameters.Set<uint32_t>("L_pjpq", l_pjpq);
        parameters.Set<uint32_t>("num_threads", threads);
        omp_set_num_threads(static_cast<int>(threads));

        efanna2e::IndexBipartite graph(
            oracle.Dimension(), oracle.NodeCount() + query_header.rows,
            efanna2e::INNER_PRODUCT, nullptr);
        graph.SetBaseBuildDistanceOracle(&oracle);
        graph.LoadLearnBaseKNN(knn_path.c_str());

        const auto build_started = std::chrono::steady_clock::now();
        graph.BuildRoarGraph(query_header.rows, nullptr, oracle.NodeCount(),
                             nullptr, parameters);
        const double graph_build_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          build_started).count();
        graph.SaveProjectionGraph(graph_path.c_str());

        if (!summary_path.empty()) {
            WriteSummary(summary_path, model_path, codes_path, side_codes_path,
                         sampled_query_path, knn_path, graph_path, oracle,
                         pair_check, query_header.rows, m_sq, threads,
                         model_load_seconds, pair_lookup_seconds,
                         graph_build_seconds);
        }
        std::cout << "code-native RoarGraph build complete\n"
                  << "graph: " << graph_path << '\n'
                  << "base vectors loaded: false\n"
                  << "materialized base vectors: false\n"
                  << "pair lookup bytes: " << oracle.PairLookupBytes() << '\n'
                  << "build seconds: " << graph_build_seconds << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "quantized RoarGraph build failed: " << error.what()
                  << std::endl;
        return 1;
    }
}
