#include <boost/program_options.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "ours/drb_model.h"

namespace po = boost::program_options;

namespace {

struct Result {
    uint32_t fanout = 0;
    uint64_t anchors = 0;
    uint64_t comparisons = 0;
    double direct_seconds = 0.0;
    double prepared_seconds = 0.0;
    double speedup = 0.0;
    double prepared_batch_seconds = 0.0;
    double batch_speedup = 0.0;
    double max_absolute_batch_score_error = 0.0;
    double max_absolute_score_error = 0.0;
    double mean_absolute_score_error = 0.0;
    double checksum_difference = 0.0;
};

double SecondsSince(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start)
        .count();
}

void WriteJson(const std::string& path, const std::vector<Result>& results,
               uint64_t node_count, uint64_t target_comparisons) {
    const std::filesystem::path output(path);
    if (!output.parent_path().empty()) {
        std::filesystem::create_directories(output.parent_path());
    }
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("cannot write benchmark JSON: " + path);
    stream << std::setprecision(17)
           << "{\n  \"version\": 1,\n"
           << "  \"method\": \"real_code_base_pair_anchor_benchmark\",\n"
           << "  \"nodes\": " << node_count << ",\n"
           << "  \"target_comparisons_per_fanout\": " << target_comparisons
           << ",\n  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const Result& result = results[i];
        stream << "    {\n"
               << "      \"fanout\": " << result.fanout << ",\n"
               << "      \"anchors\": " << result.anchors << ",\n"
               << "      \"comparisons\": " << result.comparisons << ",\n"
               << "      \"direct_seconds\": " << result.direct_seconds
               << ",\n"
               << "      \"prepared_seconds\": " << result.prepared_seconds
               << ",\n"
               << "      \"prepared_speedup\": " << result.speedup << ",\n"
               << "      \"prepared_batch_seconds\": "
               << result.prepared_batch_seconds << ",\n"
               << "      \"prepared_batch_speedup\": "
               << result.batch_speedup << ",\n"
               << "      \"max_absolute_score_error\": "
               << result.max_absolute_score_error << ",\n"
               << "      \"mean_absolute_score_error\": "
               << result.mean_absolute_score_error << ",\n"
               << "      \"max_absolute_batch_score_error\": "
               << result.max_absolute_batch_score_error << ",\n"
               << "      \"checksum_difference\": "
               << result.checksum_difference << "\n    }"
               << (i + 1 == results.size() ? "\n" : ",\n");
    }
    stream << "  ]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path;
    std::string codes_path;
    std::string side_codes_path;
    std::string output_path;
    uint64_t target_comparisons = 1000000;
    uint64_t seed = 20260904;

    po::options_description options("Arguments");
    options.add_options()
        ("help,h", "Print command-line help")
        ("model", po::value<std::string>(&model_path)->required(),
         "Exported primary-PQ plus residual-PQ online model")
        ("codes", po::value<std::string>(&codes_path)->required(),
         "Primary PQ codes")
        ("side_codes", po::value<std::string>(&side_codes_path)->required(),
         "Residual-PQ side codes")
        ("out_json", po::value<std::string>(&output_path)->required(),
         "Benchmark JSON output")
        ("comparisons", po::value<uint64_t>(&target_comparisons)
                            ->default_value(target_comparisons),
         "Target comparisons for each fanout")
        ("seed", po::value<uint64_t>(&seed)->default_value(seed),
         "Random seed");

    try {
        po::variables_map variables;
        po::store(po::parse_command_line(argc, argv, options), variables);
        if (variables.count("help")) {
            std::cout << options;
            return 0;
        }
        po::notify(variables);
        if (target_comparisons == 0) {
            throw std::invalid_argument("comparisons must be positive");
        }

        ours::DrbPqIndex index =
            ours::DrbPqIndex::Load(model_path, codes_path, side_codes_path);
        index.PrepareResidualPqBasePairLookup();
        std::mt19937_64 generator(seed);
        std::uniform_int_distribution<uint32_t> nodes(
            0, static_cast<uint32_t>(index.node_count() - 1));
        const std::vector<uint32_t> fanouts = {
            32, 64, 128, 256, 512, 1024, 2048, 4096};
        std::vector<Result> results;
        results.reserve(fanouts.size());

        for (uint32_t fanout : fanouts) {
            Result result;
            result.fanout = fanout;
            result.anchors = std::max<uint64_t>(
                16, (target_comparisons + fanout - 1) / fanout);
            result.comparisons = result.anchors * fanout;
            std::vector<uint32_t> anchors(result.anchors);
            std::vector<uint32_t> candidates(result.comparisons);
            for (uint32_t& anchor : anchors) anchor = nodes(generator);
            for (uint32_t& candidate : candidates) candidate = nodes(generator);

            std::vector<float> direct_scores(result.comparisons);
            double direct_checksum = 0.0;
            auto started = std::chrono::steady_clock::now();
            for (uint64_t anchor_index = 0; anchor_index < result.anchors;
                 ++anchor_index) {
                const uint64_t begin = anchor_index * fanout;
                for (uint32_t offset = 0; offset < fanout; ++offset) {
                    const uint64_t position = begin + offset;
                    const float score = index.ResidualPqBasePairScore(
                        candidates[position], anchors[anchor_index]);
                    direct_scores[position] = score;
                    direct_checksum += score;
                }
            }
            result.direct_seconds = SecondsSince(started);

            ours::ResidualPqBaseAnchorContext context;
            std::vector<float> prepared_scores(result.comparisons);
            double prepared_checksum = 0.0;
            started = std::chrono::steady_clock::now();
            for (uint64_t anchor_index = 0; anchor_index < result.anchors;
                 ++anchor_index) {
                index.PrepareResidualPqBaseAnchor(anchors[anchor_index],
                                                  &context);
                const uint64_t begin = anchor_index * fanout;
                for (uint32_t offset = 0; offset < fanout; ++offset) {
                    const uint64_t position = begin + offset;
                    const float score =
                        index.ResidualPqPreparedBaseAnchorScore(
                            context, candidates[position]);
                    prepared_scores[position] = score;
                    prepared_checksum += score;
                }
            }
            result.prepared_seconds = SecondsSince(started);
            result.speedup = result.direct_seconds /
                             std::max(result.prepared_seconds, 1e-12);

            std::vector<float> prepared_batch_distances(result.comparisons);
            started = std::chrono::steady_clock::now();
            for (uint64_t anchor_index = 0; anchor_index < result.anchors;
                 ++anchor_index) {
                index.PrepareResidualPqBaseAnchor(anchors[anchor_index],
                                                  &context);
                const uint64_t begin = anchor_index * fanout;
                index.ResidualPqPreparedBaseAnchorDistanceBatch(
                    context, candidates.data() + begin, fanout,
                    prepared_batch_distances.data() + begin);
            }
            result.prepared_batch_seconds = SecondsSince(started);
            result.batch_speedup = result.direct_seconds /
                std::max(result.prepared_batch_seconds, 1e-12);

            double total_error = 0.0;
            for (uint64_t position = 0; position < result.comparisons;
                 ++position) {
                const double scalar_error = std::abs(
                    static_cast<double>(prepared_scores[position]) -
                    direct_scores[position]);
                total_error += scalar_error;
                result.max_absolute_score_error = std::max(
                    result.max_absolute_score_error, scalar_error);
                const double batch_error = std::abs(
                    -static_cast<double>(prepared_batch_distances[position]) -
                    direct_scores[position]);
                result.max_absolute_batch_score_error = std::max(
                    result.max_absolute_batch_score_error, batch_error);
            }
            result.mean_absolute_score_error =
                total_error / std::max<uint64_t>(result.comparisons, 1);
            result.checksum_difference = prepared_checksum - direct_checksum;
            results.push_back(result);
            std::cout << "fanout=" << fanout
                      << " speedup=" << result.speedup
                      << " batch_speedup=" << result.batch_speedup
                      << " max_error=" << result.max_absolute_score_error
                      << std::endl;
        }
        WriteJson(output_path, results, index.node_count(),
                  target_comparisons);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "base-pair benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
