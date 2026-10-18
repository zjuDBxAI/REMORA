#include "ours/drb_model.h"
#include "ours/mapped_fbin.h"
#include "ours/roar_graph.h"
#include "ours/search_kernel.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <sys/resource.h>

namespace {

struct Arguments {
    std::string graph;
    std::string base;
    std::string queries;
    std::string model;
    std::string codes;
    std::string side_bits;
    std::string mode = "drb";
    std::string out_prefix;
    std::string events_csv;
    std::string visited_csr;
    uint32_t topk = 10;
    uint32_t beam = 100;
    uint64_t query_start = 0;
    uint64_t query_limit = 0;
    uint64_t warmup_queries = 0;
    float min_exact_gap = 1e-8f;
    float protect_adc_margin = 0.02f;
    uint32_t max_inversions_per_source = 32;
    uint32_t max_protect_per_source = 32;
    uint32_t threads = 1;
};

void Usage(const char* program) {
    std::cerr
        << "Usage: " << program << " --graph FILE [--base FILE] --queries FILE\n"
        << "  --model FILE --codes FILE [--side-bits FILE]\n"
        << "  --mode exact|adc|drb|vdrb|rpq --topk K --beam L --out-prefix PATH\n"
        << "  [--events-csv FILE] [--query-start N] [--query-limit N]\n"
        << "  [--warmup-queries N] (excluded from outputs and timing)\n"
        << "  [--visited-csr FILE]\n"
        << "  [--min-exact-gap X] [--protect-adc-margin X]\n"
        << "  [--max-inversions-per-source N]\n"
        << "  [--max-protect-per-source N]\n"
        << "  [--threads N] (default: 1)\n"
        << "\n--base is required only for exact mode or event tracing. Pure ADC/RPQ\n"
        << "search deliberately runs from graph + compressed codes without FP32 vectors.\n";
}

std::string NextValue(int argc, char** argv, int& index) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value after ") + argv[index]);
    }
    return argv[++index];
}

Arguments ParseArguments(int argc, char** argv) {
    Arguments args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            Usage(argv[0]);
            std::exit(0);
        } else if (key == "--graph") {
            args.graph = NextValue(argc, argv, i);
        } else if (key == "--base") {
            args.base = NextValue(argc, argv, i);
        } else if (key == "--queries") {
            args.queries = NextValue(argc, argv, i);
        } else if (key == "--model") {
            args.model = NextValue(argc, argv, i);
        } else if (key == "--codes") {
            args.codes = NextValue(argc, argv, i);
        } else if (key == "--side-bits") {
            args.side_bits = NextValue(argc, argv, i);
        } else if (key == "--mode") {
            args.mode = NextValue(argc, argv, i);
        } else if (key == "--out-prefix") {
            args.out_prefix = NextValue(argc, argv, i);
        } else if (key == "--events-csv") {
            args.events_csv = NextValue(argc, argv, i);
        } else if (key == "--visited-csr") {
            args.visited_csr = NextValue(argc, argv, i);
        } else if (key == "--topk") {
            args.topk = static_cast<uint32_t>(std::stoul(NextValue(argc, argv, i)));
        } else if (key == "--beam") {
            args.beam = static_cast<uint32_t>(std::stoul(NextValue(argc, argv, i)));
        } else if (key == "--query-start") {
            args.query_start = std::stoull(NextValue(argc, argv, i));
        } else if (key == "--query-limit") {
            args.query_limit = std::stoull(NextValue(argc, argv, i));
        } else if (key == "--warmup-queries") {
            args.warmup_queries = std::stoull(NextValue(argc, argv, i));
        } else if (key == "--min-exact-gap") {
            args.min_exact_gap = std::stof(NextValue(argc, argv, i));
        } else if (key == "--protect-adc-margin") {
            args.protect_adc_margin = std::stof(NextValue(argc, argv, i));
        } else if (key == "--max-inversions-per-source") {
            args.max_inversions_per_source =
                static_cast<uint32_t>(std::stoul(NextValue(argc, argv, i)));
        } else if (key == "--max-protect-per-source") {
            args.max_protect_per_source =
                static_cast<uint32_t>(std::stoul(NextValue(argc, argv, i)));
        } else if (key == "--threads") {
            args.threads = static_cast<uint32_t>(std::stoul(NextValue(argc, argv, i)));
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }
    if (args.graph.empty() || args.queries.empty() ||
        args.model.empty() || args.codes.empty() || args.out_prefix.empty()) {
        throw std::invalid_argument("missing required arguments; use --help");
    }
    if (args.min_exact_gap < 0.0f || args.protect_adc_margin < 0.0f) {
        throw std::invalid_argument("event gap thresholds must be non-negative");
    }
    if (args.threads == 0) {
        throw std::invalid_argument("threads must be at least 1");
    }
    return args;
}

template <typename T>
void WriteMatrix(const std::string& path,
                 uint32_t rows,
                 uint32_t cols,
                 const std::vector<T>& values) {
    if (values.size() != static_cast<uint64_t>(rows) * cols) {
        throw std::logic_error("output matrix shape mismatch");
    }
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot create output: " + path);
    output.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    output.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!output) throw std::runtime_error("failed writing output: " + path);
}

void WriteVisitedCsr(const std::string& path,
                     uint64_t query_start,
                     const std::vector<uint64_t>& offsets,
                     const std::vector<uint32_t>& ids) {
    if (offsets.empty() || offsets.front() != 0 || offsets.back() != ids.size()) {
        throw std::logic_error("invalid visited-candidate CSR payload");
    }
    const uint64_t query_count = offsets.size() - 1;
    if (query_count > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("visited-candidate query count exceeds format limit");
    }
    const std::filesystem::path output_path(path);
    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot create visited-candidate CSR: " + path);
    const std::array<char, 8> magic = {'D', 'R', 'B', 'V', 'S', 'T', '0', '1'};
    const uint32_t version = 1;
    const uint32_t rows = static_cast<uint32_t>(query_count);
    const uint64_t total = ids.size();
    output.write(magic.data(), static_cast<std::streamsize>(magic.size()));
    output.write(reinterpret_cast<const char*>(&version), sizeof(version));
    output.write(reinterpret_cast<const char*>(&query_start), sizeof(query_start));
    output.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    output.write(reinterpret_cast<const char*>(&total), sizeof(total));
    output.write(reinterpret_cast<const char*>(offsets.data()),
                 static_cast<std::streamsize>(offsets.size() * sizeof(uint64_t)));
    output.write(reinterpret_cast<const char*>(ids.data()),
                 static_cast<std::streamsize>(ids.size() * sizeof(uint32_t)));
    if (!output) throw std::runtime_error("failed writing visited-candidate CSR: " + path);
}

uint64_t PeakResidentBytes() {
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
#if defined(__APPLE__)
    return static_cast<uint64_t>(usage.ru_maxrss);
#else
    return static_cast<uint64_t>(usage.ru_maxrss) * 1024ULL;
#endif
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments args = ParseArguments(argc, argv);
        const ours::ScoreMode mode = ours::ParseScoreMode(args.mode);
        const bool needs_base = mode == ours::ScoreMode::kExact || !args.events_csv.empty();
        if (needs_base && args.base.empty()) {
            throw std::invalid_argument(
                "--base is required for exact mode and event tracing");
        }
        std::unique_ptr<ours::MappedFbin> base;
        if (!args.base.empty()) {
            base = std::make_unique<ours::MappedFbin>(args.base);
        }
        const ours::MappedFbin queries(args.queries);
        const ours::RoarGraph graph = ours::RoarGraph::Load(args.graph);
        const ours::DrbPqIndex model =
            ours::DrbPqIndex::Load(args.model, args.codes, args.side_bits);
        if (queries.dim() != model.dim()) {
            throw std::runtime_error("query dimension does not match model");
        }
        const ours::MatrixView base_view = base ? base->view() : ours::MatrixView{};
        const ours::DrbRoarSearch kernel(graph, model, base_view);

        if (args.query_start >= queries.rows()) {
            throw std::invalid_argument("query-start is outside the query file");
        }
        const uint64_t available = queries.rows() - args.query_start;
        const uint64_t query_count =
            args.query_limit == 0 ? available : std::min(args.query_limit, available);
        if (query_count > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("output format only supports uint32 query counts");
        }

        const std::filesystem::path prefix(args.out_prefix);
        if (!prefix.parent_path().empty()) {
            std::filesystem::create_directories(prefix.parent_path());
        }
        std::ofstream event_output;
        if (!args.events_csv.empty()) {
            const std::filesystem::path event_path(args.events_csv);
            if (!event_path.parent_path().empty()) {
                std::filesystem::create_directories(event_path.parent_path());
            }
            event_output.open(args.events_csv);
            if (!event_output) {
                throw std::runtime_error("cannot create event CSV: " + args.events_csv);
            }
            event_output << "query_id,step,source,kind,u_id,v_id,exact_gap,adc_gap,trajectory_gap\n";
            event_output << std::setprecision(9);
        }

        ours::SearchOptions options;
        options.topk = args.topk;
        options.beam_width = args.beam;
        options.mode = mode;
        options.collect_visited = !args.visited_csr.empty();
        options.trace.enabled = event_output.is_open();
        options.trace.min_exact_gap = args.min_exact_gap;
        options.trace.protect_adc_margin = args.protect_adc_margin;
        options.trace.max_inversions_per_source = args.max_inversions_per_source;
        options.trace.max_protect_per_source = args.max_protect_per_source;

        const uint64_t warmup_count = std::min(args.warmup_queries, query_count);
        if (warmup_count > 0) {
            ours::SearchOptions warmup_options = options;
            warmup_options.collect_visited = false;
            warmup_options.trace.enabled = false;
            for (uint64_t local = 0; local < warmup_count; ++local) {
                const uint64_t query_id = args.query_start + local;
                static_cast<void>(kernel.Search(
                    queries.Row(query_id), queries.dim(), query_id, warmup_options));
            }
        }

        std::vector<uint32_t> ids;
        std::vector<float> scores;
        ids.reserve(query_count * args.topk);
        scores.reserve(query_count * args.topk);
        uint64_t comparisons = 0;
        uint64_t expansions = 0;
        uint64_t visited_nodes = 0;
        uint64_t candidate_pool_size = 0;
        uint64_t event_count = 0;
        std::array<std::array<uint64_t, 2>, 3> event_counts{};
        std::vector<uint64_t> visited_offsets;
        std::vector<uint32_t> visited_ids;
        if (options.collect_visited) visited_offsets.push_back(0);

        std::vector<ours::SearchResult> query_results(static_cast<size_t>(query_count));
        std::vector<double> query_latency_ms(static_cast<size_t>(query_count), 0.0);
        const uint32_t worker_count = static_cast<uint32_t>(
            std::min<uint64_t>(args.threads, query_count));
        std::atomic<uint64_t> next_query{0};
        std::atomic<uint64_t> completed_queries{0};
        std::atomic<bool> stop_workers{false};
        std::mutex output_mutex;
        std::mutex exception_mutex;
        std::exception_ptr worker_exception;

        const auto started = std::chrono::steady_clock::now();
        const auto worker = [&]() {
            try {
                while (!stop_workers.load(std::memory_order_relaxed)) {
                    const uint64_t local = next_query.fetch_add(1, std::memory_order_relaxed);
                    if (local >= query_count) break;
                    const uint64_t query_id = args.query_start + local;
                    const auto query_started = std::chrono::steady_clock::now();
                    query_results[static_cast<size_t>(local)] = kernel.Search(
                        queries.Row(query_id), queries.dim(), query_id, options);
                    query_latency_ms[static_cast<size_t>(local)] =
                        1000.0 * std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - query_started).count();
                    const uint64_t done =
                        completed_queries.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (done % 100 == 0 || done == query_count) {
                        const std::lock_guard<std::mutex> lock(output_mutex);
                        std::cout << "[search] " << done << '/' << query_count << '\n';
                    }
                }
            } catch (...) {
                stop_workers.store(true, std::memory_order_relaxed);
                const std::lock_guard<std::mutex> lock(exception_mutex);
                if (!worker_exception) worker_exception = std::current_exception();
            }
        };

        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (uint32_t worker_index = 0; worker_index < worker_count; ++worker_index) {
            workers.emplace_back(worker);
        }
        for (std::thread& thread : workers) thread.join();
        if (worker_exception) std::rethrow_exception(worker_exception);

        for (uint64_t local = 0; local < query_count; ++local) {
            ours::SearchResult& result = query_results[static_cast<size_t>(local)];
            ids.insert(ids.end(), result.ids.begin(), result.ids.end());
            scores.insert(scores.end(), result.scores.begin(), result.scores.end());
            comparisons += result.comparisons;
            expansions += result.expansions;
            visited_nodes += result.visited_nodes;
            candidate_pool_size += result.final_candidate_pool;
            if (options.collect_visited) {
                visited_ids.insert(visited_ids.end(), result.visited_ids.begin(),
                                   result.visited_ids.end());
                visited_offsets.push_back(visited_ids.size());
            }
            for (const ours::DecisionEvent& event : result.events) {
                event_counts[static_cast<size_t>(event.source)][static_cast<size_t>(event.kind)]++;
                event_count++;
                event_output << event.query_id << ',' << event.step << ','
                             << ours::EventSourceName(event.source) << ','
                             << ours::EventKindName(event.kind) << ',' << event.u_id << ','
                             << event.v_id << ',' << event.exact_gap << ',' << event.adc_gap
                             << ',' << event.trajectory_gap << '\n';
            }
        }
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        std::vector<double> sorted_latency = query_latency_ms;
        std::sort(sorted_latency.begin(), sorted_latency.end());
        const auto percentile = [&](double fraction) {
            const size_t index = static_cast<size_t>(
                fraction * static_cast<double>(sorted_latency.size() - 1U));
            return sorted_latency[index];
        };

        const uint32_t output_rows = static_cast<uint32_t>(query_count);
        WriteMatrix(args.out_prefix + ".ids.ibin", output_rows, args.topk, ids);
        WriteMatrix(args.out_prefix + ".scores.fbin", output_rows, args.topk, scores);
        if (options.collect_visited) {
            WriteVisitedCsr(args.visited_csr, args.query_start, visited_offsets, visited_ids);
        }
        std::ofstream summary(args.out_prefix + ".summary.json");
        if (!summary) throw std::runtime_error("cannot create search summary");
        summary << std::setprecision(10)
                << "{\n"
                << "  \"mode\": \"" << ours::ScoreModeName(mode) << "\",\n"
                << "  \"query_start\": " << args.query_start << ",\n"
                << "  \"queries\": " << query_count << ",\n"
                << "  \"topk\": " << args.topk << ",\n"
                << "  \"beam_width\": " << args.beam << ",\n"
                << "  \"base_vectors_loaded\": " << (base ? "true" : "false") << ",\n"
                << "  \"storage_bytes\": {\n"
                << "    \"graph\": " << std::filesystem::file_size(args.graph) << ",\n"
                << "    \"model\": " << std::filesystem::file_size(args.model) << ",\n"
                << "    \"primary_codes\": " << std::filesystem::file_size(args.codes) << ",\n"
                << "    \"side_codes\": "
                << (args.side_bits.empty() ? 0 : std::filesystem::file_size(args.side_bits))
                << ",\n"
                << "    \"base_vectors\": "
                << (base ? std::filesystem::file_size(args.base) : 0) << ",\n"
                << "    \"hot_index_total\": "
                << (std::filesystem::file_size(args.graph) +
                    std::filesystem::file_size(args.model) +
                    std::filesystem::file_size(args.codes) +
                    (args.side_bits.empty() ? 0 : std::filesystem::file_size(args.side_bits)) +
                    (base ? std::filesystem::file_size(args.base) : 0)) << "\n"
                << "  },\n"
                << "  \"peak_rss_bytes\": " << PeakResidentBytes() << ",\n"
                << "  \"threads\": " << worker_count << ",\n"
                << "  \"warmup_queries\": " << warmup_count << ",\n"
                << "  \"seconds\": " << seconds << ",\n"
                << "  \"qps\": " << query_count / std::max(seconds, 1e-12) << ",\n"
                << "  \"mean_latency_ms\": "
                << 1000.0 * seconds / query_count << ",\n"
                << "  \"p50_query_latency_ms\": " << percentile(0.50) << ",\n"
                << "  \"p95_query_latency_ms\": " << percentile(0.95) << ",\n"
                << "  \"p99_query_latency_ms\": " << percentile(0.99) << ",\n"
                << "  \"mean_comparisons\": "
                << static_cast<double>(comparisons) / query_count << ",\n"
                << "  \"mean_expansions\": "
                << static_cast<double>(expansions) / query_count << ",\n"
                << "  \"mean_visited_nodes\": "
                << static_cast<double>(visited_nodes) / query_count << ",\n"
                << "  \"mean_final_candidate_pool\": "
                << static_cast<double>(candidate_pool_size) / query_count << ",\n"
                << "  \"events\": " << event_count << ",\n"
                << "  \"event_counts\": {\n";
        for (size_t source = 0; source < 3; ++source) {
            const auto source_enum = static_cast<ours::EventSource>(source);
            summary << "    \"" << ours::EventSourceName(source_enum) << "\": {"
                    << "\"protect\": " << event_counts[source][0] << ", "
                    << "\"inversion\": " << event_counts[source][1] << "}"
                    << (source + 1 == 3 ? "\n" : ",\n");
        }
        summary << "  }\n}\n";
        std::cout << "[done] mode=" << ours::ScoreModeName(mode)
                  << " qps=" << query_count / std::max(seconds, 1e-12)
                  << " events=" << event_count << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        Usage(argv[0]);
        return 1;
    }
}
