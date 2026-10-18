#include "ours/exact_vector_store.h"
#include "ours/fp32_inner_product.h"
#include "ours/ji_online.h"
#include "ours/generic_primary.h"
#include "ours/qvrr_primary.h"
#include "ours/mapped_fbin.h"
#include "ours/roar_graph.h"
#include "ours/terminal_selection.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/rotator.hpp"

#include "quantization/saq_data.hpp"
#include "quantization/saq_estimator.hpp"
#include "quantization/saq_quantizer.hpp"
#include "quantization/single_data.hpp"

namespace {

class Mapping {
public:
  explicit Mapping(const std::string &path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0)
      throw std::runtime_error("cannot open: " + path);
    struct stat status {};
    if (fstat(fd_, &status) != 0 || status.st_size <= 0)
      throw std::runtime_error("cannot stat or empty file: " + path);
    size_ = static_cast<size_t>(status.st_size);
    data_ = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (data_ == MAP_FAILED) {
      data_ = nullptr;
      throw std::runtime_error("cannot mmap: " + path);
    }
  }
  ~Mapping() {
    if (data_)
      munmap(data_, size_);
    if (fd_ >= 0)
      close(fd_);
  }
  Mapping(const Mapping &) = delete;
  Mapping &operator=(const Mapping &) = delete;
  const void *data() const { return data_; }
  size_t size() const { return size_; }

private:
  int fd_ = -1;
  void *data_ = nullptr;
  size_t size_ = 0;
};

#pragma pack(push, 1)
struct RabitqHeader {
  char magic[8];
  uint32_t version;
  uint32_t metric_type;
  uint32_t rotator_type;
  uint32_t total_bits;
  uint64_t num_points;
  uint64_t dim;
  uint64_t padded_dim;
  uint64_t bin_bytes_per_vec;
  uint64_t ex_bytes_per_vec;
  uint64_t rotator_bytes;
};

struct SaqCodeHeader {
  char magic[8];
  uint32_t version;
  uint32_t header_bytes;
  uint64_t rows;
  uint64_t record_bytes;
  uint64_t model_bytes;
  uint64_t reserved[3];
};
#pragma pack(pop)
static_assert(sizeof(RabitqHeader) == 72);
static_assert(sizeof(SaqCodeHeader) == 64);

struct Arguments {
  std::string primary_kind;
  std::string primary_model;
  std::string primary_codes;
  std::string rabitq_config = "faster";
  std::string ji_model;
  std::string ji_side_codes;
  std::string graph;
  std::string queries;
  std::string base;
  std::string rerank_storage = "mmap";
  std::string fp32_storage = "mmap";
  std::string out_prefix;
  std::string visited_csr;
  std::string retained_csr;
  std::string expanded_csr;
  std::string routing_pairs_csv;
  std::string batch_plan;
  bool stream_plan = false;
  uint64_t memory_budget_bytes = 0;
  bool packed_graph = false;
  bool pdf_routing_tags = false;
  uint32_t topk = 10;
  uint32_t beam = 100;
  uint32_t rerank_candidates = 0;
  uint64_t query_start = 0;
  uint64_t query_limit = 0;
  uint64_t warmup_queries = 0;
  uint32_t threads = 1;
  uint32_t routing_pairs_per_source = 64;
  bool disable_ji = false;
  bool lazy_ji = false;
  bool terminal_ji = false;
  std::string ji_query_space = "scorer";
  std::string terminal_implementation = "optimized";
  bool profile_terminal = false;
  bool profile_components = false;
};

void Usage(const char *program) {
  std::cerr << "Usage: " << program << " --primary-kind fp32|extended-rabitq|saq|opq-pq|qaq|leanvec-ood|qvrr\n"
            << "  --primary-model FILE --primary-codes FILE --ji-model FILE\n"
            << "  --ji-side-codes FILE --graph FILE --queries FILE "
               "--out-prefix PATH\n"
            << "  [--base FILE --rerank-candidates N] "
               "[--rerank-storage mmap|direct] [--rabitq-config "
               "faster|exact]\n"
            << "  [--topk K] [--beam L] [--query-start N] [--query-limit N]\n"
            << "  [--fp32-storage mmap|direct] (FP32 primary uses the same vector-store backend as rerank)\n"
            << "  [--warmup-queries N] [--threads N] [--profile-components]\n"
            << "  [--lazy-ji] (lossless primary+JI-upper-bound pruning)\n\n"
            << "  [--terminal-ji] (primary traversal, then JI-rank the retained pool)\n"
            << "  [--ji-query-space scorer|raw] (explicit decoder coordinates in both traversal modes)\n"
            << "  [--terminal-ji-query-space scorer|raw] (backward-compatible alias)\n"
            << "  [--terminal-implementation legacy|optimized] [--profile-terminal]\n"
            << "  [--visited-csr FILE] [--retained-csr FILE] [--expanded-csr FILE]\n\n"
            << "  [--routing-pairs-csv FILE --routing-pairs-per-source N]\n"
            << "  [--packed-graph] [--memory-budget-bytes N] [--pdf-routing-tags]\n"
            << "  [--batch-plan FILE] (CSV: beam,rerank_candidates,out_prefix; load index once)\n"
            << "  [--stream-plan] (same CSV on stdin; keeps the index loaded during adaptive refinement)\n"
            << "    (FP32 teacher traversal only; records decision-local pairs)\n\n"
            << "  [--disable-ji] (primary-only control; ignores JI files)\n\n"
            << "rerank-candidates=0 disables reranking.  Positive N reranks "
               "the best\n"
            << "min(N, beam) candidates from the current terminal ordering "
               "with original FP32 vectors.\n";
}

std::string NextValue(int argc, char **argv, int &index) {
  if (index + 1 >= argc)
    throw std::invalid_argument("missing value after " +
                                std::string(argv[index]));
  return argv[++index];
}

Arguments ParseArguments(int argc, char **argv) {
  Arguments args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help" || key == "-h") {
      Usage(argv[0]);
      std::exit(0);
    } else if (key == "--primary-kind")
      args.primary_kind = NextValue(argc, argv, i);
    else if (key == "--primary-model")
      args.primary_model = NextValue(argc, argv, i);
    else if (key == "--primary-codes")
      args.primary_codes = NextValue(argc, argv, i);
    else if (key == "--rabitq-config")
      args.rabitq_config = NextValue(argc, argv, i);
    else if (key == "--ji-model")
      args.ji_model = NextValue(argc, argv, i);
    else if (key == "--ji-side-codes")
      args.ji_side_codes = NextValue(argc, argv, i);
    else if (key == "--graph")
      args.graph = NextValue(argc, argv, i);
    else if (key == "--queries")
      args.queries = NextValue(argc, argv, i);
    else if (key == "--base")
      args.base = NextValue(argc, argv, i);
    else if (key == "--rerank-storage")
      args.rerank_storage = NextValue(argc, argv, i);
    else if (key == "--fp32-storage")
      args.fp32_storage = NextValue(argc, argv, i);
    else if (key == "--out-prefix")
      args.out_prefix = NextValue(argc, argv, i);
    else if (key == "--visited-csr")
      args.visited_csr = NextValue(argc, argv, i);
    else if (key == "--retained-csr")
      args.retained_csr = NextValue(argc, argv, i);
    else if (key == "--expanded-csr")
      args.expanded_csr = NextValue(argc, argv, i);
    else if (key == "--routing-pairs-csv")
      args.routing_pairs_csv = NextValue(argc, argv, i);
    else if (key == "--batch-plan")
      args.batch_plan = NextValue(argc, argv, i);
    else if (key == "--stream-plan")
      args.stream_plan = true;
    else if (key == "--memory-budget-bytes")
      args.memory_budget_bytes = std::stoull(NextValue(argc, argv, i));
    else if (key == "--packed-graph")
      args.packed_graph = true;
    else if (key == "--pdf-routing-tags")
      args.pdf_routing_tags = true;
    else if (key == "--topk")
      args.topk = std::stoul(NextValue(argc, argv, i));
    else if (key == "--beam")
      args.beam = std::stoul(NextValue(argc, argv, i));
    else if (key == "--rerank-candidates")
      args.rerank_candidates = std::stoul(NextValue(argc, argv, i));
    else if (key == "--query-start")
      args.query_start = std::stoull(NextValue(argc, argv, i));
    else if (key == "--query-limit")
      args.query_limit = std::stoull(NextValue(argc, argv, i));
    else if (key == "--warmup-queries")
      args.warmup_queries = std::stoull(NextValue(argc, argv, i));
    else if (key == "--threads")
      args.threads = std::stoul(NextValue(argc, argv, i));
    else if (key == "--routing-pairs-per-source")
      args.routing_pairs_per_source = std::stoul(NextValue(argc, argv, i));
    else if (key == "--disable-ji")
      args.disable_ji = true;
    else if (key == "--lazy-ji")
      args.lazy_ji = true;
    else if (key == "--terminal-ji")
      args.terminal_ji = true;
    else if (key == "--ji-query-space" || key == "--terminal-ji-query-space")
      args.ji_query_space = NextValue(argc, argv, i);
    else if (key == "--terminal-implementation")
      args.terminal_implementation = NextValue(argc, argv, i);
    else if (key == "--profile-terminal")
      args.profile_terminal = true;
    else if (key == "--profile-components")
      args.profile_components = true;
    else
      throw std::invalid_argument("unknown argument: " + key);
  }
  if (args.primary_kind.empty() || args.primary_model.empty() ||
      args.primary_codes.empty() ||
      (!args.disable_ji &&
       (args.ji_model.empty() || args.ji_side_codes.empty())) || args.graph.empty() ||
      args.queries.empty() || args.out_prefix.empty())
    throw std::invalid_argument("missing required arguments");
  if (args.topk == 0 || args.beam < args.topk || args.threads == 0)
    throw std::invalid_argument("requires 0 < topk <= beam and threads > 0");
  if (args.rerank_candidates > 0 && args.base.empty())
    throw std::invalid_argument("--base is required when reranking is enabled");
  if (args.rerank_candidates > 0 && args.rerank_candidates < args.topk)
    throw std::invalid_argument(
        "rerank-candidates must be zero or at least topk");
  if (args.rerank_storage != "mmap" && args.rerank_storage != "direct")
    throw std::invalid_argument("rerank-storage must be mmap or direct");
  if (args.fp32_storage != "mmap" && args.fp32_storage != "direct")
    throw std::invalid_argument("fp32-storage must be mmap or direct");
  if (args.disable_ji && args.lazy_ji)
    throw std::invalid_argument("--lazy-ji requires JI to be enabled");
  if (args.disable_ji && args.terminal_ji)
    throw std::invalid_argument("--terminal-ji requires JI to be enabled");
  if (args.ji_query_space != "scorer" && args.ji_query_space != "raw")
    throw std::invalid_argument("JI query space must be scorer or raw");
  if (args.terminal_implementation != "legacy" && args.terminal_implementation != "optimized")
    throw std::invalid_argument("terminal-implementation must be legacy or optimized");
  if (args.profile_terminal && !args.terminal_ji)
    throw std::invalid_argument("--profile-terminal requires --terminal-ji");
  if (args.lazy_ji && args.terminal_ji)
    throw std::invalid_argument(
        "--lazy-ji and --terminal-ji select different traversal modes");
  if (!args.routing_pairs_csv.empty() &&
      (args.primary_kind != "fp32" || !args.disable_ji))
    throw std::invalid_argument(
        "routing-pair tracing requires --primary-kind fp32 --disable-ji");
  return args;
}

struct Candidate {
  uint32_t id = 0;
  float score = 0.0f;
  bool expanded = false;
};
bool Better(const Candidate &left, const Candidate &right) {
  return left.score > right.score ||
         (left.score == right.score && left.id < right.id);
}

struct TerminalWorkspace {
  // One instance per worker; reused for terminal AND traversal placement.
  ours::JiQueryWorkspace query;
  std::vector<Candidate> output;
  std::vector<uint32_t> ids;
  std::vector<float> corrections;
  std::vector<uint32_t> fresh_neighbors;
  std::vector<float> fresh_scores;
  std::vector<uint32_t> lazy_ids;
  std::vector<size_t> lazy_positions;
  std::vector<float> lazy_corrections;
};

uint64_t WorkspaceBytes(const TerminalWorkspace &w) {
  return w.output.capacity() * sizeof(Candidate) +
      (w.ids.capacity() + w.fresh_neighbors.capacity() + w.lazy_ids.capacity()) * sizeof(uint32_t) +
      (w.corrections.capacity() + w.fresh_scores.capacity() + w.lazy_corrections.capacity() +
       w.query.context.lookup.capacity() + w.query.padded.capacity() + w.query.phi.capacity()) * sizeof(float) +
      w.lazy_positions.capacity() * sizeof(size_t);
}

void SelectTerminalCandidates(std::vector<Candidate> &output, size_t keep) {
  ours::SelectSortedPrefix(output, keep, Better);
}

class CandidatePool {
public:
  explicit CandidatePool(uint32_t capacity) : capacity_(capacity) {
    values_.reserve(capacity + 1);
  }
  bool empty() const { return values_.empty(); }
  bool full() const { return values_.size() == capacity_; }
  const Candidate &worst() const {
    if (!full())
      throw std::logic_error("candidate pool is not full");
    return values_.back();
  }
  std::optional<Candidate> PopBestUnexpanded() {
    for (Candidate &value : values_)
      if (!value.expanded) {
        value.expanded = true;
        return value;
      }
    return std::nullopt;
  }
  std::optional<Candidate> Insert(Candidate candidate) {
    const auto position =
        std::lower_bound(values_.begin(), values_.end(), candidate, Better);
    if (values_.size() == capacity_ && position == values_.end())
      return candidate;
    values_.insert(position, candidate);
    if (values_.size() > capacity_) {
      const Candidate rejected = values_.back();
      values_.pop_back();
      return rejected;
    }
    return std::nullopt;
  }
  const std::vector<Candidate> &values() const { return values_; }

private:
  uint32_t capacity_;
  std::vector<Candidate> values_;
};

struct RoutingPair {
  uint8_t source = 0;  // 0=exp, 1=queue, 2=stop
  uint32_t u_id = 0;
  uint32_t v_id = 0;
  float exact_margin = 0.0f;
};

class Fp32Primary {
public:
  struct Context {
    const float *query = nullptr;
    mutable ours::ExactScoreStats stats;
  };

  explicit Fp32Primary(const std::string &path, const std::string &mode = "mmap")
      : base_(ours::OpenExactVectorStore(path, mode)), mode_(mode) {}
  uint64_t nodes() const { return base_->rows(); }
  uint32_t original_dim() const { return base_->dim(); }
  uint32_t scorer_dim() const { return base_->dim(); }
  uint64_t storage_bytes() const {
    return mode_ == "mmap" ? base_->file_bytes() : 0;
  }
  Context Prepare(const float *query, uint32_t dim) const {
    if (dim != base_->dim())
      throw std::invalid_argument("FP32 query dimension mismatch");
    return Context{query, {}};
  }
  float Score(const Context &context, uint32_t id) const {
    float score;
    base_->ScoreBatch(context.query, &id, 1, &score, &context.stats);
    return score;
  }
  void ScoreBatch(const Context &context, const uint32_t *ids, size_t count,
                  float *scores) const {
    base_->ScoreBatch(context.query, ids, count, scores, &context.stats);
  }
  const float *Transformed(const Context &context) const { return context.query; }

private:
  std::unique_ptr<ours::ExactVectorStore> base_;
  std::string mode_;
};

struct QueryResult {
  std::vector<uint32_t> ids;
  std::vector<float> scores;
  std::vector<uint32_t> visited_ids;
  std::vector<uint32_t> retained_ids;
  std::vector<uint32_t> expanded_ids;
  uint32_t comparisons = 0;
  uint32_t expansions = 0;
  uint32_t visited = 0;
  uint32_t pool = 0;
  uint32_t reranked = 0;
  uint64_t workspace_bytes = 0;
  double primary_prepare_ms = 0.0;
  double ji_prepare_ms = 0.0;
  double primary_score_ms = 0.0;
  double ji_score_ms = 0.0;
  double terminal_ji_ms = 0.0;
  double terminal_ji_prepare_ms = 0.0;
  double terminal_ji_score_ms = 0.0;
  ours::JiPrepareTimings terminal_prepare_details;
  double terminal_buffer_ms = 0.0;
  double terminal_lookup_ms = 0.0;
  double terminal_selection_ms = 0.0;
  uint64_t terminal_selected = 0;
  double rerank_ms = 0.0;
  double primary_io_ms = 0.0;
  uint64_t primary_logical_bytes = 0;
  uint64_t primary_physical_bytes = 0;
  uint64_t primary_read_ops = 0;
  uint64_t exact_workspace_peak_bytes = 0;
  double rerank_io_ms = 0.0;
  double rerank_compute_ms = 0.0;
  uint64_t rerank_logical_bytes = 0;
  uint64_t rerank_physical_bytes = 0;
  uint64_t rerank_read_ops = 0;
  double graph_bookkeeping_ms = 0.0;
  uint64_t ji_candidates_seen = 0;
  uint64_t ji_bound_checks = 0;
  uint64_t ji_corrections_evaluated = 0;
  uint64_t ji_corrections_pruned = 0;
  uint64_t terminal_ji_corrections_evaluated = 0;
  std::array<std::vector<RoutingPair>, 3> routing_pairs;
};

class RabitqPrimary {
public:
  struct Context {
    std::vector<float> transformed;
    std::unique_ptr<rabitqlib::SplitSingleQuery<float>> query;
  };

  RabitqPrimary(const std::string &path, const std::string &config_name)
      : mapping_(path) {
    if (mapping_.size() < sizeof(header_))
      throw std::runtime_error("truncated RaBitQ file");
    std::memcpy(&header_, mapping_.data(), sizeof(header_));
    if (std::memcmp(header_.magic, "RBQROAR1", 8) != 0 ||
        header_.version != 1 || header_.metric_type != rabitqlib::METRIC_IP ||
        header_.total_bits < 2)
      throw std::runtime_error("unsupported Extended-RaBitQ file");
    const uint64_t expected = sizeof(header_) + header_.rotator_bytes +
                              header_.num_points * (header_.bin_bytes_per_vec +
                                                    header_.ex_bytes_per_vec);
    if (mapping_.size() != expected)
      throw std::runtime_error("RaBitQ file-size mismatch");
    rotator_.reset(rabitqlib::choose_rotator<float>(
        header_.dim, static_cast<rabitqlib::RotatorType>(header_.rotator_type),
        header_.padded_dim));
    rotator_->load(static_cast<const char *>(mapping_.data()) +
                   sizeof(header_));
    bin_begin_ = static_cast<const char *>(mapping_.data()) + sizeof(header_) +
                 header_.rotator_bytes;
    ex_begin_ = bin_begin_ + header_.num_points * header_.bin_bytes_per_vec;
    ex_bits_ = header_.total_bits - 1;
    if (config_name == "faster")
      config_ = rabitqlib::quant::faster_config(header_.padded_dim,
                                                header_.total_bits);
    else if (config_name == "exact")
      config_ = rabitqlib::quant::RabitqConfig();
    else
      throw std::invalid_argument("rabitq-config must be faster or exact");
    ip_func_ = rabitqlib::select_excode_ipfunc(ex_bits_);
  }
  uint64_t nodes() const { return header_.num_points; }
  uint32_t original_dim() const { return header_.dim; }
  uint32_t scorer_dim() const { return header_.padded_dim; }
  uint64_t storage_bytes() const { return mapping_.size(); }
  Context Prepare(const float *query, uint32_t dim) const {
    if (dim != header_.dim)
      throw std::invalid_argument("RaBitQ query dimension mismatch");
    Context result;
    result.transformed.resize(header_.padded_dim);
    rotator_->rotate(query, result.transformed.data());
    result.query = std::make_unique<rabitqlib::SplitSingleQuery<float>>(
        result.transformed.data(), header_.padded_dim, ex_bits_, config_,
        rabitqlib::METRIC_IP);
    return result;
  }
  float Score(const Context &context, uint32_t id) const {
    float distance = 0.0f, lower = 0.0f, binary_ip = 0.0f;
    rabitqlib::split_single_fulldist(
        bin_begin_ + static_cast<uint64_t>(id) * header_.bin_bytes_per_vec,
        ex_begin_ + static_cast<uint64_t>(id) * header_.ex_bytes_per_vec,
        ip_func_, *context.query, header_.padded_dim, ex_bits_, distance, lower,
        binary_ip, 0.0f, 0.0f);
    return 1.0f - distance;
  }
  void ScoreBatch(const Context &context, const uint32_t *ids, size_t count,
                  float *scores) const {
    for (size_t index = 0; index < count; ++index) {
      if (index + 8 < count) {
        __builtin_prefetch(
            bin_begin_ + static_cast<uint64_t>(ids[index + 8]) *
                             header_.bin_bytes_per_vec,
            0, 1);
        __builtin_prefetch(
            ex_begin_ + static_cast<uint64_t>(ids[index + 8]) *
                            header_.ex_bytes_per_vec,
            0, 1);
      }
      scores[index] = Score(context, ids[index]);
    }
  }
  const float *Transformed(const Context &context) const {
    return context.transformed.data();
  }

private:
  Mapping mapping_;
  RabitqHeader header_{};
  std::unique_ptr<rabitqlib::Rotator<float>> rotator_;
  const char *bin_begin_ = nullptr;
  const char *ex_begin_ = nullptr;
  size_t ex_bits_ = 0;
  rabitqlib::quant::RabitqConfig config_;
  decltype(rabitqlib::select_excode_ipfunc(1)) ip_func_;
};

struct AlignedFree {
  void operator()(uint8_t *value) const { std::free(value); }
};

class SaqPrimary {
public:
  struct Context {
    saqlib::FloatVec original;
    saqlib::FloatVec transformed;
    saqlib::SaqSingleDataWrapper wrapper;
    std::unique_ptr<uint8_t, AlignedFree> scratch;
    std::unique_ptr<saqlib::SaqSingleEstimator<saqlib::DistType::IP>> estimator;

    Context(const saqlib::SaqData &data, const float *query, uint32_t dim,
            size_t record_bytes)
        : original(dim), wrapper(data.quant_plan) {
      std::memcpy(original.data(), query, dim * sizeof(float));
      saqlib::SAQuantizerSingle transformer(&data);
      transformer.transform(original, &transformed);
      void *raw = nullptr;
      if (posix_memalign(&raw, 64, ((record_bytes + 63) / 64) * 64) != 0)
        throw std::bad_alloc();
      scratch.reset(static_cast<uint8_t *>(raw));
      wrapper.set_memory_base(scratch.get());
      saqlib::SearcherConfig config;
      config.dist_type = saqlib::DistType::IP;
      estimator =
          std::make_unique<saqlib::SaqSingleEstimator<saqlib::DistType::IP>>(
              data, config, original);
    }
  };

  SaqPrimary(const std::string &model_path, const std::string &codes_path)
      : codes_(codes_path) {
    std::ifstream input(model_path, std::ios::binary);
    if (!input)
      throw std::runtime_error("cannot open SAQ model: " + model_path);
    data_.load(input);
    if (codes_.size() < sizeof(SaqCodeHeader))
      throw std::runtime_error("truncated SAQ codes");
    std::memcpy(&header_, codes_.data(), sizeof(header_));
    if (std::memcmp(header_.magic, "SAQCODE1", 8) != 0 ||
        header_.version != 1 || header_.header_bytes != sizeof(SaqCodeHeader) ||
        header_.record_bytes == 0 ||
        header_.model_bytes != std::filesystem::file_size(model_path) ||
        codes_.size() !=
            sizeof(SaqCodeHeader) + header_.rows * header_.record_bytes)
      throw std::runtime_error("invalid SAQ full-code file");
    const size_t expected =
        saqlib::SaqSingleDataWrapper::calculate_memory_size(data_.quant_plan);
    if (header_.record_bytes != expected)
      throw std::runtime_error("SAQ code/model record mismatch");
    code_begin_ =
        static_cast<const uint8_t *>(codes_.data()) + sizeof(SaqCodeHeader);
  }
  uint64_t nodes() const { return header_.rows; }
  uint32_t original_dim() const { return data_.num_dim; }
  uint32_t scorer_dim() const { return data_.data_variance.cols(); }
  uint64_t storage_bytes() const { return codes_.size(); }
  std::unique_ptr<Context> Prepare(const float *query, uint32_t dim) const {
    if (dim != data_.num_dim)
      throw std::invalid_argument("SAQ query dimension mismatch");
    return std::make_unique<Context>(data_, query, dim, header_.record_bytes);
  }
  float Score(const std::unique_ptr<Context> &context, uint32_t id) const {
    std::memcpy(context->scratch.get(),
                code_begin_ + static_cast<uint64_t>(id) * header_.record_bytes,
                header_.record_bytes);
    return context->estimator->compAccurateDist(context->wrapper);
  }
  void ScoreBatch(const std::unique_ptr<Context> &context,
                  const uint32_t *ids, size_t count, float *scores) const {
    for (size_t index = 0; index < count; ++index) {
      if (index + 8 < count)
        __builtin_prefetch(code_begin_ +
                               static_cast<uint64_t>(ids[index + 8]) *
                                   header_.record_bytes,
                           0, 1);
      scores[index] = Score(context, ids[index]);
    }
  }
  const float *Transformed(const std::unique_ptr<Context> &context) const {
    return context->transformed.data();
  }

private:
  saqlib::SaqData data_;
  Mapping codes_;
  SaqCodeHeader header_{};
  const uint8_t *code_begin_ = nullptr;
};

class QvrrPrimary {
public:
  using Context = ours::QvrrPrimaryIndex::QueryContext;
  QvrrPrimary(const std::string& model, const std::string& codes):index_(model,codes) {}
  uint64_t nodes() const { return index_.nodes(); }
  uint32_t original_dim() const { return index_.original_dim(); }
  uint32_t scorer_dim() const { return index_.scorer_dim(); }
  uint64_t storage_bytes() const { return index_.storage_bytes(); }
  const char* scoring_kernel() const { return index_.scoring_kernel(); }
  Context Prepare(const float* q,uint32_t d) const { return index_.Prepare(q,d); }
  float Score(const Context& q,uint32_t id) const { return index_.Score(q,id); }
  void ScoreBatch(const Context& q,const uint32_t* ids,size_t n,float* out) const { index_.ScoreBatch(q,ids,n,out); }
  const float* Transformed(const Context& q) const { return q.transformed.data(); }
private:
  ours::QvrrPrimaryIndex index_;
};

class GenericPrimary {
public:
  using Context = ours::GenericPrimaryIndex::QueryContext;
  GenericPrimary(const std::string &model_path, const std::string &codes_path)
      : index_(ours::GenericPrimaryIndex::Load(model_path, codes_path)) {}
  uint64_t nodes() const { return index_.nodes(); }
  uint32_t original_dim() const { return index_.original_dim(); }
  uint32_t scorer_dim() const { return index_.scorer_dim(); }
  uint64_t storage_bytes() const { return index_.storage_bytes(); }
  const char *scoring_kernel() const { return index_.scoring_kernel(); }
  Context Prepare(const float *query, uint32_t dim) const {
    return index_.Prepare(query, dim);
  }
  float Score(const Context &context, uint32_t id) const {
    return index_.Score(context, id);
  }
  void ScoreBatch(const Context &context, const uint32_t *ids, size_t count,
                  float *scores) const {
    index_.ScoreBatch(context, ids, count, scores);
  }
  const float *Transformed(const Context &context) const {
    return context.transformed.data();
  }
private:
  ours::GenericPrimaryIndex index_;
};

template <typename Primary>
const char *PrimaryScoringKernel(const Primary &) { return "backend-native"; }
const char *PrimaryScoringKernel(const Fp32Primary &) {
  return ours::Fp32InnerProductKernel();
}
const char *PrimaryScoringKernel(const GenericPrimary &primary) {
  return primary.scoring_kernel();
}
const char *PrimaryScoringKernel(const QvrrPrimary &primary) { return primary.scoring_kernel(); }

template <class P, class C> uint64_t PrimaryQueryBytes(const P &, const C &) { return 0; }
template <class P, class C> ours::ExactScoreStats PrimaryExactStats(const P &, const C &) { return {}; }
ours::ExactScoreStats PrimaryExactStats(const Fp32Primary &, const Fp32Primary::Context &q) {
  return q.stats;
}
uint64_t PrimaryQueryBytes(const GenericPrimary &, const GenericPrimary::Context &q) {
  return (q.transformed.capacity() + q.lut.capacity()) * sizeof(float);
}
uint64_t PrimaryQueryBytes(const QvrrPrimary &, const QvrrPrimary::Context &q) {
  return (q.transformed.capacity()+q.lut.capacity())*sizeof(float);
}
uint64_t PrimaryQueryBytes(const RabitqPrimary &, const RabitqPrimary::Context &q) {
  return q.transformed.capacity() * sizeof(float);
}

template <typename Primary>
QueryResult SearchOne(const Primary &primary, const ours::JiOnlineIndex *ji,
                      const ours::RoarGraph &graph, const float *query,
                      uint32_t query_dim, uint32_t topk, uint32_t beam,
                      uint32_t rerank_candidates,
                      const ours::ExactVectorStore *base,
                      bool profile_components,
                      bool lazy_ji, bool terminal_ji, bool ji_raw_query,
                      bool collect_visited,
                      bool collect_retained,
                      bool collect_expanded,
                      uint32_t routing_pairs_per_source,
                      TerminalWorkspace *terminal_workspace = nullptr,
                      bool profile_terminal = false) {
  using Clock = std::chrono::steady_clock;
  const auto milliseconds = [](Clock::duration duration) {
    return 1000.0 * std::chrono::duration<double>(duration).count();
  };
  QueryResult result;
  const auto record_pair = [&](uint8_t source, const Candidate &left,
                               const Candidate &right) {
    if (routing_pairs_per_source == 0 || left.id == right.id ||
        left.score == right.score)
      return;
    const Candidate &u = left.score > right.score ? left : right;
    const Candidate &v = left.score > right.score ? right : left;
    RoutingPair pair{source, u.id, v.id, u.score - v.score};
    std::vector<RoutingPair> &values = result.routing_pairs[source];
    if (values.size() < routing_pairs_per_source) {
      values.push_back(pair);
      return;
    }
    const auto largest = std::max_element(
        values.begin(), values.end(), [](const RoutingPair &a,
                                         const RoutingPair &b) {
          return a.exact_margin < b.exact_margin;
        });
    if (pair.exact_margin < largest->exact_margin) *largest = pair;
  };
  const auto query_started = Clock::now();
  auto stage_started = query_started;
  auto primary_query = primary.Prepare(query, query_dim);
  if (profile_components)
    result.primary_prepare_ms = milliseconds(Clock::now() - stage_started);
  stage_started = Clock::now();
  const bool traversal_ji = ji != nullptr && !terminal_ji;
  std::optional<ours::JiQueryContext> legacy_ji_query;
  const ours::JiQueryContext *ji_query = nullptr;
  if (traversal_ji) {
    const float *input = ji_raw_query ? query : primary.Transformed(primary_query);
    const uint32_t dim = ji_raw_query ? query_dim : primary.scorer_dim();
    if (terminal_workspace) {
      ji_query = &ji->PrepareQueryReused(input, dim, terminal_workspace->query, lazy_ji);
    } else {
      legacy_ji_query.emplace(ji->PrepareQueryZeroPadded(input, dim));
      ji_query = &*legacy_ji_query;
    }
  }
  if (profile_components && traversal_ji)
    result.ji_prepare_ms = milliseconds(Clock::now() - stage_started);
  const auto score_batch = [&](const uint32_t *ids, size_t count,
                               float *scores) {
    auto scoring_started = profile_components ? Clock::now() : Clock::time_point{};
    primary.ScoreBatch(primary_query, ids, count, scores);
    if (profile_components) {
      result.primary_score_ms +=
          milliseconds(Clock::now() - scoring_started);
      scoring_started = Clock::now();
    }
    if (traversal_ji) {
      result.ji_candidates_seen += count;
      result.ji_corrections_evaluated += count;
      ji->AddCorrectionsBatch(*ji_query, ids, count, scores);
    }
    if (profile_components && traversal_ji)
      result.ji_score_ms += milliseconds(Clock::now() - scoring_started);
  };
  CandidatePool pool(beam);
  std::unordered_set<uint32_t> visited;
  visited.reserve(static_cast<size_t>(beam) * 32);
  const uint32_t entry = graph.entry_point();
  visited.insert(entry);
  float entry_score = 0.0f;
  score_batch(&entry, 1, &entry_score);
  pool.Insert(Candidate{entry, entry_score, false});
  std::optional<Candidate> best_rejected;
  std::vector<uint32_t> local_neighbors, local_lazy_ids;
  std::vector<float> local_scores, local_lazy_corrections;
  std::vector<size_t> local_lazy_positions;
  auto &fresh_neighbors = terminal_workspace ? terminal_workspace->fresh_neighbors : local_neighbors;
  auto &fresh_scores = terminal_workspace ? terminal_workspace->fresh_scores : local_scores;
  auto &lazy_ids = terminal_workspace ? terminal_workspace->lazy_ids : local_lazy_ids;
  auto &lazy_positions = terminal_workspace ? terminal_workspace->lazy_positions : local_lazy_positions;
  auto &lazy_corrections = terminal_workspace ? terminal_workspace->lazy_corrections : local_lazy_corrections;
  fresh_neighbors.reserve(128);
  fresh_scores.reserve(128);
  lazy_ids.reserve(128);
  lazy_positions.reserve(128);
  lazy_corrections.reserve(128);
  while (true) {
    if (routing_pairs_per_source != 0) {
      const Candidate *first = nullptr;
      const Candidate *second = nullptr;
      for (const Candidate &candidate : pool.values()) {
        if (candidate.expanded) continue;
        if (!first)
          first = &candidate;
        else {
          second = &candidate;
          break;
        }
      }
      if (first && second) record_pair(0, *first, *second);
    }
    const std::optional<Candidate> current = pool.PopBestUnexpanded();
    if (!current)
      break;
    if (collect_expanded)
      result.expanded_ids.push_back(current->id);
    ++result.expansions;
    fresh_neighbors.clear();
    for (uint32_t neighbor : graph.NeighborIds(current->id)) {
      if (!visited.insert(neighbor).second)
        continue;
      ++result.comparisons;
      fresh_neighbors.push_back(neighbor);
    }
    if (!fresh_neighbors.empty()) {
      fresh_scores.resize(fresh_neighbors.size());
      if (!traversal_ji || !lazy_ji) {
        score_batch(fresh_neighbors.data(), fresh_neighbors.size(),
                    fresh_scores.data());
      } else {
        auto scoring_started = profile_components ? Clock::now() : Clock::time_point{};
        primary.ScoreBatch(primary_query, fresh_neighbors.data(),
                           fresh_neighbors.size(), fresh_scores.data());
        if (profile_components) {
          result.primary_score_ms +=
              milliseconds(Clock::now() - scoring_started);
          scoring_started = Clock::now();
        }

        result.ji_candidates_seen += fresh_neighbors.size();
        lazy_ids.clear();
        lazy_positions.clear();
        const bool can_prune = pool.full();
        const float threshold = can_prune ? pool.worst().score : 0.0f;
        for (size_t index = 0; index < fresh_neighbors.size(); ++index) {
          if (can_prune) {
            ++result.ji_bound_checks;
            // max_correction_upper already bounds every realizable float JI
            // correction.  IEEE addition is monotone, so applying the same
            // final float addition as the scorer preserves that upper bound.
            const float total_upper =
                fresh_scores[index] + ji_query->max_correction_upper;
            // Strict comparison intentionally keeps all ties.  This avoids
            // depending on the candidate-id tie breaker and is conservative
            // for every finite scorer result.
            if (total_upper < threshold) {
              ++result.ji_corrections_pruned;
              continue;
            }
          }
          lazy_ids.push_back(fresh_neighbors[index]);
          lazy_positions.push_back(index);
        }
        result.ji_corrections_evaluated += lazy_ids.size();
        lazy_corrections.resize(lazy_ids.size());
        if (!lazy_ids.empty())
          ji->ComputeCorrectionsBatch(*ji_query, lazy_ids.data(),
                                      lazy_ids.size(),
                                      lazy_corrections.data());
        for (size_t local = 0; local < lazy_ids.size(); ++local)
          fresh_scores[lazy_positions[local]] += lazy_corrections[local];
        if (profile_components)
          result.ji_score_ms +=
              milliseconds(Clock::now() - scoring_started);
        for (size_t local = 0; local < lazy_ids.size(); ++local)
          pool.Insert(Candidate{lazy_ids[local],
                                fresh_scores[lazy_positions[local]], false});
        // Pruned candidates are omitted from insertion.  The fixed threshold
        // was captured before the batch; subsequent insertions can only make
        // it stricter, so batching is conservative and trajectory-preserving.
        continue;
      }
      for (size_t index = 0; index < fresh_neighbors.size(); ++index) {
        const Candidate candidate{fresh_neighbors[index], fresh_scores[index],
                                  false};
        if (routing_pairs_per_source != 0 && pool.full())
          record_pair(1, candidate, pool.worst());
        const std::optional<Candidate> rejected = pool.Insert(candidate);
        if (rejected &&
            (!best_rejected || Better(*rejected, *best_rejected)))
          best_rejected = *rejected;
      }
    }
  }
  if (routing_pairs_per_source != 0 && best_rejected && !pool.empty())
    record_pair(2, *best_rejected, pool.values().back());
  if (pool.values().size() < topk)
    throw std::runtime_error("search returned fewer than topk nodes");
  result.visited = visited.size();
  if (collect_visited) {
    result.visited_ids.assign(visited.begin(), visited.end());
    std::sort(result.visited_ids.begin(), result.visited_ids.end());
  }
  result.pool = pool.values().size();
  if (collect_retained) {
    result.retained_ids.reserve(pool.values().size());
    for (const Candidate &candidate : pool.values())
      result.retained_ids.push_back(candidate.id);
    std::sort(result.retained_ids.begin(), result.retained_ids.end());
  }
  std::vector<Candidate> local_output;
  auto &output = terminal_workspace ? terminal_workspace->output : local_output;
  output.assign(pool.values().begin(), pool.values().end());
  if (terminal_ji) {
    const auto terminal_started = Clock::now();
    auto terminal_stage_started = terminal_started;
    const float *terminal_input = ji_raw_query ? query : primary.Transformed(primary_query);
    const uint32_t terminal_dim = ji_raw_query ? query_dim : primary.scorer_dim();
    auto *prepare_timings = profile_terminal ? &result.terminal_prepare_details : nullptr;
    std::optional<ours::JiQueryContext> legacy_query;
    const ours::JiQueryContext *terminal_query = nullptr;
    if (terminal_workspace) {
      terminal_query = &ji->PrepareTerminalQuery(terminal_input, terminal_dim,
                                                terminal_workspace->query, prepare_timings);
    } else {
      legacy_query.emplace(ji->PrepareQueryZeroPadded(terminal_input, terminal_dim, true, prepare_timings));
      terminal_query = &*legacy_query;
    }
    result.terminal_ji_prepare_ms =
        milliseconds(Clock::now() - terminal_stage_started);
    terminal_stage_started = Clock::now();
    std::vector<uint32_t> local_ids;
    std::vector<float> local_corrections;
    auto &terminal_ids = terminal_workspace ? terminal_workspace->ids : local_ids;
    auto &terminal_corrections = terminal_workspace ? terminal_workspace->corrections : local_corrections;
    terminal_ids.resize(output.size());
    terminal_corrections.resize(output.size());
    for (size_t index = 0; index < output.size(); ++index)
      terminal_ids[index] = output[index].id;
    auto detail_started = Clock::time_point{};
    if (profile_terminal) {
      detail_started = Clock::now();
      result.terminal_buffer_ms = milliseconds(detail_started - terminal_stage_started);
    }
    ji->ComputeCorrectionsBatch(*terminal_query, terminal_ids.data(),
                                terminal_ids.size(),
                                terminal_corrections.data());
    for (size_t index = 0; index < output.size(); ++index)
      output[index].score += terminal_corrections[index];
    if (profile_terminal) {
      const auto now = Clock::now();
      result.terminal_lookup_ms = milliseconds(now - detail_started);
      detail_started = now;
    }
    // All retained nodes still receive corrections before selection; reducing
    // the corrected set to primary Top-L would change the algorithm.
    result.terminal_ji_corrections_evaluated = output.size();
    const size_t keep = rerank_candidates ? std::min<size_t>(rerank_candidates, output.size()) : topk;
    if (terminal_workspace) SelectTerminalCandidates(output, keep);
    else std::sort(output.begin(), output.end(), Better);
    result.terminal_selected = keep;
    if (profile_terminal) result.terminal_selection_ms = milliseconds(Clock::now() - detail_started);
    result.terminal_ji_score_ms =
        milliseconds(Clock::now() - terminal_stage_started);
    result.terminal_ji_ms = milliseconds(Clock::now() - terminal_started);
  }
  if (rerank_candidates > 0) {
    const auto rerank_started = Clock::now();
    const uint32_t count =
        std::min<uint32_t>(rerank_candidates, output.size());
    std::vector<uint32_t> local_rerank_ids;
    std::vector<float> local_rerank_scores;
    auto &rerank_ids = terminal_workspace ? terminal_workspace->ids : local_rerank_ids;
    auto &rerank_scores = terminal_workspace ? terminal_workspace->corrections : local_rerank_scores;
    rerank_ids.resize(count);
    rerank_scores.resize(count);
    for (uint32_t index = 0; index < count; ++index)
      rerank_ids[index] = output[index].id;
    ours::ExactScoreStats exact_stats;
    base->ScoreBatch(query, rerank_ids.data(), count, rerank_scores.data(),
                     &exact_stats);
    if (terminal_workspace) {
      output.resize(count);
      for (uint32_t index = 0; index < count; ++index)
        output[index] = Candidate{rerank_ids[index], rerank_scores[index], false};
      SelectTerminalCandidates(output, topk);
    } else {
      std::vector<Candidate> exact_output;
      exact_output.reserve(count);
      for (uint32_t index = 0; index < count; ++index)
        exact_output.push_back(
            Candidate{rerank_ids[index], rerank_scores[index], false});
      std::sort(exact_output.begin(), exact_output.end(), Better);
      output = std::move(exact_output);
    }
    result.reranked = count;
    result.rerank_logical_bytes = exact_stats.logical_bytes;
    result.rerank_physical_bytes = exact_stats.physical_bytes;
    result.rerank_read_ops = exact_stats.read_ops;
    result.rerank_io_ms = exact_stats.io_ms;
    result.rerank_compute_ms = exact_stats.compute_ms;
    result.exact_workspace_peak_bytes = exact_stats.workspace_peak_bytes;
    // Exact reranking is timed even when the fine-grained component profiler
    // is disabled.  This adds only one timer pair per query and lets the
    // limited-rerank experiment report measured rerank cost without enabling
    // the much more intrusive timers inside the graph-scoring loop.
    result.rerank_ms = milliseconds(Clock::now() - rerank_started);
  } else {
  }
  result.ids.reserve(topk);
  result.scores.reserve(topk);
  for (uint32_t index = 0; index < topk; ++index) {
    result.ids.push_back(output[index].id);
    result.scores.push_back(output[index].score);
  }
  if (profile_components) {
    const double total_ms = milliseconds(Clock::now() - query_started);
    result.graph_bookkeeping_ms =
        std::max(0.0, total_ms - result.primary_prepare_ms -
                          result.ji_prepare_ms - result.primary_score_ms -
                          result.ji_score_ms - result.terminal_ji_ms -
                          result.rerank_ms);
  }
  const auto primary_stats = PrimaryExactStats(primary, primary_query);
  result.primary_io_ms = primary_stats.io_ms;
  result.primary_logical_bytes = primary_stats.logical_bytes;
  result.primary_physical_bytes = primary_stats.physical_bytes;
  result.primary_read_ops = primary_stats.read_ops;
  result.exact_workspace_peak_bytes = std::max(result.exact_workspace_peak_bytes, primary_stats.workspace_peak_bytes);
  result.workspace_bytes = PrimaryQueryBytes(primary, primary_query) + result.exact_workspace_peak_bytes +
      pool.values().capacity() * sizeof(Candidate) + visited.bucket_count() * sizeof(void*) +
      visited.size() * sizeof(uint32_t); // Payload only; hash node/allocator overhead is in RSS/cap.
  if (terminal_workspace) result.workspace_bytes += WorkspaceBytes(*terminal_workspace);
  else result.workspace_bytes += local_output.capacity() * sizeof(Candidate) +
      (local_neighbors.capacity() + local_lazy_ids.capacity()) * sizeof(uint32_t) +
      (local_scores.capacity() + local_lazy_corrections.capacity()) * sizeof(float) +
      local_lazy_positions.capacity() * sizeof(size_t) +
      (legacy_ji_query ? legacy_ji_query->lookup.capacity() * sizeof(float) : 0);
  return result;
}

template <typename T>
void WriteMatrix(const std::string &path, uint32_t rows, uint32_t cols,
                 const std::vector<T> &values) {
  std::ofstream output(path, std::ios::binary);
  if (!output)
    throw std::runtime_error("cannot create output: " + path);
  output.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
  output.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
  output.write(reinterpret_cast<const char *>(values.data()),
               values.size() * sizeof(T));
  if (!output)
    throw std::runtime_error("failed writing output: " + path);
}

void WriteTraceCsr(const std::string &path, uint64_t query_start,
                   const std::vector<uint64_t> &offsets,
                   const std::vector<uint32_t> &ids) {
  if (offsets.empty() || offsets.front() != 0 || offsets.back() != ids.size())
    throw std::logic_error("invalid trajectory CSR payload");
  const uint64_t rows64 = offsets.size() - 1;
  if (rows64 > std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("trajectory CSR row count exceeds uint32");
  const std::filesystem::path output_path(path);
  if (!output_path.parent_path().empty())
    std::filesystem::create_directories(output_path.parent_path());
  std::ofstream output(path, std::ios::binary);
  if (!output)
    throw std::runtime_error("cannot create trajectory CSR: " + path);
  const std::array<char, 8> magic = {'D', 'R', 'B', 'V', 'S', 'T', '0', '1'};
  const uint32_t version = 1;
  const uint32_t rows = static_cast<uint32_t>(rows64);
  const uint64_t total = ids.size();
  output.write(magic.data(), static_cast<std::streamsize>(magic.size()));
  output.write(reinterpret_cast<const char *>(&version), sizeof(version));
  output.write(reinterpret_cast<const char *>(&query_start), sizeof(query_start));
  output.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
  output.write(reinterpret_cast<const char *>(&total), sizeof(total));
  output.write(reinterpret_cast<const char *>(offsets.data()),
               static_cast<std::streamsize>(offsets.size() * sizeof(uint64_t)));
  output.write(reinterpret_cast<const char *>(ids.data()),
               static_cast<std::streamsize>(ids.size() * sizeof(uint32_t)));
  if (!output)
    throw std::runtime_error("failed writing trajectory CSR: " + path);
}

uint64_t PeakResidentBytes() {
  struct rusage usage {};
  if (getrusage(RUSAGE_SELF, &usage) != 0)
    return 0;
  return static_cast<uint64_t>(usage.ru_maxrss) * 1024ULL;
}

std::vector<std::string> CsvFields(const std::string &line) {
  std::vector<std::string> fields;
  std::string value;
  bool quoted = false;
  for (size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (c == '"') {
      if (quoted && i + 1 < line.size() && line[i + 1] == '"') { value += c; ++i; }
      else quoted = !quoted;
    } else if (c == ',' && !quoted) { fields.push_back(value); value.clear(); }
    else if (c != '\r') value += c;
  }
  if (quoted) throw std::runtime_error("unterminated CSV field");
  fields.push_back(value);
  return fields;
}

struct MemoryParts { uint64_t codes = 0, model = 0; };
uint64_t PrimaryHotVectorBytes(const Arguments &args) {
  return args.primary_kind == "fp32" && args.fp32_storage == "mmap"
      ? std::filesystem::file_size(args.primary_codes) : 0;
}
MemoryParts PrimaryMemoryParts(const Arguments &args) {
  // FP32 is an external vector store, not a quantized code payload. The mmap
  // variant is charged separately; the direct variant owns only IO buffers.
  if (args.primary_kind == "fp32") return {};
  const uint64_t codes = std::filesystem::file_size(args.primary_codes);
  if (std::filesystem::equivalent(args.primary_model, args.primary_codes)) {
    if (args.primary_kind == "extended-rabitq") {
      std::ifstream input(args.primary_codes, std::ios::binary);
      RabitqHeader header{};
      input.read(reinterpret_cast<char*>(&header), sizeof(header));
      const uint64_t model = sizeof(header) + header.rotator_bytes;
      return {codes - model, model};
    }
    return {codes, 0};
  }
  return {codes, std::filesystem::file_size(args.primary_model)};
}

template <typename Primary>
int RunLoaded(const Arguments &args, const Primary &primary,
              const ours::MappedFbin &queries, const ours::RoarGraph &graph,
              const std::unique_ptr<ours::JiOnlineIndex> &ji,
              const std::unique_ptr<ours::ExactVectorStore> &base) {
  const uint32_t ji_input_dim = args.ji_query_space == "raw"
                                    ? queries.dim() : primary.scorer_dim();
  if (queries.dim() != primary.original_dim() ||
      graph.size() != primary.nodes() ||
      (ji && (ji->node_count() != primary.nodes() ||
              ((args.terminal_ji || args.ji_query_space == "raw") ? ji->dim() < ji_input_dim
                                : ji->dim() != primary.scorer_dim()))))
    throw std::runtime_error("query/graph/primary/JI shapes do not match");
  if (base && (base->rows() < primary.nodes() || base->dim() != queries.dim()))
    throw std::runtime_error("base/query/primary shapes do not match");
  if (args.query_start >= queries.rows())
    throw std::invalid_argument("query-start outside file");
  const uint64_t available = queries.rows() - args.query_start;
  const uint64_t query_count =
      args.query_limit ? std::min(args.query_limit, available) : available;
  if (query_count > std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("output matrix row count exceeds uint32");
  const uint64_t warmup = std::min(args.warmup_queries, query_count);
  const bool optimized_terminal = args.terminal_ji && args.terminal_implementation == "optimized";
  const bool optimized_workspace = args.terminal_implementation == "optimized";
  TerminalWorkspace warmup_workspace;
  for (uint64_t local = 0; local < warmup; ++local) {
    static_cast<void>(SearchOne(primary, ji.get(), graph,
                                queries.Row(args.query_start + local),
                                queries.dim(), args.topk, args.beam,
                                args.rerank_candidates, base.get(), false,
                                args.lazy_ji, args.terminal_ji,
                                args.ji_query_space == "raw", false, false,
                                false, 0, optimized_workspace ? &warmup_workspace : nullptr));
  }

  std::vector<QueryResult> results(query_count);
  std::vector<double> latency_ms(query_count);
  std::atomic<uint64_t> next{0};
  std::atomic<bool> stop{false};
  std::mutex exception_mutex;
  std::exception_ptr exception;
  const uint32_t workers_count = std::min<uint64_t>(args.threads, query_count);
  std::atomic<uint64_t> worker_workspace_bytes{0};
  const auto started = std::chrono::steady_clock::now();
  const auto worker = [&]() {
    try {
      TerminalWorkspace terminal_workspace;
      uint64_t workspace_peak = 0;
      while (!stop.load(std::memory_order_relaxed)) {
        const uint64_t local = next.fetch_add(1, std::memory_order_relaxed);
        if (local >= query_count)
          break;
        const auto one_started = std::chrono::steady_clock::now();
        results[local] =
            SearchOne(primary, ji.get(), graph, queries.Row(args.query_start + local),
                      queries.dim(), args.topk, args.beam,
                      args.rerank_candidates, base.get(),
                      args.profile_components, args.lazy_ji,
                      args.terminal_ji,
                      args.ji_query_space == "raw",
                      !args.visited_csr.empty(),
                      !args.retained_csr.empty(),
                      !args.expanded_csr.empty(),
                      args.routing_pairs_csv.empty()
                          ? 0
                          : args.routing_pairs_per_source,
                      optimized_workspace ? &terminal_workspace : nullptr,
                      args.profile_terminal);
        workspace_peak = std::max(workspace_peak, results[local].workspace_bytes);
        latency_ms[local] =
            1000.0 * std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - one_started)
                         .count();
      }
      worker_workspace_bytes.fetch_add(workspace_peak, std::memory_order_relaxed);
    } catch (...) {
      stop.store(true, std::memory_order_relaxed);
      std::lock_guard<std::mutex> lock(exception_mutex);
      if (!exception)
        exception = std::current_exception();
    }
  };
  std::vector<std::thread> workers;
  for (uint32_t index = 0; index < workers_count; ++index)
    workers.emplace_back(worker);
  for (std::thread &thread : workers)
    thread.join();
  if (exception)
    std::rethrow_exception(exception);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();

  std::vector<uint32_t> ids;
  std::vector<float> scores;
  std::vector<uint64_t> visited_offsets{0};
  std::vector<uint64_t> retained_offsets{0};
  std::vector<uint64_t> expanded_offsets{0};
  std::vector<uint32_t> traced_visited_ids;
  std::vector<uint32_t> traced_retained_ids;
  std::vector<uint32_t> traced_expanded_ids;
  uint64_t comparisons = 0, expansions = 0, visited = 0, pools = 0,
           reranked = 0, ji_candidates_seen = 0, ji_bound_checks = 0,
           ji_corrections_evaluated = 0, ji_corrections_pruned = 0,
           terminal_ji_corrections_evaluated = 0,
           terminal_selected = 0,
           rerank_logical_bytes = 0, rerank_physical_bytes = 0,
           rerank_read_ops = 0;
  uint64_t primary_logical_bytes = 0, primary_physical_bytes = 0, primary_read_ops = 0;
  double primary_io_ms = 0.0;
  double primary_prepare_ms = 0.0, ji_prepare_ms = 0.0,
         primary_score_ms = 0.0, ji_score_ms = 0.0, rerank_ms = 0.0,
         terminal_ji_ms = 0.0, terminal_ji_prepare_ms = 0.0,
         terminal_ji_score_ms = 0.0,
         terminal_projection_ms = 0.0, terminal_lut_ms = 0.0, terminal_bound_ms = 0.0,
         terminal_buffer_ms = 0.0, terminal_lookup_ms = 0.0, terminal_selection_ms = 0.0,
         rerank_io_ms = 0.0, rerank_compute_ms = 0.0,
         graph_bookkeeping_ms = 0.0;
  ids.reserve(query_count * args.topk);
  scores.reserve(query_count * args.topk);
  for (const QueryResult &result : results) {
    ids.insert(ids.end(), result.ids.begin(), result.ids.end());
    scores.insert(scores.end(), result.scores.begin(), result.scores.end());
    comparisons += result.comparisons;
    expansions += result.expansions;
    visited += result.visited;
    pools += result.pool;
    reranked += result.reranked;
    primary_prepare_ms += result.primary_prepare_ms;
    ji_prepare_ms += result.ji_prepare_ms;
    primary_score_ms += result.primary_score_ms;
    ji_score_ms += result.ji_score_ms;
    terminal_ji_ms += result.terminal_ji_ms;
    terminal_ji_prepare_ms += result.terminal_ji_prepare_ms;
    terminal_ji_score_ms += result.terminal_ji_score_ms;
    terminal_projection_ms += result.terminal_prepare_details.projection_ms;
    terminal_lut_ms += result.terminal_prepare_details.lut_ms;
    terminal_bound_ms += result.terminal_prepare_details.upper_bound_ms;
    terminal_buffer_ms += result.terminal_buffer_ms;
    terminal_lookup_ms += result.terminal_lookup_ms;
    terminal_selection_ms += result.terminal_selection_ms;
    terminal_selected += result.terminal_selected;
    rerank_ms += result.rerank_ms;
    rerank_io_ms += result.rerank_io_ms;
    rerank_compute_ms += result.rerank_compute_ms;
    rerank_logical_bytes += result.rerank_logical_bytes;
    rerank_physical_bytes += result.rerank_physical_bytes;
    rerank_read_ops += result.rerank_read_ops;
    primary_io_ms += result.primary_io_ms;
    primary_logical_bytes += result.primary_logical_bytes;
    primary_physical_bytes += result.primary_physical_bytes;
    primary_read_ops += result.primary_read_ops;
    graph_bookkeeping_ms += result.graph_bookkeeping_ms;
    ji_candidates_seen += result.ji_candidates_seen;
    ji_bound_checks += result.ji_bound_checks;
    ji_corrections_evaluated += result.ji_corrections_evaluated;
    ji_corrections_pruned += result.ji_corrections_pruned;
    terminal_ji_corrections_evaluated +=
        result.terminal_ji_corrections_evaluated;
    if (!args.visited_csr.empty()) {
      traced_visited_ids.insert(traced_visited_ids.end(),
                                result.visited_ids.begin(),
                                result.visited_ids.end());
      visited_offsets.push_back(traced_visited_ids.size());
    }
    if (!args.retained_csr.empty()) {
      traced_retained_ids.insert(traced_retained_ids.end(),
                                 result.retained_ids.begin(),
                                 result.retained_ids.end());
      retained_offsets.push_back(traced_retained_ids.size());
    }
    if (!args.expanded_csr.empty()) {
      traced_expanded_ids.insert(traced_expanded_ids.end(),
                                 result.expanded_ids.begin(),
                                 result.expanded_ids.end());
      expanded_offsets.push_back(traced_expanded_ids.size());
    }
  }
  const double mean_query_latency_ms =
      std::accumulate(latency_ms.begin(), latency_ms.end(), 0.0) / query_count;
  std::sort(latency_ms.begin(), latency_ms.end());
  const auto percentile = [&](double fraction) {
    return latency_ms[static_cast<size_t>(fraction * (latency_ms.size() - 1))];
  };
  const std::filesystem::path prefix(args.out_prefix);
  if (!prefix.parent_path().empty())
    std::filesystem::create_directories(prefix.parent_path());
  WriteMatrix(args.out_prefix + ".ids.ibin", static_cast<uint32_t>(query_count),
              args.topk, ids);
  WriteMatrix(args.out_prefix + ".scores.fbin",
              static_cast<uint32_t>(query_count), args.topk, scores);
  if (!args.visited_csr.empty())
    WriteTraceCsr(args.visited_csr, args.query_start, visited_offsets,
                  traced_visited_ids);
  if (!args.retained_csr.empty())
    WriteTraceCsr(args.retained_csr, args.query_start, retained_offsets,
                  traced_retained_ids);
  if (!args.expanded_csr.empty())
    WriteTraceCsr(args.expanded_csr, args.query_start, expanded_offsets,
                  traced_expanded_ids);
  uint64_t routing_pair_count = 0;
  if (!args.routing_pairs_csv.empty()) {
    const std::filesystem::path path(args.routing_pairs_csv);
    if (!path.parent_path().empty())
      std::filesystem::create_directories(path.parent_path());
    std::ofstream routing(path);
    if (!routing)
      throw std::runtime_error("cannot create routing-pair CSV: " +
                               args.routing_pairs_csv);
    routing << "query_id,event_type,u_id,v_id,exact_margin\n"
            << std::setprecision(9);
    const std::array<const char *, 3> names{"exp", "queue",
        args.pdf_routing_tags ? "stop-boundary-proxy" : "stop"};
    for (uint64_t local = 0; local < query_count; ++local) {
      for (size_t source = 0; source < names.size(); ++source) {
        for (const RoutingPair &pair : results[local].routing_pairs[source]) {
          routing << args.query_start + local << ',' << names[source] << ','
                  << pair.u_id << ',' << pair.v_id << ','
                  << pair.exact_margin << '\n';
          ++routing_pair_count;
        }
      }
    }
  }
  const uint64_t graph_bytes = std::filesystem::file_size(args.graph);
  const uint64_t primary_model_bytes =
      std::filesystem::file_size(args.primary_model);
  const uint64_t primary_code_bytes =
      std::filesystem::file_size(args.primary_codes);
  const uint64_t primary_unique_bytes =
      args.primary_model == args.primary_codes
          ? primary_code_bytes
          : primary_model_bytes + primary_code_bytes;
  const uint64_t primary_fp_bytes = args.primary_kind == "fp32" ? primary_code_bytes : 0;
  const uint64_t base_bytes = primary_fp_bytes + (base ? base->file_bytes() : 0);
  const uint64_t hot_base_bytes = PrimaryHotVectorBytes(args) +
      (base && args.rerank_storage == "mmap" ? base->file_bytes() : 0);
  const auto primary_memory = PrimaryMemoryParts(args);
  const uint64_t side_bytes = ji ? std::filesystem::file_size(args.ji_side_codes) : 0;
  const uint64_t books_bytes = primary_memory.model +
      (ji ? std::filesystem::file_size(args.ji_model) : 0);
  // Owned result buffers are counted separately from physical node payloads.
  // Process/allocator/stack overhead is additionally covered by peak RSS and
  // by the 1B driver's hard RLIMIT_AS. This is a capacity ledger, not an RSS
  // attribution of sparsely touched memory-mapped pages.
  uint64_t runtime_bytes = results.capacity() * sizeof(QueryResult) +
      latency_ms.capacity() * sizeof(double) + ids.capacity() * sizeof(uint32_t) +
      scores.capacity() * sizeof(float) + (ji ? ji->runtime_cache_bytes() : 0);
  runtime_bytes += worker_workspace_bytes.load() + WorkspaceBytes(warmup_workspace) +
      8 + queries.rows() * static_cast<uint64_t>(queries.dim()) * sizeof(float);
  for (const auto &r : results)
    runtime_bytes += (r.ids.capacity() + r.visited_ids.capacity() +
        r.retained_ids.capacity() + r.expanded_ids.capacity()) * sizeof(uint32_t) +
        r.scores.capacity() * sizeof(float);
  runtime_bytes += (visited_offsets.capacity() + retained_offsets.capacity() +
      expanded_offsets.capacity()) * sizeof(uint64_t) +
      (traced_visited_ids.capacity() + traced_retained_ids.capacity() +
       traced_expanded_ids.capacity()) * sizeof(uint32_t);
  const uint64_t accounted_bytes = graph.memory_bytes() + primary_memory.codes +
      side_bytes + books_bytes + runtime_bytes + hot_base_bytes;
  const uint64_t peak_rss = PeakResidentBytes();
  const bool budget_pass = !args.memory_budget_bytes ||
      (accounted_bytes <= args.memory_budget_bytes && peak_rss <= args.memory_budget_bytes);
  // The quantitative record is written before the auxiliary JSON summary.
  std::ofstream metrics(args.out_prefix + ".metrics.csv");
  metrics << std::setprecision(17)
      << "queries,threads,beam,rerank_depth,qps,mean_service_latency_ms,p50_ms,p95_ms,p99_ms,"
         "visited,expanded,comparisons,reranked,rerank_io_ms,rerank_physical_bytes,"
         "graph_memory_bytes,primary_code_bytes,refiner_code_bytes,codebooks_transform_bytes,"
         "runtime_buffer_bytes,fp_hot_bytes,accounted_memory_bytes,peak_rss_bytes,memory_budget_bytes,budget_pass,"
         "primary_scoring_kernel,exact_rerank_scoring_kernel,query_preparation_included,"
         "fp32_storage,rerank_storage,primary_io_ms,primary_logical_bytes,primary_physical_bytes,primary_read_ops\n"
      << query_count << ',' << workers_count << ',' << args.beam << ',' << args.rerank_candidates << ','
      << query_count / std::max(seconds, 1e-12) << ',' << mean_query_latency_ms << ','
      << percentile(.50) << ',' << percentile(.95) << ',' << percentile(.99) << ','
      << static_cast<double>(visited) / query_count << ',' << static_cast<double>(expansions) / query_count << ','
      << static_cast<double>(comparisons) / query_count << ',' << static_cast<double>(reranked) / query_count << ','
      << rerank_io_ms / query_count << ',' << static_cast<double>(rerank_physical_bytes) / query_count << ','
      << graph.memory_bytes() << ',' << primary_memory.codes << ',' << side_bytes << ',' << books_bytes << ','
      << runtime_bytes << ',' << hot_base_bytes << ',' << accounted_bytes << ',' << peak_rss << ','
      << args.memory_budget_bytes << ',' << (budget_pass ? "true" : "false") << ','
      << PrimaryScoringKernel(primary) << ',' << (base ? base->scoring_kernel() : "none")
      << ",true," << (args.primary_kind == "fp32" ? args.fp32_storage : "none") << ','
      << (base ? args.rerank_storage : "none") << ',' << primary_io_ms / query_count << ','
      << static_cast<double>(primary_logical_bytes) / query_count << ','
      << static_cast<double>(primary_physical_bytes) / query_count << ','
      << static_cast<double>(primary_read_ops) / query_count << '\n';
  metrics.flush();
  if (!metrics) throw std::runtime_error("failed writing metrics CSV");
  metrics.close();
  std::ofstream summary(args.out_prefix + ".summary.json");
  summary << std::setprecision(10) << "{\n  \"mode\": \""
          << (args.terminal_ji
                  ? "primary-traversal-terminal-ji-"
                  : args.disable_ji
                  ? "primary-only-"
                  : (args.lazy_ji ? "ji-pure-lazy-upper-bound-"
                                  : "ji-pure-"))
          << (args.rerank_candidates ? "rerank" : "no-rerank") << "\",\n"
          << "  \"primary_kind\": \"" << args.primary_kind << "\",\n"
          << "  \"primary_scoring_kernel\": \"" << PrimaryScoringKernel(primary) << "\",\n"
          << "  \"exact_rerank_scoring_kernel\": \"" << (base ? base->scoring_kernel() : "none") << "\",\n"
          << "  \"query_preparation_included\": true,\n"
          << "  \"fp32_storage\": \"" << (args.primary_kind == "fp32" ? args.fp32_storage : "none") << "\",\n"
          << "  \"mean_primary_io_ms\": " << primary_io_ms / query_count << ",\n"
          << "  \"mean_primary_physical_bytes\": " << static_cast<double>(primary_physical_bytes) / query_count << ",\n"
          << "  \"mean_primary_read_ops\": " << static_cast<double>(primary_read_ops) / query_count << ",\n"
          << "  \"search_kernel_version\": \"traversal-fast-v1\",\n"
          << "  \"graph_layout\": \"" << (args.packed_graph ? "csr" : "adjacency-vectors") << "\",\n"
          << "  \"graph_memory_bytes\": " << graph.memory_bytes() << ",\n"
          << "  \"memory_budget_bytes\": " << args.memory_budget_bytes << ",\n"
          << "  \"memory_budget_pass\": " << (budget_pass ? "true" : "false") << ",\n"
          << "  \"stop_inv_status\": \"not-applicable: termination is queue exhaustion without a score comparison\",\n"
          << "  \"search_workspace_reused\": " << (optimized_workspace ? "true" : "false") << ",\n"
          << "  \"ji_traversal_upper_bound_built\": "
          << (ji && !args.terminal_ji && (args.lazy_ji || !optimized_workspace) ? "true" : "false") << ",\n"
          << "  \"ji_batch_add_inplace\": " << (ji && !args.terminal_ji && !args.lazy_ji ? "true" : "false") << ",\n"
          << "  \"ji_query_space\": \"" << args.ji_query_space << "\",\n"
          << "  \"ji_traversal_enabled\": "
          << (ji && !args.terminal_ji ? "true" : "false") << ",\n"
          << "  \"query_start\": " << args.query_start << ",\n"
          << "  \"queries\": " << query_count << ",\n"
          << "  \"topk\": " << args.topk << ",\n"
          << "  \"beam_width\": " << args.beam << ",\n"
          << "  \"rerank_enabled\": "
          << (args.rerank_candidates ? "true" : "false") << ",\n"
          << "  \"rerank_candidates_requested\": " << args.rerank_candidates
          << ",\n"
          << "  \"rerank_storage\": \"" << args.rerank_storage << "\",\n"
          << "  \"base_vectors_loaded\": "
          << (hot_base_bytes ? "true" : "false") << ",\n"
          << "  \"base_vectors_external\": " << (base || primary_fp_bytes ? "true" : "false")
          << ",\n"
          << "  \"storage_bytes\": {\n"
          << "    \"graph\": " << graph_bytes << ",\n"
          << "    \"primary_model_and_codes\": " << primary_unique_bytes
          << ",\n"
          << "    \"ji_model_and_side_codes\": "
          << (ji ? ji->storage_bytes() : 0) << ",\n"
          << "    \"ji_runtime_cache\": "
          << (ji ? ji->runtime_cache_bytes() : 0) << ",\n"
          << "    \"base_vectors\": " << base_bytes << ",\n"
          << "    \"base_vectors_hot\": " << hot_base_bytes << ",\n"
          << "    \"base_vectors_cold\": "
          << (base_bytes - hot_base_bytes) << ",\n"
          << "    \"hot_index_total\": "
          << graph.memory_bytes() + primary_memory.codes + primary_memory.model +
                 (ji ? ji->storage_bytes() + ji->runtime_cache_bytes() : 0) +
                 hot_base_bytes
          << "\n  },\n"
          << "  \"peak_rss_bytes\": " << PeakResidentBytes() << ",\n"
          << "  \"threads\": " << workers_count << ",\n"
          << "  \"warmup_queries\": " << warmup << ",\n"
          << "  \"seconds\": " << seconds << ",\n"
          << "  \"qps\": " << query_count / std::max(seconds, 1e-12) << ",\n"
          << "  \"mean_latency_ms\": " << 1000.0 * seconds / query_count
          << ",\n"
          << "  \"amortized_wall_ms_per_query\": " << 1000.0 * seconds / query_count << ",\n"
          << "  \"mean_query_latency_ms\": " << mean_query_latency_ms << ",\n"
          << "  \"p50_query_latency_ms\": " << percentile(0.50) << ",\n"
          << "  \"p95_query_latency_ms\": " << percentile(0.95) << ",\n"
          << "  \"p99_query_latency_ms\": " << percentile(0.99) << ",\n"
          << "  \"ji_online_format_version\": "
          << (ji ? ji->format_version() : 0) << ",\n"
          << "  \"ji_declared_groups\": " << (ji ? ji->groups() : 0)
          << ",\n"
          << "  \"ji_active_groups\": " << (ji ? ji->active_groups() : 0)
          << ",\n"
          << "  \"ji_score_groups\": " << (ji ? ji->score_groups() : 0)
          << ",\n"
          << "  \"trajectory_trace\": {\n"
          << "    \"visited_csr\": "
          << (args.visited_csr.empty() ? "null" : "\"" + args.visited_csr + "\"")
          << ",\n"
          << "    \"retained_csr\": "
          << (args.retained_csr.empty()
                  ? "null"
                  : "\"" + args.retained_csr + "\"")
          << ",\n"
          << "    \"expanded_csr\": "
          << (args.expanded_csr.empty() ? "null" : "\"" + args.expanded_csr + "\"")
          << ",\n    \"routing_pairs_csv\": "
          << (args.routing_pairs_csv.empty()
                  ? "null"
                  : "\"" + args.routing_pairs_csv + "\"")
          << ",\n    \"routing_pairs\": " << routing_pair_count
          << "\n  },\n"
          << "  \"lazy_ji\": {\n"
          << "    \"enabled\": " << (args.lazy_ji ? "true" : "false")
          << ",\n"
          << "    \"bound\": \"primary_score + upward-rounded(mean_bias + sum_group_max_lut)\",\n"
          << "    \"strict_prune_rule\": \"upper_score < current_worst_beam_score\",\n"
          << "    \"candidates_seen\": " << ji_candidates_seen << ",\n"
          << "    \"bound_checks\": " << ji_bound_checks << ",\n"
          << "    \"corrections_evaluated\": "
          << ji_corrections_evaluated << ",\n"
          << "    \"corrections_pruned\": " << ji_corrections_pruned
          << ",\n"
          << "    \"prune_rate_all_candidates\": "
          << static_cast<double>(ji_corrections_pruned) /
                 std::max<uint64_t>(ji_candidates_seen, 1)
          << ",\n"
          << "    \"prune_rate_checked_candidates\": "
          << static_cast<double>(ji_corrections_pruned) /
                 std::max<uint64_t>(ji_bound_checks, 1)
          << "\n  },\n"
          << "  \"terminal_ji\": {\n"
          << "    \"implementation\": \"" << args.terminal_implementation << "\",\n"
          << "    \"workspace_reused\": " << (optimized_terminal ? "true" : "false") << ",\n"
          << "    \"upper_bound_built\": " << (args.terminal_ji && !optimized_terminal ? "true" : "false") << ",\n"
          << "    \"selection\": \"" << (optimized_terminal ? "required-prefix-only" : "full-sort") << "\",\n"
          << "    \"mean_selected_candidates\": " << static_cast<double>(terminal_selected) / query_count << ",\n"
          << "    \"kernel\": \"" << (ji ? ji->scoring_kernel() : "none") << "\",\n"
          << "    \"primary_scores_reused\": " << (args.terminal_ji ? "true" : "false") << ",\n"
          << "    \"primary_terminal_rescores\": 0,\n"
          << "    \"query_lut_preparation_included_in_timing\": true,\n"
          << "    \"query_space\": \"" << args.ji_query_space << "\",\n"
          << "    \"enabled\": " << (args.terminal_ji ? "true" : "false")
          << ",\n"
          << "    \"corrections_evaluated\": "
          << terminal_ji_corrections_evaluated << ",\n"
          << "    \"mean_corrections_evaluated\": "
          << static_cast<double>(terminal_ji_corrections_evaluated) /
                 query_count
          << ",\n"
          << "    \"mean_total_ms\": " << terminal_ji_ms / query_count
          << ",\n"
          << "    \"mean_prepare_ms\": "
          << terminal_ji_prepare_ms / query_count << ",\n"
          << "    \"mean_score_ms\": "
          << terminal_ji_score_ms / query_count << "\n  },\n"
          << "  \"terminal_profile\": {\n"
          << "    \"enabled\": " << (args.profile_terminal ? "true" : "false") << ",\n"
          << "    \"mean_projection_ms\": " << terminal_projection_ms / query_count << ",\n"
          << "    \"mean_lut_ms\": " << terminal_lut_ms / query_count << ",\n"
          << "    \"mean_upper_bound_ms\": " << terminal_bound_ms / query_count << ",\n"
          << "    \"mean_candidate_buffer_ms\": " << terminal_buffer_ms / query_count << ",\n"
          << "    \"mean_lookup_and_add_ms\": " << terminal_lookup_ms / query_count << ",\n"
          << "    \"mean_topk_selection_ms\": " << terminal_selection_ms / query_count << ",\n"
          << "    \"note\": \"Instrumented runs are for attribution, not headline QPS. Lookup includes code reads and adding cached primary scores.\"\n  },\n"
          << "  \"component_profile\": {\n"
          << "    \"enabled\": "
          << (args.profile_components ? "true" : "false") << ",\n"
          << "    \"mean_primary_prepare_ms\": "
          << primary_prepare_ms / query_count << ",\n"
          << "    \"mean_ji_prepare_ms\": " << ji_prepare_ms / query_count
          << ",\n"
          << "    \"mean_primary_scoring_ms\": "
          << primary_score_ms / query_count << ",\n"
          << "    \"mean_ji_correction_ms\": " << ji_score_ms / query_count
          << ",\n    \"mean_terminal_ji_ms\": "
          << terminal_ji_ms / query_count
          << ",\n    \"mean_terminal_ji_prepare_ms\": "
          << terminal_ji_prepare_ms / query_count
          << ",\n    \"mean_terminal_ji_score_ms\": "
          << terminal_ji_score_ms / query_count << ",\n"
          << "    \"mean_exact_rerank_ms\": " << rerank_ms / query_count
          << ",\n    \"mean_exact_rerank_io_ms\": "
          << rerank_io_ms / query_count
          << ",\n    \"mean_exact_rerank_compute_ms\": "
          << rerank_compute_ms / query_count
          << ",\n"
          << "    \"mean_graph_bookkeeping_ms\": "
          << graph_bookkeeping_ms / query_count << "\n"
          << "  },\n"
          << "  \"mean_comparisons\": "
          << static_cast<double>(comparisons) / query_count << ",\n"
          << "  \"mean_expansions\": "
          << static_cast<double>(expansions) / query_count << ",\n"
          << "  \"mean_visited_nodes\": "
          << static_cast<double>(visited) / query_count << ",\n"
          << "  \"mean_final_candidate_pool\": "
          << static_cast<double>(pools) / query_count << ",\n"
          << "  \"mean_reranked_candidates\": "
          << static_cast<double>(reranked) / query_count << ",\n"
          << "  \"mean_exact_rerank_comparisons\": "
          << static_cast<double>(reranked) / query_count << ",\n"
          << "  \"mean_exact_rerank_ms\": " << rerank_ms / query_count
          << ",\n"
          << "  \"mean_exact_rerank_logical_bytes\": "
          << static_cast<double>(rerank_logical_bytes) / query_count << ",\n"
          << "  \"mean_exact_rerank_physical_bytes\": "
          << static_cast<double>(rerank_physical_bytes) / query_count << ",\n"
          << "  \"mean_exact_rerank_read_ops\": "
          << static_cast<double>(rerank_read_ops) / query_count << ",\n"
          << "  \"mean_exact_rerank_io_ms\": "
          << rerank_io_ms / query_count << ",\n"
          << "  \"mean_exact_rerank_compute_ms\": "
          << rerank_compute_ms / query_count
          << "\n}\n";
  std::cout << "[done] primary=" << args.primary_kind
            << " rerank=" << (args.rerank_candidates ? "yes" : "no")
            << " beam=" << args.beam << " qps=" << query_count / seconds
            << '\n';
  return 0;
}

Arguments TimingPoint(const Arguments &args, const std::string &line,
                      std::unordered_set<std::string> &prefixes) {
  const auto fields = CsvFields(line);
  if (fields.size() != 3) throw std::runtime_error("invalid timing plan row");
  Arguments point = args;
  size_t beam_end = 0, depth_end = 0;
  const auto beam = std::stoull(fields[0], &beam_end);
  const auto depth = std::stoull(fields[1], &depth_end);
  if (beam_end != fields[0].size() || depth_end != fields[1].size() ||
      beam > UINT32_MAX || depth > UINT32_MAX)
    throw std::runtime_error("timing integer outside uint32 range");
  point.beam = static_cast<uint32_t>(beam);
  point.rerank_candidates = static_cast<uint32_t>(depth);
  point.out_prefix = fields[2];
  if (point.beam < args.topk || point.rerank_candidates > point.beam ||
      (point.rerank_candidates && point.rerank_candidates < args.topk) ||
      point.out_prefix.empty() || !prefixes.insert(point.out_prefix).second)
    throw std::runtime_error("invalid or duplicate timing point");
  if (args.stream_plan && bool(point.rerank_candidates) != bool(args.rerank_candidates))
    throw std::runtime_error("stream point changes the configured rerank IO mode");
  return point;
}

template <typename Primary>
int Run(const Arguments &args, const Primary &primary) {
  if (args.stream_plan && !args.batch_plan.empty())
    throw std::invalid_argument("stream-plan and batch-plan are mutually exclusive");
  std::vector<Arguments> points;
  std::unordered_set<std::string> prefixes;
  if (args.stream_plan || !args.batch_plan.empty()) {
    if (!args.routing_pairs_csv.empty() || !args.visited_csr.empty() ||
        !args.retained_csr.empty() || !args.expanded_csr.empty())
      throw std::invalid_argument("plan timing cannot share trajectory output paths");
  }
  if (!args.stream_plan && args.batch_plan.empty()) points.push_back(args);
  else if (!args.stream_plan) {
    std::ifstream plan(args.batch_plan);
    std::string line;
    if (!std::getline(plan, line) || CsvFields(line) !=
        std::vector<std::string>{"beam", "rerank_candidates", "out_prefix"})
      throw std::runtime_error("batch plan header must be beam,rerank_candidates,out_prefix");
    while (std::getline(plan, line)) {
      if (line.empty()) continue;
      points.push_back(TimingPoint(args, line, prefixes));
    }
    if (points.empty()) throw std::runtime_error("empty batch plan");
  }
  const ours::MappedFbin queries(args.queries);
  std::unique_ptr<ours::JiOnlineIndex> ji;
  if (!args.disable_ji)
    ji = std::make_unique<ours::JiOnlineIndex>(
        ours::JiOnlineIndex::Load(args.ji_model, args.ji_side_codes));
  const bool needs_base = (args.stream_plan && args.rerank_candidates != 0) || std::any_of(points.begin(), points.end(),
      [](const Arguments &p) { return p.rerank_candidates != 0; });
  std::unique_ptr<ours::ExactVectorStore> base;
  if (needs_base) base = ours::OpenExactVectorStore(args.base, args.rerank_storage);
  const auto memory = PrimaryMemoryParts(args);
  const uint64_t graph_file = std::filesystem::file_size(args.graph);
  if (graph_file < 8 + 4 * primary.nodes()) throw std::runtime_error("truncated graph header/degrees");
  const uint64_t graph_estimate = args.packed_graph ? graph_file + 4 * primary.nodes()
      : graph_file + 20 * primary.nodes() - 8;
  const uint64_t fixed = graph_estimate + memory.codes + memory.model + PrimaryHotVectorBytes(args) +
      (ji ? ji->storage_bytes() + ji->runtime_cache_bytes() : 0) +
      (base && args.rerank_storage == "mmap" ? base->file_bytes() : 0);
  if (args.memory_budget_bytes && fixed > args.memory_budget_bytes)
    throw std::runtime_error("memory budget admission failed: index capacity " +
        std::to_string(fixed) + " exceeds " + std::to_string(args.memory_budget_bytes));
  const ours::RoarGraph graph = args.packed_graph ? ours::RoarGraph::LoadPacked(args.graph)
                                               : ours::RoarGraph::Load(args.graph);
  const std::unique_ptr<ours::ExactVectorStore> no_base;
  if (args.stream_plan) {
    std::string line;
    if (!std::getline(std::cin, line) || CsvFields(line) !=
        std::vector<std::string>{"beam", "rerank_candidates", "out_prefix"})
      throw std::runtime_error("stream header must be beam,rerank_candidates,out_prefix");
    std::cout << "[stream ready] index loaded once\n" << std::flush;
    size_t count = 0;
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      const Arguments point = TimingPoint(args, line, prefixes);
      RunLoaded(point, primary, queries, graph, ji, point.rerank_candidates ? base : no_base);
      // RunLoaded has closed all result files; this marker lets the controller
      // distinguish incomplete writes from a completed invalid result.
      std::ofstream complete(point.out_prefix + ".complete");
      complete << "done\n";
      complete.close();
      if (!complete) throw std::runtime_error("cannot write timing completion marker");
      std::cout << "[stream " << ++count << "] CSV=" << point.out_prefix
                << ".metrics.csv\n" << std::flush;
    }
    return 0;
  }
  for (size_t i = 0; i < points.size(); ++i) {
    RunLoaded(points[i], primary, queries, graph, ji,
              points[i].rerank_candidates ? base : no_base);
    std::cout << "[batch " << i + 1 << '/' << points.size() << "] beam="
              << points[i].beam << " L=" << points[i].rerank_candidates
              << " CSV=" << points[i].out_prefix << ".metrics.csv\n" << std::flush;
  }
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Arguments args = ParseArguments(argc, argv);
    if (args.primary_kind == "qvrr") {
      QvrrPrimary primary(args.primary_model,args.primary_codes);
      return Run(args,primary);
    }
    if (args.primary_kind == "fp32") {
      Fp32Primary primary(args.primary_codes, args.fp32_storage);
      return Run(args, primary);
    }
    if (args.primary_kind == "extended-rabitq") {
      RabitqPrimary primary(args.primary_codes, args.rabitq_config);
      return Run(args, primary);
    }
    if (args.primary_kind == "saq") {
      SaqPrimary primary(args.primary_model, args.primary_codes);
      return Run(args, primary);
    }
    if (args.primary_kind == "opq-pq" || args.primary_kind == "qaq" ||
        args.primary_kind == "leanvec-ood") {
      GenericPrimary primary(args.primary_model, args.primary_codes);
      return Run(args, primary);
    }
    throw std::invalid_argument(
        "unsupported primary-kind");
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    Usage(argv[0]);
    return 1;
  }
}
